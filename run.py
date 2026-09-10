"""The round loop.

Load config, run N rounds. Each round: artists produce concepts in parallel,
then critics evaluate them in parallel, each seeing this round's concepts plus
a windowed view of the prior feed. Every Record is appended to the shared feed
and written to a per-run .jsonl log. When the run completes, the analysis
(analyze.py) and the archive page (report.py) are refreshed automatically — one
command does it all.

The log is append-only and complete; what agents SEE is narrower. At round r the
visible feed carries every concept plus critiques from the last FEED_WINDOW
completed rounds — see agents.windowed_feed. The window that produced a run is
recorded in logs/<run_stem>.meta.json, because it is not recoverable from the
log itself.

Usage:
    uv run python run.py --rounds 2
    uv run python run.py --rounds 2 --feed-window 5
    uv run python run.py --rounds 2 --no-analyze   # just the run, skip post-processing
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agents import FEED_WINDOW, MODEL, Agent, UsageTally, windowed_feed
from schema import Record

# Defaults — overridable on the command line. Never hardcoded inside the loop.
DEFAULT_ROUNDS = 4
AGENTS_FILE = Path("agents.yaml")
LOG_DIR = Path("logs")

# Max Claude calls in flight at once. Large rosters can otherwise burst past
# the API's rate limits (starter tier: 50 requests/min and 8,000 output
# tokens/min; the bucket counts each call's max_tokens, not what a call
# actually returns).
#
# That output bucket is the hard ceiling: 8,000 / max_tokens = 10 calls/min at
# max_tokens=800, whatever the concurrency. A 5-round 2x5 run measured ~3.8
# calls/min at MAX_CONCURRENT=3 — well under the ceiling, so the run was
# latency-bound rather than limit-bound, and widening the pipe buys real
# wall-clock. Past the ceiling nothing errors: requests just 429, the SDK backs
# off and retries, and throughput flattens while individual calls stall. If the
# console shows retry delays, come back down.
MAX_CONCURRENT = 6


async def throttled(sem: asyncio.Semaphore, coro):
    """Run an agent call under the shared concurrency cap.

    Also prints one progress line per completed call — the run is network-
    bound, so console output costs nothing.
    """
    async with sem:
        record = await coro
    if record.kind == "concept":
        print(f"    {record.agent} made “{record.title}”")
    else:
        print(f"    {record.agent} critiqued {record.concept_id} (score {record.score})")
    return record


def load_agents(client: AsyncAnthropic,
                tally: UsageTally | None = None) -> list[Agent]:
    """Build the agent roster from agents.yaml."""
    spec = yaml.safe_load(AGENTS_FILE.read_text())
    return [
        Agent(a["name"], a["role"], a["disposition"], client, tally)
        for a in spec["agents"]
    ]


# --- Run provenance -----------------------------------------------------------
# The window, the model and the roster all change what a log means, and none of
# them are recoverable from the log itself. They go in a sidecar rather than a
# header line: control.py and analyze.py both parse every line of the .jsonl as
# a Record, so a metadata line would break them.
def meta_path_for(log_path: Path) -> Path:
    """logs/run_X.jsonl -> logs/run_X.meta.json"""
    return log_path.with_suffix(".meta.json")


def agents_file_sha256(path: Path | None = None) -> str:
    """Hash of the roster a run used, so drift since is detectable."""
    return hashlib.sha256((path or AGENTS_FILE).read_bytes()).hexdigest()


def write_run_meta(log_path: Path, **fields) -> Path:
    """Write the sidecar. Callers add whatever else identifies their arm."""
    meta = {
        "run_id": log_path.stem,
        "model": MODEL,
        "agents_file": str(AGENTS_FILE),
        "agents_file_sha256": agents_file_sha256(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **fields,
    }
    meta_path = meta_path_for(log_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def read_run_meta(log_path: Path) -> dict | None:
    """The sidecar for a log, or None for logs written before it existed."""
    meta_path = meta_path_for(log_path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def append_records(records: list[Record], feed: list[Record], log_path: Path) -> None:
    """Append records to the in-memory feed and to the .jsonl log on disk."""
    feed.extend(records)
    with log_path.open("a") as f:
        for r in records:
            f.write(json.dumps(r.model_dump()) + "\n")


async def run(rounds: int, feed_window: int = FEED_WINDOW) -> Path:
    load_dotenv()  # reads ANTHROPIC_API_KEY into the environment
    # Extra retries ride out 429s (the SDK backs off and retries on its own).
    client = AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tally = UsageTally()

    agents = load_agents(client, tally)
    artists = [a for a in agents if a.role == "artist"]
    critics = [a for a in agents if a.role == "critic"]

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    write_run_meta(log_path, rounds=rounds, feed_window=feed_window,
                   n_artists=len(artists), n_critics=len(critics),
                   max_concurrent=MAX_CONCURRENT)

    feed: list[Record] = []  # the shared, append-only history
    print(f"Running {rounds} rounds with {len(artists)} artists + "
          f"{len(critics)} critics on {MODEL}, feed window {feed_window}. "
          f"Logging to {log_path}")

    for round_idx in range(rounds):
        # The feed is the full append-only history; what agents SEE is the
        # windowed view of it — all concepts, critiques from the last
        # `feed_window` completed rounds. Windowing happens here rather than in
        # Agent.act so the prompt blocks and their cache breakpoints are
        # untouched: act() still receives a plain list and renders it as-is.
        #
        # 1) Artists each invent one concept, in parallel. They see the feed.
        artist_feed = windowed_feed(feed, round_idx, feed_window)
        new_concepts = await asyncio.gather(
            *(throttled(sem, a.act(round_idx, artist_feed)) for a in artists)
        )
        append_records(list(new_concepts), feed, log_path)

        # 2) Each critic writes one critique per concept — every
        #    (critic, concept) pair, all in parallel. Re-window now that this
        #    round's concepts are in the feed; concepts are never windowed, so
        #    critics see all of them, and no critique from this round exists
        #    yet either way.
        critic_feed = windowed_feed(feed, round_idx, feed_window)
        log_size = len(feed)   # before this round's critiques land, so the
                               # visible/total ratio below compares like for like
        evaluations = await asyncio.gather(
            *(throttled(sem, c.act(round_idx, critic_feed, concept))
              for c in critics for concept in new_concepts)
        )
        append_records(list(evaluations), feed, log_path)

        print(f"  round {round_idx}: {len(new_concepts)} concepts, "
              f"{len(evaluations)} evaluations, "
              f"{len(critic_feed)}/{log_size} records visible to critics")

    print(f"Done. {len(feed)} records in {log_path}")
    print(f"Metadata in {meta_path_for(log_path)}")
    print()
    print(tally.report())
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the observation kernel.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"number of rounds (default {DEFAULT_ROUNDS})")
    parser.add_argument("--feed-window", type=int, default=FEED_WINDOW,
                        help="rounds of critiques an agent can still see "
                             f"(default {FEED_WINDOW}; 0 = concepts only)")
    parser.add_argument("--no-analyze", action="store_true",
                        help="skip the automatic analyze.py + report.py after the run")
    args = parser.parse_args()
    if args.feed_window < 0:
        parser.error("--feed-window must be >= 0")
    log_path = asyncio.run(run(args.rounds, args.feed_window))

    # Post-process: analyze this run and rebuild the archive page. Run as
    # subprocesses so run.py itself never imports the heavy NLP stack; the
    # run's data is already safely on disk if either step fails.
    if not args.no_analyze:
        print("\nAnalyzing...")
        subprocess.run([sys.executable, "analyze.py", str(log_path)], check=True)
        print("Rebuilding archive page...")
        subprocess.run([sys.executable, "report.py"], check=True)
        print("Open site/index.html to browse the run.")


if __name__ == "__main__":
    main()
