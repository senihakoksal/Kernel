"""The round loop.

Load config, run N rounds. Each round: artists produce concepts in parallel,
then critics evaluate them in parallel (seeing this round's concepts + the full
prior feed). Every Record is appended to the shared feed and written to a
per-run .jsonl log. When the run completes, the analysis (analyze.py) and the
archive page (report.py) are refreshed automatically — one command does it all.

Usage:
    uv run python run.py --rounds 2
    uv run python run.py --rounds 2 --no-analyze   # just the run, skip post-processing
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agents import MODEL, Agent
from schema import Record

# Defaults — overridable on the command line. Never hardcoded inside the loop.
DEFAULT_ROUNDS = 4
AGENTS_FILE = Path("agents.yaml")
LOG_DIR = Path("logs")

# Max Claude calls in flight at once. Large rosters can otherwise burst past
# the API's rate limits (starter tier: 50 requests/min and 8,000 output
# tokens/min; the bucket counts each call's max_tokens). Raise this if your
# tier allows more.
MAX_CONCURRENT = 3


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


def load_agents(client: AsyncAnthropic) -> list[Agent]:
    """Build the agent roster from agents.yaml."""
    spec = yaml.safe_load(AGENTS_FILE.read_text())
    return [
        Agent(a["name"], a["role"], a["disposition"], client)
        for a in spec["agents"]
    ]


def append_records(records: list[Record], feed: list[Record], log_path: Path) -> None:
    """Append records to the in-memory feed and to the .jsonl log on disk."""
    feed.extend(records)
    with log_path.open("a") as f:
        for r in records:
            f.write(json.dumps(r.model_dump()) + "\n")


async def run(rounds: int) -> Path:
    load_dotenv()  # reads ANTHROPIC_API_KEY into the environment
    # Extra retries ride out 429s (the SDK backs off and retries on its own).
    client = AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    agents = load_agents(client)
    artists = [a for a in agents if a.role == "artist"]
    critics = [a for a in agents if a.role == "critic"]

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    feed: list[Record] = []  # the shared, append-only history
    print(f"Running {rounds} rounds with {len(artists)} artists + "
          f"{len(critics)} critics on {MODEL}. Logging to {log_path}")

    for round_idx in range(rounds):
        # 1) Artists each invent one concept, in parallel. They see the feed.
        new_concepts = await asyncio.gather(
            *(throttled(sem, a.act(round_idx, feed)) for a in artists)
        )
        append_records(list(new_concepts), feed, log_path)

        # 2) Each critic writes one critique per concept — every
        #    (critic, concept) pair, all in parallel. The feed already
        #    includes this round's concepts, so critics have full context.
        evaluations = await asyncio.gather(
            *(throttled(sem, c.act(round_idx, feed, concept))
              for c in critics for concept in new_concepts)
        )
        append_records(list(evaluations), feed, log_path)

        print(f"  round {round_idx}: {len(new_concepts)} concepts, "
              f"{len(evaluations)} evaluations")

    print(f"Done. {len(feed)} records in {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the observation kernel.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"number of rounds (default {DEFAULT_ROUNDS})")
    parser.add_argument("--no-analyze", action="store_true",
                        help="skip the automatic analyze.py + report.py after the run")
    args = parser.parse_args()
    log_path = asyncio.run(run(args.rounds))

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
