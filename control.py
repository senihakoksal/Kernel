"""Control condition: replay a treatment run's artworks, re-run the critics in
isolation.

The "treatment" is an ordinary run (run.py), where each critic sees the shared
studio history — every artwork AND every other critic's recent critiques, the
latter bounded by the rolling feed window. Convergence
there could come from critics reading each other (the effect we care about), or
just from all of them judging the same evolving artworks, or shared model
priors. To isolate the first channel we hold the other two fixed:

  - REPLAY: the artworks are not regenerated. We feed the critics the exact same
    concepts the treatment produced, round by round, so the stimulus is
    identical across conditions.
  - ISOLATE: each critic's feed contains only the artworks and its OWN prior
    critiques — never another critic's. The single variable removed versus the
    treatment is the peer-critique channel.
  - SAME WINDOW: critiques age out of both arms after the same number of
    rounds, using the same rule (agents.evaluation_round_bounds). A window
    mismatch is a hard error, not a warning — see check_feed_window.

So treatment-minus-control on the same artworks isolates direct critic-to-critic
influence. It also gives the propagation metric a null baseline: an isolated
critic cannot have read the coiner of a descriptor, so anything the metric still
flags here is independent re-coinage (a false positive).

Run analyze.py on the resulting control_*.jsonl to get its numbers, then compare
them against the treatment run's.

Usage:
    uv run python control.py                  # newest run_*.jsonl as treatment
    uv run python control.py logs/run_X.jsonl
    uv run python control.py logs/run_X.jsonl --feed-window 5
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agents import FEED_WINDOW, UsageTally, evaluation_round_bounds
# Reuse the run machinery verbatim so the control behaves identically except for
# the feed each critic sees.
from run import (LOG_DIR, MAX_CONCURRENT, append_records, load_agents,
                 meta_path_for, read_run_meta, throttled, write_run_meta)
from schema import Record


def load_treatment(path: Path) -> tuple[dict[int, list[Record]], list[int], list[str]]:
    """Read a treatment log into (concepts_by_round, rounds, critic_names).

    `rounds` are only the rounds the treatment actually critiqued — replaying a
    round the treatment never evaluated (e.g. a half-finished final round) would
    add control data with no treatment counterpart to compare against.
    """
    concepts_by_round: dict[int, list[Record]] = defaultdict(list)
    critiqued_rounds: set[int] = set()
    critic_names: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = Record(**json.loads(line))
        if rec.kind == "concept":
            concepts_by_round[rec.round].append(rec)
        elif rec.kind == "evaluation":
            critiqued_rounds.add(rec.round)
            if rec.agent not in critic_names:
                critic_names.append(rec.agent)
    rounds = sorted(r for r in concepts_by_round if r in critiqued_rounds)
    return concepts_by_round, rounds, critic_names


def isolated_feed(concepts_by_round: dict[int, list[Record]],
                  own_critiques: dict[str, list[Record]],
                  rounds: list[int], upto: int, critic: str,
                  window: int = FEED_WINDOW) -> list[Record]:
    """The feed one critic sees in round `upto`: all artworks so far + ONLY this
    critic's own critiques, from rounds still inside the window.

    Two bounds, for two different reasons. No peer critiques — that is the
    manipulation. And critiques only from rounds [upto-window, upto) — that is
    the same rolling window the treatment applies, and it must match exactly or
    treatment-minus-control stops isolating the peer channel and starts
    measuring a difference in how much history each arm could see.

    The bound comes from agents.evaluation_round_bounds, the same rule
    agents.windowed_feed uses for the treatment, so there is one definition of
    the window and not two. It is applied inside this loop rather than by
    post-filtering the assembled feed, so the round range is visible where the
    feed is built.

    Artworks are never windowed: the control replays the treatment's stimulus
    and the replay is only valid if that stimulus is complete in both arms.
    """
    lo, hi = evaluation_round_bounds(upto, window)
    feed: list[Record] = []
    for r in rounds:
        if r > upto:
            break
        feed.extend(concepts_by_round[r])
        if lo <= r < hi:
            feed.extend(rec for rec in own_critiques[critic] if rec.round == r)
    return feed


def check_feed_window(treatment_path: Path, feed_window: int) -> dict | None:
    """Refuse to run a control whose window differs from its treatment's.

    Mismatched windows across arms invalidate the comparison silently — both
    runs complete, both produce plausible numbers, and the gap between them is
    partly an artefact of one arm remembering more. So it is a hard error.

    A treatment with no sidecar predates the sidecar and cannot be checked;
    that is a warning, not an error, because failing would lock out every
    existing log.
    """
    meta = read_run_meta(treatment_path)
    if meta is None:
        print(f"WARNING: no {meta_path_for(treatment_path).name} — cannot verify "
              f"the treatment's feed window. Proceeding with {feed_window}; if the "
              f"treatment ran unwindowed, this comparison is not valid.")
        return None
    if "feed_window" not in meta:
        print(f"WARNING: {meta_path_for(treatment_path).name} records no feed_window. "
              f"Proceeding with {feed_window}, unverified.")
        return meta
    if meta["feed_window"] != feed_window:
        sys.exit(
            f"Feed window mismatch — refusing to run.\n"
            f"  treatment {treatment_path.name} ran with feed_window="
            f"{meta['feed_window']}\n"
            f"  this control would use feed_window={feed_window}\n"
            f"Re-run with --feed-window {meta['feed_window']} to match, or use a "
            f"treatment that matches the window you want."
        )
    return meta


async def run_control(treatment_path: Path,
                      feed_window: int = FEED_WINDOW) -> Path:
    # The window check first, before the client or anything else: it is cheap,
    # it is a hard error, and failing on it should not be masked by a missing
    # API key surfacing earlier.
    treat_meta = check_feed_window(treatment_path, feed_window)

    load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tally = UsageTally()

    concepts_by_round, rounds, treat_critics = load_treatment(treatment_path)
    if not rounds:
        sys.exit(f"{treatment_path} has no critiqued rounds to replay.")

    # Use the same critic population the treatment used, with dispositions from
    # the current agents.yaml. Warn if the roster has drifted since.
    critics_by_name = {a.name: a for a in load_agents(client, tally)
                       if a.role == "critic"}
    critics = [critics_by_name[n] for n in treat_critics if n in critics_by_name]
    missing = [n for n in treat_critics if n not in critics_by_name]
    if missing:
        print(f"WARNING: treatment critics not in agents.yaml, skipped: {missing}")
    if not critics:
        sys.exit("None of the treatment's critics exist in agents.yaml.")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"control_{treatment_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    # Deliberately NOT `rounds`: in run.py's sidecar that key is the number of
    # rounds requested, and a control replays whatever the treatment critiqued.
    # Reusing the key for a different quantity would make the two sidecars look
    # comparable when they are not.
    write_run_meta(log_path, n_replayed_rounds=len(rounds), feed_window=feed_window,
                   replayed_rounds=rounds,
                   # No artists are invoked — their concepts are replayed from
                   # the treatment, not regenerated.
                   n_artists_invoked=0, n_critics=len(critics),
                   max_concurrent=MAX_CONCURRENT,
                   condition="control-isolated",
                   treatment_run=treatment_path.stem,
                   treatment_feed_window=(treat_meta or {}).get("feed_window"))
    print(f"Control of {treatment_path.name}: {len(critics)} isolated critics "
          f"replaying rounds {rounds}, feed window {feed_window}. "
          f"Logging to {log_path}")

    written: list[Record] = []                       # only for the final count
    own_critiques: dict[str, list[Record]] = defaultdict(list)

    for r in rounds:
        concepts = concepts_by_round[r]
        # Log the replayed artworks so the control file is self-contained.
        append_records(concepts, written, log_path)

        # Each critic critiques each artwork of this round, in parallel, from its
        # own isolated feed. Critics never see this round's critiques (own or
        # peer), matching the treatment's within-round parallelism.
        tasks = []
        for critic in critics:
            feed = isolated_feed(concepts_by_round, own_critiques, rounds, r,
                                 critic.name, feed_window)
            tasks.extend(throttled(sem, critic.act(r, feed, concept)) for concept in concepts)
        new_critiques = await asyncio.gather(*tasks)
        append_records(list(new_critiques), written, log_path)

        for rec in new_critiques:                    # visible to self next round
            own_critiques[rec.agent].append(rec)
        print(f"  round {r}: {len(concepts)} artworks, {len(new_critiques)} isolated critiques")

    print(f"Done. {len(written)} records in {log_path}")
    print(f"Metadata in {meta_path_for(log_path)}")
    print(f"Analyze with: uv run python analyze.py {log_path}")
    print()
    print(tally.report())
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a treatment run's artworks to isolated critics.")
    parser.add_argument("treatment", nargs="?", type=Path,
                        help="treatment log (default: the newest run_*.jsonl)")
    parser.add_argument("--feed-window", type=int, default=FEED_WINDOW,
                        help="rounds of critiques a critic can still see; must "
                             f"match the treatment's (default {FEED_WINDOW})")
    args = parser.parse_args()
    if args.feed_window < 0:
        parser.error("--feed-window must be >= 0")

    treatment = args.treatment
    if treatment is None:
        logs = sorted(LOG_DIR.glob("run_*.jsonl"))
        if not logs:
            sys.exit("No run_*.jsonl logs found to use as treatment.")
        treatment = logs[-1]
    asyncio.run(run_control(treatment, args.feed_window))


if __name__ == "__main__":
    main()
