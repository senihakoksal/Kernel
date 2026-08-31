"""Control condition: replay a treatment run's artworks, re-run the critics in
isolation.

The "treatment" is an ordinary run (run.py), where each critic sees the whole
studio history — every artwork AND every other critic's critiques. Convergence
there could come from critics reading each other (the effect we care about), or
just from all of them judging the same evolving artworks, or shared model
priors. To isolate the first channel we hold the other two fixed:

  - REPLAY: the artworks are not regenerated. We feed the critics the exact same
    concepts the treatment produced, round by round, so the stimulus is
    identical across conditions.
  - ISOLATE: each critic's feed contains only the artworks and its OWN prior
    critiques — never another critic's. The single variable removed versus the
    treatment is the peer-critique channel.

So treatment-minus-control on the same artworks isolates direct critic-to-critic
influence. It also gives the propagation metric a null baseline: an isolated
critic cannot have read the coiner of a descriptor, so anything the metric still
flags here is independent re-coinage (a false positive).

Run analyze.py on the resulting control_*.jsonl to get its numbers, then compare
them against the treatment run's.

Usage:
    uv run python control.py                  # newest run_*.jsonl as treatment
    uv run python control.py logs/run_X.jsonl
"""

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Reuse the run machinery verbatim so the control behaves identically except for
# the feed each critic sees.
from run import LOG_DIR, MAX_CONCURRENT, append_records, load_agents, throttled
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
                  rounds: list[int], upto: int, critic: str) -> list[Record]:
    """The feed one critic sees in round `upto`: all artworks so far + ONLY this
    critic's own critiques from earlier rounds. No peer critiques, and (like the
    treatment, where critics act in parallel) no critiques from the current
    round — own_critiques is filled in only after each round completes.
    """
    feed: list[Record] = []
    for r in rounds:
        if r > upto:
            break
        feed.extend(concepts_by_round[r])
        feed.extend(rec for rec in own_critiques[critic] if rec.round == r)
    return feed


async def run_control(treatment_path: Path) -> Path:
    load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    concepts_by_round, rounds, treat_critics = load_treatment(treatment_path)
    if not rounds:
        sys.exit(f"{treatment_path} has no critiqued rounds to replay.")

    # Use the same critic population the treatment used, with dispositions from
    # the current agents.yaml. Warn if the roster has drifted since.
    critics_by_name = {a.name: a for a in load_agents(client) if a.role == "critic"}
    critics = [critics_by_name[n] for n in treat_critics if n in critics_by_name]
    missing = [n for n in treat_critics if n not in critics_by_name]
    if missing:
        print(f"WARNING: treatment critics not in agents.yaml, skipped: {missing}")
    if not critics:
        sys.exit("None of the treatment's critics exist in agents.yaml.")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"control_{treatment_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Control of {treatment_path.name}: {len(critics)} isolated critics "
          f"replaying rounds {rounds}. Logging to {log_path}")

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
            feed = isolated_feed(concepts_by_round, own_critiques, rounds, r, critic.name)
            tasks.extend(throttled(sem, critic.act(r, feed, concept)) for concept in concepts)
        new_critiques = await asyncio.gather(*tasks)
        append_records(list(new_critiques), written, log_path)

        for rec in new_critiques:                    # visible to self next round
            own_critiques[rec.agent].append(rec)
        print(f"  round {r}: {len(concepts)} artworks, {len(new_critiques)} isolated critiques")

    print(f"Done. {len(written)} records in {log_path}")
    print(f"Analyze with: uv run python analyze.py {log_path}")
    return log_path


def main() -> None:
    if len(sys.argv) > 1:
        treatment = Path(sys.argv[1])
    else:
        logs = sorted(LOG_DIR.glob("run_*.jsonl"))
        if not logs:
            sys.exit("No run_*.jsonl logs found to use as treatment.")
        treatment = logs[-1]
    asyncio.run(run_control(treatment))


if __name__ == "__main__":
    main()
