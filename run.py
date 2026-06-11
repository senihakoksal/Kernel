"""The round loop.

Load config, run N rounds. Each round: artists produce concepts in parallel,
then critics evaluate them in parallel (seeing this round's concepts + the full
prior feed). Every Record is appended to the shared feed and written to a
per-run .jsonl log.

Usage:
    uv run python run.py --rounds 2
"""

import argparse
import asyncio
import json
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
    client = AsyncAnthropic()

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
            *(a.act(round_idx, feed) for a in artists)
        )
        append_records(list(new_concepts), feed, log_path)

        # 2) Each critic writes one critique per concept — every
        #    (critic, concept) pair, all in parallel. The feed already
        #    includes this round's concepts, so critics have full context.
        evaluations = await asyncio.gather(
            *(c.act(round_idx, feed, concept)
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
    args = parser.parse_args()
    asyncio.run(run(args.rounds))


if __name__ == "__main__":
    main()
