# Observation Kernel

A small multi-agent "observation kernel." LLM **artist** agents produce short
artwork *concepts* (text prompts, never images). LLM **critic** agents evaluate
those concepts in writing. It runs over a fixed number of rounds.

All agents use a single model: Anthropic's Claude (`claude-sonnet-4-6`), set in
one constant — `MODEL` in [agents.py](agents.py).

## Research question

Does a new aesthetic descriptor coined by one critic — present in no starting
prompt — propagate to other critics over rounds, and do critics' judgments
converge or split?

## How it works

- **`run.py`** runs the round loop. Each round, all artists propose one concept
  each (in parallel via `asyncio.gather`), then all critics evaluate this
  round's concepts (also in parallel), seeing the full prior feed. Every action
  is logged as one JSON line to `logs/run_<timestamp>.jsonl`.
- **`analyze.py`** reads a run log, extracts descriptors (spaCy noun phrases)
  from critic evaluations, clusters them by exact match + embedding similarity
  (sentence-transformers), detects when a descriptor first coined by one critic
  is later picked up by a *different* critic, and writes a plotly figure of
  descriptor propagation and per-critic score trajectories to `figures/`.

### Files

- [schema.py](schema.py) — `Record` (one logged action) and `ModelOutput` (what
  Claude returns).
- [agents.py](agents.py) — the `Agent` class, the shared system prompt, the
  feed formatter, and JSON parsing. `MODEL` lives here.
- [agents.yaml](agents.yaml) — the agent roster (name, role, disposition).
  **Dispositions are placeholders — write them yourself.**
- [run.py](run.py) — the round loop.
- [analyze.py](analyze.py) — descriptor-propagation and score analysis. The
  similarity threshold (`SIMILARITY_THRESHOLD`) is a marked constant to tune.
- [report.py](report.py) — generates a static HTML archive of all runs
  (`site/index.html`): when each run happened, the concepts made, and the
  critiques given. Stdlib only; no server.

## Setup

```sh
# 1. Install runtime + analysis dependencies.
uv sync --extra analysis

# 2. spaCy English model (needed by analyze.py).
uv run python -m spacy download en_core_web_sm

# 3. Provide your API key.
cp .env.example .env   # then edit .env and set ANTHROPIC_API_KEY
```

## Run

```sh
# A run (writes logs/run_<timestamp>.jsonl).
uv run python run.py --rounds 4

# Analyze the newest run (writes figures/analysis_<timestamp>.html).
uv run python analyze.py
# or point at a specific log:
uv run python analyze.py logs/run_20260609_120000.jsonl

# Regenerate the run archive page (open site/index.html in a browser).
uv run python report.py
```

### Smoke test

The default [agents.yaml](agents.yaml) is exactly 2 artists + 2 critics, so a
tiny end-to-end check is:

```sh
uv run python run.py --rounds 2
uv run python analyze.py
```

This is ~12 Claude calls. Once it works, write real dispositions and scale up
rounds and agents.

## Non-goals

No token economy, coins, costs, or earnings. No survival or elimination of
agents. No marketplace, prices, buying/selling, or collectors. No blockchain,
smart contracts, or NFTs. No prediction markets, governance, or attention
budgets. No image generation. No web server or UI framework (the run archive
is a generated static HTML page). No orchestration framework. No database.
Just one runnable program plus analysis and report scripts.
