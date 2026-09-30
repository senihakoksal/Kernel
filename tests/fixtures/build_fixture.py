"""Carve a small, real subset out of a run log to pin the pipeline against.

Real text, not synthetic: the descriptor extractor works on spaCy noun phrases,
so invented filler produces no descriptors and would exercise nothing. Small
enough to commit and to run in a test: 2 artists, 3 critics, 5 rounds — enough
rounds that the feed window engages at round 4, which is where the adoption
rate's recency bound starts to bite.

    uv run python tests/fixtures/build_fixture.py
"""
import json
import pathlib
import shutil

SOURCE = pathlib.Path("logs/run_20260910_154044.jsonl")
HERE = pathlib.Path(__file__).parent
ROUNDS, N_ARTISTS, N_CRITICS = 5, 2, 3

records = [json.loads(l) for l in SOURCE.read_text().splitlines() if l.strip()]
artists = sorted({r["agent"] for r in records if r["role"] == "artist"})[:N_ARTISTS]
critics = sorted({r["agent"] for r in records if r["role"] == "critic"})[:N_CRITICS]

kept, keep_ids = [], set()
for r in records:
    if r["round"] >= ROUNDS:
        continue
    if r["kind"] == "concept" and r["agent"] in artists:
        keep_ids.add(r["concept_id"])
        kept.append(r)
    elif r["kind"] == "evaluation" and r["agent"] in critics and r["concept_id"] in keep_ids:
        kept.append(r)

out = HERE / "regression_run.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in kept))
# The window the fixture is measured against has to travel with it.
(HERE / "regression_run.meta.json").write_text(json.dumps(
    {"run_id": "regression_run", "model": "claude-sonnet-4-6", "rounds": ROUNDS,
     "feed_window": 3, "n_artists": N_ARTISTS, "n_critics": N_CRITICS,
     "note": "subset of run_20260910_154044 — see build_fixture.py"}, indent=2) + "\n")
# prior_vocabulary() reads the live roster, so pin that too or every roster
# edit moves the expected numbers.
shutil.copyfile("agents.yaml", HERE / "regression_agents.yaml")

print(f"{len(kept)} records -> {out} ({out.stat().st_size / 1024:.0f} KB)")
print(f"  artists: {artists}")
print(f"  critics: {critics}")
print(f"  rounds:  {sorted({r['round'] for r in kept})}")
