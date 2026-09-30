"""Recompute the pinned pipeline outputs for the regression fixture.

Run this deliberately, then READ THE DIFF. A change here means the numbers the
project reports have moved — that is either the point of the change you just
made, or a bug. It is not a formality.

    uv run python tests/fixtures/regenerate_expected.py
"""
import json
import pathlib

import analyze
import spacy
from sentence_transformers import SentenceTransformer

HERE = pathlib.Path(__file__).parent
LOG = HERE / "regression_run.jsonl"


def measure() -> dict:
    # The roster is pinned with the fixture: prior_vocabulary() reads the live
    # agents.yaml, so without this every roster edit would move these numbers.
    analyze.AGENTS_FILE = HERE / "regression_agents.yaml"
    nlp = spacy.load(analyze.SPACY_MODEL)
    model = SentenceTransformer(analyze.EMBED_MODEL)

    evals = analyze.critic_evaluations(LOG)
    window, _ = analyze.feed_window_for(LOG)
    occ, seeds, n_sub = analyze.descriptor_pipeline(evals, nlp, model)
    prop_df, propagated = analyze.find_propagation(occ)
    vocab = analyze.vocabulary_convergence(occ)
    rate = analyze.adoption_rate(occ, window)
    spread = analyze.score_spread(analyze.score_trajectories(evals))

    return {
        "feed_window": window,
        "n_critiques": len(evals),
        "n_descriptors": int(occ["descriptor"].nunique()),
        "n_clusters": int(occ["cluster"].nunique()),
        "n_seed_terms": len(seeds),
        "n_subtracted": int(n_sub),
        "n_propagated": len(propagated),
        "propagated_sample": sorted(propagated)[:5],
        "overlap": [{"round": int(r), "dice": round(float(d), 6)}
                    for r, d in zip(vocab["round"], vocab["dice"])],
        "adoption_rate": [{"round": int(r.round), "uses": int(r.uses),
                           "opportunities": int(r.opportunities),
                           "rate": round(float(r.rate), 6)}
                          for r in rate.itertuples()],
        "score_spread": [{"round": int(r), "spread": round(float(v), 6)}
                         for r, v in zip(spread["round"], spread["spread"])],
    }


if __name__ == "__main__":
    out = HERE / "expected.json"
    new = measure()
    if out.exists():
        old = json.loads(out.read_text())
        changed = [k for k in new if old.get(k) != new[k]]
        print("changed keys:" if changed else "no change", changed or "")
    out.write_text(json.dumps(new, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in new.items()
                      if not isinstance(v, list)}, indent=2))
