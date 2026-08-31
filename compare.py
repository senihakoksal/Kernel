"""Overlay a treatment run against its isolated-critic control on one chart.

Computes the two convergence metrics — vocabulary overlap (avg pairwise Jaccard
of critics' descriptor sets per round) and score spread (std of critics' scores
per round) — for BOTH the treatment run and the control produced by control.py,
and plots each metric with both conditions on the same axes.

Descriptors from the two conditions are clustered ONCE, in a single pooled
vocabulary space, so the two overlap curves are measured against the same
cluster structure and the gap between them is not an artifact of one run having
been clustered more coarsely than the other.

The treatment lets critics read each other; the control replays the same
artworks but isolates each critic. So the gap between the two lines is the part
of convergence attributable to critics actually reading one another, with the
shared-stimulus and shared-prior channels held fixed.

Usage:
    uv run python compare.py logs/run_X.jsonl logs/control_run_X_Y.jsonl
    uv run python compare.py logs/control_run_X_Y.jsonl   # treatment inferred from name
"""

import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from analyze import (EMBED_MODEL, SPACY_MODEL, _early_late, critic_evaluations,
                     make_figure, pooled_descriptor_pipeline, score_spread,
                     score_trajectories, top_descriptor_usage,
                     vocabulary_convergence)

FIGURE_DIR = Path("figures")


def metrics_for_pair(treatment_path: Path, control_path: Path, nlp,
                     embed_model) -> tuple[dict, dict]:
    """Convergence metrics for both conditions, clustered in ONE shared space.

    The two conditions must be clustered together, not separately: independent
    clusterings give each run its own vocabulary structure, and the run with
    more descriptors can end up with coarser clusters, which inflates its
    pairwise Jaccard for purely mechanical reasons. Overlaying two such lines
    would plot the difference between two cluster geometries and read it as an
    effect of the manipulation.
    """
    evals = {"treatment": critic_evaluations(treatment_path),
             "control": critic_evaluations(control_path)}
    by_condition, _, _ = pooled_descriptor_pipeline(evals, nlp, embed_model)
    for name, path in (("treatment", treatment_path), ("control", control_path)):
        if by_condition[name].empty:
            sys.exit(f"{path}: no descriptors survived the pipeline.")

    def metrics(name: str) -> dict:
        occ = by_condition[name]
        return {"vocab": vocabulary_convergence(occ),
                "spread": score_spread(score_trajectories(evals[name])),
                "occ": occ}

    return metrics("treatment"), metrics("control")


def _infer_treatment(control_path: Path) -> Path:
    """control_<treatment-stem>_<timestamp>.jsonl -> logs/<treatment-stem>.jsonl."""
    stem = control_path.stem.removeprefix("control_")          # run_X_YYYYmmdd_HHMMSS
    parts = stem.split("_")
    treatment_stem = "_".join(parts[:-2])                       # drop the control timestamp
    return control_path.parent / f"{treatment_stem}.jsonl"


def _overlap_trend(vocab_df: pd.DataFrame) -> dict:
    """Early/late average pairwise overlap (as fractions) for one condition."""
    early, late, _ = _early_late(vocab_df.sort_values("round")["jaccard"].tolist())
    return {"early": round(early, 4), "late": round(late, 4)}


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if len(args) == 2:
        treatment_path, control_path = args
    elif len(args) == 1:
        control_path = args[0]
        treatment_path = _infer_treatment(control_path)
    else:
        sys.exit("Usage: compare.py <treatment.jsonl> <control.jsonl>  (or just the control)")
    if not treatment_path.exists():
        sys.exit(f"Treatment log not found: {treatment_path}")
    print(f"Treatment: {treatment_path.name}\nControl:   {control_path.name}")

    nlp = spacy.load(SPACY_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    treat, ctrl = metrics_for_pair(treatment_path, control_path, nlp, embed_model)
    t_usage, t_top = top_descriptor_usage(treat["occ"], top_n=10)
    c_usage, c_top = top_descriptor_usage(ctrl["occ"], top_n=10)

    treat_trend, ctrl_trend = _overlap_trend(treat["vocab"]), _overlap_trend(ctrl["vocab"])
    print(f"  vocabulary overlap  treatment {treat_trend['early']*100:.1f}% -> "
          f"{treat_trend['late']*100:.1f}% | control {ctrl_trend['early']*100:.1f}% -> "
          f"{ctrl_trend['late']*100:.1f}%")

    # The comparison figure is the same layout as the per-run analysis figure,
    # with the control overlaid on the two top panels. Keyed by the TREATMENT
    # run id so report.py can attach it to that run in the archive.
    FIGURE_DIR.mkdir(exist_ok=True)
    fig = make_figure(treat["vocab"], treat["spread"], t_usage, t_top,
                      control_vocab=ctrl["vocab"], control_spread=ctrl["spread"],
                      control_usage=c_usage, control_top=c_top)
    fig.write_html(FIGURE_DIR / f"compare_{treatment_path.stem}.html")
    summary = {
        "treatment_run_id": treatment_path.stem,
        "control_run_id": control_path.stem,
        "treatment": treat_trend,
        "control": ctrl_trend,
        # Both conditions were clustered together; the overlap figures above are
        # only comparable because of it.
        "shared_vocabulary_space": True,
        "n_clusters": int(pd.concat([treat["occ"], ctrl["occ"]])["cluster"].nunique()),
    }
    (FIGURE_DIR / f"compare_{treatment_path.stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote figures/compare_{treatment_path.stem}.html and .json")


if __name__ == "__main__":
    main()
