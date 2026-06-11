"""Analyze a run: did a coined descriptor propagate across critics, and did
critics' scores converge or split?

Pipeline:
  1. Read a run's .jsonl into a DataFrame.
  2. Extract descriptors (adjective-bearing noun phrases) from each critic
     evaluation with spaCy.
  3. Subtract the prior vocabulary: any descriptor already present in the
     shared system prompt or an agent's disposition (exact or
     embedding-similar) is removed from the candidate pool — this encodes
     "present in no starting prompt".
  4. Canonicalize the survivors into clusters by exact match + embedding
     similarity (sentence-transformers), so "ghostly light" and "spectral
     glow" can count as the same descriptor.
  5. A descriptor "propagated" if, after first appearing in one critic's
     writing, it (or a semantically-similar variant) later appears in a
     DIFFERENT critic's writing. Output a plotly figure of propagation over
     rounds + per-critic score trajectories.

Usage:
    uv run python analyze.py                 # newest run in logs/
    uv run python analyze.py logs/run_X.jsonl
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
import yaml
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer

from agents import SHARED_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Similarity threshold for treating two descriptors as "the same" idea.
# This is the key knob for the propagation metric — TUNE THIS.
# Higher = stricter (only near-identical phrases match); lower = looser.
SIMILARITY_THRESHOLD = 0.72

# Threshold for matching a candidate against the prior vocabulary (descriptors
# seeded by the system prompt / dispositions). Candidates at or above this
# similarity to any seed term are excluded — TUNE THIS independently if the
# subtraction is too aggressive or too lax.
PRIOR_VOCAB_THRESHOLD = 0.72
# ---------------------------------------------------------------------------

AGENTS_FILE = Path("agents.yaml")
FIGURE_DIR = Path("figures")
EMBED_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"


def load_records(path: Path) -> pd.DataFrame:
    """Read a .jsonl run log into a DataFrame."""
    return pd.read_json(path, lines=True)


def extract_descriptors(text: str, nlp) -> list[str]:
    """Extract candidate aesthetic descriptors (noun phrases) from text.

    Only noun phrases containing at least one adjective qualify ("spectral
    glow", "brutal flatness"): a bare common-noun phrase ("the piece",
    "score") is studio shop talk, and a lone adjective carries little meaning
    outside its phrase — both are dropped.
    """
    doc = nlp(text)
    descriptors: list[str] = []
    for chunk in doc.noun_chunks:
        # Drop leading determiners/pronouns ("a haunting glow" -> "haunting glow").
        tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON")]
        if not any(t.pos_ == "ADJ" for t in tokens):
            continue
        phrase = " ".join(t.text for t in tokens).strip().lower()
        if len(phrase) >= 3 and not all(t.is_stop for t in tokens):
            descriptors.append(phrase)
    return descriptors


def prior_vocabulary(nlp) -> list[str]:
    """Every descriptor seeded by the starting prompts.

    Runs the same extractor over the shared system prompt and each agent's
    disposition. NOTE: reads the *current* agents.py / agents.yaml — run logs
    don't store the prompts they were generated with, so re-analyzing an old
    run after editing prompts will subtract the new vocabulary, not the old.
    """
    texts = [SHARED_SYSTEM_PROMPT]
    spec = yaml.safe_load(AGENTS_FILE.read_text())
    texts += [a["disposition"] for a in spec["agents"]]
    seeds: set[str] = set()
    for text in texts:
        seeds.update(extract_descriptors(text, nlp))
    return sorted(seeds)


def subtract_prior_vocabulary(occ: pd.DataFrame, seeds: list[str],
                              model: SentenceTransformer,
                              threshold: float) -> pd.DataFrame:
    """Drop candidates already present in the starting prompts.

    This encodes the research question's "present in no starting prompt": a
    candidate is removed if it exactly matches a seed descriptor or sits at or
    above `threshold` cosine similarity to any of them.
    """
    if not seeds or occ.empty:
        return occ
    uniq = sorted(set(occ["descriptor"]))
    seed_emb = model.encode(seeds, normalize_embeddings=True)
    cand_emb = model.encode(uniq, normalize_embeddings=True)
    sims = cand_emb @ seed_emb.T  # cosine similarity, candidates x seeds
    seed_set = set(seeds)
    seeded = {d for d, row in zip(uniq, sims)
              if d in seed_set or float(row.max()) >= threshold}
    return occ[~occ["descriptor"].isin(seeded)].copy()


def canonicalize(descriptors: list[str], model: SentenceTransformer,
                 threshold: float) -> dict[str, str]:
    """Map each unique descriptor to a cluster label.

    Greedy single-pass clustering: each descriptor joins the first existing
    cluster whose representative is within `threshold` cosine similarity,
    otherwise it starts a new cluster (and labels it). Embeddings are normalized
    so a dot product is cosine similarity.
    """
    uniq = sorted(set(descriptors))
    if not uniq:
        return {}
    embeddings = model.encode(uniq, normalize_embeddings=True)

    cluster_reps: list[np.ndarray] = []  # representative embedding per cluster
    cluster_labels: list[str] = []       # label (first descriptor) per cluster
    assignment: dict[str, str] = {}

    for i, descriptor in enumerate(uniq):
        best_idx, best_sim = -1, 0.0
        for ci, rep in enumerate(cluster_reps):
            sim = float(np.dot(embeddings[i], rep))
            if sim > best_sim:
                best_idx, best_sim = ci, sim
        if best_idx >= 0 and best_sim >= threshold:
            assignment[descriptor] = cluster_labels[best_idx]
        else:
            cluster_reps.append(embeddings[i])
            cluster_labels.append(descriptor)
            assignment[descriptor] = descriptor
    return assignment


def build_occurrences(critic_evals: pd.DataFrame, nlp) -> pd.DataFrame:
    """Flatten critic evaluations into (round, critic, descriptor) rows."""
    rows = []
    for _, ev in critic_evals.iterrows():
        for descriptor in extract_descriptors(ev["content"], nlp):
            rows.append({
                "round": ev["round"],
                "critic": ev["agent"],
                "descriptor": descriptor,
            })
    return pd.DataFrame(rows)


def find_propagation(occ: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Identify clusters that propagated to a DIFFERENT critic.

    Returns (propagation_df, propagated_labels) where propagation_df has, per
    propagated cluster per round, the number of distinct critics using it.
    """
    propagated_labels: list[str] = []
    for label, grp in occ.groupby("cluster"):
        first_round = grp["round"].min()
        first_critic = grp.loc[grp["round"].idxmin(), "critic"]
        # Did a different critic pick it up at or after first appearance?
        adopted = grp[(grp["critic"] != first_critic) & (grp["round"] >= first_round)]
        if not adopted.empty:
            propagated_labels.append(label)

    prop = occ[occ["cluster"].isin(propagated_labels)]
    propagation_df = (
        prop.groupby(["cluster", "round"])["critic"]
        .nunique()
        .reset_index(name="n_critics")
    )
    return propagation_df, propagated_labels


def convergence_summary(scores_df: pd.DataFrame) -> dict | None:
    """Did critics' judgments converge or split?

    Measured as the spread (max - min) of critics' mean scores per round,
    comparing the first round to the last. Returns None if no round has two or
    more scoring critics. The 0.01 band treats tiny drifts as 'stable'.
    """
    spread = scores_df.groupby("round")["mean_score"].agg(lambda s: s.max() - s.min())
    n_critics = scores_df.groupby("round")["critic"].nunique()
    spread = spread[n_critics >= 2]
    if spread.empty:
        return None
    first, last = float(spread.iloc[0]), float(spread.iloc[-1])
    if last < first - 0.01:
        verdict = "converging"
    elif last > first + 0.01:
        verdict = "splitting"
    else:
        verdict = "stable"
    return {"first_spread": round(first, 3), "last_spread": round(last, 3),
            "verdict": verdict}


def score_trajectories(critic_evals: pd.DataFrame) -> pd.DataFrame:
    """Per-critic mean score per round (drops evals with no score)."""
    scored = critic_evals.dropna(subset=["score"])
    return (
        scored.groupby(["agent", "round"])["score"]
        .mean()
        .reset_index(name="mean_score")
        .rename(columns={"agent": "critic"})
    )


def make_figure(propagation_df: pd.DataFrame, scores_df: pd.DataFrame,
                propagated_labels: list[str]) -> go.Figure:
    """Two stacked panels: descriptor propagation, then score trajectories."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Descriptor propagation (distinct critics using a descriptor, per round)",
            "Per-critic score trajectories",
        ),
    )

    # Panel 1: one line per propagated descriptor cluster.
    if propagated_labels:
        for label in propagated_labels:
            sub = propagation_df[propagation_df["cluster"] == label].sort_values("round")
            fig.add_trace(
                go.Scatter(x=sub["round"], y=sub["n_critics"], mode="lines+markers",
                           name=f"“{label}”"),
                row=1, col=1,
            )
    else:
        fig.add_annotation(text="No descriptor propagated across critics.",
                           xref="x domain", yref="y domain", x=0.5, y=0.5,
                           showarrow=False, row=1, col=1)
    fig.update_yaxes(title_text="# critics", dtick=1, row=1, col=1)
    fig.update_xaxes(title_text="round", row=1, col=1)

    # Panel 2: one line per critic.
    for critic, sub in scores_df.groupby("critic"):
        sub = sub.sort_values("round")
        fig.add_trace(
            go.Scatter(x=sub["round"], y=sub["mean_score"], mode="lines+markers",
                       name=str(critic)),
            row=2, col=1,
        )
    fig.update_yaxes(title_text="mean score", row=2, col=1)
    fig.update_xaxes(title_text="round", row=2, col=1)

    fig.update_layout(height=800, title_text="Observation kernel — propagation & convergence")
    return fig


def main() -> None:
    # Pick the run to analyze: a path argument, or the newest log.
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        logs = sorted(Path("logs").glob("run_*.jsonl"))
        if not logs:
            sys.exit("No logs found in logs/. Run run.py first.")
        log_path = logs[-1]
    print(f"Analyzing {log_path}")

    df = load_records(log_path)
    critic_evals = df[(df["role"] == "critic") & (df["kind"] == "evaluation")].copy()
    if critic_evals.empty:
        sys.exit("No critic evaluations in this run.")

    nlp = spacy.load(SPACY_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    # 1) descriptors per (round, critic)
    occ = build_occurrences(critic_evals, nlp)
    if occ.empty:
        sys.exit("No descriptors extracted from critic evaluations.")

    # 2) subtract the prior vocabulary — only descriptors present in NO
    #    starting prompt can count as coined
    seeds = prior_vocabulary(nlp)
    n_before = occ["descriptor"].nunique()
    occ = subtract_prior_vocabulary(occ, seeds, embed_model, PRIOR_VOCAB_THRESHOLD)
    n_subtracted = n_before - occ["descriptor"].nunique()
    print(f"  prior vocabulary: {len(seeds)} seed descriptors; "
          f"subtracted {n_subtracted} of {n_before} candidates")
    if occ.empty:
        sys.exit("All descriptors were already in the starting prompts.")

    # 3) cluster the survivors (exact + embedding similarity), label each row
    assignment = canonicalize(occ["descriptor"].tolist(), embed_model, SIMILARITY_THRESHOLD)
    occ["cluster"] = occ["descriptor"].map(assignment)

    # 4) propagation + scores
    propagation_df, propagated_labels = find_propagation(occ)
    scores_df = score_trajectories(critic_evals)

    print(f"  {occ['cluster'].nunique()} descriptor clusters; "
          f"{len(propagated_labels)} propagated across critics: {propagated_labels}")

    # 5) outputs, keyed by run id so report.py can attach them to the run:
    #    the figure, plus a small JSON summary of what propagated.
    FIGURE_DIR.mkdir(exist_ok=True)
    fig = make_figure(propagation_df, scores_df, propagated_labels)
    fig_path = FIGURE_DIR / f"analysis_{log_path.stem}.html"
    fig.write_html(fig_path)
    summary = {
        "run_id": log_path.stem,
        "n_clusters": int(occ["cluster"].nunique()),
        "propagated": propagated_labels,
        "threshold": SIMILARITY_THRESHOLD,
        "prior_vocab_size": len(seeds),
        "prior_vocab_subtracted": int(n_subtracted),
        "convergence": convergence_summary(scores_df),
    }
    summary_path = FIGURE_DIR / f"analysis_{log_path.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {fig_path} and {summary_path}")


if __name__ == "__main__":
    main()
