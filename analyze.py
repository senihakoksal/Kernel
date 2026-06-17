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
     DIFFERENT critic's writing.
  6. Output a two-panel plotly figure: vocabulary convergence between critics
     (avg pairwise Jaccard overlap of their descriptor sets per round) and
     score spread between critics (std of their scores per round).

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


def vocabulary_convergence(occ: pd.DataFrame) -> pd.DataFrame:
    """Average pairwise vocabulary overlap between critics, per round.

    For each round, take each critic's set of descriptor clusters and compute
    the Jaccard overlap |A∩B| / |A∪B| for every pair of critics, then average.
    Rising overlap means critics are increasingly reaching for the same
    descriptors — vocabulary convergence. Rounds with fewer than two critics
    (no pair to compare) are skipped.
    """
    rows = []
    for rnd, grp in occ.groupby("round"):
        sets = {c: set(g["cluster"]) for c, g in grp.groupby("critic")}
        critics = list(sets)
        if len(critics) < 2:
            continue
        pair_jaccard = []
        for i in range(len(critics)):
            for j in range(i + 1, len(critics)):
                a, b = sets[critics[i]], sets[critics[j]]
                union = a | b
                pair_jaccard.append(len(a & b) / len(union) if union else 0.0)
        rows.append({"round": int(rnd), "jaccard": sum(pair_jaccard) / len(pair_jaccard)})
    return pd.DataFrame(rows)


def score_spread(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Spread of critics' judgments per round = std of the critics' mean scores.

    Lower spread means the critics agree more closely on quality that round.
    """
    return (
        scores_df.groupby("round")["mean_score"]
        .std(ddof=0)
        .reset_index(name="spread")
        .dropna()
    )


def top_descriptor_usage(occ: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """The most-used descriptor clusters and how many critics used each per round.

    "Most-used" ranks clusters by total distinct-critic usage summed over all
    rounds. Returns (usage_df with columns cluster/round/n_critics for the top
    clusters, ordered top labels most-used first).
    """
    per = (occ.groupby(["cluster", "round"])["critic"]
              .nunique().reset_index(name="n_critics"))
    totals = per.groupby("cluster")["n_critics"].sum().sort_values(ascending=False)
    top = totals.head(top_n).index.tolist()
    return per[per["cluster"].isin(top)].copy(), top


def _early_late(values: list[float]) -> tuple[float, float, float]:
    """Mean of the first two and last two points, and the % change between them.

    With ≤2 points it falls back to first vs. last so the annotation still works.
    """
    if len(values) >= 4:
        early = sum(values[:2]) / 2
        late = sum(values[-2:]) / 2
    else:
        early, late = values[0], values[-1]
    pct = (late - early) / early * 100 if early else 0.0
    return early, late, pct


def make_figure(vocab_df: pd.DataFrame, spread_df: pd.DataFrame,
                usage_df: pd.DataFrame, top_labels: list[str]) -> go.Figure:
    """Two convergence panels on top, a top-descriptor usage heatmap below.

    Top-left: average pairwise Jaccard overlap of critics' descriptor sets per
    round (rising = converging vocabulary), with dashed early/late reference
    lines. Top-right: std of critics' scores per round (falling = agreeing more
    on quality). Bottom: the 10 most-used descriptors × round, colored by how
    many critics used each that round.
    """
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.16,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Vocabulary convergence between critics",
            "Score spread between critics (lower = agreement)",
            "Top 10 descriptors — how many critics used each, by round",
        ),
    )

    # Left panel: vocabulary overlap as a percentage, with early/late lines.
    vocab_df = vocab_df.sort_values("round")
    overlap_pct = vocab_df["jaccard"] * 100
    fig.add_trace(
        go.Scatter(x=vocab_df["round"], y=overlap_pct, mode="lines+markers",
                   line=dict(color="#3b4cb8", width=2.5), marker=dict(size=9),
                   hovertemplate="round %{x}: %{y:.1f}%<extra></extra>",
                   showlegend=False),
        row=1, col=1,
    )
    if len(vocab_df) >= 2:
        early, late, pct = _early_late(vocab_df["jaccard"].tolist())
        fig.add_hline(y=early * 100, line_dash="dash", line_color="#9aa0a6", line_width=1,
                      annotation_text=f"early avg {early * 100:.1f}%",
                      annotation_position="bottom left",
                      annotation_font=dict(color="#9aa0a6"), row=1, col=1)
        fig.add_hline(y=late * 100, line_dash="dash", line_color="#9aa0a6", line_width=1,
                      annotation_text=f"late avg {late * 100:.1f}% ({pct:+.0f}% relative)",
                      annotation_position="top left",
                      annotation_font=dict(color="#3b4cb8"), row=1, col=1)
    fig.update_yaxes(title_text="avg pairwise overlap", ticksuffix="%",
                     rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="round", row=1, col=1)

    # Right panel: score spread per round.
    spread_df = spread_df.sort_values("round")
    fig.add_trace(
        go.Scatter(x=spread_df["round"], y=spread_df["spread"], mode="lines+markers",
                   line=dict(color="#8a4b2f", width=2.5), marker=dict(size=9),
                   showlegend=False),
        row=1, col=2,
    )
    fig.update_yaxes(title_text="std of the critics' scores", rangemode="tozero",
                     row=1, col=2)
    fig.update_xaxes(title_text="round", row=1, col=2)

    # Bottom panel: top-descriptor usage heatmap (descriptor × round → #critics).
    if top_labels:
        rounds = sorted(usage_df["round"].unique())
        # Most-used at the top: reverse so the heatmap's first row sits highest.
        labels = list(reversed(top_labels))
        pivot = (usage_df.pivot(index="cluster", columns="round", values="n_critics")
                         .reindex(index=labels, columns=rounds).fillna(0).astype(int))
        # Truncate long phrases so the y-axis stays readable.
        ticks = [f"“{l[:34]}{'…' if len(l) > 34 else ''}”" for l in labels]
        fig.add_trace(
            go.Heatmap(z=pivot.values, x=[f"r{r}" for r in rounds], y=ticks,
                       colorscale="Blues", zmin=0, text=pivot.values,
                       texttemplate="%{text}", textfont=dict(size=11),
                       colorbar=dict(title="# critics", len=0.45, y=0.22)),
            row=2, col=1,
        )
    else:
        fig.add_annotation(text="No descriptors to rank.", xref="x domain",
                           yref="y domain", x=0.5, y=0.5, showarrow=False,
                           row=2, col=1)
    fig.update_xaxes(title_text="round", row=2, col=1)

    fig.update_layout(height=900, plot_bgcolor="white",
                      title_text="Observation kernel — convergence between critics")
    fig.update_xaxes(showgrid=True, gridcolor="#eee", row=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", row=1)
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

    # 4) propagation, scores, and the two convergence metrics the figure plots
    propagation_df, propagated_labels = find_propagation(occ)
    scores_df = score_trajectories(critic_evals)
    vocab_df = vocabulary_convergence(occ)
    spread_df = score_spread(scores_df)
    usage_df, top_labels = top_descriptor_usage(occ, top_n=10)

    print(f"  {occ['cluster'].nunique()} descriptor clusters; "
          f"{len(propagated_labels)} propagated across critics: {propagated_labels}")

    # Vocabulary-convergence headline (early vs. late pairwise overlap).
    vocab_trend = None
    if len(vocab_df) >= 2:
        early, late, pct = _early_late(vocab_df.sort_values("round")["jaccard"].tolist())
        vocab_trend = {"early": round(early, 3), "late": round(late, 3),
                       "pct_change": round(pct, 1)}
        print(f"  vocabulary overlap: early {early:.3f} -> late {late:.3f} ({pct:+.0f}%)")

    # 5) outputs, keyed by run id so report.py can attach them to the run:
    #    the figure, plus a small JSON summary of what propagated.
    FIGURE_DIR.mkdir(exist_ok=True)
    fig = make_figure(vocab_df, spread_df, usage_df, top_labels)
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
        "vocab_trend": vocab_trend,
    }
    summary_path = FIGURE_DIR / f"analysis_{log_path.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {fig_path} and {summary_path}")


if __name__ == "__main__":
    main()
