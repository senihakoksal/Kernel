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
    # a treatment/control pair, clustered in ONE shared vocabulary space:
    uv run python analyze.py --treatment logs/run_A.jsonl --control logs/control_A_B.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import spacy
import yaml
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from agents import SHARED_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Similarity threshold for treating two descriptors as "the same" idea.
# Swept 0.45–0.90 (docs/threshold-sweep.md); kept at 0.72 because merges here
# are clean synonym groups and below 0.65 distinct ideas begin to combine.
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

    Agglomerative clustering with complete linkage over cosine distance. Two
    properties matter, and neither held for the greedy single pass this
    replaced:

      - Order independence. The greedy version walked descriptors in
        alphabetical order and froze each cluster's founder as its permanent
        representative, so the partition depended on spelling rather than on
        the data.
      - No chaining. Complete linkage requires EVERY pair in a cluster to sit
        within the threshold, so A and C cannot land together merely because
        both resemble B. (single linkage would reintroduce exactly that, and
        ward does not accept a cosine metric at all.)

    `threshold` is a cosine SIMILARITY. sklearn wants a DISTANCE, so it is
    converted here: 0.72 similarity -> 0.28 distance. This one line is
    load-bearing — passing a similarity straight through as distance_threshold
    merges nearly everything into a single cluster, which does not look like an
    error, it looks like a spectacular convergence result.

    Labels are the most frequent member of each cluster, ties broken
    alphabetically. That is why `descriptors` is taken WITH duplicates: a
    cluster should be named for the phrase critics actually reached for, not
    for whichever member happened to sort first.
    """
    counts = Counter(descriptors)
    uniq = sorted(counts)
    if not uniq:
        return {}
    if len(uniq) == 1:
        # AgglomerativeClustering needs at least two samples to fit.
        return {uniq[0]: uniq[0]}

    embeddings = model.encode(uniq, normalize_embeddings=True)
    labels = AgglomerativeClustering(
        n_clusters=None,                     # mandatory when a threshold is set
        distance_threshold=1.0 - threshold,  # DISTANCE, not similarity
        metric="cosine",
        linkage="complete",
    ).fit(embeddings).labels_

    members: dict[int, list[str]] = defaultdict(list)
    for descriptor, label in zip(uniq, labels):
        members[label].append(descriptor)
    names = {label: sorted(group, key=lambda d: (-counts[d], d))[0]
             for label, group in members.items()}
    return {descriptor: names[label] for descriptor, label in zip(uniq, labels)}


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
    # Explicit columns so an empty result still concatenates cleanly when
    # several conditions are pooled.
    return pd.DataFrame(rows, columns=["round", "critic", "descriptor"])


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


def usage_heatmap(usage_df: pd.DataFrame, top_labels: list[str],
                  colorscale: str = "Blues", colorbar_y: float = 0.5,
                  colorbar_len: float = 0.45):
    """A descriptor × round heatmap of how many critics used each top descriptor.

    Reused by both the per-run figure and the baseline/control comparison.
    Returns a go.Heatmap (most-used descriptor on top), or None if there's
    nothing to rank.
    """
    if not top_labels:
        return None
    rounds = sorted(usage_df["round"].unique())
    labels = list(reversed(top_labels))  # reverse so most-used sits highest
    pivot = (usage_df.pivot(index="cluster", columns="round", values="n_critics")
                     .reindex(index=labels, columns=rounds).fillna(0).astype(int))
    ticks = [f"“{l[:34]}{'…' if len(l) > 34 else ''}”" for l in labels]  # keep y readable
    return go.Heatmap(z=pivot.values, x=[f"r{r}" for r in rounds], y=ticks,
                      colorscale=colorscale, zmin=0, text=pivot.values,
                      texttemplate="%{text}", textfont=dict(size=11),
                      colorbar=dict(title="# critics", len=colorbar_len, y=colorbar_y))


CONTROL_COLOR = "#b06c3f"  # copper — isolated-critic control lines


def make_figure(vocab_df: pd.DataFrame, spread_df: pd.DataFrame,
                usage_df: pd.DataFrame, top_labels: list[str],
                control_vocab: pd.DataFrame | None = None,
                control_spread: pd.DataFrame | None = None,
                control_usage: pd.DataFrame | None = None,
                control_top: list[str] | None = None) -> go.Figure:
    """Two convergence panels on top, a top-descriptor usage heatmap below.

    Top-left: average pairwise Jaccard overlap of critics' descriptor sets per
    round (rising = converging vocabulary), with dashed early/late reference
    lines. Top-right: std of critics' scores per round (falling = agreeing more
    on quality). Bottom: the 10 most-used descriptors × round, colored by how
    many critics used each that round.

    If control_vocab / control_spread are given (the isolated-critic control),
    they are overlaid as copper lines on the two top panels and a legend is
    shown distinguishing baseline from control.
    """
    has_control = control_vocab is not None
    base_name = "baseline" if has_control else None
    # A second heatmap row appears only when the control's descriptors are given.
    two_heatmaps = has_control and control_top is not None

    if two_heatmaps:
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.34, 0.33, 0.33],
            vertical_spacing=0.12, horizontal_spacing=0.13,
            specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
            subplot_titles=(
                "Vocabulary convergence", "Score spread (lower = agreement)",
                "Baseline — top 10 descriptors, critics using each by round",
                "Control (isolated) — top 10 descriptors, critics using each by round",
            ),
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.45, 0.55],
            vertical_spacing=0.16, horizontal_spacing=0.13,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=(
                "Vocabulary convergence",
                "Score spread (lower = agreement)",
                "Top 10 descriptors — critics using each, by round",
            ),
        )

    # Left panel: vocabulary overlap as a percentage, with early/late lines.
    vocab_df = vocab_df.sort_values("round")
    fig.add_trace(
        go.Scatter(x=vocab_df["round"], y=vocab_df["jaccard"] * 100, mode="lines+markers",
                   name=base_name, legendgroup="baseline", showlegend=has_control,
                   line=dict(color="#3b4cb8", width=2.5), marker=dict(size=9),
                   hovertemplate="round %{x}: %{y:.1f}%<extra>baseline</extra>"),
        row=1, col=1,
    )
    if has_control:
        cv = control_vocab.sort_values("round")
        fig.add_trace(
            go.Scatter(x=cv["round"], y=cv["jaccard"] * 100, mode="lines+markers",
                       name="control (isolated)", legendgroup="control",
                       line=dict(color=CONTROL_COLOR, width=2.5), marker=dict(size=9),
                       hovertemplate="round %{x}: %{y:.1f}%<extra>control</extra>"),
            row=1, col=1,
        )
    elif len(vocab_df) >= 2:
        # Reference lines only in the solo view; they'd clutter the overlay.
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
                   name=base_name, legendgroup="baseline", showlegend=False,
                   line=dict(color="#3b4cb8" if has_control else "#8a4b2f", width=2.5),
                   marker=dict(size=9)),
        row=1, col=2,
    )
    if control_spread is not None:
        cs = control_spread.sort_values("round")
        fig.add_trace(
            go.Scatter(x=cs["round"], y=cs["spread"], mode="lines+markers",
                       name="control (isolated)", legendgroup="control", showlegend=False,
                       line=dict(color=CONTROL_COLOR, width=2.5), marker=dict(size=9)),
            row=1, col=2,
        )
    fig.update_yaxes(title_text="std of the critics' scores", rangemode="tozero",
                     row=1, col=2)
    fig.update_xaxes(title_text="round", row=1, col=2)

    # Heatmap panel(s): top-descriptor usage (descriptor × round → #critics).
    # Baseline in blue; when a control is present, its descriptors go in a
    # second copper heatmap below.
    base_cbar_y = 0.30 if two_heatmaps else 0.22
    heatmap = usage_heatmap(usage_df, top_labels, colorscale="Blues",
                            colorbar_y=base_cbar_y, colorbar_len=0.3 if two_heatmaps else 0.45)
    if heatmap is not None:
        fig.add_trace(heatmap, row=2, col=1)
    else:
        fig.add_annotation(text="No descriptors to rank.", xref="x domain",
                           yref="y domain", x=0.5, y=0.5, showarrow=False,
                           row=2, col=1)
    fig.update_xaxes(title_text="round", row=2, col=1)

    if two_heatmaps:
        ctrl_heatmap = usage_heatmap(control_usage, control_top, colorscale="Oranges",
                                     colorbar_y=-0.02, colorbar_len=0.3)
        if ctrl_heatmap is not None:
            fig.add_trace(ctrl_heatmap, row=3, col=1)
        fig.update_xaxes(title_text="round", row=3, col=1)

    title = ("Observation kernel — baseline vs. control" if has_control
             else "Observation kernel — convergence between critics")
    # Extra top margin so the title, legend, and the two subplot titles each get
    # their own band instead of stacking on top of one another.
    fig.update_layout(
        height=1240 if two_heatmaps else 920, plot_bgcolor="white",
        title=dict(text=title, y=0.99, yanchor="top"),
        margin=dict(t=150 if has_control else 110),
        legend=dict(orientation="h", y=1.09, yanchor="bottom", x=0.5, xanchor="center")
        if has_control else {})
    fig.update_xaxes(showgrid=True, gridcolor="#eee", row=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", row=1)
    return fig


def pooled_descriptor_pipeline(
        evals_by_condition: dict[str, pd.DataFrame], nlp,
        embed_model: SentenceTransformer) -> tuple[dict[str, pd.DataFrame], list[str], int]:
    """Extract → subtract prior vocabulary → cluster, for several conditions at once.

    Clustering conditions separately does not produce comparable numbers. Each
    run would get its own cluster structure, and a run with more descriptors can
    end up with coarser clusters, which mechanically raises its pairwise Jaccard
    overlap whether or not anything real happened. Jaccard measured in two
    differently-shaped vocabulary spaces compares the shapes, not the runs.

    So both steps that define the space run exactly once over the pooled
    descriptors — the prior-vocabulary subtraction and the clustering — and the
    resulting assignment is applied to each condition afterwards. Every returned
    frame therefore carries cluster labels drawn from one shared space.

    Returns ({condition: occ}, seeds, n_subtracted). Note the caveat carried by
    prior_vocabulary(): it reads the CURRENT agents.py / agents.yaml, so pooling
    is only meaningful for logs generated against the same prompt files.
    """
    per_condition = {name: build_occurrences(evals, nlp)
                     for name, evals in evals_by_condition.items()}
    pooled = pd.concat(
        [occ.assign(condition=name) for name, occ in per_condition.items()],
        ignore_index=True,
    )
    if pooled.empty:
        return per_condition, [], 0

    seeds = prior_vocabulary(nlp)
    n_before = pooled["descriptor"].nunique()
    pooled = subtract_prior_vocabulary(pooled, seeds, embed_model, PRIOR_VOCAB_THRESHOLD)
    n_subtracted = n_before - pooled["descriptor"].nunique()
    if pooled.empty:
        return {name: pooled.copy() for name in per_condition}, seeds, n_subtracted

    assignment = canonicalize(pooled["descriptor"].tolist(), embed_model,
                              SIMILARITY_THRESHOLD)
    pooled["cluster"] = pooled["descriptor"].map(assignment)
    split = {name: pooled[pooled["condition"] == name].drop(columns="condition").copy()
             for name in per_condition}
    return split, seeds, n_subtracted


def descriptor_pipeline(critic_evals: pd.DataFrame, nlp,
                        embed_model: SentenceTransformer) -> tuple[pd.DataFrame, list[str], int]:
    """Single-condition pipeline. Returns (occ, seeds, n_subtracted).

    A thin wrapper over pooled_descriptor_pipeline so there is only one
    implementation of the pipeline. Use this when analyzing one log on its own;
    anything that compares two conditions must use the pooled entry point, or
    the two sides end up in different vocabulary spaces.
    """
    by_condition, seeds, n_subtracted = pooled_descriptor_pipeline(
        {"single": critic_evals}, nlp, embed_model)
    return by_condition["single"], seeds, n_subtracted


def critic_evaluations(log_path: Path) -> pd.DataFrame:
    """The critic evaluations of one run log, or exit if there are none."""
    df = load_records(log_path)
    critic_evals = df[(df["role"] == "critic") & (df["kind"] == "evaluation")].copy()
    if critic_evals.empty:
        sys.exit(f"No critic evaluations in {log_path}.")
    return critic_evals


def write_condition_outputs(log_path: Path, critic_evals: pd.DataFrame,
                            occ: pd.DataFrame, seeds: list[str],
                            n_subtracted: int, pooled_with: str | None = None) -> None:
    """Compute one condition's metrics and write its figure + summary JSON.

    `pooled_with` records the other run whose descriptors shared this run's
    vocabulary space, so nobody later mistakes a jointly-clustered number for an
    independently-clustered one.
    """
    propagation_df, propagated_labels = find_propagation(occ)
    scores_df = score_trajectories(critic_evals)
    vocab_df = vocabulary_convergence(occ)
    spread_df = score_spread(scores_df)
    usage_df, top_labels = top_descriptor_usage(occ, top_n=10)

    print(f"  {log_path.stem}: {occ['cluster'].nunique()} descriptor clusters; "
          f"{len(propagated_labels)} propagated across critics: {propagated_labels}")

    vocab_trend = None
    if len(vocab_df) >= 2:
        early, late, pct = _early_late(vocab_df.sort_values("round")["jaccard"].tolist())
        vocab_trend = {"early": round(early, 3), "late": round(late, 3),
                       "pct_change": round(pct, 1)}
        print(f"    vocabulary overlap: early {early:.3f} -> late {late:.3f} ({pct:+.0f}%)")

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
        # None for a standalone analysis; the paired run id when clustered jointly.
        "pooled_with": pooled_with,
    }
    summary_path = FIGURE_DIR / f"analysis_{log_path.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"    wrote {fig_path} and {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a run, or a treatment/control pair in one vocabulary space.")
    parser.add_argument("logfile", nargs="?", type=Path,
                        help="a single run log (default: the newest in logs/)")
    parser.add_argument("--treatment", type=Path,
                        help="treatment log; pair with --control to cluster jointly")
    parser.add_argument("--control", type=Path, help="control log")
    args = parser.parse_args()

    paired = args.treatment is not None or args.control is not None
    if paired:
        if not (args.treatment and args.control):
            sys.exit("--treatment and --control must be given together.")
        if args.logfile:
            sys.exit("Give either a single log or --treatment/--control, not both.")
        conditions = {"treatment": args.treatment, "control": args.control}
    else:
        log_path = args.logfile
        if log_path is None:
            logs = sorted(Path("logs").glob("run_*.jsonl"))
            if not logs:
                sys.exit("No logs found in logs/. Run run.py first.")
            log_path = logs[-1]
        conditions = {"single": log_path}

    for path in conditions.values():
        if not path.exists():
            sys.exit(f"Log not found: {path}")
    if paired:
        print(f"Analyzing treatment {conditions['treatment'].name} and "
              f"control {conditions['control'].name} in one shared vocabulary space")
    else:
        print(f"Analyzing {conditions['single']}")

    nlp = spacy.load(SPACY_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    evals = {name: critic_evaluations(path) for name, path in conditions.items()}
    by_condition, seeds, n_subtracted = pooled_descriptor_pipeline(evals, nlp, embed_model)
    if all(occ.empty for occ in by_condition.values()):
        sys.exit("No descriptors survived (none extracted, or all were seeded).")
    print(f"  prior vocabulary: {len(seeds)} seed descriptors; "
          f"subtracted {n_subtracted} candidates (pooled across all conditions)")
    if paired:
        pooled_clusters = pd.concat(by_condition.values())["cluster"].nunique()
        print(f"  shared vocabulary space: {pooled_clusters} clusters total")

    for name, path in conditions.items():
        occ = by_condition[name]
        if occ.empty:
            print(f"  {path.stem}: no descriptors survived; skipped")
            continue
        other = next((conditions[o].stem for o in conditions if o != name), None)
        write_condition_outputs(path, evals[name], occ, seeds, n_subtracted,
                                pooled_with=other)


if __name__ == "__main__":
    main()
