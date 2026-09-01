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
        # Everyone using it in its first round is a co-coiner, not an adopter:
        # critics within a round run in parallel and cannot read each other.
        coiners = set(grp.loc[grp["round"] == first_round, "critic"])
        # Genuine adoption: a critic who did not coin it, in a strictly later round.
        adopted = grp[(~grp["critic"].isin(coiners)) & (grp["round"] > first_round)]
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


# --- Palette ----------------------------------------------------------------
# Validated categorical slots 1 and 2 (adjacent-CVD ΔE 24.7 light / 26.8 dark
# against a ≥8 target; normal-vision 33.6 / 31.8 against a ≥15 floor; ≥3:1
# contrast on both surfaces). Do not substitute: the dark values are a selected
# set of steps, not an automatic inversion of the light ones.
#
# A plotly HTML export renders ONE theme — it cannot respond to the reader's
# prefers-color-scheme the way the archive page can. The light palette is what
# ships; PALETTE_DARK is kept here so a dark export is a one-line change and the
# validated pairing is not lost.
PALETTE_LIGHT = {
    "treatment": "#2a78d6", "control": "#eb6834", "inherited": "#9a9992",
    "surface": "#fcfcfb", "text": "#0b0b0b", "text_secondary": "#52514e",
    "text_muted": "#7c7b76", "grid": "#e6e5e1",
}
PALETTE_DARK = {
    "treatment": "#3987e5", "control": "#d95926", "inherited": "#6d6c66",
    "surface": "#1a1a19", "text": "#ffffff", "text_secondary": "#c3c2b7",
    "text_muted": "#94938b", "grid": "#33322f",
}
PALETTE = PALETTE_LIGHT

# Region fill opacities, per the figure spec.
INHERITED_ALPHA = 0.13   # row 1, below the control line
BAND_ALPHA = 0.17        # rows 1 and 2, between the two conditions
FALSE_POSITIVE_ALPHA = 0.12  # row 2, below the control line

TREATMENT_NAME = "treatment — critics see each other"
CONTROL_NAME = "control (isolated) — peer critiques hidden"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def adoption_rate(occ: pd.DataFrame) -> pd.DataFrame:
    """Share of borrowable vocabulary actually in use, per round.

    A rate, not a count: a raw count of adoptions is uninterpretable without
    knowing how many chances there were, and the pool of adoptable vocabulary
    grows every round, so a rising count can reflect nothing but a bigger pool.
    Per round, not cumulative: a running total can only rise — it climbs whether
    adoption is accelerating, steady, or dying — so it cannot answer the one
    question this panel exists for.

    For a round r, with first_round(k) the earliest round cluster k appears and
    coiners(k) every critic who used it in that round:

      eligible(c, r) = clusters k where first_round(k) < r and c not in coiners(k)
      opportunities(r) = sum over critics of |eligible(c, r)|
      uses(r)          = (critic, cluster) pairs from those eligible sets that
                         the critic used at round r
      rate(r)          = uses(r) / opportunities(r)

    This is PREVALENCE, not incidence: every round a critic uses a borrowed
    cluster counts, not only the first. A term adopted once and dropped is a
    failed transmission; a convention is a term that keeps being reproduced.
    Capped at once per (critic, cluster) per round, so one critic with a verbal
    tic cannot dominate the measure.

    Rounds where opportunities(r) == 0 are OMITTED, never plotted as zero. That
    always includes round 0, where the eligible pool is empty and the rate is
    0/0 — undefined, not zero. Returns columns round/uses/opportunities/rate.
    """
    columns = ["round", "uses", "opportunities", "rate"]
    if occ.empty:
        return pd.DataFrame(columns=columns)

    critics = sorted(occ["critic"].unique())
    first_round: dict = {}
    coiners: dict = {}
    for cluster, grp in occ.groupby("cluster"):
        fr = grp["round"].min()
        first_round[cluster] = fr
        coiners[cluster] = set(grp.loc[grp["round"] == fr, "critic"])

    # Deduplicated: repeated use of a cluster inside one round counts once.
    used = set(zip(occ["critic"], occ["cluster"], occ["round"]))

    rows = []
    for r in sorted(occ["round"].unique()):
        eligible = [(c, k) for k, fr in first_round.items() if fr < r
                    for c in critics if c not in coiners[k]]
        if not eligible:
            continue
        uses = sum(1 for c, k in eligible if (c, k, r) in used)
        rows.append({"round": int(r), "uses": uses, "opportunities": len(eligible),
                     "rate": uses / len(eligible)})
    return pd.DataFrame(rows, columns=columns)


def _series_marker(color: str) -> dict:
    """≥8px markers ringed in the surface colour so overlaps stay legible."""
    return dict(size=9, color=color, line=dict(width=2, color=PALETTE["surface"]))


def _panel_heading(fig, row: int, title: str, definition: str) -> None:
    """Panel title plus a muted one-line definition of what the y value counts.

    Domain-referenced so both stay pinned to their panel regardless of how the
    row heights or tick label widths change.
    """
    axis = "" if row == 1 else str(row)
    for text, shift, size, color, weight in (
        (title, 27, 12.5, PALETTE["text"], "bold"),
        (definition, 11, 10.5, PALETTE["text_muted"], "normal"),
    ):
        fig.add_annotation(
            text=f"<b>{text}</b>" if weight == "bold" else text,
            xref=f"x{axis} domain", x=0, xanchor="left",
            yref=f"y{axis} domain", y=1, yanchor="bottom", yshift=shift,
            showarrow=False, font=dict(size=size, color=color),
        )


def _region_label(fig, row: int, x, y, text: str) -> None:
    """Region annotation in text ink — never the series colour."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, row=row, col=1,
                       font=dict(size=11.5, color=PALETTE["text_secondary"]),
                       bgcolor=_rgba(PALETTE["surface"], 0.55), borderpad=2)


def _midpoint_label_position(upper: pd.DataFrame, lower: pd.DataFrame | None,
                             xcol: str, ycol: str, scale: float = 1.0):
    """A point inside a shaded region: middle round, halfway up the band.

    Placing labels at the band's midpoint rather than a fixed coordinate is what
    keeps them off the lines when the data changes.
    """
    if upper.empty:
        return None
    rounds = sorted(upper[xcol])
    x = rounds[len(rounds) // 2]
    top = float(upper.loc[upper[xcol] == x, ycol].iloc[0]) * scale
    if lower is None:
        return x, top / 2
    match = lower.loc[lower[xcol] == x, ycol]
    if match.empty:
        return None
    return x, (top + float(match.iloc[0]) * scale) / 2


def make_figure(vocab_df: pd.DataFrame, spread_df: pd.DataFrame,
                rate_df: pd.DataFrame,
                control_vocab: pd.DataFrame | None = None,
                control_spread: pd.DataFrame | None = None,
                control_rate: pd.DataFrame | None = None) -> go.Figure:
    """Three stacked panels on one shared x-axis: shared vocabulary, borrowed
    vocabulary in use, and score spread.

    The argument runs down the column. Row 1: how much vocabulary critics share,
    split into what they shared without ever reading each other and what peer
    visibility added. Row 2: how much of the borrowable vocabulary was actually
    in use — overlap alone cannot answer the research question, because critics
    judging the same artworks share vocabulary by coincidence with no influence
    involved. Row 3: whether judgment moved with the vocabulary.

    Rows 1 and 2 carry shaded bands with the same meaning, so a reader learns
    the grammar once. Row 3 deliberately carries NO band: there is no consistent
    gap between the conditions there, and shading an inconsistent gap as though
    it were an effect would be a lie. The eye should read effect, effect,
    nothing. Removing that asymmetry "for consistency" is a regression.

    The panels share one x-axis object rather than merely matching ranges — if
    they only matched, alignment would drift the moment a y tick label changed
    width, and reading vertically down a round is the whole point.
    """
    has_control = control_vocab is not None and not control_vocab.empty
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.11, row_heights=[0.42, 0.31, 0.27],
    )

    # --- Row 1: shared vocabulary -------------------------------------------
    # Trace order is load-bearing: the control is added first with fill to zero,
    # then the treatment fills "tonexty" — to the previous trace — which yields
    # exactly the two regions with no double counting.
    vocab_df = vocab_df.sort_values("round")
    if has_control:
        cv = control_vocab.sort_values("round")
        fig.add_trace(
            go.Scatter(x=cv["round"], y=cv["jaccard"] * 100, mode="lines+markers",
                       name=CONTROL_NAME, legendgroup="control", legendrank=2,
                       fill="tozeroy", fillcolor=_rgba(PALETTE["inherited"], INHERITED_ALPHA),
                       line=dict(color=PALETTE["control"], width=2),
                       marker=_series_marker(PALETTE["control"]),
                       hovertemplate="%{y:.1f}%<extra>control</extra>"),
            row=1, col=1)
    fig.add_trace(
        go.Scatter(x=vocab_df["round"], y=vocab_df["jaccard"] * 100, mode="lines+markers",
                   name=TREATMENT_NAME, legendgroup="treatment", legendrank=1,
                   fill="tonexty" if has_control else None,
                   fillcolor=_rgba(PALETTE["treatment"], BAND_ALPHA) if has_control else None,
                   line=dict(color=PALETTE["treatment"], width=2),
                   marker=_series_marker(PALETTE["treatment"]),
                   hovertemplate="%{y:.1f}%<extra>treatment</extra>"),
        row=1, col=1)

    if has_control:
        band = _midpoint_label_position(vocab_df, control_vocab, "round", "jaccard", 100)
        if band:
            _region_label(fig, 1, band[0], band[1], "added by peer visibility")
        inherited = _midpoint_label_position(control_vocab, None, "round", "jaccard", 100)
        if inherited:
            _region_label(fig, 1, inherited[0], inherited[1],
                          "shared without contact — inherited from the model")

    fig.update_yaxes(title_text="avg pairwise overlap", ticksuffix="%",
                     rangemode="tozero", row=1, col=1)
    _panel_heading(fig, 1, "Vocabulary shared between critics",
                   "share of descriptor vocabulary two critics have in common, "
                   "averaged over all critic pairs")

    # --- Row 2: borrowed vocabulary in use ----------------------------------
    rate_df = rate_df.sort_values("round") if not rate_df.empty else rate_df
    if has_control and control_rate is not None and not control_rate.empty:
        cr = control_rate.sort_values("round")
        fig.add_trace(
            go.Scatter(x=cr["round"], y=cr["rate"] * 100, mode="lines+markers",
                       name=CONTROL_NAME, legendgroup="control", showlegend=False,
                       fill="tozeroy",
                       fillcolor=_rgba(PALETTE["control"], FALSE_POSITIVE_ALPHA),
                       line=dict(color=PALETTE["control"], width=2),
                       marker=_series_marker(PALETTE["control"]),
                       customdata=cr[["uses", "opportunities"]].to_numpy(),
                       hovertemplate="%{y:.2f}%  (%{customdata[0]} of "
                                     "%{customdata[1]})<extra>control</extra>"),
            row=2, col=1)
    # A control that never adopts anything is NOT absent from this panel: it
    # reaches adoption_rate() with a real denominator and comes back as a row of
    # 0.0, so the branch above draws a flat line along the axis. An absent line
    # would read as missing data rather than as a null result. control_rate is
    # empty only when opportunities were 0 — the rate undefined, not zero — and
    # fabricating zeros there would invent a measurement.
    if not rate_df.empty:
        fig.add_trace(
            go.Scatter(x=rate_df["round"], y=rate_df["rate"] * 100, mode="lines+markers",
                       name=TREATMENT_NAME, legendgroup="treatment", showlegend=False,
                       fill="tonexty" if has_control else None,
                       fillcolor=_rgba(PALETTE["treatment"], BAND_ALPHA) if has_control else None,
                       line=dict(color=PALETTE["treatment"], width=2),
                       marker=_series_marker(PALETTE["treatment"]),
                       customdata=rate_df[["uses", "opportunities"]].to_numpy(),
                       hovertemplate="%{y:.2f}%  (%{customdata[0]} of "
                                     "%{customdata[1]})<extra>treatment</extra>"),
            row=2, col=1)
        if has_control and control_rate is not None and not control_rate.empty:
            band = _midpoint_label_position(rate_df, control_rate, "round", "rate", 100)
            if band:
                _region_label(fig, 2, band[0], band[1], "genuine adoption")
            fp = _midpoint_label_position(control_rate, None, "round", "rate", 100)
            if fp:
                _region_label(fig, 2, fp[0], fp[1], "false positives")

    fig.update_yaxes(title_text="borrowed vocabulary in use", ticksuffix="%",
                     rangemode="tozero", row=2, col=1)
    _panel_heading(fig, 2, "Borrowed vocabulary in use, per round",
                   "of all the vocabulary a critic could have borrowed from others, "
                   "the share actually in use that round")
    # The left edge carries the structural gap, so the missing round 0 does not
    # read as missing data.
    fig.add_annotation(xref="x2 domain", x=0.005, xanchor="left",
                       yref="y2 domain", y=0.04, yanchor="bottom",
                       text="round 0: no earlier<br>coinages to adopt",
                       showarrow=False, align="left",
                       font=dict(size=10.5, color=PALETTE["text_muted"]))

    # --- Row 3: score spread, deliberately unshaded -------------------------
    if has_control and control_spread is not None and not control_spread.empty:
        cs = control_spread.sort_values("round")
        fig.add_trace(
            go.Scatter(x=cs["round"], y=cs["spread"], mode="lines+markers",
                       name=CONTROL_NAME, legendgroup="control", showlegend=False,
                       line=dict(color=PALETTE["control"], width=2),
                       marker=_series_marker(PALETTE["control"]),
                       hovertemplate="%{y:.3f}<extra>control</extra>"),
            row=3, col=1)
    spread_df = spread_df.sort_values("round")
    fig.add_trace(
        go.Scatter(x=spread_df["round"], y=spread_df["spread"], mode="lines+markers",
                   name=TREATMENT_NAME, legendgroup="treatment", showlegend=False,
                   line=dict(color=PALETTE["treatment"], width=2),
                   marker=_series_marker(PALETTE["treatment"]),
                   hovertemplate="%{y:.3f}<extra>treatment</extra>"),
        row=3, col=1)
    fig.update_yaxes(title_text="std of the critics' scores", rangemode="tozero",
                     row=3, col=1)
    _panel_heading(fig, 3, "Score spread (lower = agreement)",
                   "standard deviation of the critics' mean scores that round")

    # --- Shared axis and chrome ---------------------------------------------
    fig.update_xaxes(dtick=1, showgrid=False, showspikes=True, spikemode="across",
                     spikethickness=1, spikecolor=PALETTE["text_muted"], spikedash="dot")
    fig.update_xaxes(title_text="round", row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False,
                     title_font=dict(size=11, color=PALETTE["text_secondary"]),
                     tickfont=dict(size=11, color=PALETTE["text_muted"]))

    layout = dict(
        height=900, plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font=dict(family="ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
                  color=PALETTE["text"]),
        # A crosshair spanning all three panels is what turns three stacked
        # charts into one figure.
        hovermode="x unified", hoversubplots="axis",
        # Top margin carries three stacked bands: legend, panel title, definition
        # line. The legend sits well clear of row 1's heading — at a smaller gap
        # the two collide, since the heading is pinned to the panel domain while
        # the legend is pinned to the paper.
        margin=dict(l=96, r=32, t=145, b=56),
        legend=dict(orientation="h", y=1.09, yanchor="bottom", x=0, xanchor="left",
                    font=dict(size=12, color=PALETTE["text_secondary"]),
                    bgcolor="rgba(0,0,0,0)"),
        showlegend=has_control,
    )
    fig.update_layout(**layout)
    fig.update_xaxes(tickfont=dict(size=11, color=PALETTE["text_muted"]),
                     title_font=dict(size=11, color=PALETTE["text_secondary"]))
    return fig


def make_timeline_figure(occ: pd.DataFrame, top_n: int = 12) -> go.Figure | None:
    """Descriptor adoption timeline — a supporting figure, deliberately separate.

    Rows are the most-propagated clusters, x is round, one mark per
    (cluster, round, critic) usage. A FILLED mark is a coinage; a HOLLOW ringed
    mark is adoption by a critic who did not coin it — coined-vs-adopted is the
    distinction that matters, so it gets the visual channel. Critic identity
    rides as text initials beside the mark, not as colour: five categorical hues
    cannot pass the all-pairs colourblind floor in a dot plot.

    Returns None when nothing propagated.
    """
    if occ.empty:
        return None
    first_round, coiners = {}, {}
    for cluster, grp in occ.groupby("cluster"):
        fr = grp["round"].min()
        first_round[cluster] = fr
        coiners[cluster] = set(grp.loc[grp["round"] == fr, "critic"])

    adopters = {k: set(occ[(occ["cluster"] == k) & (occ["round"] > first_round[k])]["critic"])
                - coiners[k] for k in first_round}
    ranked = sorted((k for k in first_round if adopters[k]),
                    key=lambda k: (-len(adopters[k]), first_round[k], k))[:top_n]
    if not ranked:
        return None

    order = list(reversed(ranked))               # most-adopted at the top
    y_of = {k: i for i, k in enumerate(order)}
    coin_x, coin_y, coin_t, adopt_x, adopt_y, adopt_t = [], [], [], [], [], []
    for (cluster, rnd), grp in occ[occ["cluster"].isin(ranked)].groupby(["cluster", "round"]):
        critics = sorted(grp["critic"].unique())
        # Spread co-users of the same round vertically so marks do not overlap.
        offsets = [(i - (len(critics) - 1) / 2) * 0.18 for i in range(len(critics))]
        for critic, off in zip(critics, offsets):
            initials = "".join(part[0] for part in critic.split("_"))[:2].upper()
            if critic in coiners[cluster] and rnd == first_round[cluster]:
                coin_x.append(rnd); coin_y.append(y_of[cluster] + off); coin_t.append(initials)
            else:
                adopt_x.append(rnd); adopt_y.append(y_of[cluster] + off); adopt_t.append(initials)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coin_x, y=coin_y, mode="markers+text", text=coin_t, textposition="middle right",
        textfont=dict(size=9, color=PALETTE["text_muted"]), name="coined",
        marker=dict(size=11, color=PALETTE["treatment"],
                    line=dict(width=2, color=PALETTE["surface"])),
        hovertemplate="round %{x} — coined<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=adopt_x, y=adopt_y, mode="markers+text", text=adopt_t, textposition="middle right",
        textfont=dict(size=9, color=PALETTE["text_muted"]), name="adopted (did not coin)",
        marker=dict(size=11, color=PALETTE["surface"],
                    line=dict(width=2, color=PALETTE["treatment"])),
        hovertemplate="round %{x} — adopted<extra></extra>"))

    ticks = [f"“{k[:38]}{'…' if len(k) > 38 else ''}”" for k in order]
    fig.update_layout(
        height=max(320, 46 * len(order) + 150),
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"]),
        title=dict(text="<b>Descriptor adoption timeline</b>", x=0, xanchor="left",
                   font=dict(size=13)),
        margin=dict(l=300, r=40, t=90, b=56),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left",
                    font=dict(size=11, color=PALETTE["text_secondary"])))
    fig.update_xaxes(title_text="round", dtick=1, showgrid=True, gridcolor=PALETTE["grid"],
                     tickfont=dict(size=11, color=PALETTE["text_muted"]))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=ticks,
                     showgrid=True, gridcolor=PALETTE["grid"],
                     tickfont=dict(size=10, color=PALETTE["text_secondary"]),
                     range=[-0.6, len(order) - 0.4])
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
    rate_df = adoption_rate(occ)
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
    fig = make_figure(vocab_df, spread_df, rate_df)
    fig_path = FIGURE_DIR / f"analysis_{log_path.stem}.html"
    fig.write_html(fig_path)

    # Supporting figures live in their own files rather than being crammed into
    # the main one: the adoption timeline, and the top-descriptor usage heatmap
    # the three-panel figure no longer carries.
    timeline = make_timeline_figure(occ)
    if timeline is not None:
        timeline.write_html(FIGURE_DIR / f"timeline_{log_path.stem}.html")
    heatmap = usage_heatmap(usage_df, top_labels)
    if heatmap is not None:
        go.Figure(heatmap).update_layout(
            height=max(320, 34 * len(top_labels) + 140),
            title=dict(text="<b>Top descriptors — critics using each, by round</b>",
                       x=0, xanchor="left", font=dict(size=13)),
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            margin=dict(l=300, r=40, t=80, b=56),
        ).write_html(FIGURE_DIR / f"usage_{log_path.stem}.html")
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
        # The plotted numbers, so the archive page can render them as a table and
        # identity never depends on reading a colour off a chart.
        "series": {
            "overlap": [{"round": int(r), "jaccard": round(float(j), 5)}
                        for r, j in zip(vocab_df["round"], vocab_df["jaccard"])],
            "adoption_rate": rate_df.to_dict("records"),
            "spread": [{"round": int(r), "spread": round(float(v), 5)}
                       for r, v in zip(spread_df["round"], spread_df["spread"])],
        },
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
