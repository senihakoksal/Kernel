"""Generate a static HTML report of all runs.

Reads every logs/run_*.jsonl and writes one self-contained site/index.html:
an archive of runs (when, how many rounds/concepts/critiques) and, per run,
the full round-by-round record — concepts by artists, critiques + scores by
critics — plus that run's analysis report if analyze.py has produced one.
No server, no dependencies beyond the stdlib; the run data is embedded in the
page as JSON and rendered by a small inline script.

Usage:
    uv run python report.py        # then open site/index.html
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
FIGURE_DIR = Path("figures")
SITE_DIR = Path("site")
DOCS_DIR = Path("docs")


def figure_name(kind: str, run_id: str) -> str | None:
    """The figure's filename if it was actually written, else None.

    Not every run produces every figure: a run where nothing propagated has no
    timeline, and a run with no critic evaluations has none at all. The page
    used to link all of them unconditionally, which published dead links.
    """
    name = f"{kind}_{run_id}.html"
    return name if (FIGURE_DIR / name).exists() else None


def load_analysis(run_id: str) -> dict | None:
    """Load the analyze.py summary for a run, if that run has been analyzed."""
    summary_path = FIGURE_DIR / f"analysis_{run_id}.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    # Bare filenames: the page prepends whichever prefix the build is for, so
    # the same markup serves site/ (../figures/) and docs/ (figures/).
    summary["figure"] = figure_name("analysis", run_id)
    summary["timeline"] = figure_name("timeline", run_id)
    summary["usage"] = figure_name("usage", run_id)
    return summary


def load_comparison(run_id: str) -> dict | None:
    """Load the compare.py summary attaching an isolated-critic control to a run."""
    summary_path = FIGURE_DIR / f"compare_{run_id}.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    summary["figure"] = figure_name("compare", run_id)
    return summary


def load_runs() -> list[dict]:
    """Read every run log into {id, started, records, analysis} dicts, newest first."""
    runs: list[dict] = []
    for path in sorted(LOG_DIR.glob("run_*.jsonl"), reverse=True):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # The run's start time is encoded in its filename: run_YYYYMMDD_HHMMSS.
        stamp = path.stem.removeprefix("run_")
        dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        # A run is identified to the reader by when it happened, not by its
        # filename. The stem is still carried for the analyze.py command line.
        # %-d is not portable, so the day is composed rather than formatted.
        runs.append({
            "id": path.stem,
            "name": f"{dt.day} {dt.strftime('%B %Y')}",
            "clock": dt.strftime("%H:%M"),
            "started": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "records": records,
            "analysis": load_analysis(path.stem),
            "comparison": load_comparison(path.stem),
        })
    return runs


# The page shell. Run data is injected as JSON at the __RUNS_JSON__ marker
# (plain .replace, so braces below need no escaping); the inline script renders
# the archive and the per-run gallery.
#
# The page is built around one unit: an artwork, then the reception of that
# artwork. Two facts drive every decision here.
#
# First, the medium is text, so a work is a passage — it gets the largest type on
# the page and nothing competes with it. Second, each work carries roughly ten
# times more words of criticism than of art, so the critiques are COLLAPSED by
# default. Open, they bury the thing they are about; what stays visible is the
# shape of the reception (six scores on a track, mean, range), which is scannable
# down the gallery in a way six blocks of prose never are.
#
# Colours are analyze.PALETTE_LIGHT verbatim, so the page ground is the figures'
# own paper colour and an embedded plot has no visible edge.
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Observation Kernel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
  /* Palette: analyze.PALETTE_LIGHT verbatim. The page ground IS the figures'
     paper colour, so an embedded plot has no visible edge — it sits on the wall
     instead of being framed on it. */
  :root {
    --wall: #fcfcfb;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --ink-3: #7c7b76;
    --rule: #e6e5e1;
    --accent: #2a78d6;
    --text: "Inter", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  * { box-sizing: border-box; }
  html { background: var(--wall); }
  body { margin: 0; background: var(--wall); color: var(--ink); font-family: var(--text);
         font-weight: 400; font-size: 16px; line-height: 1.6;
         -webkit-font-smoothing: antialiased; }

  /* Two widths: a reading column for the works, a wide one for figures. */
  .col  { max-width: 43rem; margin: 0 auto; padding: 0 1.75rem; }
  .wide { max-width: 76rem; margin: 0 auto; padding: 0 1.75rem; }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .quiet { color: var(--ink-3); font-size: .8rem; }
  .m { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  header { padding: 5.5rem 0 0; }
  .kernel { font-family: var(--mono); font-size: .7rem; color: var(--ink-3);
            letter-spacing: .04em; }
  h1 { font-weight: 300; font-size: 2.4rem; letter-spacing: -.025em; line-height: 1.1;
       margin: .9rem 0 .6rem; }
  h1 .at { color: var(--ink-3); font-size: .55em; letter-spacing: 0;
           font-family: var(--mono); margin-left: .5rem; vertical-align: .18em; }
  .lede { color: var(--ink-3); font-size: .95rem; margin: 0 0 4.5rem; max-width: 34rem; }

  /* --- archive: a quiet typographic list, not cards --------------------- */
  .runs { border-top: 1px solid var(--rule); }
  .runs-head { border-bottom: 0; }
  /* The numbers sit on ONE baseline row with the run name, and the work titles
     get a row of their own spanning every column. Previously the titles shared
     column one and the row was end-aligned, so a long or missing title line
     moved the numbers up and down and nothing could be read down the column.
     Their column header carries the unit, so no label repeats per row. */
  .runs-cols { display: grid; grid-template-columns: 1fr 4.5rem 5.5rem 4.5rem;
               gap: .5rem 2.25rem; }
  .runs-head { padding-bottom: .8rem; font-size: .66rem; color: var(--ink-3); }
  .runs-head div:not(:first-child) { text-align: right; }
  .run { padding: 2rem 0; border-bottom: 1px solid var(--rule); cursor: pointer; }
  .run:hover .run-name { color: var(--accent); }
  .run-name { font-size: 1.02rem; font-weight: 400; transition: color .12s ease; }
  .run-name .clock { color: var(--ink-3); font-family: var(--mono); font-size: .8rem;
                     margin-left: .6rem; }
  .run-titles { grid-column: 1 / -1; color: var(--ink-3); font-size: .85rem;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .run-n { text-align: right; font-family: var(--mono); font-size: .88rem;
           color: var(--ink-2); font-variant-numeric: tabular-nums; }

  /* --- the gallery ------------------------------------------------------ */
  .round { margin: 5.5rem 0 3rem; font-family: var(--mono); font-size: .7rem;
           color: var(--ink-3); }
  .work { margin: 0 0 6.5rem; }
  .work-by { font-family: var(--mono); font-size: .72rem; color: var(--ink-3);
             margin-bottom: .9rem; }
  .work-title { font-weight: 500; font-size: 1.85rem; letter-spacing: -.02em;
                line-height: 1.2; margin: 0 0 1.5rem; }
  /* The artwork. Near-black, large, generously led — the only thing on the page
     set at this size, because it is the subject and everything else is about it. */
  .work-text { font-size: 1.28rem; line-height: 1.68; color: var(--ink);
               font-weight: 300; white-space: pre-wrap; margin: 0; }
  /* The `reasoning` field: the agent's own stated rationale, not part of the
     work or the critique. Unlabelled it read as a trailing paragraph of the
     thing above it — after an artwork especially, as more artwork. The rule and
     the key name it, and the key uses the field's own name so a reader can map
     the page back onto the .jsonl. */
  .why { margin-top: 1.4rem; padding-left: 1rem; border-left: 1px solid var(--rule);
         max-width: 34rem; }
  .why-k { display: block; font-family: var(--mono); font-size: .64rem;
           color: var(--ink-3); letter-spacing: .04em; margin-bottom: .3rem; }
  .why-t { font-size: .88rem; color: var(--ink-3); line-height: 1.6; }
  .crit .why { margin-top: .9rem; }
  .crit .why-t { font-size: .84rem; }

  /* Reception: the shape of six judgements, before any of their words. */
  .recv { margin-top: 2.5rem; border-top: 1px solid var(--rule); padding-top: 1.1rem; }
  .recv-row { display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
  .track { position: relative; flex: 1 1 14rem; height: 1.5rem; min-width: 11rem; }
  .track .axis { position: absolute; top: 50%; left: 0; right: 0; height: 1px;
                 background: var(--rule); }
  .track .dot { position: absolute; top: 50%; width: 8px; height: 8px; border-radius: 50%;
                background: var(--accent); transform: translate(-50%, -50%);
                box-shadow: 0 0 0 2px var(--wall); }
  .track .cap { position: absolute; top: 50%; width: 1px; height: 7px; background: var(--rule);
                transform: translate(-50%, -50%); }
  .recv-stat { font-family: var(--mono); font-size: .78rem; color: var(--ink-2);
               font-variant-numeric: tabular-nums; text-align: right; }
  .recv-stat.mean { width: 5.5rem; }
  .recv-stat.range { width: 8.5rem; }
  .recv-stat em { font-style: normal; color: var(--ink-3); font-family: var(--text);
                  font-size: .7rem; }
  .recv details { margin-top: 1.1rem; }
  .recv summary { cursor: pointer; font-size: .8rem; color: var(--accent); list-style: none; }
  .recv summary::-webkit-details-marker { display: none; }
  .recv summary::before { content: "+ "; }
  .recv details[open] summary::before { content: "\2212 "; }
  .recv summary:hover { text-decoration: underline; }

  .crit { padding: 1.4rem 0; border-bottom: 1px solid var(--rule); }
  .crit:last-child { border-bottom: 0; padding-bottom: 0; }
  .crit-head { display: flex; align-items: baseline; gap: .9rem; margin-bottom: .6rem; }
  .crit-who { font-family: var(--mono); font-size: .74rem; color: var(--ink-2); }
  .crit-score { font-family: var(--mono); font-size: .74rem; color: var(--ink);
                margin-left: auto; font-variant-numeric: tabular-nums; }
  .crit-text { font-size: .95rem; line-height: 1.65; color: var(--ink-2);
               white-space: pre-wrap; }


  /* --- analysis: reference, not the headline --------------------------- */
  .analysis { border-top: 1px solid var(--rule); margin-top: 1rem; padding-top: 1.1rem; }
  .analysis > summary { cursor: pointer; font-size: .85rem; color: var(--accent);
                        list-style: none; }
  .analysis > summary::-webkit-details-marker { display: none; }
  .analysis > summary::before { content: "+ "; }
  .analysis[open] > summary::before { content: "\2212 "; }
  .analysis p { font-size: .92rem; color: var(--ink-2); max-width: 40rem; }
  .analysis .caveat { color: var(--ink-3); font-size: .85rem; }
  .fig { margin: 1.75rem 0 .5rem; }
  .fig iframe { width: 100%; border: 0; display: block; }
  .links { font-size: .8rem; color: var(--ink-3); }
  .themes { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .9rem; }
  .theme { font-family: var(--mono); font-size: .72rem; color: var(--ink-2);
           border: 1px solid var(--rule); padding: .15rem .5rem; }
  table { border-collapse: collapse; margin-top: .9rem; font-family: var(--mono);
          font-size: .72rem; }
  th, td { border-bottom: 1px solid var(--rule); padding: .35rem .8rem .35rem 0;
           text-align: right; color: var(--ink-2); font-weight: 400;
           font-variant-numeric: tabular-nums; }
  th { color: var(--ink-3); font-size: .66rem; }
  th:first-child, td:first-child { text-align: left; }
  code { font-family: var(--mono); font-size: .85rem; color: var(--ink-2); }
  .back { font-family: var(--mono); font-size: .72rem; color: var(--ink-3);
          display: inline-block; padding: 3.5rem 0 2rem; }
  .back:hover { color: var(--accent); text-decoration: none; }
  #detail { display: none; }
  @media (max-width: 40rem) {
    h1 { font-size: 1.9rem; }
    .work-title { font-size: 1.45rem; }
    .work-text { font-size: 1.12rem; }
    .work { margin-bottom: 4.5rem; }
  }
</style>
</head>
<body>

<div id="archive">
  <header class="col">
    <div class="kernel">Observation Kernel</div>
    <h1>Studio archive</h1>
    <p class="lede">Artists invent artworks in language; critics judge them and read
    each other. Each run is a studio that ran for a number of rounds.</p>
  </header>
  <div class="col">
    <div class="runs-head runs-cols"><div>run</div><div>works</div>
      <div>critiques</div><div>rounds</div></div>
    <div class="runs" id="run-list"></div>
  </div>
</div>

<div id="detail">
  <div class="col"><a class="back" href="#" onclick="showArchive(); return false;">&larr; archive</a></div>
  <div id="detail-body"></div>
</div>

<script>
const RUNS = __RUNS_JSON__;
// Where the figures live relative to this page. site/ and docs/ nest
// differently, so the build supplies it rather than the markup assuming it.
const FIG = "__FIG_PREFIX__";

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : s;
  return d.innerHTML;
}

function counts(run) {
  const c = {rounds: 0, concepts: 0, evals: 0, artists: new Set(), critics: new Set()};
  for (const r of run.records) {
    c.rounds = Math.max(c.rounds, r.round + 1);
    if (r.kind === "concept") { c.concepts++; c.artists.add(r.agent); }
    else { c.evals++; c.critics.add(r.agent); }
  }
  return {rounds: c.rounds, concepts: c.concepts, evals: c.evals,
          artists: c.artists.size, critics: c.critics.size};
}

function showArchive() {
  document.getElementById("archive").style.display = "block";
  document.getElementById("detail").style.display = "none";
  window.scrollTo(0, 0);
}

/* Six judgements on a 0-1 track. Single hue, not six: a dot plot is an
   all-pairs form, which caps at three categorical hues before colourblind
   separation fails — so identity rides on the hover label, the same choice the
   adoption-timeline figure makes. A 2px wall-coloured ring keeps overlapping
   dots readable. */
function track(crits) {
  const scored = crits.filter(c => c.score != null);
  if (!scored.length) return '<div class="track"><div class="axis"></div></div>';
  const dots = scored.map(c =>
    `<span class="dot" style="left:${(c.score * 100).toFixed(1)}%"
           title="${esc(c.agent)} — ${c.score.toFixed(2)}"></span>`).join("");
  return `<div class="track"><div class="axis"></div>
            <span class="cap" style="left:0"></span><span class="cap" style="left:100%"></span>
            ${dots}</div>`;
}

/* The `reasoning` field, labelled. One key for artists and critics alike: it is
   the agent's own account of what it just did, and naming it that way keeps the
   page mapping onto the log, where both roles write to the same field. */
function why(text) {
  if (!text) return "";
  return `<div class="why"><span class="why-k">Agent's reasoning</span>
            <span class="why-t">${esc(text)}</span></div>`;
}

/* One artwork, then its reception. The critiques are collapsed because there
   are ten times more words of criticism than of art, and open by default the
   commentary buries the work it is about. */
function workBlock(entry) {
  const w = entry.work, crits = entry.critiques;
  const scored = crits.map(c => c.score).filter(s => s != null);
  const mean = scored.length ? scored.reduce((a, b) => a + b, 0) / scored.length : null;

  let stat = "";
  if (mean != null) {
    stat = `<span class="recv-stat mean">${mean.toFixed(2)} <em>mean</em></span>`
         + `<span class="recv-stat range">${Math.min(...scored).toFixed(2)}&ndash;${Math.max(...scored).toFixed(2)} <em>range</em></span>`;
  }

  const body = crits.map(c => `
    <div class="crit">
      <div class="crit-head"><span class="crit-who">${esc(c.agent)}</span>
        ${c.score != null ? `<span class="crit-score">${c.score.toFixed(2)}</span>` : ""}</div>
      <div class="crit-text">${esc(c.content)}</div>
      ${why(c.reasoning)}
    </div>`).join("");

  const reception = crits.length ? `
    <div class="recv">
      <div class="recv-row">${track(crits)}${stat}</div>
      <details><summary>${crits.length} critique${crits.length === 1 ? "" : "s"}</summary>
        ${body}
      </details>
    </div>` : `<div class="recv"><span class="quiet">No critiques recorded.</span></div>`;

  return `<article class="work">
      <div class="work-by">${esc(w.agent)}<span class="quiet"> · ${esc(w.concept_id || "")}</span></div>
      <h2 class="work-title">${w.title ? esc(w.title) : "Untitled"}</h2>
      <p class="work-text">${esc(w.content)}</p>
      ${why(w.reasoning)}
      ${reception}
    </article>`;
}

function dataTable(a, cmp) {
  const s = a && a.series;
  if (!s) return "";
  const cs = cmp && cmp.series;
  const rounds = [...new Set([
    ...(s.overlap || []).map(r => r.round),
    ...(s.adoption_rate || []).map(r => r.round),
    ...(s.spread || []).map(r => r.round),
  ])].sort((x, y) => x - y);
  if (!rounds.length) return "";
  const at = (rows, rnd, key) => {
    const hit = (rows || []).find(r => r.round === rnd);
    return hit === undefined ? null : hit[key];
  };
  // Two formatters, because the two quantities are an order of magnitude apart:
  // overlap runs 7-15% and reads wrong at 2dp, the adoption rate runs 1-3% and
  // loses its resolution at 1dp. One shared formatter mis-set both.
  const pctOv = v => v === null ? "&mdash;" : (v * 100).toFixed(1) + "%";
  const pctRt = v => v === null ? "&mdash;" : (v * 100).toFixed(2) + "%";
  const num = v => v === null ? "&mdash;" : v.toFixed(3);
  const frac = (u, o) => u === null ? "&mdash;" : u + " / " + o;
  const paired = !!cs;
  const head = paired
    ? ["round", "dice T", "dice C", "borrowed T", "borrowed C",
       "uses/opps T", "uses/opps C", "spread T", "spread C"]
    : ["round", "overlap (dice)", "borrowed in use", "uses / opportunities", "score spread"];
  const rows = rounds.map(rnd => {
    if (!paired) {
      return [rnd, pctOv(at(s.overlap, rnd, "dice")), pctRt(at(s.adoption_rate, rnd, "rate")),
              frac(at(s.adoption_rate, rnd, "uses"), at(s.adoption_rate, rnd, "opportunities")),
              num(at(s.spread, rnd, "spread"))];
    }
    const tr = cs.adoption_rate.treatment, cr = cs.adoption_rate.control;
    return [rnd, pctOv(at(cs.overlap.treatment, rnd, "dice")),
            pctOv(at(cs.overlap.control, rnd, "dice")),
            pctRt(at(tr, rnd, "rate")), pctRt(at(cr, rnd, "rate")),
            frac(at(tr, rnd, "uses"), at(tr, rnd, "opportunities")),
            frac(at(cr, rnd, "uses"), at(cr, rnd, "opportunities")),
            num(at(cs.spread.treatment, rnd, "spread")),
            num(at(cs.spread.control, rnd, "spread"))];
  });
  return `<details><summary class="links">The figure's numbers as a table</summary>
    <table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>
    <p class="caveat">Dashes at round 0 in the adoption columns are structural: no critique
    precedes the round, so the rate is undefined rather than zero.</p></details>`;
}

function analysisBlock(run) {
  const a = run.analysis, cmp = run.comparison;
  if (!a) {
    return `<div class="col"><div class="analysis"><p class="quiet">Not analysed yet &mdash;
      <code>uv run python analyze.py logs/${esc(run.id)}.jsonl</code></p></div></div>`;
  }
  let findings = `${a.n_clusters} descriptor clusters survived the pipeline;
    ${a.propagated.length} were later used by a critic who had not coined them.`;
  if (a.vocab_trend) {
    const pc = a.vocab_trend.pct_change;
    findings += ` Average pairwise overlap moved from ${(a.vocab_trend.early * 100).toFixed(1)}%
      to ${(a.vocab_trend.late * 100).toFixed(1)}% (${pc >= 0 ? "+" : ""}${pc}% relative),
      comparing the first two rounds with the last two.`;
  }
  if (a.feed_window != null) findings += ` Critics saw critiques from the last ${a.feed_window} rounds.`;

  const control = cmp ? `<p>Replaying the identical artworks with every critic isolated —
    each seeing the works and its own recent critiques, never a peer's — overlap grew
    ${(cmp.control.early * 100).toFixed(1)}% to ${(cmp.control.late * 100).toFixed(1)}%
    against ${(cmp.treatment.early * 100).toFixed(1)}% to
    ${(cmp.treatment.late * 100).toFixed(1)}% with peers visible. Both arms are clustered
    together in one pooled space of ${cmp.n_clusters} clusters.</p>
    <p class="caveat">Round 0 is identical in both arms by construction and their prompts are
    verified byte-identical, so any gap there is noise. From round 1 the isolated critics carry
    their own regenerated history, so the arms differ in more than the peer channel alone.</p>`
    : "";
  const themes = a.propagated.length
    ? `<div class="themes">${a.propagated.map(t => `<span class="theme">${esc(t)}</span>`).join("")}</div>`
    : "<p class='quiet'>Nothing propagated across critics.</p>";

  // Link only figures that were actually written. A run where nothing
  // propagated has no timeline; one with no critic evaluations has no figures
  // at all. Linking them regardless published dead links.
  const main = (cmp && cmp.figure) || a.figure;
  const extra = [["adoption timeline", a.timeline], ["top descriptors", a.usage]]
    .filter(pair => pair[1])
    .map(pair => ` &nbsp;·&nbsp; <a href="${FIG}${esc(pair[1])}" target="_blank">${pair[0]}</a>`)
    .join("");

  return `<div class="col"><details class="analysis">
      <summary>Analysis</summary>
      <p>${findings}</p>${control}
    </details></div>
    ${main ? `<div class="wide"><div class="fig">
      <iframe src="${FIG}${esc(main)}" style="height:940px" loading="lazy"></iframe>
    </div></div>` : ""}
    <div class="col">
      <p class="links">${main ? `<a href="${FIG}${esc(main)}" target="_blank">full figure</a>` : ""}${extra}</p>
      ${dataTable(a, cmp)}
      <details><summary class="links">Propagated descriptors (${a.propagated.length})</summary>${themes}</details>
    </div>`;
}

function showRun(i) {
  const run = RUNS[i];
  const c = counts(run);

  // Key every concept, including any written without an id — keying only on
  // r.concept_id would drop the work itself on a lookup miss, silently.
  const byWork = {};
  run.records.forEach((r, n) => {
    if (r.kind !== "concept") return;
    byWork[r.concept_id || `unkeyed-${n}`] = {work: r, critiques: []};
  });
  // Critiques with no concept_id predate per-work critiques. Shown at the end
  // rather than dropped by a silent lookup miss.
  const orphans = [];
  for (const r of run.records) {
    if (r.kind !== "evaluation") continue;
    if (r.concept_id && byWork[r.concept_id]) byWork[r.concept_id].critiques.push(r);
    else orphans.push(r);
  }
  const rounds = {};
  for (const id of Object.keys(byWork)) (rounds[byWork[id].work.round] ??= []).push(id);

  let out = `<header class="col">
      <div class="kernel">${esc(run.id)}</div>
      <h1>${esc(run.name)} <span class="at">${esc(run.clock)}</span></h1>
      <p class="lede">${c.concepts} works by ${c.artists} artists, judged
        ${c.evals} times by ${c.critics} critics over ${c.rounds} rounds.</p>
    </header>`;

  out += analysisBlock(run);
  out += '<div class="col">';
  for (const idx of Object.keys(rounds).sort((x, y) => x - y)) {
    out += `<div class="round">Round ${idx} &nbsp;/&nbsp; ${rounds[idx].length} works</div>`;
    for (const id of rounds[idx]) out += workBlock(byWork[id]);
  }
  if (orphans.length) {
    out += `<div class="round">Unattributed critiques</div>
      <p class="quiet">${orphans.length} critique${orphans.length === 1 ? "" : "s"} in this log
      carry no artwork id — written before critiques were linked to one work — so they cannot be
      placed under a work.</p>
      ${orphans.map(o => `<div class="crit">
        <div class="crit-head"><span class="crit-who">${esc(o.agent)}</span>
          <span class="quiet">round ${o.round}</span>
          ${o.score != null ? `<span class="crit-score">${o.score.toFixed(2)}</span>` : ""}</div>
        <div class="crit-text">${esc(o.content)}</div></div>`).join("")}`;
  }
  out += "</div>";

  document.getElementById("detail-body").innerHTML = out;
  document.getElementById("archive").style.display = "none";
  document.getElementById("detail").style.display = "block";
  window.scrollTo(0, 0);
}

const list = document.getElementById("run-list");
RUNS.forEach((run, i) => {
  const c = counts(run);
  const titles = run.records.filter(r => r.kind === "concept" && r.title)
                            .slice(0, 3).map(r => r.title).join(" · ");
  const el = document.createElement("div");
  el.className = "run runs-cols";
  el.innerHTML = `
    <div class="run-name">${esc(run.name)}<span class="clock">${esc(run.clock)}</span></div>
    <div class="run-n">${c.concepts}</div>
    <div class="run-n">${c.evals}</div>
    <div class="run-n">${c.rounds}</div>
    <div class="run-titles">${titles ? esc(titles) : "&mdash;"}</div>`;
  el.onclick = () => showRun(i);
  list.appendChild(el);
});
</script>
</body>
</html>
"""


def render(runs: list[dict], fig_prefix: str) -> str:
    """The page, with the run data and the figure prefix substituted in."""
    return (PAGE.replace("__RUNS_JSON__", json.dumps(runs))
                .replace("__FIG_PREFIX__", fig_prefix))


def referenced_figures(runs: list[dict]) -> set[str]:
    """Every figure the page actually links, and nothing else.

    Copying all of figures/ would drag in the control-arm plots the page never
    links; copying from the page's own references means what ships is exactly
    what is reachable.
    """
    names: set[str] = set()
    for run in runs:
        for summary in (run.get("analysis"), run.get("comparison")):
            if not summary:
                continue
            for key in ("figure", "timeline", "usage"):
                if summary.get(key):
                    names.add(summary[key])
    return names


def publish(runs: list[dict]) -> Path:
    """Assemble a self-contained docs/ for GitHub Pages.

    Pages serves committed files, and logs/, figures/ and site/ are all ignored
    — so publishing means writing a directory that IS committed, holding the
    rendered page and the figures it links. docs/ on the default branch is the
    one Pages source that needs no second branch and no workflow.
    """
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "figures").mkdir(exist_ok=True)
    # Figures sit beside the page here, not a level up as they do from site/.
    (DOCS_DIR / "index.html").write_text(render(runs, "figures/"))

    wanted = referenced_figures(runs)
    for name in sorted(wanted):
        shutil.copyfile(FIGURE_DIR / name, DOCS_DIR / "figures" / name)
    # Drop anything a previous publish left behind that is no longer linked.
    for stale in (DOCS_DIR / "figures").glob("*.html"):
        if stale.name not in wanted:
            stale.unlink()

    size = sum(f.stat().st_size for f in DOCS_DIR.rglob("*") if f.is_file())
    print(f"Published {DOCS_DIR}/ — 1 page + {len(wanted)} figures, "
          f"{size / 1e6:.1f} MB total.")
    print("Commit docs/, then enable GitHub Pages: Settings -> Pages -> "
          "Source: main, folder /docs")
    return DOCS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the run archive page.")
    parser.add_argument("--publish", action="store_true",
                        help="also assemble docs/ for GitHub Pages")
    args = parser.parse_args()

    runs = load_runs()
    if not runs:
        raise SystemExit("No logs found in logs/. Run run.py first.")

    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(render(runs, "../figures/"))
    print(f"Wrote {out} ({len(runs)} runs). Open it in a browser.")

    if args.publish:
        publish(runs)


if __name__ == "__main__":
    main()
