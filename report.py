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

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
FIGURE_DIR = Path("figures")
SITE_DIR = Path("site")


def load_analysis(run_id: str) -> dict | None:
    """Load the analyze.py summary for a run, if that run has been analyzed."""
    summary_path = FIGURE_DIR / f"analysis_{run_id}.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    # The figure sits next to the summary; the page links to it relative to site/.
    summary["figure"] = f"../figures/analysis_{run_id}.html"
    return summary


def load_runs() -> list[dict]:
    """Read every run log into {id, started, records, analysis} dicts, newest first."""
    runs: list[dict] = []
    for path in sorted(LOG_DIR.glob("run_*.jsonl"), reverse=True):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # The run's start time is encoded in its filename: run_YYYYMMDD_HHMMSS.
        stamp = path.stem.removeprefix("run_")
        started = datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        runs.append({
            "id": path.stem,
            "started": started,
            "records": records,
            "analysis": load_analysis(path.stem),
        })
    return runs


# The page shell. Run data is injected as JSON at the __RUNS_JSON__ marker
# (plain .replace, so braces below need no escaping); the inline script renders
# the archive and the per-run detail. Aesthetic: warm cream paper, hairline
# rules, large old-style serif numerals, letterspaced small-cap labels.
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Observation Kernel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f4efe7;        /* warm cream paper */
    --panel: #faf7f1;     /* slightly lighter panel */
    --ink: #2b2620;       /* warm near-black */
    --muted: #94896f;     /* warm gray for labels */
    --soft: #6f675c;      /* secondary text */
    --hairline: #e0d8c8;  /* thin rules */
    --accent: #b06c3f;    /* copper (status dot, scores, links) */
    --sage: #7d8b6f;      /* concepts */
    --serif: "Cormorant Garamond", Georgia, serif;
    --sans: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  body { background: var(--bg); color: var(--ink); font-family: var(--sans);
         max-width: 62rem; margin: 3rem auto 6rem; padding: 0 1.5rem;
         line-height: 1.55; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Letterspaced small-cap labels, as in the reference design. */
  .label { font-family: var(--sans); font-size: .68rem; font-weight: 600;
           letter-spacing: .18em; text-transform: uppercase; color: var(--muted); }
  .dot { display: inline-block; width: .5rem; height: .5rem; border-radius: 50%;
         background: var(--accent); margin-right: .55rem; vertical-align: 1px; }

  h1 { font-family: var(--serif); font-weight: 500; font-size: 2.6rem;
       margin: .4rem 0 2.2rem; letter-spacing: .01em; }
  h2 { font-family: var(--serif); font-weight: 500; font-size: 1.9rem;
       margin: .2rem 0 .2rem; }
  .when { color: var(--soft); font-size: .85rem; margin-bottom: 1.6rem; }

  /* Stat cards: big old-style serif numerals over hairlines. */
  /* One row always: each stat gets an equal column, however many there are. */
  .stats { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr;
           column-gap: 2rem; margin: 1.5rem 0 2.5rem; }
  .stat { border-bottom: 1px solid var(--hairline); padding: 1.1rem 0 1.3rem; }
  .stat .num { font-family: var(--serif); font-size: 3.2rem; font-weight: 400;
               line-height: 1.05; font-variant-numeric: oldstyle-nums; }
  .stat .sub { color: var(--muted); font-size: .85rem; margin-top: .15rem; }

  /* Run archive table. */
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th { text-align: left; padding: .55rem .8rem .55rem 0; border-bottom: 1px solid var(--hairline); }
  td { text-align: left; padding: .8rem .8rem .8rem 0; border-bottom: 1px solid var(--hairline);
       font-variant-numeric: oldstyle-nums; }
  td.run-date { font-family: var(--serif); font-size: 1.25rem; }
  tr.run-row { cursor: pointer; }
  tr.run-row:hover { background: var(--panel); }

  /* Report panel. */
  .report { background: var(--panel); border: 1px solid var(--hairline);
            padding: 1.6rem 1.8rem; margin: 1.8rem 0 2.4rem; }
  .report p { margin: .5rem 0 1.1rem; color: var(--soft); max-width: 46rem; }
  .report .verdict { color: var(--ink); font-family: var(--serif);
                     font-size: 1.15em; font-style: italic; }
  ul.themes { list-style: none; padding: 0; margin: .5rem 0 1.2rem; columns: 2; }
  ul.themes li { padding: .22rem 0 .22rem 1.1rem; position: relative;
                 break-inside: avoid; color: var(--ink); }
  ul.themes li::before { content: ""; position: absolute; left: 0; top: .72em;
                         width: .38rem; height: .38rem; border-radius: 50%;
                         background: #cfc6b2; }
  .report iframe { width: 100%; height: 520px; border: 1px solid var(--hairline);
                   background: #fff; margin-top: .6rem; }
  code { background: var(--panel); border: 1px solid var(--hairline);
         padding: .1rem .35rem; font-size: .85em; }

  /* Round-by-round record. */
  .round-label { margin: 2.2rem 0 .8rem; padding-top: 1rem;
                 border-top: 1px solid var(--hairline); }
  .card { background: var(--panel); border: 1px solid var(--hairline);
          border-left: 3px solid var(--hairline);
          padding: .9rem 1.2rem; margin: .8rem 0; }
  .card.concept { border-left-color: var(--sage); }
  .card.evaluation { border-left-color: var(--accent); }
  .who { font-family: var(--serif); font-size: 1.2rem; }
  .role-tag { color: var(--muted); font-size: .72rem; letter-spacing: .14em;
              text-transform: uppercase; margin-left: .5rem; }
  .score { float: right; font-family: var(--serif); font-size: 1.5rem;
           color: var(--accent); font-variant-numeric: oldstyle-nums; }
  .content { margin-top: .45rem; }
  .reasoning { color: var(--soft); font-style: italic; margin-top: .45rem;
               font-family: var(--serif); font-size: 1.05rem; }

  #detail { display: none; }
  a.back { display: inline-block; margin-bottom: 1.4rem; font-size: .8rem;
           letter-spacing: .14em; text-transform: uppercase; }
</style>
</head>
<body>

<div class="label"><span class="dot"></span>Observation Kernel</div>
<h1>Run archive</h1>

<div id="archive">
  <div class="stats" id="overview"></div>
  <div class="label">Runs</div>
  <table>
    <thead><tr><th class="label">started</th><th class="label">rounds</th>
               <th class="label">works</th><th class="label">critiques</th>
               <th class="label">artists</th><th class="label">critics</th>
               <th class="label">report</th></tr></thead>
    <tbody id="run-list"></tbody>
  </table>
</div>

<div id="detail">
  <a class="back" href="#" onclick="showArchive(); return false;">&larr; all runs</a>
  <div id="detail-body"></div>
</div>

<script>
const RUNS = __RUNS_JSON__;

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function counts(run) {
  return {
    concepts: run.records.filter(r => r.kind === "concept").length,
    evals: run.records.filter(r => r.kind === "evaluation").length,
    rounds: new Set(run.records.map(r => r.round)).size,
    artists: new Set(run.records.filter(r => r.role === "artist").map(r => r.agent)).size,
    critics: new Set(run.records.filter(r => r.role === "critic").map(r => r.agent)).size,
  };
}

function stat(num, label, sub) {
  return `<div class="stat"><div class="label">${label}</div>
          <div class="num">${num}</div><div class="sub">${sub}</div></div>`;
}

function showArchive() {
  document.getElementById("archive").style.display = "block";
  document.getElementById("detail").style.display = "none";
}

function showRun(i) {
  const run = RUNS[i];
  const c = counts(run);
  const rounds = {};
  for (const r of run.records) (rounds[r.round] ??= []).push(r);

  let out = `<div class="label"><span class="dot"></span>Run</div>
             <h2>${esc(run.id)}</h2>
             <div class="when">started ${esc(run.started)}</div>
             <div class="stats">
               ${stat(c.rounds, "rounds", "completed")}
               ${stat(c.concepts, "works generated", "this run")}
               ${stat(c.evals, "critiques published", "this run")}
               ${stat(c.artists, "artists", "active")}
               ${stat(c.critics, "critics", "active")}
             </div>`;

  // Report section: this run's analysis, if analyze.py has been run on it.
  if (run.analysis) {
    const a = run.analysis;
    const themes = a.propagated.length
      ? `<ul class="themes">${a.propagated.map(d => `<li>${esc(d)}</li>`).join("")}</ul>`
      : `<p><em>No descriptor propagated across critics.</em></p>`;

    // Findings, stated as the computed facts behind the research question.
    let findings = `${a.propagated.length} of ${a.n_clusters} descriptor clusters
      coined by one critic later appeared in a different critic's writing
      (similarity threshold ${a.threshold}).`;
    if (a.prior_vocab_size != null) {
      findings += ` Descriptors already present in the starting prompts were
        excluded first: ${a.prior_vocab_subtracted} candidates matched the
        ${a.prior_vocab_size} seeded terms and were removed.`;
    }
    if (a.convergence) {
      findings += ` Critics' judgments were <span class="verdict">${esc(a.convergence.verdict)}</span>:
        the score spread across critics went from ${a.convergence.first_spread}
        in the first round to ${a.convergence.last_spread} in the last.`;
    } else {
      findings += ` Not enough scored critiques to assess convergence.`;
    }

    out += `<div class="report">
            <div class="label">What we're looking for</div>
            <p>Does a new aesthetic descriptor coined by one critic &mdash; present in no
            starting prompt &mdash; propagate to other critics over rounds, and do critics'
            judgments converge or split?</p>
            <div class="label">Findings</div>
            <p>${findings}</p>
            <div class="label">Propagated descriptors</div>
            ${themes}
            <div class="label">Figure</div>
            <p><a href="${esc(a.figure)}" target="_blank">open full figure &nearr;</a></p>
            <iframe src="${esc(a.figure)}" loading="lazy"></iframe></div>`;
  } else {
    out += `<div class="report"><div class="label">Report</div>
            <p>Not analyzed yet. Run <code>uv run python analyze.py logs/${esc(run.id)}.jsonl</code>,
            then regenerate this page.</p></div>`;
  }

  for (const idx of Object.keys(rounds).sort((a, b) => a - b)) {
    out += `<div class="round-label label">Round ${idx}</div>`;
    for (const r of rounds[idx]) {
      const score = r.score != null ? `<span class="score">${r.score.toFixed(2)}</span>` : "";
      out += `<div class="card ${r.kind}">${score}<span class="who">${esc(r.agent)}</span>
              <span class="role-tag">${r.role}</span>
              <div class="content">${esc(r.content)}</div>
              <div class="reasoning">${esc(r.reasoning)}</div></div>`;
    }
  }
  document.getElementById("detail-body").innerHTML = out;
  document.getElementById("archive").style.display = "none";
  document.getElementById("detail").style.display = "block";
  window.scrollTo(0, 0);
}

// Overview stats across all runs, in the spirit of the reference design.
const totals = RUNS.map(counts);
document.getElementById("overview").innerHTML =
  stat(RUNS.length, "runs", "archived") +
  stat(totals.reduce((s, c) => s + c.concepts, 0), "works generated", "all runs") +
  stat(totals.reduce((s, c) => s + c.evals, 0), "critiques published", "all runs") +
  stat(RUNS.length ? totals[0].artists : 0, "artists", "latest run") +
  stat(RUNS.length ? totals[0].critics : 0, "critics", "latest run");

const tbody = document.getElementById("run-list");
RUNS.forEach((run, i) => {
  const c = counts(run);
  const tr = document.createElement("tr");
  tr.className = "run-row";
  tr.innerHTML = `<td class="run-date">${esc(run.started)}</td><td>${c.rounds}</td>
                  <td>${c.concepts}</td><td>${c.evals}</td>
                  <td>${c.artists}</td><td>${c.critics}</td>
                  <td>${run.analysis ? "analyzed" : "&mdash;"}</td>`;
  tr.onclick = () => showRun(i);
  tbody.appendChild(tr);
});
</script>
</body>
</html>
"""


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("No logs found in logs/. Run run.py first.")
    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(PAGE.replace("__RUNS_JSON__", json.dumps(runs)))
    print(f"Wrote {out} ({len(runs)} runs). Open it in a browser.")


if __name__ == "__main__":
    main()
