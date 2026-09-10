"""Is a run's log usable data? Structural checks only, no model calls.

Separate from analyze.py by design. analyze.py answers "what does this run
show?"; this answers "is there anything to show, or was the run malformed?" —
the question worth answering in five seconds before spending an hour of
analysis on a log that is missing a third of its critiques.

Every check here is true or false without judgement: completeness, score
health, duplicate text, length against the band the system prompt asks for,
and provenance from the sidecar. Nothing in here asks whether the critiques are
any GOOD. That would need a model call per record and would quietly become a
second experiment inside the validation step.

Depends only on the base install — no spaCy, no torch — so it runs anywhere the
kernel itself runs. Works on treatment and control logs alike.

Exit code is 0 when nothing is at ERROR level and 1 otherwise, so it can gate a
pipeline. Warnings never fail the run.

Usage:
    uv run python check_run.py                       # newest run_*.jsonl
    uv run python check_run.py logs/run_X.jsonl
    uv run python check_run.py logs/*.jsonl          # several at once
    uv run python check_run.py logs/run_X.jsonl -v   # include INFO detail
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from agents import SHARED_SYSTEM_PROMPT
from schema import Record

LOG_DIR = Path("logs")
AGENTS_FILE = Path("agents.yaml")

# The word band the system prompt asks for. Mirrored here rather than parsed at
# import time so the thresholds are visible; check_prompt_band() below warns if
# the prompt and these numbers drift apart.
WORD_MIN, WORD_MAX = 80, 180

# A critique far under the floor is the shape a clipped response leaves behind.
# Note that outright truncation cannot reach the log at all: a response cut
# mid-JSON fails to parse, Agent.act retries three times and then raises, so
# clipping shows up as a CRASHED run — missing records — not as bad records.
RUNT_WORDS = 40

# Score degeneracy. If critics stop discriminating, the score-spread panel is
# measuring nothing, and vocabulary convergence loses its counterweight.
MIN_SCORE_STD = 0.05
MAX_MODAL_SHARE = 0.5      # more than half the scores at one value
MIN_DISTINCT_SCORES = 3


@dataclass
class Finding:
    level: str        # ERROR | WARN | INFO
    check: str
    detail: str


def error(check, detail): return Finding("ERROR", check, detail)
def warn(check, detail): return Finding("WARN", check, detail)
def info(check, detail): return Finding("INFO", check, detail)


# --- Loading -------------------------------------------------------------------
def load(path: Path) -> tuple[list[Record], list[Finding]]:
    """Parse the log, reporting bad lines rather than dying on them.

    A log with three unparseable lines is still worth checking; refusing to
    look would hide what else is wrong with it.
    """
    findings, records = [], []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(Record(**json.loads(line)))
        except Exception as exc:
            findings.append(error("parse", f"line {lineno}: {type(exc).__name__} {exc}"))
    if not records:
        findings.append(error("parse", "no records parsed — nothing to check"))
    return records, findings


def load_meta(path: Path) -> dict | None:
    meta_path = path.with_suffix(".meta.json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else None


# --- Checks --------------------------------------------------------------------
def check_prompt_band() -> list[Finding]:
    """Guard the guard: warn if the system prompt no longer asks for this band."""
    match = re.search(r'between (\d+) and (\d+) words', SHARED_SYSTEM_PROMPT)
    if not match:
        return [warn("thresholds", "system prompt no longer states a word band; "
                                   f"still checking against {WORD_MIN}-{WORD_MAX}")]
    lo, hi = int(match.group(1)), int(match.group(2))
    if (lo, hi) != (WORD_MIN, WORD_MAX):
        return [warn("thresholds", f"system prompt asks for {lo}-{hi} words but this "
                                   f"check uses {WORD_MIN}-{WORD_MAX} — update "
                                   f"WORD_MIN/WORD_MAX in check_run.py")]
    return []


def check_completeness(records: list[Record]) -> list[Finding]:
    """Every critic critiqued every concept, every round, and nothing dangles."""
    findings = []
    rounds = sorted({r.round for r in records})
    if rounds != list(range(len(rounds))):
        findings.append(error("rounds", f"rounds are not contiguous from 0: {rounds}"))

    critics = sorted({r.agent for r in records if r.role == "critic"})
    if not critics:
        findings.append(error("rounds", "no critic evaluations at all"))
        return findings

    concept_ids = {r.concept_id for r in records if r.kind == "concept"}
    for rnd in rounds:
        concepts = [r for r in records if r.kind == "concept" and r.round == rnd]
        evals = [r for r in records if r.kind == "evaluation" and r.round == rnd]
        if not concepts:
            findings.append(error("completeness", f"round {rnd}: no concepts"))
            continue
        if not evals:
            # The last round of an aborted run; earlier rounds are the problem.
            level = warn if rnd == rounds[-1] else error
            findings.append(level("completeness", f"round {rnd}: concepts but no "
                                                  f"critiques ({len(concepts)} artworks)"))
            continue
        expected = {(c, k.concept_id) for c in critics for k in concepts}
        actual = {(r.agent, r.concept_id) for r in evals}
        missing = expected - actual
        if missing:
            findings.append(error(
                "completeness",
                f"round {rnd}: {len(missing)} of {len(expected)} (critic, artwork) "
                f"pairs missing, e.g. {sorted(missing)[:3]}"))
        extra = Counter(f"{r.agent}|{r.concept_id}" for r in evals)
        dupes = {k: n for k, n in extra.items() if n > 1}
        if dupes:
            findings.append(error("completeness",
                                  f"round {rnd}: duplicate critiques {dupes}"))
        findings.append(info("completeness", f"round {rnd}: {len(concepts)} artworks, "
                                             f"{len(evals)} critiques, {len(critics)} critics"))

    dangling = {r.concept_id for r in records if r.kind == "evaluation"} - concept_ids
    if dangling:
        findings.append(error("integrity", f"critiques reference concepts not in the "
                                           f"log: {sorted(dangling)[:5]}"))
    return findings


def check_lengths(records: list[Record]) -> list[Finding]:
    """Length distribution against the band, and runts that suggest clipping."""
    findings = []
    for kind in ("concept", "evaluation"):
        words = [len(r.content.split()) for r in records if r.kind == kind]
        if not words:
            continue
        runts = [w for w in words if w < RUNT_WORDS]
        over = [w for w in words if w > WORD_MAX]
        under = [w for w in words if w < WORD_MIN]
        findings.append(info("length", f"{kind}: n={len(words)} "
                                       f"min={min(words)} median={int(statistics.median(words))} "
                                       f"max={max(words)}"))
        if runts:
            findings.append(warn("length", f"{kind}: {len(runts)} under {RUNT_WORDS} words "
                                           f"— check for clipped responses"))
        # Not an error: the model routinely overruns a soft word instruction.
        # It matters because it drives output tokens, which is the rate limit.
        if len(over) > len(words) * 0.5:
            findings.append(warn("length", f"{kind}: {len(over)}/{len(words)} over "
                                           f"{WORD_MAX} words — the band is not holding"))
        elif under and len(under) > len(words) * 0.5:
            findings.append(warn("length", f"{kind}: {len(under)}/{len(words)} under "
                                           f"{WORD_MIN} words"))
    return findings


def check_scores(records: list[Record]) -> list[Finding]:
    """Are critics discriminating, or rubber-stamping?"""
    findings = []
    scores = [r.score for r in records if r.kind == "evaluation" and r.score is not None]
    missing = [r for r in records if r.kind == "evaluation" and r.score is None]
    if missing:
        findings.append(error("scores", f"{len(missing)} critiques have no score"))
    if not scores:
        return findings

    out_of_range = [s for s in scores if not 0.0 <= s <= 1.0]
    if out_of_range:
        findings.append(error("scores", f"{len(out_of_range)} scores outside 0.0-1.0: "
                                        f"{sorted(set(out_of_range))[:5]}"))

    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    modal_value, modal_n = Counter(scores).most_common(1)[0]
    findings.append(info("scores", f"n={len(scores)} min={min(scores):.2f} "
                                   f"median={statistics.median(scores):.2f} "
                                   f"max={max(scores):.2f} std={std:.3f} "
                                   f"distinct={len(set(scores))}"))
    if std < MIN_SCORE_STD:
        findings.append(warn("scores", f"std {std:.3f} below {MIN_SCORE_STD} — critics are "
                                       f"barely discriminating; score spread will be noise"))
    if modal_n > len(scores) * MAX_MODAL_SHARE:
        findings.append(warn("scores", f"{modal_n}/{len(scores)} scores are exactly "
                                       f"{modal_value} — degenerate"))
    if len(set(scores)) < MIN_DISTINCT_SCORES:
        findings.append(warn("scores", f"only {len(set(scores))} distinct score values"))

    # A critic with no variance is not evaluating, whatever the run average says.
    per_critic = defaultdict(list)
    for r in records:
        if r.kind == "evaluation" and r.score is not None:
            per_critic[r.agent].append(r.score)
    for critic, vals in sorted(per_critic.items()):
        if len(vals) > 2 and statistics.pstdev(vals) == 0.0:
            findings.append(warn("scores", f"{critic} gave the same score "
                                           f"({vals[0]}) to all {len(vals)} artworks"))
    return findings


def check_duplicates(records: list[Record]) -> list[Finding]:
    """Identical text is a bug, not convergence."""
    findings = []
    by_text = defaultdict(list)
    for r in records:
        by_text[" ".join(r.content.split())].append(r)
    for text, group in by_text.items():
        if len(group) < 2:
            continue
        who = sorted({f"{r.agent}@r{r.round}" for r in group})
        # Concepts repeat legitimately in a control log: the artworks are
        # replayed, and each replayed round re-logs them.
        kinds = {r.kind for r in group}
        level = warn if kinds == {"concept"} else error
        findings.append(level("duplicates", f"{len(group)} records share identical "
                                            f"content ({', '.join(who[:4])}): "
                                            f"{text[:60]!r}"))
    return findings


def check_provenance(path: Path, records: list[Record], meta: dict | None) -> list[Finding]:
    """The sidecar, and whether it agrees with the log."""
    findings = []
    if meta is None:
        findings.append(warn("provenance", f"no {path.with_suffix('.meta.json').name} — "
                                           f"the feed window and roster this run used are "
                                           f"not recoverable"))
        return findings

    window = meta.get("feed_window")
    findings.append(info("provenance",
                         f"model={meta.get('model')} feed_window={window} "
                         f"started={meta.get('started_at')}"))
    n_rounds = len({r.round for r in records})
    if window is not None and n_rounds <= window:
        findings.append(info("provenance",
                             f"{n_rounds} rounds at window {window}: the window never "
                             f"engaged — nothing aged out of any feed"))
    declared = meta.get("rounds") or meta.get("n_replayed_rounds")
    if declared is not None and declared != n_rounds:
        findings.append(warn("provenance", f"sidecar says {declared} rounds, log has "
                                           f"{n_rounds} — the run did not finish"))
    if AGENTS_FILE.exists() and meta.get("agents_file_sha256"):
        import hashlib
        current = hashlib.sha256(AGENTS_FILE.read_bytes()).hexdigest()
        if current != meta["agents_file_sha256"]:
            findings.append(warn("provenance",
                                 "agents.yaml has changed since this run — re-analysis "
                                 "will subtract the CURRENT prior vocabulary, not the "
                                 "one the run was produced under"))
    return findings


def check_log(path: Path) -> list[Finding]:
    records, findings = load(path)
    if not records:
        return findings
    findings += check_prompt_band()
    findings += check_completeness(records)
    findings += check_lengths(records)
    findings += check_scores(records)
    findings += check_duplicates(records)
    findings += check_provenance(path, records, load_meta(path))
    return findings


# --- Reporting -----------------------------------------------------------------
ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


def report(path: Path, findings: list[Finding], verbose: bool) -> int:
    """Print one log's findings, worst first. Returns the number of errors."""
    counts = Counter(f.level for f in findings)
    print(f"\n{path.name}")
    shown = [f for f in findings if verbose or f.level != "INFO"]
    for f in sorted(shown, key=lambda f: (ORDER[f.level], f.check)):
        print(f"  {f.level:<5} {f.check:<13} {f.detail}")
    if not shown:
        print("  (nothing to report; -v for detail)")
    verdict = "USABLE" if not counts["ERROR"] else "MALFORMED"
    print(f"  -> {verdict}  ({counts['ERROR']} errors, {counts['WARN']} warnings)")
    return counts["ERROR"]


def main() -> None:
    p = argparse.ArgumentParser(description="Check whether a run's log is usable data.")
    p.add_argument("logs", nargs="*", type=Path,
                   help="log files (default: the newest run_*.jsonl)")
    p.add_argument("-v", "--verbose", action="store_true", help="include INFO findings")
    args = p.parse_args()

    paths = args.logs
    if not paths:
        found = sorted(LOG_DIR.glob("run_*.jsonl"))
        if not found:
            sys.exit("No run_*.jsonl logs found.")
        paths = [found[-1]]

    errors = 0
    for path in paths:
        if not path.exists():
            print(f"\n{path}\n  ERROR missing       file not found")
            errors += 1
            continue
        errors += report(path, check_log(path), args.verbose)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
