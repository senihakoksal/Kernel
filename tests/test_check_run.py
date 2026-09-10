"""The run health check must be able to fail.

A validator that only ever says USABLE is worse than none: it converts an
unexamined log into a log that looks examined. So each check gets a fixture
that trips it.

    uv run pytest tests/test_check_run.py -v
"""

import json

import pytest

import check_run
from check_run import check_log

CRITICS = ["critic_x", "critic_y"]
ARTISTS = ["artist_a", "artist_b"]


def rec(**kw) -> dict:
    base = {"round": 0, "agent": "artist_a", "role": "artist", "model": "stub",
            "kind": "concept", "concept_id": "r0-artist_a", "title": "t",
            "content": " ".join(["word"] * 120), "score": None, "reasoning": "r"}
    return {**base, **kw}


def healthy_rows(rounds: int = 2) -> list[dict]:
    """A well-formed log: every critic critiques every artwork, scores vary."""
    rows, n = [], 0
    for rnd in range(rounds):
        concepts = [rec(round=rnd, agent=a, concept_id=f"r{rnd}-{a}") for a in ARTISTS]
        rows += concepts
        for critic in CRITICS:
            for k in concepts:
                n += 1
                rows.append(rec(round=rnd, agent=critic, role="critic",
                                kind="evaluation", concept_id=k["concept_id"],
                                title=None, score=round(0.3 + 0.07 * n, 2),
                                content=" ".join(["crit"] * (100 + n))))
    return rows


def write_log(tmp_path, rows) -> "object":
    path = tmp_path / "run_stub.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def levels(findings, level) -> list[str]:
    return [f"{f.check}: {f.detail}" for f in findings if f.level == level]


# --- the happy path -----------------------------------------------------------
def test_a_healthy_log_has_no_errors(tmp_path):
    findings = check_log(write_log(tmp_path, healthy_rows()))
    assert levels(findings, "ERROR") == []


# --- each check trips ---------------------------------------------------------
def test_missing_critic_artwork_pair_is_an_error(tmp_path):
    rows = healthy_rows()
    dropped = next(r for r in rows if r["kind"] == "evaluation" and r["round"] == 0)
    rows.remove(dropped)
    errs = levels(check_log(write_log(tmp_path, rows)), "ERROR")
    assert any("pairs missing" in e for e in errs), errs


def test_duplicate_critique_text_is_an_error(tmp_path):
    rows = healthy_rows()
    evals = [r for r in rows if r["kind"] == "evaluation"]
    evals[1]["content"] = evals[0]["content"]
    errs = levels(check_log(write_log(tmp_path, rows)), "ERROR")
    assert any("identical" in e for e in errs), errs


def test_replayed_concepts_are_only_a_warning(tmp_path):
    """A control log re-logs the same artworks each round; that is by design."""
    rows = healthy_rows()
    concepts = [r for r in rows if r["kind"] == "concept"]
    concepts[1]["content"] = concepts[0]["content"]
    findings = check_log(write_log(tmp_path, rows))
    assert not any("identical" in e for e in levels(findings, "ERROR"))
    assert any("identical" in w for w in levels(findings, "WARN"))


def test_dangling_concept_id_is_an_error(tmp_path):
    rows = healthy_rows()
    for r in rows:
        if r["kind"] == "evaluation":
            r["concept_id"] = None
    errs = levels(check_log(write_log(tmp_path, rows)), "ERROR")
    assert any("not in the log" in e for e in errs), errs


def test_score_out_of_range_is_an_error(tmp_path):
    rows = healthy_rows()
    next(r for r in rows if r["kind"] == "evaluation")["score"] = 1.4
    errs = levels(check_log(write_log(tmp_path, rows)), "ERROR")
    assert any("outside 0.0-1.0" in e for e in errs), errs


def test_missing_score_is_an_error(tmp_path):
    rows = healthy_rows()
    next(r for r in rows if r["kind"] == "evaluation")["score"] = None
    errs = levels(check_log(write_log(tmp_path, rows)), "ERROR")
    assert any("no score" in e for e in errs), errs


def test_degenerate_scores_are_warned(tmp_path):
    """Every critique scored the same: the spread panel would be measuring noise."""
    rows = healthy_rows()
    for r in rows:
        if r["kind"] == "evaluation":
            r["score"] = 0.5
    warns = levels(check_log(write_log(tmp_path, rows)), "WARN")
    assert any("degenerate" in w for w in warns), warns
    assert any("barely discriminating" in w for w in warns), warns


def test_a_rubber_stamping_critic_is_warned(tmp_path):
    """One critic with zero variance, hidden inside a healthy run average."""
    rows = healthy_rows(rounds=3)
    for r in rows:
        if r["kind"] == "evaluation" and r["agent"] == "critic_x":
            r["score"] = 0.7
    warns = levels(check_log(write_log(tmp_path, rows)), "WARN")
    assert any("critic_x gave the same score" in w for w in warns), warns


def test_runt_content_is_warned(tmp_path):
    rows = healthy_rows()
    next(r for r in rows if r["kind"] == "evaluation")["content"] = "far too short"
    warns = levels(check_log(write_log(tmp_path, rows)), "WARN")
    assert any("clipped" in w for w in warns), warns


def test_unparseable_line_is_an_error(tmp_path):
    path = tmp_path / "run_stub.jsonl"
    rows = healthy_rows()
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{not json\n")
    errs = levels(check_log(path), "ERROR")
    assert any("parse" in e for e in errs), errs


def test_missing_round_of_critiques_is_an_error_but_a_trailing_one_is_not(tmp_path):
    """An aborted final round is a stopped run; a hole in the middle is a bug."""
    rows = [r for r in healthy_rows(rounds=3)
            if not (r["kind"] == "evaluation" and r["round"] == 2)]
    findings = check_log(write_log(tmp_path, rows))
    assert not any("no critiques" in e for e in levels(findings, "ERROR"))
    assert any("no critiques" in w for w in levels(findings, "WARN"))

    rows = [r for r in healthy_rows(rounds=3)
            if not (r["kind"] == "evaluation" and r["round"] == 1)]
    errs = levels(check_log(write_log(tmp_path, rows)), "ERROR")
    assert any("no critiques" in e for e in errs), errs


# --- provenance ---------------------------------------------------------------
def test_sidecar_is_read_and_a_short_run_is_noted(tmp_path):
    path = write_log(tmp_path, healthy_rows(rounds=2))
    (tmp_path / "run_stub.meta.json").write_text(json.dumps(
        {"model": "m", "feed_window": 3, "rounds": 2, "started_at": "now"}))
    infos = levels(check_log(path), "INFO")
    assert any("feed_window=3" in i for i in infos), infos
    assert any("never engaged" in i for i in infos), infos


def test_unfinished_run_is_warned(tmp_path):
    path = write_log(tmp_path, healthy_rows(rounds=2))
    (tmp_path / "run_stub.meta.json").write_text(json.dumps(
        {"model": "m", "feed_window": 3, "rounds": 8}))
    warns = levels(check_log(path), "WARN")
    assert any("did not finish" in w for w in warns), warns


def test_missing_sidecar_is_warned(tmp_path):
    warns = levels(check_log(write_log(tmp_path, healthy_rows())), "WARN")
    assert any("meta.json" in w for w in warns), warns


# --- the guard on the guard ---------------------------------------------------
def test_word_band_drift_is_warned(monkeypatch):
    """If the system prompt's band changes, this checker's thresholds are stale."""
    monkeypatch.setattr(check_run, "SHARED_SYSTEM_PROMPT",
                        'Keep "content" between 20 and 40 words.')
    warns = levels(check_run.check_prompt_band(), "WARN")
    assert any("update WORD_MIN/WORD_MAX" in w for w in warns), warns


def test_word_band_matches_the_shipped_prompt():
    assert check_run.check_prompt_band() == []
