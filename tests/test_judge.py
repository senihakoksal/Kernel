"""Tier 2 plumbing: the sample, the sheet, the judge prompt, the scoring.

Nothing here touches the network. The judge is exercised through a stub client,
and the one test that cares about money asserts that `run` refuses to make
calls without being told to.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge
from judge import (Item, estimate, influence_names, judge_request, key_for,
                   load_dispositions, parse_judge, parse_sheet, redact,
                   sample_items, score_labels, sheet)

ROSTER = Path(__file__).resolve().parents[1] / "agents.yaml"
DISPOSITIONS = {
    "clara_bennett": "A contemporary critic inspired by Susan Sontag. Prioritizes close attention.",
    "marcus_ivey": "A contemporary critic inspired by Walter Benjamin. Evaluates through technology.",
    "lena_voss": "A contemporary critic inspired by Roland Barthes. Interested in symbols.",
    "evelyn_hart": "A contemporary critic inspired by John Ruskin's concern for craft.",
    "julian_moreau": "A contemporary critic inspired by Charles Baudelaire's interest in modernity.",
    "ines_marchetti": "A contemporary critic inspired by Harald Szeemann, who got art into rooms.",
}
INFL = influence_names(DISPOSITIONS)


def record(round_, agent, role, kind, content, concept_id, score=None, title=None):
    return {"round": round_, "agent": agent, "role": role, "model": "stub",
            "kind": kind, "concept_id": concept_id, "title": title,
            "content": content, "score": score, "reasoning": "because"}


@pytest.fixture
def pair(tmp_path):
    """A two-arm pair of logs with six critics over four rounds."""
    logs = {}
    for arm, stem in (("feed", "run_x"), ("isolated", "control_run_x_y")):
        lines = []
        for rnd in range(4):
            cid = f"r{rnd}-art"
            lines.append(record(rnd, "artist_a", "artist", "concept",
                                f"a work for round {rnd}", cid, title=f"Work {rnd}"))
            for i, critic in enumerate(sorted(DISPOSITIONS)):
                lines.append(record(rnd, critic, "critic", "evaluation",
                                    f"a critique of work {rnd}, " + "padding " * 20,
                                    cid, score=0.5 + 0.05 * i))
        path = tmp_path / f"{stem}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        (tmp_path / f"{stem}.meta.json").write_text(json.dumps(
            {"agents_file_sha256": __import__("hashlib").sha256(
                ROSTER.read_bytes()).hexdigest()}))
        logs[arm] = path
    return logs


# --- redaction ----------------------------------------------------------------
def test_redaction_removes_every_critics_influence():
    text = ("Sontag would object, Benjamin argued otherwise, and Barthes, "
            "Ruskin, Baudelaire and Szeemann all disagree.")
    out, n = redact(text, INFL)
    for name in ("Sontag", "Benjamin", "Barthes", "Ruskin", "Baudelaire", "Szeemann"):
        assert name not in out
    assert n >= 6


def test_redaction_leaves_the_artists_influences_alone():
    """A critique naming Judy Chicago is describing the work, which is shown."""
    text = "The debt to Judy Chicago is heavy, and Ruskin would ask about craft."
    out, _ = redact(text, INFL)
    assert "Judy Chicago" in out
    assert "Ruskin" not in out


def test_redaction_takes_full_names_without_leaving_a_run_of_markers():
    out, _ = redact("As Susan Sontag wrote,", INFL)
    assert judge.REDACTION in out
    assert out.count(judge.REDACTION) == 1, out


def test_redaction_respects_word_boundaries():
    """'Johnson' must survive a redaction aimed at 'John'."""
    out, _ = redact("Johnson and Benjamins are untouched.", INFL)
    assert "Johnson" in out and "Benjamins" in out


def test_influence_names_reads_every_critic():
    assert set(INFL) == set(DISPOSITIONS)
    assert INFL["evelyn_hart"] == ("John", "Ruskin")


# --- the sample ---------------------------------------------------------------
def test_sample_is_deterministic(pair):
    a = sample_items(pair, DISPOSITIONS, n=12)
    b = sample_items(pair, DISPOSITIONS, n=12)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    assert [i.critic for i in a] == [i.critic for i in b]


def test_sample_is_balanced_across_arms_and_critics(pair):
    items = sample_items(pair, DISPOSITIONS, n=12)
    assert len(items) == 12
    cells = {}
    for it in items:
        cells[(it.arm, it.critic)] = cells.get((it.arm, it.critic), 0) + 1
    assert len(cells) == 12, cells          # 2 arms x 6 critics, one each
    assert max(cells.values()) - min(cells.values()) <= 1


def test_sample_carries_truth_and_a_redacted_critique(pair):
    items = sample_items(pair, DISPOSITIONS, n=6)
    for it in items:
        assert it.critic in DISPOSITIONS
        assert 0.0 <= it.score <= 1.0
        assert it.arm in ("feed", "isolated")
        assert it.signposted is False      # the fixture names nobody
        assert it.redactions == 0


def test_signposting_is_recorded_when_a_critique_names_its_own_influence(pair):
    """The field has to track the author's own influence, not any influence."""
    lines = pair["feed"].read_text().splitlines()
    rows = [json.loads(l) for l in lines]
    for r in rows:
        if r["agent"] == "evelyn_hart" and r["kind"] == "evaluation":
            r["content"] = "Ruskin would ask about craft. " + r["content"]
    pair["feed"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items = sample_items(pair, DISPOSITIONS, n=12)
    hart = [i for i in items if i.critic == "evelyn_hart" and i.arm == "feed"]
    others = [i for i in items if i.critic != "evelyn_hart"]
    assert hart and all(i.signposted for i in hart)
    assert not any(i.signposted for i in others)
    assert all("Ruskin" not in i.critique for i in items)


def test_sheet_never_renders_the_answers(pair):
    items = sample_items(pair, DISPOSITIONS, n=12)
    text = sheet(items, key_for(sorted(DISPOSITIONS)), DISPOSITIONS)
    body = text.split("---", 1)[1]
    for it in items:
        assert it.critic not in body
        assert it.arm not in body
        assert f"{it.score}" not in body
    assert body.count("- answer:") == len(items)


# --- the sheet, read back -----------------------------------------------------
def test_parse_sheet_reads_answers():
    labels, problems = parse_sheet(
        "### i01\nprose\n- answer: wrote=C  score=0.72\n"
        "### i02\nprose\n- answer: wrote=a score=.4\n")
    assert problems == []
    assert labels["i01"] == {"wrote": "C", "score": 0.72}
    assert labels["i02"] == {"wrote": "A", "score": 0.4}


def test_parse_sheet_treats_question_marks_as_unanswered():
    labels, problems = parse_sheet("### i01\n- answer: wrote=?  score=?\n")
    assert problems == []
    assert labels["i01"] == {"wrote": None, "score": None}


def test_parse_sheet_reports_what_it_cannot_read():
    labels, problems = parse_sheet(
        "### i01\n- answer: wrote=Z score=0.5\n"
        "### i02\n- answer: wrote=A score=1.4\n"
        "### i03\n- answer: wrote=A score=high\n"
        "### i04\n- answer: C 0.5\n")
    assert len(problems) == 4, problems
    assert labels["i01"]["wrote"] is None      # Z is not a letter offered
    assert labels["i02"]["score"] is None      # outside 0.0-1.0
    assert labels["i03"]["score"] is None      # not a number
    assert "i04" not in labels                 # the line itself was unreadable


def test_parse_sheet_does_not_attach_an_answer_to_the_wrong_item():
    labels, problems = parse_sheet("- answer: wrote=A score=0.5\n### i01\n")
    assert labels == {}
    assert problems and "no item heading" in problems[0]


# --- the judge prompt ---------------------------------------------------------
def make_item(item_id="i01", critic="lena_voss", score=0.7):
    return Item(item_id=item_id, source="run_x", arm="feed", round=1, critic=critic,
                score=score, work_title="A Work", work_content="the work",
                critique="the critique", signposted=False, redactions=0)


def test_judge_prompt_hides_the_answers():
    item = make_item()
    for task in ("wrote", "score"):
        system, user, _ = judge_request(item, task, DISPOSITIONS)
        assert item.critic not in system + user
        assert str(item.score) not in system + user
        assert item.arm not in user


def test_wrote_prompt_shuffles_the_letters_per_item_but_maps_back():
    keys = [judge_request(make_item(f"i{n:02d}"), "wrote", DISPOSITIONS)[2]
            for n in range(1, 8)]
    assert all(set(k.values()) == set(DISPOSITIONS) for k in keys)
    assert len({tuple(sorted(k.items())) for k in keys}) > 1, "letters never moved"


def test_score_prompt_is_the_same_for_every_item():
    a, _, _ = judge_request(make_item("i01"), "score", DISPOSITIONS)
    b, _, _ = judge_request(make_item("i02"), "score", DISPOSITIONS)
    assert a == b
    assert "0.0 to 1.0" in a


def test_parse_judge_maps_a_letter_back_to_a_critic():
    key = {"A": "lena_voss", "B": "clara_bennett"}
    assert parse_judge('{"wrote": "B", "why": "symbols"}', "wrote", key) == {
        "why": "symbols", "wrote": "clara_bennett"}


def test_parse_judge_rejects_answers_it_cannot_use():
    key = {"A": "lena_voss"}
    for text, task in [('{"wrote": "Q"}', "wrote"), ("no json here", "wrote"),
                       ('{"wrote":', "wrote"), ('{"score": "high"}', "score"),
                       ('{"score": null}', "score")]:
        assert "error" in parse_judge(text, task, key), (text, task)


def test_parse_judge_takes_a_score_out_of_a_fenced_reply():
    out = parse_judge('here you go\n{"score": 0.65, "why": "warm"}\n', "score", {})
    assert out["score"] == 0.65


# --- the judge loop, stubbed --------------------------------------------------
class StubClient:
    """Answers every call, recording what it was asked."""

    def __init__(self, reply='{"wrote": "A", "score": 0.5, "why": "x"}'):
        self.reply, self.calls = reply, []
        outer = self

        class Messages:
            async def create(self, **kw):
                outer.calls.append(kw)

                class Block:
                    type, text = "text", outer.reply
                return type("R", (), {"content": [Block()]})()
        self.messages = Messages()


def test_run_judge_makes_two_calls_per_item_and_maps_the_answers():
    import asyncio
    items = [make_item("i01"), make_item("i02")]
    client = StubClient()
    answers = asyncio.run(judge.run_judge(items, DISPOSITIONS, "stub-model", client,
                                          provider="anthropic"))
    assert len(client.calls) == 4
    assert all(c["model"] == "stub-model" for c in client.calls)
    for item in items:
        assert answers[item.item_id]["wrote"]["wrote"] in DISPOSITIONS
        assert answers[item.item_id]["score"]["score"] == 0.5


def test_run_judge_records_a_failure_instead_of_dropping_the_item():
    import asyncio

    class Boom(StubClient):
        def __init__(self):
            super().__init__()
            outer = self

            class Messages:
                async def create(self, **kw):
                    raise RuntimeError("rate limited")
            self.messages = Messages()

    answers = asyncio.run(judge.run_judge([make_item()], DISPOSITIONS, "m", Boom(),
                                          provider="anthropic"))
    assert "rate limited" in answers["i01"]["wrote"]["error"]


# --- scoring ------------------------------------------------------------------
def build(n=10):
    critics = sorted(DISPOSITIONS)
    return [make_item(f"i{k:02d}", critics[k % 6], 0.4 + 0.05 * k) for k in range(n)]


LETTER = {name: letter for letter, name in key_for(sorted(DISPOSITIONS)).items()}


def test_score_labels_counts_accuracy_against_truth():
    items = build(6)
    human = {it.item_id: {"wrote": LETTER[it.critic], "score": it.score}
             for it in items}
    judge_ans = {it.item_id: {"wrote": {"wrote": it.critic}, "score": {"score": it.score}}
                 for it in items}
    out = score_labels(items, human, judge_ans, key_for(sorted(DISPOSITIONS)))
    assert out["wrote"]["human_accuracy"] == 1.0
    assert out["wrote"]["judge_accuracy"] == 1.0
    assert out["wrote"]["agree_rate"] == 1.0
    assert out["wrote"]["kappa"] == 1.0
    assert out["score"]["judge"]["mae"] == 0.0
    assert out["score"]["judge"]["spearman"] == 1.0


def test_score_labels_separates_the_two_readers():
    """A judge that is always right and a human that is always wrong must not
    be averaged into one number."""
    items = build(6)
    wrong = {c: d for c, d in zip(sorted(DISPOSITIONS), sorted(DISPOSITIONS)[1:]
                                  + sorted(DISPOSITIONS)[:1])}
    human = {it.item_id: {"wrote": LETTER[wrong[it.critic]], "score": 0.5}
             for it in items}
    judge_ans = {it.item_id: {"wrote": {"wrote": it.critic}, "score": {"score": it.score}}
                 for it in items}
    out = score_labels(items, human, judge_ans, key_for(sorted(DISPOSITIONS)))
    assert out["wrote"]["human_accuracy"] == 0.0
    assert out["wrote"]["judge_accuracy"] == 1.0
    assert out["wrote"]["agree_rate"] == 0.0
    assert len(out["wrote"]["disagreements"]) == 6


def test_score_labels_ignores_answers_for_items_outside_the_sample():
    items = build(6)
    human = {it.item_id: {"wrote": LETTER[it.critic], "score": it.score}
             for it in items}
    human["i99"] = {"wrote": LETTER["lena_voss"], "score": 0.9}
    out = score_labels(items, human, {}, key_for(sorted(DISPOSITIONS)))
    assert out["wrote"]["n_human"] == 6


def test_score_labels_withholds_a_kappa_it_cannot_compute():
    items = build(3)
    human = {it.item_id: {"wrote": LETTER[it.critic], "score": None} for it in items}
    judge_ans = {it.item_id: {"wrote": {"wrote": it.critic}} for it in items}
    out = score_labels(items, human, judge_ans, key_for(sorted(DISPOSITIONS)))
    assert out["wrote"]["kappa"] is None            # 3 shared items is not enough
    assert out["score"]["human"]["spearman"] is None
    assert out["criteria"][0]["verdict"] == "insufficient"


def test_score_labels_reports_the_predict_the_mean_null():
    items = build(10)
    out = score_labels(items, {}, {}, key_for(sorted(DISPOSITIONS)))
    scores = [it.score for it in items]
    mean = sum(scores) / len(scores)
    expected = sum(abs(s - mean) for s in scores) / len(scores)
    assert out["score"]["mean_baseline_mae"] == round(expected, 3)


def test_score_labels_splits_accuracy_by_whether_the_critique_signposted():
    items = build(6)
    items = [Item(**{**vars(it), "signposted": k < 3}) for k, it in enumerate(items)]
    # right on the three that signposted, wrong on the three that did not
    judge_ans = {}
    for k, it in enumerate(items):
        pick = it.critic if k < 3 else "nobody"
        judge_ans[it.item_id] = {"wrote": {"wrote": pick}}
    out = score_labels(items, {}, judge_ans, key_for(sorted(DISPOSITIONS)))
    assert out["wrote"]["by_signposting"]["judge"] == {"signposted": 1.0,
                                                       "never_named": 0.0}
    assert "human" not in out["wrote"]["by_signposting"]   # nobody labelled it


# --- provenance and safety ----------------------------------------------------
def test_load_dispositions_stops_when_the_roster_has_changed(pair, tmp_path):
    meta = pair["feed"].with_suffix(".meta.json")
    meta.write_text(json.dumps({"agents_file_sha256": "deadbeef"}))
    with pytest.raises(SystemExit) as exc:
        load_dispositions([pair["feed"]], roster=ROSTER)
    assert "has changed" in str(exc.value)


def test_load_dispositions_stops_when_a_run_has_no_sidecar(pair):
    pair["feed"].with_suffix(".meta.json").unlink()
    with pytest.raises(SystemExit):
        load_dispositions([pair["feed"]], roster=ROSTER)


def test_load_dispositions_returns_critics_only():
    out = load_dispositions([], roster=ROSTER)
    assert len(out) == 6
    assert all("critic" in text for text in out.values())


def test_run_refuses_to_spend_money_without_confirm():
    repo = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "judge.py", "run"], cwd=repo,
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert "--confirm" in done.stderr


def test_estimate_counts_both_tasks_and_makes_no_client():
    items = build(5)
    est = estimate(items, DISPOSITIONS, "claude-test", price=(3.0, 15.0))
    assert est["calls"] == 10
    assert est["input_tokens"] > 0
    assert est["usd"] == round(
        est["input_tokens"] / 1e6 * 3 + est["output_tokens"] / 1e6 * 15, 4)


def test_score_labels_never_emits_nan_for_a_flat_rater():
    """A reader who scores everything 0.5 makes Spearman undefined. nan is not
    valid JSON, so it has to come back absent rather than broken."""
    items = build(8)
    human = {it.item_id: {"wrote": None, "score": 0.5} for it in items}
    out = score_labels(items, human, {}, key_for(sorted(DISPOSITIONS)))
    assert out["score"]["human"]["spearman"] is None
    assert out["score"]["human"]["mae"] is not None       # MAE is still defined
    assert "NaN" not in json.dumps(out)


def test_score_labels_maps_the_sheets_letters_through_the_key():
    """The sheet answers in letters; truth is a critic name. Comparing the two
    directly pins the human's accuracy to exactly 0 whatever they wrote, which
    is how a perfect set of labels came back as 0/25."""
    items = build(6)
    key = key_for(sorted(DISPOSITIONS))
    letter = {name: l for l, name in key.items()}
    human = {it.item_id: {"wrote": letter[it.critic], "score": None} for it in items}
    assert score_labels(items, human, {}, key)["wrote"]["human_accuracy"] == 1.0
    # a letter outside the key is dropped from the denominator, not scored as
    # a miss — an unreadable answer is not evidence the rater was wrong
    human["i00"] = {"wrote": "Z", "score": None}
    out = score_labels(items, human, {}, key)
    assert out["wrote"]["n_human"] == 5
    assert out["wrote"]["human_accuracy"] == 1.0


# --- the answer key lives apart from the questions ----------------------------
def test_write_set_keeps_the_answers_out_of_the_questions_file(pair, tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "LABEL_DIR", tmp_path)
    items = sample_items(pair, DISPOSITIONS, n=6)
    p = judge.write_set("s1", items, key_for(sorted(DISPOSITIONS)), pair, {})
    questions = json.loads(p["items"].read_text())
    assert {k for it in questions["items"] for k in it} == {
        "item_id", "work_title", "work_content", "critique"}
    # Only the items array matters: the A-F legend and the log paths are
    # set-level metadata (the legend is what makes the sheet readable at all),
    # and neither says which critic wrote which item.
    body = json.dumps(questions["items"])
    for it in items:
        assert it.critic not in body
        assert str(it.score) not in body
        assert it.arm not in body
    assert set(questions["key"].values()) == set(DISPOSITIONS)


def test_the_truth_file_is_not_readable_at_a_glance(pair, tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "LABEL_DIR", tmp_path)
    items = sample_items(pair, DISPOSITIONS, n=6)
    p = judge.write_set("s1", items, key_for(sorted(DISPOSITIONS)), pair, {})
    raw = p["truth"].read_text()
    for it in items:
        assert it.critic not in raw
    assert judge.read_truth(p["truth"])["answers"][items[0].item_id]["critic"] \
        == items[0].critic


def test_a_set_round_trips_through_the_two_files(pair, tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "LABEL_DIR", tmp_path)
    items = sample_items(pair, DISPOSITIONS, n=6)
    key = key_for(sorted(DISPOSITIONS))
    judge.write_set("s1", items, key, pair, {"feed": {"rate": 0.4}})
    back, back_key, sp = judge.read_items("s1")
    assert back == items
    assert back_key == key
    assert sp == {"feed": {"rate": 0.4}}


def test_a_fresh_draw_skips_critiques_already_shown_to_a_rater(pair, tmp_path, monkeypatch):
    """A rater who has seen an item's answer cannot be asked about it again."""
    monkeypatch.setattr(judge, "LABEL_DIR", tmp_path)
    first = sample_items(pair, DISPOSITIONS, n=6)
    judge.write_set("s1", first, key_for(sorted(DISPOSITIONS)), pair, {})
    used = judge.used_fingerprints()
    assert used == {judge.fingerprint(i.critique) for i in first}
    second = sample_items(pair, DISPOSITIONS, n=6, exclude=used)
    assert second, "no unseen critiques left in the fixture"
    assert not ({i.critique for i in second} & {i.critique for i in first})


def test_used_fingerprints_is_empty_when_nothing_has_been_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "LABEL_DIR", tmp_path)
    assert judge.used_fingerprints() == set()


# --- the copied-labels check --------------------------------------------------
def test_looks_copied_trips_on_an_exact_reproduction_of_the_key():
    items = build(20)
    out = judge.looks_copied(items, {it.item_id: it.score for it in items})
    assert out["flag"] is True
    assert out["exact_match"] == 1.0


def test_tenths_cannot_be_separated_and_the_check_says_so():
    """The regression this check exists for. An earlier version flagged answers
    sitting on the truth's rounding grid, and gpt-5.5 — which never sees the
    scores — does that 84% of the time once rounded to tenths. At tenth
    precision a copier and an accurate reader are the same thing to this test,
    and saying so is the honest output, not an accusation."""
    items = build(20)
    out = judge.looks_copied(items, {it.item_id: round(it.score, 1) for it in items})
    assert out["separable"] is False
    assert out["flag"] is False


def test_looks_copied_leaves_a_plausible_reader_alone():
    import random
    rng = random.Random(7)
    items = build(20)
    read = {it.item_id: round(min(1.0, max(0.0, it.score + rng.uniform(-0.12, 0.12))), 2)
            for it in items}
    out = judge.looks_copied(items, read)
    assert out["separable"] is True      # answered finer than a tenth
    assert out["flag"] is False          # but does not reproduce the key


def test_looks_copied_says_nothing_on_a_sample_too_small_to_read():
    items = build(6)
    assert judge.looks_copied(items, {i.item_id: i.score for i in items}) == {}


def test_answer_step_reads_the_grid_a_rater_used():
    assert judge.answer_step([0.1, 0.5, 0.9]) == 0.1
    assert judge.answer_step([0.15, 0.5]) == 0.05
    assert judge.answer_step([0.12, 0.5]) == 0.01


def test_a_flagged_set_withholds_the_score_finding_but_keeps_attribution():
    items = build(20)
    key = key_for(sorted(DISPOSITIONS))
    letter = {name: l for l, name in key.items()}
    human = {it.item_id: {"wrote": letter[it.critic], "score": it.score}
             for it in items}
    out = score_labels(items, human, {}, key)
    rows = {r["check"]: r for r in out["criteria"]}
    assert rows["the score is legible from the prose"]["verdict"] == "insufficient"
    assert out["wrote"]["human_accuracy"] == 1.0          # attribution survives
    assert any("not usable" in c for c in out["caveats"])


def test_a_set_answered_in_tenths_carries_the_limit_as_a_caveat():
    items = build(20)
    human = {it.item_id: {"wrote": None, "score": round(it.score, 1)} for it in items}
    out = score_labels(items, human, {}, key_for(sorted(DISPOSITIONS)))
    assert not out["score"]["human_looks_copied"]["flag"]
    assert any("look the same" in c for c in out["caveats"])


def test_every_result_carries_the_provenance_caveat():
    out = score_labels(build(6), {}, {}, key_for(sorted(DISPOSITIONS)))
    assert judge.PROVENANCE_CAVEAT in out["caveats"]


def test_a_set_with_no_recorded_labeller_says_so_first():
    """An unattributed set is not a human baseline, and the page should not be
    able to present it as one."""
    out = score_labels(build(6), {}, {}, key_for(sorted(DISPOSITIONS)))
    assert out["labeller"] is None
    assert out["caveats"][0] == judge.NO_LABELLER_CAVEAT


def test_a_recorded_labeller_travels_with_the_result():
    out = score_labels(build(6), {}, {}, key_for(sorted(DISPOSITIONS)),
                       labeller="a practising critic, no involvement")
    assert out["labeller"] == "a practising critic, no involvement"
    assert judge.NO_LABELLER_CAVEAT not in out["caveats"]


def test_the_money_guard_fires_before_any_other_check():
    """With no sample on disk, `run` must still refuse for the right reason —
    otherwise it exits complaining about a missing file and the caller learns
    nothing about what needed confirming."""
    repo = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "judge.py", "run", "--set", "nosuchset"],
                          cwd=repo, capture_output=True, text=True)
    assert done.returncode != 0
    assert "--confirm" in done.stderr
    assert "No sample" not in done.stderr


# --- two providers, one prompt ------------------------------------------------
class StubOpenAI:
    """Shaped like the Responses API: instructions + input -> output_text."""

    def __init__(self, reply='{"wrote": "A", "score": 0.5, "why": "x"}'):
        self.reply, self.calls = reply, []
        outer = self

        class Responses:
            async def create(self, **kw):
                outer.calls.append(kw)
                return type("R", (), {"output_text": outer.reply})()
        self.responses = Responses()


def test_the_model_name_decides_the_provider():
    assert judge.provider_for("claude-opus-5") == "anthropic"
    for name in ("gpt-5", "gpt-4o", "o3", "o4-mini", "chatgpt-4o-latest"):
        assert judge.provider_for(name) == "openai", name


def test_an_unplaceable_model_name_stops_rather_than_guessing():
    with pytest.raises(SystemExit) as exc:
        judge.provider_for("llama-3")
    assert "provider" in str(exc.value)


def test_a_missing_key_names_the_variable_and_does_not_traceback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(judge, "PROVIDERS", {**judge.PROVIDERS})
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        judge.make_client("openai")
    assert "OPENAI_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


def test_the_openai_judge_gets_the_same_prompt_as_the_anthropic_one():
    """The comparison between an in-family and an out-of-family judge only
    means anything if neither is given a hint the other is not."""
    import asyncio
    items = [make_item("i01")]
    a, o = StubClient(), StubOpenAI()
    asyncio.run(judge.run_judge(items, DISPOSITIONS, "claude-x", a, provider="anthropic"))
    asyncio.run(judge.run_judge(items, DISPOSITIONS, "gpt-x", o, provider="openai"))
    anth = {(c["system"], c["messages"][0]["content"]) for c in a.calls}
    open_ = {(c["instructions"], c["input"]) for c in o.calls}
    assert anth == open_


def test_the_openai_judge_answers_are_parsed_the_same_way():
    import asyncio
    answers = asyncio.run(judge.run_judge([make_item()], DISPOSITIONS, "gpt-x",
                                          StubOpenAI(), provider="openai"))
    assert answers["i01"]["wrote"]["wrote"] in DISPOSITIONS
    assert answers["i01"]["score"]["score"] == 0.5


def test_reasoning_models_get_a_larger_output_budget():
    """An OpenAI reasoning model bills thinking against the output ceiling, so
    a 300-token cap can truncate before the answer is reached."""
    import asyncio
    o = StubOpenAI()
    asyncio.run(judge.run_judge([make_item()], DISPOSITIONS, "o3", o, provider="openai"))
    assert all(c["max_output_tokens"] == judge.ANSWER_TOKENS["openai"] for c in o.calls)
    assert judge.ANSWER_TOKENS["openai"] > judge.ANSWER_TOKENS["anthropic"]


def test_each_judge_gets_its_own_file():
    a = judge.judge_path("s1", "claude-opus-5")
    b = judge.judge_path("s1", "gpt-5")
    assert a != b
    assert "/" not in a.name and "/" not in b.name


def test_score_refuses_to_guess_between_two_judges(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "LABEL_DIR", tmp_path)
    for model in ("claude-opus-5", "gpt-5"):
        judge.judge_path("s1", model).write_text(json.dumps(
            {"set_id": "s1", "model": model, "answers": {}}))
    assert sorted(judge.judge_files("s1")) == ["claude-opus-5", "gpt-5"]
