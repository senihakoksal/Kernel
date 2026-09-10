"""The rolling feed window: what an agent can still see, and both arms agreeing.

The window bounds critique visibility only. Concepts are never windowed,
because the control replays the treatment's artworks and that replay is valid
only if the stimulus is complete in both arms.

The test that earns its place here is test_both_arms_agree_on_the_round_range.
Everything else pins the rule; that one pins the thing most likely to go wrong,
which is the two arms drifting apart after someone fixes a bug in one of them.

    uv run pytest tests/test_feed_window.py -v
"""

import pytest

import control
from agents import FEED_WINDOW, evaluation_round_bounds, windowed_feed
from schema import Record

ARTISTS = ["artist_a", "artist_b"]
CRITICS = ["critic_x", "critic_y", "critic_z"]
ROUNDS = 8


def concept(rnd: int, agent: str) -> Record:
    return Record(round=rnd, agent=agent, role="artist", model="stub", kind="concept",
                  concept_id=f"r{rnd}-{agent}", title=f"work {rnd} {agent}",
                  content="a concept", reasoning="stub")


def evaluation(rnd: int, critic: str, concept_id: str) -> Record:
    return Record(round=rnd, agent=critic, role="critic", model="stub",
                  kind="evaluation", concept_id=concept_id, content="a critique",
                  score=0.5, reasoning="stub")


@pytest.fixture
def full_log() -> list[Record]:
    """An append-ordered treatment log: concepts then critiques, every round."""
    feed: list[Record] = []
    for rnd in range(ROUNDS):
        concepts = [concept(rnd, a) for a in ARTISTS]
        feed.extend(concepts)
        feed.extend(evaluation(rnd, c, k.concept_id) for c in CRITICS for k in concepts)
    return feed


def feed_as_of(log: list[Record], round_idx: int) -> list[Record]:
    """The unwindowed feed as it stood when round `round_idx`'s critics acted.

    That is the log prefix ending before this round's first evaluation — read
    off recorded order rather than reconstructed, so a reordering bug would
    surface instead of being reproduced.
    """
    for i, rec in enumerate(log):
        if rec.kind == "evaluation" and rec.round == round_idx:
            return log[:i]
    return list(log)


def eval_rounds(feed: list[Record]) -> set[int]:
    return {r.round for r in feed if r.kind == "evaluation"}


def concept_rounds(feed: list[Record]) -> set[int]:
    return {r.round for r in feed if r.kind == "concept"}


# --- the rule -----------------------------------------------------------------
def test_nothing_has_aged_out_before_the_window_fills(full_log):
    """For r <= FEED_WINDOW the windowed feed IS the unwindowed feed."""
    for round_idx in range(FEED_WINDOW + 1):
        unwindowed = feed_as_of(full_log, round_idx)
        assert windowed_feed(unwindowed, round_idx) == unwindowed, \
            f"round {round_idx} should be unaffected by a window of {FEED_WINDOW}"


def test_old_and_current_critiques_are_both_excluded(full_log):
    """For r > FEED_WINDOW: nothing older than r-window, nothing from r onward."""
    for round_idx in range(FEED_WINDOW + 1, ROUNDS):
        windowed = windowed_feed(feed_as_of(full_log, round_idx), round_idx)
        rounds = eval_rounds(windowed)
        assert rounds == set(range(round_idx - FEED_WINDOW, round_idx)), \
            f"round {round_idx} sees evaluation rounds {sorted(rounds)}"
        assert min(rounds) >= round_idx - FEED_WINDOW
        assert max(rounds) < round_idx


@pytest.mark.parametrize("window", [0, 1, 2, 3, 5, 99])
def test_every_concept_survives_at_every_window(full_log, window):
    """Concepts are never windowed, at any window value."""
    for round_idx in range(ROUNDS):
        unwindowed = feed_as_of(full_log, round_idx)
        windowed = windowed_feed(unwindowed, round_idx, window)
        assert concept_rounds(windowed) == concept_rounds(unwindowed)
        assert [r for r in windowed if r.kind == "concept"] == \
               [r for r in unwindowed if r.kind == "concept"]


def test_window_zero_yields_concepts_only(full_log):
    """The artists-blind-to-critique configuration — representable, not default."""
    for round_idx in range(ROUNDS):
        windowed = windowed_feed(feed_as_of(full_log, round_idx), round_idx, 0)
        assert windowed, "concepts should still be there"
        assert all(r.kind == "concept" for r in windowed)


def test_window_preserves_feed_order(full_log):
    """format_feed renders the list as-is, so filtering must not reshuffle it."""
    unwindowed = feed_as_of(full_log, 6)
    windowed = windowed_feed(unwindowed, 6)
    assert windowed == [r for r in unwindowed if r in windowed]


def test_bounds_are_half_open_at_the_current_round():
    assert evaluation_round_bounds(0) == (0, 0)
    assert evaluation_round_bounds(2, 3) == (0, 2)
    assert evaluation_round_bounds(7, 3) == (4, 7)
    assert evaluation_round_bounds(7, 0) == (7, 7)


# --- the invariant that matters ----------------------------------------------
def test_both_arms_agree_on_the_round_range(full_log):
    """The treatment feed and the control's isolated_feed must window alike.

    They differ in WHOSE critiques are visible — that is the manipulation — but
    not in WHICH ROUNDS are visible. If those diverge, treatment-minus-control
    stops isolating the peer channel and starts measuring a difference in how
    much history each arm could see, and nothing about the run would look wrong.
    """
    concepts_by_round = {rnd: [r for r in full_log
                               if r.kind == "concept" and r.round == rnd]
                         for rnd in range(ROUNDS)}
    own_critiques = {c: [r for r in full_log
                         if r.kind == "evaluation" and r.agent == c]
                     for c in CRITICS}
    rounds = list(range(ROUNDS))

    for window in (0, 1, 3, 5):
        for round_idx in range(ROUNDS):
            treatment = windowed_feed(feed_as_of(full_log, round_idx), round_idx, window)
            for critic in CRITICS:
                isolated = control.isolated_feed(concepts_by_round, own_critiques,
                                                 rounds, round_idx, critic, window)
                assert eval_rounds(isolated) <= eval_rounds(treatment), (
                    f"window={window} round={round_idx} {critic}: control sees "
                    f"evaluation rounds {sorted(eval_rounds(isolated))}, treatment "
                    f"{sorted(eval_rounds(treatment))}")
                # With every critic writing every round, the control's own-only
                # feed covers exactly the same round range as the treatment's.
                assert eval_rounds(isolated) == eval_rounds(treatment), (
                    f"window={window} round={round_idx} {critic}: round ranges differ")
                # And the stimulus is identical, which is what makes the replay valid.
                assert concept_rounds(isolated) == concept_rounds(treatment)


def test_control_sees_only_its_own_critiques(full_log):
    """The manipulation itself, alongside the shared window."""
    concepts_by_round = {rnd: [r for r in full_log
                               if r.kind == "concept" and r.round == rnd]
                         for rnd in range(ROUNDS)}
    own_critiques = {c: [r for r in full_log
                         if r.kind == "evaluation" and r.agent == c]
                     for c in CRITICS}
    isolated = control.isolated_feed(concepts_by_round, own_critiques,
                                     list(range(ROUNDS)), 6, "critic_x", FEED_WINDOW)
    authors = {r.agent for r in isolated if r.kind == "evaluation"}
    assert authors == {"critic_x"}


# --- the guard against a silent invalidation ---------------------------------
def test_control_refuses_a_window_mismatch(tmp_path, capsys):
    """Mismatched windows across arms is a hard error, not a warning."""
    log = tmp_path / "run_stub.jsonl"
    log.write_text("")
    (tmp_path / "run_stub.meta.json").write_text('{"feed_window": 5}')
    with pytest.raises(SystemExit) as exit_info:
        control.check_feed_window(log, 3)
    assert "mismatch" in str(exit_info.value).lower()
    assert "--feed-window 5" in str(exit_info.value)


def test_control_accepts_a_matching_window(tmp_path):
    log = tmp_path / "run_stub.jsonl"
    log.write_text("")
    (tmp_path / "run_stub.meta.json").write_text('{"feed_window": 3}')
    assert control.check_feed_window(log, 3) == {"feed_window": 3}


def test_control_warns_but_proceeds_without_a_sidecar(tmp_path, capsys):
    """Logs predating the sidecar cannot be checked; failing would lock them out."""
    log = tmp_path / "run_stub.jsonl"
    log.write_text("")
    assert control.check_feed_window(log, 3) is None
    assert "WARNING" in capsys.readouterr().out
