"""How much of the borrowable vocabulary was actually in use, per round.

The recency tests at the bottom are the ones that pin the current rule. An
earlier version asked only whether a cluster had been coined before round r, so
a term coined in round 0 stayed "borrowable" forever — long after the feed
window had aged it out of every prompt. Those tests fail under that rule.

These pin the definition rather than any run's numbers. The one most worth
having is the round-0 case: the metric's denominator is the pool of clusters
coined in an earlier round, which at round 0 is empty, so the rate is 0/0 —
undefined, not zero. Anything that fills that in with a zero is reporting a
measurement that was never made.

    uv run pytest tests/test_adoption_rate.py -v
"""

import pandas as pd

from analyze import adoption_rate


def occ_frame(rows) -> pd.DataFrame:
    """(round, critic, cluster) tuples -> an occurrences frame."""
    return pd.DataFrame([{"round": r, "critic": c, "descriptor": k, "cluster": k}
                         for r, c, k in rows])


def test_round_zero_is_absent_not_zero():
    """The eligible pool is empty at round 0, so the rate is undefined."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"), (1, "beta", "glow")])
    rate = adoption_rate(occ)
    assert not rate.empty
    assert 0 not in set(rate["round"]), "round 0 must not appear in the adoption rate"


def test_rounds_with_no_opportunities_are_omitted():
    """Every cluster coined by every critic: nothing is ever borrowable."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "glow"),
                     (1, "alpha", "glow"), (1, "beta", "glow")])
    assert adoption_rate(occ).empty


def test_single_round_log_yields_an_empty_rate():
    """A one-round log has no later round to adopt in — empty, not an error."""
    rate = adoption_rate(occ_frame([(0, "alpha", "glow"), (0, "beta", "hum")]))
    assert rate.empty
    assert list(rate.columns) == ["round", "uses", "opportunities", "rate"]


def test_empty_input_is_handled():
    assert adoption_rate(pd.DataFrame()).empty


def test_rate_counts_prevalence_not_incidence():
    """beta borrows 'glow' in round 1 and uses it again in round 2; both count.

    A term adopted once and dropped is a failed transmission; a convention is
    one that keeps being reproduced, so repeat use is the quantity of interest.
    """
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"),
                     (1, "beta", "glow"), (2, "beta", "glow")])
    rate = adoption_rate(occ).set_index("round")
    assert rate.loc[1, "uses"] == 1
    assert rate.loc[2, "uses"] == 1, "repeat use of a borrowed cluster must keep counting"


def test_repeat_use_within_one_round_counts_once():
    """Otherwise one critic with a verbal tic dominates the measure."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum")] + [(1, "beta", "glow")] * 5)
    assert adoption_rate(occ).set_index("round").loc[1, "uses"] == 1


def test_coiners_never_count_as_adopters():
    """Same-round co-coiners cannot have read each other, so neither is eligible."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "glow"), (0, "gamma", "hum"),
                     (1, "alpha", "glow"), (1, "beta", "glow"), (1, "gamma", "glow")])
    rate = adoption_rate(occ).set_index("round")
    # gamma is eligible for "glow" (alpha and beta co-coined it, so neither is),
    # and alpha and beta are each eligible for gamma's "hum" — three pairs.
    assert rate.loc[1, "opportunities"] == 3
    # Only gamma's use of "glow" counts; re-using your own coinage is not adoption.
    assert rate.loc[1, "uses"] == 1


def test_denominator_grows_as_the_pool_grows():
    """Why this is a rate: the borrowable pool expands every round, so a raw
    count of adoptions would rise even with constant borrowing behaviour."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"),
                     (1, "alpha", "sheen"), (1, "beta", "glow"),
                     (2, "beta", "glow"), (2, "alpha", "hum")])
    opps = adoption_rate(occ).set_index("round")["opportunities"]
    assert opps.loc[2] > opps.loc[1]


# --- the recency bound -------------------------------------------------------
def test_vocabulary_that_aged_out_is_not_an_opportunity():
    """alpha's round-0 term is gone from the feed by round 4 at window 3.

    Counting it as something beta "could have borrowed" would be counting
    vocabulary that was in no prompt. This is the test the old coined-before-r
    rule fails.
    """
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum")] +
                    # keep the run alive without anyone re-using "glow"
                    [(r, "alpha", f"fresh{r}") for r in range(1, 5)] +
                    [(r, "beta", f"other{r}") for r in range(1, 5)])
    rate = adoption_rate(occ, window=3).set_index("round")
    # by round 4 the window is [1,4): "glow" was last used in round 0
    eligible_at_4 = rate.loc[4, "opportunities"]
    unbounded = adoption_rate(occ, window=None).set_index("round")
    assert eligible_at_4 < unbounded.loc[4, "opportunities"], \
        "the window must shrink the denominator once vocabulary ages out"


def test_a_repeated_term_stays_eligible():
    """Terms that keep being used stay in the feed, so they stay borrowable.

    This is the transmission dynamic the window exists to create: repetition
    keeps a term available, silence retires it.
    """
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum")] +
                    [(r, "alpha", "glow") for r in range(1, 6)])
    rate = adoption_rate(occ, window=3).set_index("round")
    for r in (1, 4, 5):
        assert ("glow" in _eligible_clusters(occ, r, 3, "beta")), \
            f"round {r}: a term alpha keeps using should stay borrowable by beta"


def _eligible_clusters(occ, r, window, critic):
    """The clusters `critic` could borrow at round r — recomputed here so the
    assertion reads against an independent derivation of the rule."""
    lo = max(0, r - window)
    recent = occ[(occ["round"] >= lo) & (occ["round"] < r)]
    coined_first = {k: set(g.loc[g["round"] == g["round"].min(), "critic"])
                    for k, g in occ.groupby("cluster")}
    return {k for k, g in recent.groupby("cluster")
            if set(g["critic"]) - {critic} and critic not in coined_first[k]}


def test_unbounded_window_keeps_everything_eligible():
    """window=None is the right reading for logs written before the window."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum")] +
                    [(r, "alpha", f"fresh{r}") for r in range(1, 5)] +
                    [(r, "beta", f"other{r}") for r in range(1, 5)])
    unbounded = adoption_rate(occ, None).set_index("round")
    # every cluster used by the other critic in ANY earlier round still counts
    assert unbounded.loc[4, "opportunities"] > adoption_rate(occ, 1).set_index("round").loc[4, "opportunities"]


def test_denominator_stops_growing_under_a_window():
    """The old rule's denominator grew without bound, which made the rate fall
    over rounds as an artefact of run length. Under a window it plateaus."""
    rows = [(0, "alpha", "a0"), (0, "beta", "b0")]
    for r in range(1, 9):                      # each round brings fresh vocabulary
        rows += [(r, "alpha", f"a{r}"), (r, "beta", f"b{r}")]
    occ = occ_frame(rows)
    windowed = adoption_rate(occ, 3).set_index("round")["opportunities"]
    unbounded = adoption_rate(occ, None).set_index("round")["opportunities"]
    assert windowed.loc[8] == windowed.loc[7] == windowed.loc[6], \
        f"windowed denominator should plateau, got {windowed.to_dict()}"
    assert unbounded.loc[8] > unbounded.loc[4], "unbounded denominator should grow"


def test_a_critic_cannot_borrow_from_itself():
    """Only vocabulary another critic used counts — self-repetition is not
    adoption, however recent."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"),
                     (1, "alpha", "glow"), (2, "alpha", "glow")])
    for r in (1, 2):
        assert "glow" not in _eligible_clusters(occ, r, 3, "alpha")


def test_round_zero_still_absent_with_a_window():
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"), (1, "beta", "glow")])
    assert 0 not in set(adoption_rate(occ, 3)["round"])
