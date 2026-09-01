"""How much of the borrowable vocabulary was actually in use, per round.

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
