"""Dice overlap, and its relationship to the Jaccard it replaced.

vocabulary_convergence() reports Dice only. Jaccard is no longer surfaced
anywhere, so these tests compute it from their own fixtures — the point is to
pin Dice against an independently-derived reference rather than against another
column of the same frame. A wrong denominator (|A∪B| instead of |A|+|B|) or a
dropped factor of 2 shows up immediately either way.

    uv run pytest tests/test_overlap_measures.py -v
"""

import itertools

import pandas as pd
import pytest

from analyze import vocabulary_convergence


def occ_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([{"round": r, "critic": c, "descriptor": k, "cluster": k}
                         for r, c, k in rows])


def mean_jaccard(occ: pd.DataFrame, rnd: int) -> float:
    """Average pairwise |A∩B| / |A∪B| for one round, computed here.

    Deliberately independent of analyze.py: the production code no longer
    reports Jaccard, so this is a reference implementation the assertions can be
    checked against, not a second reading of the same number.
    """
    sets = {c: set(g["cluster"])
            for c, g in occ[occ["round"] == rnd].groupby("critic")}
    pairs = [(sets[a], sets[b]) for a, b in itertools.combinations(sorted(sets), 2)]
    values = [len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in pairs]
    return sum(values) / len(values)


def test_dice_equals_2j_over_1_plus_j_for_a_single_pair():
    """The exact identity, pinned where it actually holds.

    With two critics there is exactly one pair per round, so the round average
    IS the pair value and dice = 2j/(1+j) to floating-point tolerance. A wrong
    denominator — |A∪B| instead of |A|+|B|, or the factor of 2 dropped — breaks
    this immediately.
    """
    occ = occ_frame([
        (0, "alpha", "glow"), (0, "alpha", "hum"), (0, "beta", "glow"), (0, "beta", "sheen"),
        (1, "alpha", "glow"), (1, "alpha", "hum"), (1, "alpha", "wash"),
        (1, "beta", "glow"), (1, "beta", "hum"),
        (2, "alpha", "glow"), (2, "beta", "murk"),
    ])
    conv = vocabulary_convergence(occ)
    assert not conv.empty
    for _, row in conv.iterrows():
        j = mean_jaccard(occ, int(row["round"]))
        assert row["dice"] == pytest.approx(2 * j / (1 + j), rel=1e-12, abs=1e-12), \
            f"round {row['round']}: dice != 2j/(1+j)"


def test_known_values_for_one_pair():
    """|A∩B| = 1, |A| = |B| = 2 -> J = 1/3, Dice = 1/2."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "alpha", "hum"),
                     (0, "beta", "glow"), (0, "beta", "sheen")])
    row = vocabulary_convergence(occ).iloc[0]
    assert mean_jaccard(occ, 0) == pytest.approx(1 / 3)
    assert row["dice"] == pytest.approx(0.5)


def test_averaged_dice_does_not_equal_the_transformed_average():
    """With unequal pairs the identity breaks — and it breaks in one direction.

    2j/(1+j) is concave, so by Jensen's inequality the mean of the per-pair Dice
    values is at most the transform of the mean Jaccard, with equality only when
    every pair overlaps identically. This is why the two columns are computed
    separately instead of one being derived from the other.
    """
    occ = occ_frame([
        (0, "alpha", "glow"), (0, "alpha", "hum"),
        (0, "beta", "glow"), (0, "beta", "hum"),       # identical to alpha
        (0, "gamma", "glow"), (0, "gamma", "murk"),    # half-overlapping with both
    ])
    row = vocabulary_convergence(occ).iloc[0]
    # pairs: alpha-beta J=1 D=1; alpha-gamma J=1/3 D=1/2; beta-gamma J=1/3 D=1/2
    j = mean_jaccard(occ, 0)
    assert j == pytest.approx(5 / 9)
    assert row["dice"] == pytest.approx(2 / 3)
    transformed = 2 * j / (1 + j)                             # = 5/7
    assert transformed == pytest.approx(5 / 7)
    assert row["dice"] < transformed, "Jensen's inequality should make averaged Dice smaller"


def test_dice_is_at_least_jaccard():
    """Dice does not penalise the union twice, so it never reads lower."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"), (0, "gamma", "glow"),
                     (1, "alpha", "glow"), (1, "beta", "glow"), (1, "gamma", "wash")])
    for _, row in vocabulary_convergence(occ).iterrows():
        assert row["dice"] >= mean_jaccard(occ, int(row["round"])) - 1e-12


def test_dice_is_the_only_reported_measure():
    """Jaccard is computed nowhere in the returned frame — Dice is the measure."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "glow")])
    columns = list(vocabulary_convergence(occ).columns)
    assert columns == ["round", "dice"]
    assert "jaccard" not in columns


def test_rounds_with_one_critic_are_skipped():
    """No pair to compare."""
    occ = occ_frame([(0, "alpha", "glow"), (1, "alpha", "glow"), (1, "beta", "glow")])
    assert list(vocabulary_convergence(occ)["round"]) == [1]
