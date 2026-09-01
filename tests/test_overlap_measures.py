"""Jaccard and Dice, and the relationship between them.

vocabulary_convergence() returns both measures. Dice is what the figure plots;
Jaccard is kept so a reader can check the transform. These tests pin the
transform where it is exact and pin the direction of the error where it is not —
either way a wrong denominator shows up immediately.

    uv run pytest tests/test_overlap_measures.py -v
"""

import pandas as pd
import pytest

from analyze import vocabulary_convergence


def occ_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([{"round": r, "critic": c, "descriptor": k, "cluster": k}
                         for r, c, k in rows])


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
        j = row["jaccard"]
        assert row["dice"] == pytest.approx(2 * j / (1 + j), rel=1e-12, abs=1e-12), \
            f"round {row['round']}: dice != 2j/(1+j)"


def test_known_values_for_one_pair():
    """|A∩B| = 1, |A| = |B| = 2 -> J = 1/3, Dice = 1/2."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "alpha", "hum"),
                     (0, "beta", "glow"), (0, "beta", "sheen")])
    row = vocabulary_convergence(occ).iloc[0]
    assert row["jaccard"] == pytest.approx(1 / 3)
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
    assert row["jaccard"] == pytest.approx(5 / 9)
    assert row["dice"] == pytest.approx(2 / 3)
    transformed = 2 * row["jaccard"] / (1 + row["jaccard"])   # = 5/7
    assert transformed == pytest.approx(5 / 7)
    assert row["dice"] < transformed, "Jensen's inequality should make averaged Dice smaller"


def test_dice_is_at_least_jaccard():
    """Dice does not penalise the union twice, so it never reads lower."""
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"), (0, "gamma", "glow"),
                     (1, "alpha", "glow"), (1, "beta", "glow"), (1, "gamma", "wash")])
    conv = vocabulary_convergence(occ)
    assert (conv["dice"] >= conv["jaccard"] - 1e-12).all()


def test_both_columns_are_returned():
    occ = occ_frame([(0, "alpha", "glow"), (0, "beta", "glow")])
    assert list(vocabulary_convergence(occ).columns) == ["round", "jaccard", "dice"]


def test_rounds_with_one_critic_are_skipped():
    """No pair to compare."""
    occ = occ_frame([(0, "alpha", "glow"), (1, "alpha", "glow"), (1, "beta", "glow")])
    assert list(vocabulary_convergence(occ)["round"]) == [1]
