"""Acceptance criteria for the three-panel figure.

Structural properties, not any run's numbers: the panels share one x-axis, the
shading grammar means the same thing in rows 1 and 2, row 2 never plots a point
at round 0, and row 3 deliberately carries no band. That last one is the most
likely to be "fixed" by someone tidying up for visual consistency, which is why
it is a test and not a comment.

The metric these panels draw is tested separately, in test_adoption_rate.py.

    uv run pytest tests/test_figure_spec.py -v
"""

import pandas as pd
import pytest

from analyze import adoption_rate, make_figure, make_timeline_figure


def occ_frame(rows) -> pd.DataFrame:
    """(round, critic, cluster) tuples -> an occurrences frame."""
    return pd.DataFrame([{"round": r, "critic": c, "descriptor": k, "cluster": k}
                         for r, c, k in rows])


@pytest.fixture
def simple_occ():
    """alpha coins 'glow' in round 0; beta picks it up in rounds 1 and 2."""
    return occ_frame([
        (0, "alpha", "glow"), (0, "beta", "hum"),
        (1, "alpha", "glow"), (1, "beta", "glow"),
        (2, "beta", "glow"), (2, "alpha", "hum"),
    ])


# --- the figure ---------------------------------------------------------------
def figure_for(occ, control_occ=None):
    vocab = pd.DataFrame({"round": [0, 1, 2], "jaccard": [0.05, 0.07, 0.09]})
    spread = pd.DataFrame({"round": [0, 1, 2], "spread": [0.04, 0.05, 0.04]})
    kwargs = {}
    if control_occ is not None:
        kwargs = dict(
            control_vocab=pd.DataFrame({"round": [0, 1, 2], "jaccard": [0.03, 0.04, 0.05]}),
            control_spread=pd.DataFrame({"round": [0, 1, 2], "spread": [0.05, 0.04, 0.05]}),
            control_rate=adoption_rate(control_occ))
    return make_figure(vocab, spread, adoption_rate(occ), **kwargs)


def traces_in_row(fig, row: int) -> list:
    """Row n's traces, found by the y-axis the subplot assigned them."""
    axis = "y" if row == 1 else f"y{row}"
    return [t for t in fig.data if (t.yaxis or "y") == axis]


def test_panels_share_one_x_axis(simple_occ):
    """Not merely matching ranges: alignment must survive a tick-width change."""
    fig = figure_for(simple_occ)
    assert fig.layout.xaxis.matches or fig.layout.xaxis2.matches or \
        fig.layout.xaxis3.matches, "rows are not bound to a shared x-axis"


def test_row_one_fill_order_produces_two_regions(simple_occ):
    """Control first with tozeroy, treatment second with tonexty — no double count."""
    fig = figure_for(simple_occ, control_occ=simple_occ)
    fills = [t.fill for t in traces_in_row(fig, 1)]
    assert fills == ["tozeroy", "tonexty"]


def test_row_three_carries_no_band(simple_occ):
    """Deliberate: there is no consistent gap, and shading one would be a lie."""
    fig = figure_for(simple_occ, control_occ=simple_occ)
    row3 = traces_in_row(fig, 3)
    assert row3, "score spread traces are missing"
    assert all(t.fill in (None, "none") for t in row3), \
        "row 3 must carry no shaded band — see the figure spec"


def test_control_is_drawn_when_it_never_adopts():
    """A flat zero line is a null result; an absent line reads as missing data."""
    treat = occ_frame([(0, "alpha", "glow"), (0, "beta", "hum"), (1, "beta", "glow")])
    # gamma/delta coin separately and never borrow, but the pool is non-empty.
    control = occ_frame([(0, "gamma", "glow"), (0, "delta", "hum"),
                         (1, "gamma", "glow"), (1, "delta", "hum")])
    control_rate = adoption_rate(control)
    assert not control_rate.empty and (control_rate["rate"] == 0).all()
    fig = figure_for(treat, control_occ=control)
    assert len(traces_in_row(fig, 2)) == 2, "the all-zero control series must still be drawn"


def test_row_two_has_no_round_zero_point(simple_occ):
    fig = figure_for(simple_occ, control_occ=simple_occ)
    for trace in traces_in_row(fig, 2):
        assert 0 not in list(trace.x), "row 2 must not plot a point at round 0"


def test_timeline_returns_none_when_nothing_propagated():
    occ = occ_frame([(0, "alpha", "glow"), (1, "alpha", "glow")])
    assert make_timeline_figure(occ) is None


def test_timeline_separates_coinage_from_adoption(simple_occ):
    fig = make_timeline_figure(simple_occ)
    assert fig is not None
    names = [t.name for t in fig.data]
    assert names == ["coined", "adopted (did not coin)"]
    # Coinage is filled; adoption is hollow with a ring.
    assert fig.data[1].marker.color != fig.data[1].marker.line.color
