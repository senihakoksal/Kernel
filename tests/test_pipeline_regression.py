"""The numbers the pipeline reports, pinned against a real log.

Every other test in this suite builds synthetic records and checks the code's
behaviour. None of them touch real text, so none of them would notice if the
descriptor extractor, the clustering threshold, the prior-vocabulary
subtraction or the embedding model changed what the project actually reports.
The suite would stay green while every figure on the published site moved.

That has already happened twice — the greedy-to-agglomerative clustering swap
and the adoption-denominator fix both changed published numbers, and both were
caught by an ad-hoc before/after run by hand rather than by a test.

So: one small real log, committed, with its outputs written down. A failure
here is not necessarily a bug. It means the reported numbers moved, and you
should know which change moved them and whether that was the intent. If it was,
regenerate and read the diff:

    uv run python tests/fixtures/regenerate_expected.py

Skipped when the analysis extra is absent, since it needs spaCy and
sentence-transformers:

    uv sync --extra analysis --extra dev
"""

import json
import pathlib

import pytest

pytest.importorskip("spacy", reason="needs the analysis extra")
pytest.importorskip("sentence_transformers", reason="needs the analysis extra")

import analyze  # noqa: E402  (after importorskip, by design)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
LOG = FIXTURES / "regression_run.jsonl"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())
REGEN = "uv run python tests/fixtures/regenerate_expected.py"


@pytest.fixture(scope="module")
def measured():
    """Run the real pipeline once over the fixture log.

    Module-scoped: spaCy and the embedding model take seconds to load, and
    every assertion below reads from the same run, so they cannot disagree
    about what the pipeline produced.
    """
    import spacy
    from sentence_transformers import SentenceTransformer

    # prior_vocabulary() reads the live agents.yaml. Pinned to the roster
    # captured with the fixture, or every roster edit fails this test for a
    # reason that has nothing to do with the pipeline.
    original = analyze.AGENTS_FILE
    analyze.AGENTS_FILE = FIXTURES / "regression_agents.yaml"
    try:
        nlp = spacy.load(analyze.SPACY_MODEL)
        model = SentenceTransformer(analyze.EMBED_MODEL)
        evals = analyze.critic_evaluations(LOG)
        window, _ = analyze.feed_window_for(LOG)
        occ, seeds, n_sub = analyze.descriptor_pipeline(evals, nlp, model)
        _, propagated = analyze.find_propagation(occ)
        yield {
            "window": window, "evals": evals, "occ": occ, "seeds": seeds,
            "n_sub": n_sub, "propagated": propagated,
            "vocab": analyze.vocabulary_convergence(occ),
            "rate": analyze.adoption_rate(occ, window),
            "spread": analyze.score_spread(analyze.score_trajectories(evals)),
        }
    finally:
        analyze.AGENTS_FILE = original


def moved(what: str) -> str:
    return f"{what} changed — the reported numbers moved. If intended: {REGEN}"


# --- the fixture itself must not drift ---------------------------------------
def test_fixture_log_is_intact():
    """If the log changes, every expectation below is measuring something else."""
    records = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    assert len(records) == 40
    assert sorted({r["round"] for r in records}) == [0, 1, 2, 3, 4]
    assert len({r["agent"] for r in records if r["role"] == "artist"}) == 2
    assert len({r["agent"] for r in records if r["role"] == "critic"}) == 3


def test_feed_window_comes_from_the_sidecar(measured):
    assert measured["window"] == EXPECTED["feed_window"]


# --- the descriptor pipeline -------------------------------------------------
def test_descriptor_extraction_and_subtraction(measured):
    assert measured["evals"].shape[0] == EXPECTED["n_critiques"]
    assert measured["occ"]["descriptor"].nunique() == EXPECTED["n_descriptors"], \
        moved("descriptor extraction")
    assert len(measured["seeds"]) == EXPECTED["n_seed_terms"], moved("prior vocabulary")
    assert measured["n_sub"] == EXPECTED["n_subtracted"], moved("prior-vocabulary subtraction")


def test_clustering(measured):
    assert measured["occ"]["cluster"].nunique() == EXPECTED["n_clusters"], \
        moved("clustering")


def test_propagation(measured):
    assert len(measured["propagated"]) == EXPECTED["n_propagated"], moved("propagation")
    assert sorted(measured["propagated"])[:5] == EXPECTED["propagated_sample"], \
        moved("which clusters propagated")


# --- the three plotted series ------------------------------------------------
def test_vocabulary_overlap(measured):
    got = [{"round": int(r), "dice": round(float(d), 6)}
           for r, d in zip(measured["vocab"]["round"], measured["vocab"]["dice"])]
    assert got == EXPECTED["overlap"], moved("vocabulary overlap")


def test_adoption_rate(measured):
    got = [{"round": int(r.round), "uses": int(r.uses),
            "opportunities": int(r.opportunities), "rate": round(float(r.rate), 6)}
           for r in measured["rate"].itertuples()]
    assert got == EXPECTED["adoption_rate"], moved("adoption rate")


def test_score_spread(measured):
    got = [{"round": int(r), "spread": round(float(v), 6)}
           for r, v in zip(measured["spread"]["round"], measured["spread"]["spread"])]
    assert got == EXPECTED["score_spread"], moved("score spread")


# --- properties that must hold on real data too ------------------------------
def test_adoption_rate_still_omits_round_zero(measured):
    """Pinned on synthetic data elsewhere; confirmed here on real text."""
    assert 0 not in set(measured["rate"]["round"])


def test_the_window_bites_on_this_fixture(measured):
    """Five rounds at window 3, so the denominator's growth must slow at round 4.
    If it does not, the fixture is too short to be testing the window at all."""
    opps = list(measured["rate"]["opportunities"])
    assert len(opps) >= 4
    early_growth = opps[1] - opps[0]
    late_growth = opps[-1] - opps[-2]
    assert late_growth < early_growth, (
        f"denominator growth did not slow ({opps}) — the window is not engaging")
