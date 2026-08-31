"""Clustering invariants for analyze.canonicalize().

These pin the three properties the agglomerative rewrite exists to provide, and
one that would fail loudly in the worst possible way if it broke:

  - no chaining: A and C must not share a cluster just because both resemble B
  - order independence: the partition follows the geometry, not the spelling
  - threshold direction: SIMILARITY in, DISTANCE to sklearn

That last one is why this file exists. Passing the similarity straight through
as distance_threshold does not raise, does not warn, and does not look wrong —
it merges nearly every descriptor into one cluster, which shows up downstream as
a dramatic jump in vocabulary overlap. A silent bug that fabricates the result
you were hoping for is worth a test.

The embedding model is stubbed with hand-placed unit vectors. That is the point:
these are assertions about the clustering algorithm, and they should not depend
on what all-MiniLM-L6-v2 happens to think two phrases mean.

    uv run pytest tests/test_canonicalize.py -v
"""

import math

import numpy as np
import pytest

from analyze import SIMILARITY_THRESHOLD, canonicalize

THRESHOLD = 0.72  # matches SIMILARITY_THRESHOLD; kept explicit for readability


class StubEncoder:
    """Returns fixed unit vectors, so cosine similarities are exact by construction."""

    def __init__(self, vectors: dict[str, tuple[float, float]]):
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        arr = np.array([self._vectors[t] for t in texts], dtype=float)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def at(degrees: float) -> tuple[float, float]:
    """A unit vector at an angle — cos(a - b) is the similarity of two of these."""
    rad = math.radians(degrees)
    return (math.cos(rad), math.sin(rad))


def partition(assignment: dict[str, str]) -> set[frozenset[str]]:
    """The grouping alone, ignoring which member supplied the label."""
    groups: dict[str, set[str]] = {}
    for descriptor, label in assignment.items():
        groups.setdefault(label, set()).add(descriptor)
    return {frozenset(g) for g in groups.values()}


def test_threshold_is_similarity_not_distance():
    """Two descriptors at 0.5 similarity are below a 0.72 threshold: 2 clusters.

    Under the inverted bug (similarity passed as distance_threshold) their 0.5
    cosine distance falls under 0.72 and they merge into one.
    """
    model = StubEncoder({"first_phrase": at(0), "second_phrase": at(60)})  # cos 60° = 0.5
    assignment = canonicalize(["first_phrase", "second_phrase"], model, THRESHOLD)
    assert len(set(assignment.values())) == 2, (
        "descriptors at 0.5 similarity merged under a 0.72 threshold — "
        "distance_threshold is probably receiving a similarity"
    )


def test_dissimilar_descriptors_do_not_collapse():
    """The blunt version of the same guard: orthogonal phrases stay separate."""
    model = StubEncoder({f"phrase_{i}": at(i * 90) for i in range(4)})
    assignment = canonicalize(list(model._vectors), model, THRESHOLD)
    assert len(set(assignment.values())) == 4


def test_no_chaining_through_an_intermediate():
    """A-B and B-C are both above threshold; A-C is not. All three must not merge.

    Geometry: left and right sit 36.87° either side of center, so each is at 0.8
    similarity to center but only 0.28 to each other. The greedy implementation
    walked these alphabetically, made 'aa_center' the founder, and let both
    others join it — putting two descriptors at 0.28 similarity in one cluster.
    Complete linkage requires every pair to be within threshold, so it cannot.
    """
    model = StubEncoder({
        "aa_center": at(0),
        "mm_left": at(-36.87),   # cos 36.87° ≈ 0.8 to center
        "zz_right": at(36.87),   # cos 73.74° ≈ 0.28 to mm_left
    })
    assignment = canonicalize(list(model._vectors), model, THRESHOLD)
    assert assignment["mm_left"] != assignment["zz_right"], (
        "chained: two descriptors at 0.28 similarity share a cluster"
    )
    assert len(set(assignment.values())) == 2


def test_partition_is_independent_of_spelling():
    """Same geometry, two naming schemes with opposite alphabetical order.

    The greedy pass froze whichever descriptor sorted first as its cluster's
    permanent representative, so renaming the phrases could change the answer.
    The partition must depend on the vectors alone.
    """
    geometry = [at(0), at(5), at(90), at(95)]
    forward = ["aa", "bb", "cc", "dd"]
    reverse = ["dd_x", "cc_x", "bb_x", "aa_x"]

    p1 = partition(canonicalize(forward, StubEncoder(dict(zip(forward, geometry))), THRESHOLD))
    p2 = partition(canonicalize(reverse, StubEncoder(dict(zip(reverse, geometry))), THRESHOLD))

    shape = lambda p: sorted(sorted(g) for g in p)
    assert shape(p1) == [["aa", "bb"], ["cc", "dd"]]
    assert [len(g) for g in shape(p1)] == [len(g) for g in shape(p2)]
    # Same members pair up under either naming: index 0 with 1, index 2 with 3.
    assert {frozenset({"dd_x", "cc_x"}), frozenset({"bb_x", "aa_x"})} == p2


def test_cluster_label_is_the_most_frequent_member():
    """Labels name the phrase critics actually used, not the alphabetical first."""
    model = StubEncoder({"amber_glow": at(0), "zebra_glow": at(5)})  # ~0.996 similar
    assignment = canonicalize(["amber_glow"] + ["zebra_glow"] * 3, model, THRESHOLD)
    assert set(assignment.values()) == {"zebra_glow"}


def test_label_ties_break_alphabetically():
    model = StubEncoder({"amber_glow": at(0), "zebra_glow": at(5)})
    assignment = canonicalize(["amber_glow"] * 2 + ["zebra_glow"] * 2, model, THRESHOLD)
    assert set(assignment.values()) == {"amber_glow"}


@pytest.mark.parametrize("descriptors,expected", [([], {}), (["only"], {"only": "only"})])
def test_degenerate_inputs(descriptors, expected):
    """One sample is below AgglomerativeClustering's minimum, so it is special-cased."""
    model = StubEncoder({"only": at(0)})
    assert canonicalize(descriptors, model, THRESHOLD) == expected


def test_module_threshold_is_a_similarity():
    """Sanity: the shipped constant is in similarity units, as canonicalize expects."""
    assert 0.0 < SIMILARITY_THRESHOLD < 1.0
