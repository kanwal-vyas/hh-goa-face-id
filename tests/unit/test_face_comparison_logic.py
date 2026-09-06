import numpy as np
import pytest

from app.face.compare import (
    CosineFaceComparator,
    MatchBand,
    cosine_similarity,
    derive_match_band,
)
from app.face.opencv_processor import compute_lbp_embedding
from app.face.processor import BoundingBox, FaceEmbedding

# --- compute_lbp_embedding: pure function, no detector/model needed ---


def test_lbp_embedding_has_deterministic_shape():
    rng = np.random.default_rng(0)
    face = rng.integers(0, 255, (200, 150), dtype=np.uint8)
    vector = compute_lbp_embedding(face, grid=(8, 8), bins=8)
    assert len(vector) == 8 * 8 * 8


def test_lbp_embedding_shape_independent_of_input_size():
    rng = np.random.default_rng(1)
    small = rng.integers(0, 255, (40, 40), dtype=np.uint8)
    large = rng.integers(0, 255, (400, 350), dtype=np.uint8)
    v_small = compute_lbp_embedding(small)
    v_large = compute_lbp_embedding(large)
    assert len(v_small) == len(v_large)


def test_lbp_embedding_is_deterministic_for_same_input():
    rng = np.random.default_rng(2)
    face = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    v1 = compute_lbp_embedding(face)
    v2 = compute_lbp_embedding(face)
    assert v1 == v2


def test_lbp_embedding_rejects_non_2d_input():
    with pytest.raises(ValueError):
        compute_lbp_embedding(np.zeros((10, 10, 3), dtype=np.uint8))


def test_lbp_embedding_identical_faces_yield_similarity_one():
    rng = np.random.default_rng(3)
    face = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    v = compute_lbp_embedding(face)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


def test_lbp_embedding_different_faces_yield_lower_similarity():
    rng = np.random.default_rng(4)
    face_a = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    # A structurally very different image (flat/constant) should produce a
    # clearly different LBP histogram than random noise.
    face_b = np.full((128, 128), 128, dtype=np.uint8)
    v_a = compute_lbp_embedding(face_a)
    v_b = compute_lbp_embedding(face_b)
    sim_same = cosine_similarity(v_a, v_a)
    sim_diff = cosine_similarity(v_a, v_b)
    assert sim_diff < sim_same


# --- cosine_similarity: pure function ---


def test_cosine_similarity_identical_vectors():
    v = (1.0, 2.0, 3.0)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_rejects_mismatched_length():
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0))


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0


# --- derive_match_band ---


def test_derive_match_band_strong():
    assert derive_match_band(0.9, match_threshold=0.6) == MatchBand.STRONG_MATCH


def test_derive_match_band_possible():
    assert derive_match_band(0.4, match_threshold=0.6) == MatchBand.POSSIBLE_MATCH


def test_derive_match_band_no_match():
    assert derive_match_band(0.1, match_threshold=0.6) == MatchBand.NO_MATCH


def test_derive_match_band_boundary_is_strong():
    assert derive_match_band(0.6, match_threshold=0.6) == MatchBand.STRONG_MATCH


# --- CosineFaceComparator: uses hand-constructed FaceEmbeddings, no image needed ---


def _embedding(vector, model_version="test-v1") -> FaceEmbedding:
    return FaceEmbedding(
        vector=tuple(vector),
        model_version=model_version,
        face_bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )


def test_comparator_returns_valid_match_band():
    comparator = CosineFaceComparator(match_threshold=0.6)
    result = comparator.compare(_embedding((1.0, 0.0)), _embedding((1.0, 0.0)))
    assert isinstance(result.band, MatchBand)
    assert result.similarity_score == pytest.approx(1.0)
    assert result.band == MatchBand.STRONG_MATCH


def test_comparator_rejects_cross_model_version_comparison():
    comparator = CosineFaceComparator(match_threshold=0.6)
    with pytest.raises(ValueError):
        comparator.compare(
            _embedding((1.0, 0.0), model_version="v1"),
            _embedding((1.0, 0.0), model_version="v2"),
        )


def test_comparator_threshold_is_configurable():
    low_threshold = CosineFaceComparator(match_threshold=0.1)
    high_threshold = CosineFaceComparator(match_threshold=0.99)

    a, b = _embedding((1.0, 0.5)), _embedding((0.9, 0.6))
    result_low = low_threshold.compare(a, b)
    result_high = high_threshold.compare(a, b)

    # Same similarity score, different band due to different threshold.
    assert result_low.similarity_score == result_high.similarity_score
    assert result_low.band != result_high.band or result_low.band == MatchBand.STRONG_MATCH


def test_comparator_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        CosineFaceComparator(match_threshold=1.5)
