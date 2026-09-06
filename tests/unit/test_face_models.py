import pytest

from app.face.compare import FaceComparator, FaceComparisonResult, MatchBand
from app.face.processor import BoundingBox, FaceEmbedding, FaceProcessor


def test_face_processor_is_abstract():
    with pytest.raises(TypeError):
        FaceProcessor()  # type: ignore[abstract]


def test_face_comparator_is_abstract():
    with pytest.raises(TypeError):
        FaceComparator()  # type: ignore[abstract]


def test_bounding_box_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        BoundingBox(x=0, y=0, width=0, height=10)


def test_face_embedding_constructs():
    emb = FaceEmbedding(
        vector=(0.1, 0.2, 0.3),
        model_version="test-model-v1",
        face_bbox=BoundingBox(x=0, y=0, width=100, height=100),
        quality_score=0.9,
    )
    assert emb.model_version == "test-model-v1"


def test_face_embedding_rejects_empty_vector():
    with pytest.raises(ValueError):
        FaceEmbedding(
            vector=(),
            model_version="test-model-v1",
            face_bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )


def test_face_comparison_result_constructs():
    result = FaceComparisonResult(
        similarity_score=0.83,
        band=MatchBand.STRONG_MATCH,
        model_version="test-model-v1",
    )
    assert result.similarity_score == 0.83


def test_face_comparison_result_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        FaceComparisonResult(
            similarity_score=1.2,
            band=MatchBand.STRONG_MATCH,
            model_version="test-model-v1",
        )
