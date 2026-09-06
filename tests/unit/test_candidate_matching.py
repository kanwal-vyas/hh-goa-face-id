from __future__ import annotations

from pathlib import Path

from app.face.compare import CosineFaceComparator, MatchBand
from app.face.processor import (
    BoundingBox,
    FaceEmbedding,
    FaceProcessingError,
    FaceProcessor,
)
from app.matching.candidate_matcher import (
    AuthorizedCorpusCandidateMatcher,
    CandidateMatcher,
    MatchResult,
)
from app.search.authorized_corpus import AuthorizedCorpusSearchProvider
from app.search.interface import ContentType, SearchCandidate, SearchQuery


def _embedding(vector: tuple[float, ...]) -> FaceEmbedding:
    return FaceEmbedding(
        vector=vector,
        model_version="test-v1",
        face_bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )


class StubFaceProcessor(FaceProcessor):
    """Deterministic local processor used to test matcher orchestration."""

    def __init__(self, responses: dict[bytes, list[FaceEmbedding] | Exception]) -> None:
        self.responses = responses
        self.processed_bytes: list[bytes] = []

    def process(self, image_bytes: bytes) -> list[FaceEmbedding]:
        self.processed_bytes.append(image_bytes)
        response = self.responses[image_bytes]
        if isinstance(response, Exception):
            raise response
        return response


def _candidate(candidate_id: str, confidence: float | None = 0.5) -> SearchCandidate:
    return SearchCandidate(
        candidate_id=candidate_id,
        source_reference=f"demo://authorized-social/posts/{candidate_id}",
        content_type=ContentType.IMAGE,
        preview_available=True,
        provider_confidence=confidence,
    )


def _image_paths(tmp_path: Path, **images: bytes | str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for candidate_id, image_bytes in images.items():
        path = tmp_path / f"{candidate_id}.img"
        path.write_bytes(image_bytes.encode() if isinstance(image_bytes, str) else image_bytes)
        paths[candidate_id] = path
    return paths


def _matcher(
    reference: FaceEmbedding,
    processor: FaceProcessor,
    paths: dict[str, Path],
    threshold: float = 0.8,
) -> AuthorizedCorpusCandidateMatcher:
    return AuthorizedCorpusCandidateMatcher(
        reference_embedding=reference,
        face_processor=processor,
        face_comparator=CosineFaceComparator(threshold),
        candidate_image_paths=paths,
    )


def test_one_genuine_strong_match_is_selected(tmp_path):
    reference = _embedding((1.0, 0.0))
    paths = _image_paths(tmp_path, natural="one-face")
    processor = StubFaceProcessor({b"one-face": [_embedding((1.0, 0.0))]})

    result = _matcher(reference, processor, paths).match([_candidate("natural")])[0]

    assert result.usable is True
    assert result.match_band == MatchBand.STRONG_MATCH
    assert result.similarity_score == 1.0
    assert result.is_selected is True
    assert result.candidate_id == "natural"
    assert "not an identity claim" in (result.selection_reason or "")


def test_no_match_is_ranked_but_never_selected(tmp_path):
    paths = _image_paths(tmp_path, distant="distant-face")
    processor = StubFaceProcessor({b"distant-face": [_embedding((0.0, 1.0))]})

    result = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("distant")]
    )[0]

    assert result.usable is True
    assert result.match_band == MatchBand.NO_MATCH
    assert result.is_selected is False
    assert "no usable candidate reached STRONG_MATCH" in (result.selection_reason or "")


def test_multiple_candidates_are_ranked_by_band_then_similarity_then_id(tmp_path):
    paths = _image_paths(
        tmp_path,
        no="no-face",
        possible="possible-face",
        strong="strong-face",
    )
    processor = StubFaceProcessor(
        {
            b"no-face": [_embedding((0.0, 1.0))],
            b"possible-face": [_embedding((0.5, 0.8660254037844386))],
            b"strong-face": [_embedding((1.0, 0.0))],
        }
    )

    results = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("no"), _candidate("possible"), _candidate("strong")]
    )

    assert [result.candidate_id for result in results] == ["strong", "possible", "no"]
    assert [result.match_band for result in results] == [
        MatchBand.STRONG_MATCH,
        MatchBand.POSSIBLE_MATCH,
        MatchBand.NO_MATCH,
    ]
    assert [result.is_selected for result in results] == [True, False, False]


def test_tied_top_strong_candidates_use_id_order_and_are_not_selected(tmp_path):
    paths = _image_paths(tmp_path, beta="beta-face", alpha="alpha-face")
    processor = StubFaceProcessor(
        {
            b"beta-face": [_embedding((1.0, 0.0))],
            b"alpha-face": [_embedding((1.0, 0.0))],
        }
    )

    results = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("beta"), _candidate("alpha")]
    )

    assert [result.candidate_id for result in results] == ["alpha", "beta"]
    assert not any(result.is_selected for result in results)
    assert all("tied and ambiguous" in (result.selection_reason or "") for result in results)


def test_zero_candidates_returns_empty_ranked_list(tmp_path):
    matcher = _matcher(_embedding((1.0, 0.0)), StubFaceProcessor({}), {})

    assert matcher.match([]) == []


def test_all_invalid_candidates_are_rejected_without_selection(tmp_path):
    paths = _image_paths(tmp_path, zero="zero-face", multi="multi-face")
    processor = StubFaceProcessor(
        {
            b"zero-face": [],
            b"multi-face": [_embedding((1.0, 0.0)), _embedding((1.0, 0.0))],
        }
    )

    results = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("zero"), _candidate("multi")]
    )

    assert [result.candidate_id for result in results] == ["multi", "zero"]
    assert all(result.usable is False for result in results)
    assert all(result.face_similarity is None for result in results)
    assert "Multiple faces" in (results[0].rejection_reason or "")
    assert "No face" in (results[1].rejection_reason or "")
    assert all("all candidates were unusable" in (result.selection_reason or "") for result in results)


def test_missing_candidate_image_is_rejected(tmp_path):
    missing_path = tmp_path / "gone.img"
    matcher = _matcher(
        _embedding((1.0, 0.0)),
        StubFaceProcessor({}),
        {"gone": missing_path},
    )

    result = matcher.match([_candidate("gone")])[0]

    assert result.usable is False
    assert result.similarity_score is None
    assert "missing or not a file" in (result.rejection_reason or "")


def test_unreadable_candidate_image_is_rejected(tmp_path, monkeypatch):
    paths = _image_paths(tmp_path, blocked="blocked-image")

    def _unreadable(_path: Path) -> bytes:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", _unreadable)
    result = _matcher(
        _embedding((1.0, 0.0)), StubFaceProcessor({}), paths
    ).match([_candidate("blocked")])[0]

    assert result.usable is False
    assert "unreadable" in (result.rejection_reason or "")


def test_processor_failure_is_rejected_cleanly(tmp_path):
    paths = _image_paths(tmp_path, corrupt="corrupt-image")
    processor = StubFaceProcessor(
        {b"corrupt-image": FaceProcessingError("could not decode image")}
    )

    result = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("corrupt")]
    )[0]

    assert result.usable is False
    assert "could not be processed" in (result.rejection_reason or "")


def test_possible_match_is_reported_but_not_selected(tmp_path):
    paths = _image_paths(tmp_path, possible="possible-face")
    processor = StubFaceProcessor(
        {b"possible-face": [_embedding((0.5, 0.8660254037844386))]}
    )

    result = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("possible")]
    )[0]

    assert result.match_band == MatchBand.POSSIBLE_MATCH
    assert result.is_selected is False
    assert "no usable candidate reached STRONG_MATCH" in (result.selection_reason or "")


def test_unique_highest_score_is_selected_among_multiple_strong_matches(tmp_path):
    paths = _image_paths(tmp_path, highest="highest-face", lower="lower-face")
    processor = StubFaceProcessor(
        {
            b"highest-face": [_embedding((1.0, 0.0))],
            b"lower-face": [_embedding((0.9, 0.4358898943540673))],
        }
    )

    results = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("lower"), _candidate("highest")]
    )

    assert [result.candidate_id for result in results] == ["highest", "lower"]
    assert [result.is_selected for result in results] == [True, False]


def test_matcher_uses_reference_embedding_and_reference_change_changes_result(tmp_path):
    paths = _image_paths(tmp_path, sample="candidate-face")
    processor = StubFaceProcessor({b"candidate-face": [_embedding((1.0, 0.0))]})

    matching = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("sample")]
    )[0]
    nonmatching = _matcher(_embedding((0.0, 1.0)), processor, paths).match(
        [_candidate("sample")]
    )[0]

    assert matching.match_band == MatchBand.STRONG_MATCH
    assert nonmatching.match_band == MatchBand.NO_MATCH
    assert matching.similarity_score != nonmatching.similarity_score


def test_winner_is_based_on_similarity_not_hardcoded_candidate_id(tmp_path):
    paths = _image_paths(tmp_path, unrelated="no-face", arbitrary="strong-face")
    processor = StubFaceProcessor(
        {
            b"no-face": [_embedding((0.0, 1.0))],
            b"strong-face": [_embedding((1.0, 0.0))],
        }
    )

    results = _matcher(_embedding((1.0, 0.0)), processor, paths).match(
        [_candidate("unrelated"), _candidate("arbitrary")]
    )

    assert [result.candidate_id for result in results if result.is_selected] == ["arbitrary"]


def test_matcher_implements_interface_and_returns_structured_results(tmp_path):
    paths = _image_paths(tmp_path, sample="candidate-face")
    processor = StubFaceProcessor({b"candidate-face": [_embedding((1.0, 0.0))]})
    matcher = _matcher(_embedding((1.0, 0.0)), processor, paths)

    results = matcher.match([_candidate("sample")])

    assert isinstance(matcher, CandidateMatcher)
    assert isinstance(results[0], MatchResult)
    assert results[0].source_reference.endswith("/sample")
    assert results[0].content_relevance == 0.5


def test_matcher_accepts_candidates_returned_by_authorized_provider():
    provider = AuthorizedCorpusSearchProvider()
    candidates = provider.search(
        SearchQuery(
            reference_id="authorized-demo-run",
            face_embedding_ref="opaque-handle",
            query_hints={"tag": "workshop"},
        )
    )
    responses = {
        path.read_bytes(): [_embedding((1.0, 0.0))]
        for path in provider.candidate_image_paths.values()
    }
    processor = StubFaceProcessor(responses)
    matcher = _matcher(
        _embedding((1.0, 0.0)), processor, provider.candidate_image_paths
    )

    results = matcher.match(candidates)

    assert [result.candidate_id for result in results] == [
        "demo-post-coast-02",
        "demo-post-harbor-01",
    ]
    assert not any(result.is_selected for result in results)
    assert all("tied and ambiguous" in (result.selection_reason or "") for result in results)
