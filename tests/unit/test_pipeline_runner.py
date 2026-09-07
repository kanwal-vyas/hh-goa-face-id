from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import app.pipeline.runner as runner_module
from app.blockchain.interface import (
    BlockchainProvider,
    BlockchainRecord,
    OnChainPayload,
)
from app.blockchain.local_provider import (
    BlockchainConnectionError,
    BlockchainRegistrationError,
)
from app.content.canonicalize import (
    CanonicalContent,
    Canonicalizer,
    DeterministicCanonicalizer,
)
from app.content.extractor import (
    ContentExtractionError,
    ContentExtractor,
    DiscoveredContent,
)
from app.face.compare import CosineFaceComparator, FaceComparisonResult, MatchBand
from app.face.processor import BoundingBox, FaceEmbedding, FaceProcessor
from app.matching.candidate_matcher import CandidateMatcher, MatchResult
from app.pipeline.runner import STAGE_NAMES, PipelineRunner
from app.search.interface import (
    ContentType,
    SearchCandidate,
    SearchProvider,
    SearchQuery,
)
from app.verification.verifier import BlockchainVerifier


def _embedding() -> FaceEmbedding:
    return FaceEmbedding(
        vector=(1.0, 0.0),
        model_version="test-v1",
        face_bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )


def _candidate(candidate_id: str = "candidate-1") -> SearchCandidate:
    return SearchCandidate(
        candidate_id=candidate_id,
        source_reference=f"demo://authorized/{candidate_id}",
        content_type=ContentType.MIXED,
        preview_available=True,
        provider_confidence=1.0,
    )


class StubFaceProcessor(FaceProcessor):
    def __init__(self, response: list[FaceEmbedding] | Exception) -> None:
        self.response = response

    def process(self, image_bytes: bytes) -> list[FaceEmbedding]:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class StubSearchProvider(SearchProvider):
    def __init__(self, response: list[SearchCandidate] | Exception) -> None:
        self.response = response

    def search(self, query: SearchQuery) -> list[SearchCandidate]:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class StubMatcher(CandidateMatcher):
    def __init__(self, results: list[MatchResult]) -> None:
        self.results = results

    def match(self, candidates: list[SearchCandidate]) -> list[MatchResult]:
        return self.results


class StubExtractor(ContentExtractor):
    def __init__(self, response: DiscoveredContent | Exception) -> None:
        self.response = response

    def extract(self, candidate_id: str, source_reference: str) -> DiscoveredContent:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FailingCanonicalizer(Canonicalizer):
    def canonicalize(self, content: DiscoveredContent) -> CanonicalContent:
        raise ValueError("canonicalization unavailable")


class StubBlockchainProvider(BlockchainProvider):
    def __init__(
        self,
        *,
        registration_error: Exception | None = None,
        retrieve_mutates_hash: bool = False,
    ) -> None:
        self.registration_error = registration_error
        self.retrieve_mutates_hash = retrieve_mutates_hash
        self.registered_payload: OnChainPayload | None = None

    def register(self, payload: OnChainPayload) -> BlockchainRecord:
        if self.registration_error is not None:
            raise self.registration_error
        self.registered_payload = payload
        return self._record(payload)

    def retrieve(self, tx_hash: str) -> BlockchainRecord | None:
        assert self.registered_payload is not None
        payload = self.registered_payload
        if self.retrieve_mutates_hash:
            payload = OnChainPayload(
                content_hash="f" * 64,
                algorithm=payload.algorithm,
                version=payload.version,
                source_reference_hash=payload.source_reference_hash,
            )
        return self._record(payload)

    @staticmethod
    def _record(payload: OnChainPayload) -> BlockchainRecord:
        return BlockchainRecord(
            tx_hash="0x" + "a" * 64,
            block_number=7,
            timestamp=datetime.now(timezone.utc),
            on_chain_payload=payload,
        )


def _content() -> DiscoveredContent:
    return DiscoveredContent(
        candidate_id="candidate-1",
        raw_bytes=b"synthetic-content",
        text="authorized demo content",
        metadata={"kind": "demo"},
        source_reference="demo://authorized/candidate-1",
    )


def _selected_result() -> MatchResult:
    candidate = _candidate()
    return MatchResult(
        candidate=candidate,
        face_similarity=FaceComparisonResult(1.0, MatchBand.STRONG_MATCH, "test-v1"),
        content_relevance=1.0,
        usable=True,
        rejection_reason=None,
        is_selected=True,
        selection_reason="Selected by policy.",
    )


def _runner(
    tmp_path: Path,
    *,
    face_response: list[FaceEmbedding] | Exception | None = None,
    search_response: list[SearchCandidate] | Exception | None = None,
    match_results: list[MatchResult] | None = None,
    extractor: ContentExtractor | None = None,
    canonicalizer: Canonicalizer | None = None,
    blockchain_factory=lambda: StubBlockchainProvider(),
) -> tuple[PipelineRunner, Path]:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"the stub processor does not decode this")
    processor = StubFaceProcessor(face_response if face_response is not None else [_embedding()])
    candidates = search_response if search_response is not None else [_candidate()]
    matcher = StubMatcher(match_results if match_results is not None else [_selected_result()])
    return (
        PipelineRunner(
            face_processor=processor,
            face_comparator=CosineFaceComparator(0.6),
            search_provider=StubSearchProvider(candidates),
            candidate_matcher_factory=lambda reference, processor, comparator: matcher,
            content_extractor=extractor or StubExtractor(_content()),
            canonicalizer=canonicalizer or DeterministicCanonicalizer(),
            blockchain_provider_factory=blockchain_factory,
            verifier=BlockchainVerifier(),
        ),
        reference,
    )


def test_missing_reference_returns_closed_stage_one_failure(tmp_path):
    runner, reference = _runner(tmp_path)

    report = runner.run(reference.with_name("missing.jpg"))

    assert report.succeeded is False
    assert len(report.stages) == 1
    assert report.stages[0].succeeded is False
    assert "not found" in report.failure_reason.lower()


def test_reference_zero_and_multiple_face_failures_stop_at_stage_one(tmp_path):
    for faces, expected in (([], "No face"), ([_embedding(), _embedding()], "Multiple faces")):
        runner, reference = _runner(tmp_path, face_response=faces)
        report = runner.run(reference)
        assert report.succeeded is False
        assert len(report.stages) == 1
        assert expected in report.failure_reason


def test_search_empty_result_stops_before_matching(tmp_path):
    runner, reference = _runner(tmp_path, search_response=[])

    report = runner.run(reference)

    assert report.succeeded is False
    assert len(report.stages) == 2
    assert "no candidates" in report.failure_reason.lower()


def test_no_selectable_and_ambiguous_matches_stop_at_stage_three(tmp_path):
    candidate = _candidate()
    no_selection = MatchResult(
        candidate=candidate,
        face_similarity=None,
        content_relevance=1.0,
        usable=False,
        rejection_reason="No face detected",
        selection_reason="No candidate selected: all candidates were unusable.",
    )
    ambiguous = MatchResult(
        candidate=candidate,
        face_similarity=FaceComparisonResult(0.9, MatchBand.STRONG_MATCH, "test-v1"),
        content_relevance=1.0,
        usable=True,
        rejection_reason=None,
        selection_reason="No candidate selected: top STRONG_MATCH candidates are tied and ambiguous.",
    )
    for result, expected in ((no_selection, "unusable"), (ambiguous, "ambiguous")):
        runner, reference = _runner(tmp_path, match_results=[result])
        report = runner.run(reference)
        assert report.succeeded is False
        assert len(report.stages) == 3
        assert expected in report.failure_reason.lower()


def test_extraction_canonicalization_and_fingerprint_failures_stop_closed(tmp_path, monkeypatch):
    runner, reference = _runner(
        tmp_path, extractor=StubExtractor(ContentExtractionError("missing content"))
    )
    assert runner.run(reference).stages[-1].index == 4

    runner, reference = _runner(tmp_path, canonicalizer=FailingCanonicalizer())
    assert runner.run(reference).stages[-1].index == 5

    def _broken_fingerprint(canonical_content):
        raise RuntimeError("hash backend failed")

    monkeypatch.setattr(runner_module, "hash_canonical_content", _broken_fingerprint)
    runner, reference = _runner(tmp_path)
    report = runner.run(reference)
    assert report.stages[-1].index == 6
    assert "fingerprinting failed" in report.failure_reason.lower()


def test_blockchain_unavailable_and_registration_failure_stop_at_stage_seven(tmp_path):
    runner, reference = _runner(
        tmp_path,
        blockchain_factory=lambda: (_ for _ in ()).throw(BlockchainConnectionError("offline")),
    )
    assert runner.run(reference).stages[-1].index == 7

    runner, reference = _runner(
        tmp_path,
        blockchain_factory=lambda: StubBlockchainProvider(
            registration_error=BlockchainRegistrationError("reverted")
        ),
    )
    assert runner.run(reference).stages[-1].index == 7


def test_verification_mismatch_is_not_reported_as_success(tmp_path):
    runner, reference = _runner(
        tmp_path,
        blockchain_factory=lambda: StubBlockchainProvider(retrieve_mutates_hash=True),
    )

    report = runner.run(reference)

    assert report.succeeded is False
    assert report.verification_result is not None
    assert report.verification_result.match is False
    assert report.stages[-1].index == 8
    assert "mismatch" in report.failure_reason.lower()


def test_successful_pipeline_returns_every_stage_and_matching_fingerprint(tmp_path):
    provider = StubBlockchainProvider()
    runner, reference = _runner(tmp_path, blockchain_factory=lambda: provider)

    report = runner.run(reference)

    assert report.succeeded is True
    assert len(report.stages) == len(STAGE_NAMES)
    assert all(stage.succeeded for stage in report.stages)
    assert report.selected_match is not None
    assert report.fingerprint is not None
    assert report.blockchain_record is not None
    assert report.verification_result is not None
    assert report.verification_result.local_hash == report.fingerprint.hash
    assert provider.registered_payload is not None
    assert provider.registered_payload.content_hash == report.fingerprint.hash
