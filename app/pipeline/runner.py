"""Real eight-stage orchestration for the bounded authorized demo."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.blockchain.interface import (
    BlockchainProvider,
    BlockchainRecord,
    OnChainPayload,
)
from app.content.canonicalize import CanonicalContent, Canonicalizer
from app.content.extractor import ContentExtractor, DiscoveredContent
from app.content.fingerprint import (
    ContentFingerprint,
    hash_canonical_content,
    hash_source_reference,
)
from app.face.compare import FaceComparator
from app.face.opencv_processor import require_single_face
from app.face.processor import FaceEmbedding, FaceProcessingError, FaceProcessor
from app.matching.candidate_matcher import CandidateMatcher, MatchResult
from app.search.interface import SearchCandidate, SearchProvider, SearchQuery
from app.verification.verifier import VerificationResult, Verifier

logger = logging.getLogger(__name__)

STAGE_NAMES: tuple[str, ...] = (
    "Face processing",
    "Authorized search",
    "Candidate matching",
    "Content extraction",
    "Canonicalization",
    "SHA-256 fingerprint",
    "Blockchain registration",
    "Verification",
)

CandidateMatcherFactory = Callable[[FaceEmbedding, FaceProcessor, FaceComparator], CandidateMatcher]
BlockchainProviderFactory = Callable[[], BlockchainProvider]


class ReferenceImageNotFoundError(Exception):
    """Retained for callers that need a distinct missing-reference error type."""


@dataclass(frozen=True)
class StageStatus:
    """Status of an attempted pipeline stage.

    Stages after a failure are intentionally absent: the pipeline fails closed
    and never represents unexecuted downstream work as successful.
    """

    name: str
    index: int
    total: int
    succeeded: bool
    detail: str

    @property
    def label(self) -> str:
        return f"[{self.index}/{self.total}] {self.name}"

    @property
    def implemented(self) -> bool:
        """Compatibility signal: every recorded stage is a real implementation."""
        return True


@dataclass(frozen=True)
class PipelineRunReport:
    """Structured result of an authorized end-to-end pipeline run."""

    reference_image: Path
    stages: tuple[StageStatus, ...] = field(default_factory=tuple)
    reference_embedding: FaceEmbedding | None = None
    candidates: tuple[SearchCandidate, ...] = field(default_factory=tuple)
    ranked_matches: tuple[MatchResult, ...] = field(default_factory=tuple)
    selected_match: MatchResult | None = None
    extracted_content: DiscoveredContent | None = None
    canonical_content: CanonicalContent | None = None
    fingerprint: ContentFingerprint | None = None
    source_reference_hash: str | None = None
    blockchain_record: BlockchainRecord | None = None
    verification_result: VerificationResult | None = None
    failure_reason: str | None = None

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def any_implemented(self) -> bool:
        """Compatibility helper retained from the former scaffold report."""
        return bool(self.stages)

    @property
    def succeeded(self) -> bool:
        return (
            self.failure_reason is None
            and self.verification_result is not None
            and self.verification_result.match
        )


class PipelineRunner:
    """Compose existing components into a real, bounded pipeline.

    The caller injects concrete implementations instead of the runner
    inventing alternatives. The matcher factory receives the ephemeral
    reference embedding and creates the existing corpus matcher for that run;
    the blockchain factory defers the RPC connection until stage seven.
    """

    def __init__(
        self,
        *,
        face_processor: FaceProcessor,
        face_comparator: FaceComparator,
        search_provider: SearchProvider,
        candidate_matcher_factory: CandidateMatcherFactory,
        content_extractor: ContentExtractor,
        canonicalizer: Canonicalizer,
        blockchain_provider_factory: BlockchainProviderFactory,
        verifier: Verifier,
    ) -> None:
        self._face_processor = face_processor
        self._face_comparator = face_comparator
        self._search_provider = search_provider
        self._candidate_matcher_factory = candidate_matcher_factory
        self._content_extractor = content_extractor
        self._canonicalizer = canonicalizer
        self._blockchain_provider_factory = blockchain_provider_factory
        self._verifier = verifier

    def run(
        self,
        reference_image: str | Path,
        *,
        query_hints: Mapping[str, str] | None = None,
        max_results: int = 5,
    ) -> PipelineRunReport:
        """Run all stages, returning a complete report or a closed failure."""
        reference_path = Path(reference_image)
        statuses: list[StageStatus] = []
        state: dict[str, object] = {}

        def succeed(index: int, detail: str) -> None:
            statuses.append(
                StageStatus(STAGE_NAMES[index - 1], index, len(STAGE_NAMES), True, detail)
            )

        def fail(index: int, detail: str) -> PipelineRunReport:
            statuses.append(
                StageStatus(STAGE_NAMES[index - 1], index, len(STAGE_NAMES), False, detail)
            )
            logger.warning("%s failed: %s", statuses[-1].label, detail)
            return PipelineRunReport(
                reference_image=reference_path,
                stages=tuple(statuses),
                failure_reason=detail,
                **state,
            )

        if not reference_path.is_file():
            return fail(1, f"Reference image not found or not a file: {reference_path}")

        try:
            reference_embedding = require_single_face(
                self._face_processor.process(reference_path.read_bytes()),
                context="reference image",
            )
        except (FaceProcessingError, OSError) as exc:
            return fail(1, str(exc))
        state["reference_embedding"] = reference_embedding
        succeed(1, "Exactly one reference face detected.")

        try:
            candidates = self._search_provider.search(
                SearchQuery(
                    reference_id=reference_path.name,
                    face_embedding_ref="ephemeral-reference-embedding",
                    query_hints=dict(query_hints or {}),
                    max_results=max_results,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- provider failures must be reported closed
            return fail(2, f"Authorized search failed: {exc}")
        state["candidates"] = tuple(candidates)
        if not candidates:
            return fail(2, "Authorized search returned no candidates.")
        succeed(2, f"Authorized search returned {len(candidates)} candidate(s).")

        try:
            matcher = self._candidate_matcher_factory(
                reference_embedding, self._face_processor, self._face_comparator
            )
            ranked_matches = matcher.match(candidates)
        except Exception as exc:  # noqa: BLE001 -- matcher failures must stop downstream work
            return fail(3, f"Candidate matching failed: {exc}")
        state["ranked_matches"] = tuple(ranked_matches)
        selected_match = next((result for result in ranked_matches if result.is_selected), None)
        if selected_match is None:
            reason = next(
                (result.selection_reason for result in ranked_matches if result.selection_reason),
                "No candidate was selected.",
            )
            return fail(3, reason)
        state["selected_match"] = selected_match
        succeed(3, f"Selected candidate {selected_match.candidate_id} by matching policy.")

        try:
            content = self._content_extractor.extract(
                selected_match.candidate_id, selected_match.source_reference
            )
        except Exception as exc:  # noqa: BLE001 -- extraction failures must stop hashing
            return fail(4, f"Content extraction failed: {exc}")
        state["extracted_content"] = content
        succeed(4, "Selected authorized content extracted.")

        try:
            canonical_content = self._canonicalizer.canonicalize(content)
        except Exception as exc:  # noqa: BLE001 -- never hash a failed canonicalization
            return fail(5, f"Canonicalization failed: {exc}")
        state["canonical_content"] = canonical_content
        succeed(5, f"Canonicalized with {canonical_content.canonicalization_version}.")

        try:
            fingerprint = hash_canonical_content(canonical_content)
            source_reference_hash = hash_source_reference(content.source_reference)
        except Exception as exc:  # noqa: BLE001 -- hashing failures must stop registration
            return fail(6, f"Fingerprinting failed: {exc}")
        state["fingerprint"] = fingerprint
        state["source_reference_hash"] = source_reference_hash
        succeed(6, f"Computed {fingerprint.algorithm} content fingerprint.")

        try:
            blockchain_provider = self._blockchain_provider_factory()
            blockchain_record = blockchain_provider.register(
                OnChainPayload(
                    content_hash=fingerprint.hash,
                    algorithm=fingerprint.algorithm,
                    version=fingerprint.canonicalization_version,
                    source_reference_hash=source_reference_hash,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- registration is a hard boundary
            return fail(7, f"Blockchain registration failed: {exc}")
        state["blockchain_record"] = blockchain_record
        succeed(7, f"Registered fingerprint in block {blockchain_record.block_number}.")

        try:
            verification_result = self._verifier.verify(
                canonical_content, blockchain_record.tx_hash, blockchain_provider
            )
        except Exception as exc:  # noqa: BLE001 -- unreadable records fail closed
            return fail(8, f"Blockchain verification failed: {exc}")
        state["verification_result"] = verification_result
        if not verification_result.match:
            return fail(8, "Blockchain verification mismatch: local and on-chain hashes differ.")
        succeed(8, "On-chain content hash matches the recomputed local fingerprint.")
        return PipelineRunReport(reference_image=reference_path, stages=tuple(statuses), **state)
