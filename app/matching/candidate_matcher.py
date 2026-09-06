"""Candidate matching over content from the authorized local corpus.

The matcher deliberately keeps three independent signals visible:

* content relevance, produced by ``SearchProvider``;
* face similarity, produced by ``FaceComparator``; and
* whether candidate content was usable for face processing.

It returns ranked candidates rather than an identity decision. A selected
candidate means only that the deterministic selection policy was met; it is
never a claim that the candidate depicts a particular real-world person.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from app.face.compare import FaceComparator, FaceComparisonResult, MatchBand
from app.face.processor import FaceEmbedding, FaceProcessingError, FaceProcessor
from app.search.interface import SearchCandidate


@dataclass(frozen=True)
class MatchResult:
    """Outcome of evaluating a single candidate against the reference.

    Attributes:
        candidate: The SearchCandidate being evaluated.
        face_similarity: Result of comparing the candidate's face
            embedding against the reference, or None if the candidate
            had no usable face (e.g. text-only content, or no face
            detected in a candidate image).
        content_relevance: The provider's own relevance signal for this
            candidate (SearchCandidate.provider_confidence), surfaced
            here explicitly rather than re-derived, so it's clear this
            is a separate concept from face_similarity.
        usable: Whether the candidate image produced exactly one usable
            face embedding and a comparison result.
        rejection_reason: Explicit reason a candidate was unusable. It is
            ``None`` for a usable candidate; no score is fabricated for a
            rejected candidate.
        is_selected: Whether this candidate met the deterministic
            selection policy. This is a candidate-selection signal only,
            never a real-world identity claim.
        selection_reason: Run-level selection outcome repeated on ranked
            results so callers receiving only this list can explain why
            nothing was selected or why a candidate was selected.
    """

    candidate: SearchCandidate
    face_similarity: FaceComparisonResult | None
    content_relevance: float | None
    usable: bool = False
    rejection_reason: str | None = "Not evaluated."
    is_selected: bool = False
    selection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.usable and self.face_similarity is None:
            raise ValueError("A usable MatchResult requires face_similarity")
        if self.usable and self.rejection_reason is not None:
            raise ValueError("A usable MatchResult cannot have rejection_reason")
        if not self.usable and self.face_similarity is not None:
            raise ValueError("An unusable MatchResult cannot have face_similarity")
        if not self.usable and not self.rejection_reason:
            raise ValueError("An unusable MatchResult requires rejection_reason")
        if self.is_selected and not self.usable:
            raise ValueError("An unusable MatchResult cannot be selected")

    @property
    def candidate_id(self) -> str:
        """Stable identifier of the evaluated search candidate."""
        return self.candidate.candidate_id

    @property
    def source_reference(self) -> str:
        """Off-chain source reference supplied by the search provider."""
        return self.candidate.source_reference

    @property
    def similarity_score(self) -> float | None:
        """Face similarity, or ``None`` when this candidate was rejected."""
        return (
            self.face_similarity.similarity_score
            if self.face_similarity is not None
            else None
        )

    @property
    def match_band(self) -> MatchBand | None:
        """Qualitative face-similarity band, or ``None`` when rejected."""
        return self.face_similarity.band if self.face_similarity is not None else None


class CandidateMatcher(ABC):
    """Abstract interface for turning a list of SearchCandidates into
    ranked MatchResults.

    Concrete implementations receive their reference embedding and local
    candidate-image resolver through construction, keeping this method's
    established candidate-list interface intact.
    """

    @abstractmethod
    def match(self, candidates: list[SearchCandidate]) -> list[MatchResult]:
        """Evaluate candidates and return MatchResults.

        Implementations must not fabricate face_similarity or
        content_relevance values for candidates that could not
        actually be evaluated (e.g. unfetchable image) — such cases
        should surface as None with the reason logged, not a
        placeholder score.
        """
        raise NotImplementedError


class AuthorizedCorpusCandidateMatcher(CandidateMatcher):
    """Compare one reference embedding with images from an authorized corpus.

    Args:
        reference_embedding: The already processed, explicitly supplied
            reference face. It is used in every comparison and is not persisted.
        face_processor: Existing local ``FaceProcessor`` implementation.
        face_comparator: Existing ``FaceComparator`` implementation.
        candidate_image_paths: Local authorized image path for each candidate
            id. No URLs, platform APIs, or external fetches are supported.

    Results are ordered by match band (STRONG, POSSIBLE, NO), then descending
    similarity, then ``candidate_id``. Rejected candidates follow usable ones
    in stable ``candidate_id`` order. A candidate is selected only if it is the
    sole highest-scoring STRONG_MATCH; tied top strong candidates are marked
    ambiguous, and possible/no-match candidates are never selected.
    """

    def __init__(
        self,
        reference_embedding: FaceEmbedding,
        face_processor: FaceProcessor,
        face_comparator: FaceComparator,
        candidate_image_paths: Mapping[str, str | Path],
    ) -> None:
        self._reference_embedding = reference_embedding
        self._face_processor = face_processor
        self._face_comparator = face_comparator
        self._candidate_image_paths = {
            candidate_id: Path(path) for candidate_id, path in candidate_image_paths.items()
        }

    def match(self, candidates: list[SearchCandidate]) -> list[MatchResult]:
        """Evaluate, deterministically rank, and cautiously select candidates."""
        results = [self._evaluate(candidate) for candidate in candidates]
        ranked = sorted(results, key=self._rank_key)
        return self._apply_selection_policy(ranked)

    def _evaluate(self, candidate: SearchCandidate) -> MatchResult:
        image_path = self._candidate_image_paths.get(candidate.candidate_id)
        if image_path is None:
            return self._rejected(candidate, "No authorized image path is registered.")
        if not image_path.is_file():
            return self._rejected(candidate, "Authorized candidate image is missing or not a file.")

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            return self._rejected(candidate, f"Authorized candidate image is unreadable: {exc}")

        try:
            embeddings = self._face_processor.process(image_bytes)
        except FaceProcessingError as exc:
            return self._rejected(candidate, f"Candidate image could not be processed: {exc}")

        if len(embeddings) == 0:
            return self._rejected(candidate, "No face detected in candidate image.")
        if len(embeddings) > 1:
            return self._rejected(
                candidate,
                "Multiple faces detected in candidate image; no face was selected.",
            )

        try:
            comparison = self._face_comparator.compare(
                self._reference_embedding, embeddings[0]
            )
        except ValueError as exc:
            return self._rejected(candidate, f"Candidate face could not be compared: {exc}")

        return MatchResult(
            candidate=candidate,
            face_similarity=comparison,
            content_relevance=candidate.provider_confidence,
            usable=True,
            rejection_reason=None,
        )

    @staticmethod
    def _rejected(candidate: SearchCandidate, reason: str) -> MatchResult:
        return MatchResult(
            candidate=candidate,
            face_similarity=None,
            content_relevance=candidate.provider_confidence,
            usable=False,
            rejection_reason=reason,
        )

    @staticmethod
    def _rank_key(result: MatchResult) -> tuple[int, float, str]:
        if not result.usable:
            return (1, 0.0, result.candidate.candidate_id)
        assert result.face_similarity is not None
        band_rank = {
            "strong_match": 0,
            "possible_match": 1,
            "no_match": 2,
        }[result.face_similarity.band.value]
        return (band_rank, -result.face_similarity.similarity_score, result.candidate.candidate_id)

    @staticmethod
    def _apply_selection_policy(results: list[MatchResult]) -> list[MatchResult]:
        strong_results = [
            result
            for result in results
            if result.usable and result.face_similarity is not None
            and result.face_similarity.band.value == "strong_match"
        ]
        usable_results = [result for result in results if result.usable]

        if not usable_results:
            reason = "No candidate selected: all candidates were unusable."
            return [replace(result, selection_reason=reason) for result in results]
        if not strong_results:
            reason = "No candidate selected: no usable candidate reached STRONG_MATCH."
            return [replace(result, selection_reason=reason) for result in results]

        top = strong_results[0]
        assert top.face_similarity is not None
        tied_top = [
            result
            for result in strong_results
            if result.face_similarity is not None
            and result.face_similarity.similarity_score == top.face_similarity.similarity_score
        ]
        if len(tied_top) > 1:
            reason = "No candidate selected: top STRONG_MATCH candidates are tied and ambiguous."
            return [replace(result, selection_reason=reason) for result in results]

        reason = (
            "Selected by policy: sole highest-scoring STRONG_MATCH candidate; "
            "this is not an identity claim."
        )
        return [
            replace(result, is_selected=result is top, selection_reason=reason)
            for result in results
        ]
