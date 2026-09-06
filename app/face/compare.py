"""
Face comparison scaffold.

Defines the data model and interface for comparing two FaceEmbeddings and
producing a similarity score. No concrete similarity metric is wired in
at this milestone.

Similarity scores are always reported as continuous values plus a
qualitative band — never as a binary "this is the same person" claim.
See FaceComparator.compare() for the banding contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.face.processor import FaceEmbedding


class MatchBand(str, Enum):
    """Qualitative band for a similarity score, relative to a configured
    threshold. Intentionally avoids identity-confirmation language."""

    NO_MATCH = "no_match"
    POSSIBLE_MATCH = "possible_match"
    STRONG_MATCH = "strong_match"


@dataclass(frozen=True)
class FaceComparisonResult:
    """Result of comparing a reference embedding against a candidate
    embedding.

    Attributes:
        similarity_score: Continuous similarity value in [0.0, 1.0].
            Higher means more similar. The exact metric (e.g. cosine
            similarity) is defined by the concrete FaceComparator.
        band: Qualitative band derived from similarity_score and the
            configured match threshold(s).
        model_version: Embedding model version the comparison was
            performed under (both embeddings must share this).
    """

    similarity_score: float
    band: MatchBand
    model_version: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("similarity_score must be between 0.0 and 1.0")


class FaceComparator(ABC):
    """Abstract interface for scoring similarity between two embeddings.

    Concrete implementations (e.g. cosine similarity with a configurable
    threshold) are added in a later milestone.
    """

    @abstractmethod
    def compare(
        self, reference: FaceEmbedding, candidate: FaceEmbedding
    ) -> FaceComparisonResult:
        """Compare two embeddings and return a similarity result.

        Raises:
            ValueError: if the embeddings were produced by different
                model versions (comparisons across model versions are
                not meaningful).
        """
        raise NotImplementedError


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Pure cosine similarity between two equal-length vectors, clamped
    to [0.0, 1.0].

    Unit-testable without any image, detector, or model — operates on
    plain float tuples.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length to compare")
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    score = float(np.dot(va, vb) / denom)
    # LBP-histogram vectors are non-negative so cosine similarity should
    # already fall in [0, 1], but clamp defensively against floating
    # point drift.
    return max(0.0, min(1.0, score))


def derive_match_band(similarity_score: float, match_threshold: float) -> MatchBand:
    """Map a similarity score to a qualitative band given a configured
    threshold.

    Banding is intentionally coarse and heuristic — see README
    "Thresholds" section for why these boundaries are not a scientific
    guarantee for this milestone's classical (non-deep-learned)
    embedding, and must be tuned against your own authorized test
    images before relying on them for a demo.

    Boundaries (relative to match_threshold, e.g. 0.6 by default):
        similarity_score < match_threshold * 0.5   -> NO_MATCH
        match_threshold * 0.5 <= score < threshold  -> POSSIBLE_MATCH
        similarity_score >= match_threshold          -> STRONG_MATCH
    """
    if similarity_score >= match_threshold:
        return MatchBand.STRONG_MATCH
    if similarity_score >= match_threshold * 0.5:
        return MatchBand.POSSIBLE_MATCH
    return MatchBand.NO_MATCH


class CosineFaceComparator(FaceComparator):
    """Concrete FaceComparator using cosine similarity and a configurable
    match threshold (see app.config.settings.Settings.match_threshold)."""

    def __init__(self, match_threshold: float) -> None:
        if not (0.0 <= match_threshold <= 1.0):
            raise ValueError("match_threshold must be between 0.0 and 1.0")
        self._match_threshold = match_threshold

    def compare(
        self, reference: FaceEmbedding, candidate: FaceEmbedding
    ) -> FaceComparisonResult:
        if reference.model_version != candidate.model_version:
            raise ValueError(
                "Cannot compare embeddings from different model versions: "
                f"{reference.model_version!r} vs {candidate.model_version!r}"
            )
        score = cosine_similarity(reference.vector, candidate.vector)
        band = derive_match_band(score, self._match_threshold)
        return FaceComparisonResult(
            similarity_score=score,
            band=band,
            model_version=reference.model_version,
        )
