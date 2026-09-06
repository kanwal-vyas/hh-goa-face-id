"""
Candidate matching scaffold.

Combines a face-similarity signal (from app.face.compare) and a
content-relevance signal (from the SearchCandidate itself) into a single
MatchResult, WITHOUT collapsing them into one vague confidence number.
The two signals answer different questions and are kept separate per the
architecture:

    - face_similarity: does the candidate's face resemble the reference?
    - content_relevance: does the discovered content correspond to the
      search target at all (independent of any face comparison)?

No concrete decision logic (thresholding, ranking across multiple
candidates) is implemented at this milestone — only the data model and
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.face.compare import FaceComparisonResult
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
        is_selected: Whether this candidate was chosen as the best match
            for the run. Decision logic (thresholds, ranking) is
            implemented in a later milestone.
    """

    candidate: SearchCandidate
    face_similarity: FaceComparisonResult | None
    content_relevance: float | None
    is_selected: bool = False


class CandidateMatcher(ABC):
    """Abstract interface for turning a list of SearchCandidates into
    ranked MatchResults.

    Concrete decision/ranking logic is added in a later milestone.
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
