"""Candidate-matching abstractions and authorized-corpus implementation."""

from app.matching.candidate_matcher import (
    AuthorizedCorpusCandidateMatcher,
    CandidateMatcher,
    MatchResult,
)

__all__ = [
    "AuthorizedCorpusCandidateMatcher",
    "CandidateMatcher",
    "MatchResult",
]
