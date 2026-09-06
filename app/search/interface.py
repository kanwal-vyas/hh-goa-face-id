"""
SearchProvider abstraction.

SCOPE NOTE (read before extending this module):
This interface intentionally does NOT define or imply a general-purpose
open-web person-identification, reverse-face-search, or social-media
scraping capability. Per the approved architecture, the eventual concrete
implementation of SearchProvider will operate only against an authorized,
bounded demo corpus/target that the project operator controls and has
explicit permission to use (e.g. a small local dataset seeded for the
demo, or a single previously-agreed source to re-check).

No concrete SearchProvider is implemented in this milestone. Only the
provider-agnostic data structures and interface are defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ContentType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    MIXED = "mixed"


@dataclass(frozen=True)
class SearchQuery:
    """Input to a SearchProvider.search() call.

    Attributes:
        reference_id: Opaque handle identifying this pipeline run. Not a
            biometric payload — just a correlation id for logging/tracing.
        face_embedding_ref: A reference/token pointing at a face embedding
            held elsewhere in the pipeline (e.g. an in-memory handle or a
            short-lived cache key). Concrete providers must not require or
            accept the raw embedding vector or raw image bytes in this
            field; it exists so a provider *implementation* can decide, in
            a scoped and documented way, whether/how to use embedding
            similarity within its authorized corpus.
        query_hints: Optional, caller-supplied hints such as a known
            source URL, dataset id, or filename to check. This field is
            for narrowing a search within an authorized target — it is
            not an instruction to discover unknown sources on the open
            web.
        max_results: Upper bound on returned candidates.
    """

    reference_id: str
    face_embedding_ref: str
    query_hints: dict[str, str] = field(default_factory=dict)
    max_results: int = 5

    def __post_init__(self) -> None:
        if self.max_results <= 0:
            raise ValueError("max_results must be a positive integer")


@dataclass(frozen=True)
class SearchCandidate:
    """A single candidate result returned by a SearchProvider.

    Attributes:
        candidate_id: Stable identifier for this candidate within the run.
        source_reference: Where the candidate content came from (URL,
            dataset path, identifier). Kept off-chain downstream.
        content_type: The kind of content this candidate represents.
        preview_available: Whether a preview/thumbnail could be retrieved
            without a full content extraction pass.
        provider_confidence: Optional relevance score reported by the
            provider itself (distinct from the face-similarity score
            computed later in Candidate Matching). None if the provider
            does not produce its own confidence score.
    """

    candidate_id: str
    source_reference: str
    content_type: ContentType
    preview_available: bool
    provider_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.provider_confidence is not None and not (
            0.0 <= self.provider_confidence <= 1.0
        ):
            raise ValueError("provider_confidence must be between 0.0 and 1.0")


class SearchProvider(ABC):
    """Abstract, provider-agnostic search interface.

    Concrete subclasses are added in a later milestone once the
    authorized demo corpus/target has been decided. This class must
    remain abstract — it should never be instantiated directly, and
    this milestone does not ship any concrete subclass.
    """

    @abstractmethod
    def search(self, query: SearchQuery) -> list[SearchCandidate]:
        """Return candidates relevant to the given query.

        Implementations MUST perform genuine retrieval/ranking against
        their authorized target — never return hardcoded or fabricated
        candidates.
        """
        raise NotImplementedError
