"""
Content extraction scaffold.

Defines the data model and interface for pulling the exact
bytes/text/metadata that will later be canonicalized and fingerprinted
from a selected search candidate. No concrete fetch/extraction logic is
implemented at this milestone.

The eventual implementation will only operate on candidates returned by
an authorized SearchProvider (see app/search/interface.py) — it does not
independently discover or crawl sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


class ContentExtractionError(Exception):
    """Raised when a candidate's content cannot be retrieved/extracted."""


@dataclass(frozen=True)
class DiscoveredContent:
    """Raw content and metadata extracted from a selected candidate.

    Attributes:
        candidate_id: Identifies which SearchCandidate this content was
            extracted from.
        raw_bytes: Raw binary content (e.g. an image), if applicable.
        text: Text content (e.g. a caption or post body), if applicable.
        metadata: Source-provided metadata (e.g. caption, source
            timestamp). Kept as a plain dict of primitive values so it
            can be canonicalized deterministically downstream.
        source_reference: Where this content came from (URL, dataset
            path, identifier).
        retrieved_at: When extraction occurred. NOTE: this field is
            explicitly excluded from canonicalization (see
            canonicalize.py) because it is not deterministic across
            re-fetches of unchanged content.
    """

    candidate_id: str
    raw_bytes: bytes | None
    text: str | None
    metadata: dict[str, str] = field(default_factory=dict)
    source_reference: str = ""
    retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.raw_bytes is None and self.text is None:
            raise ValueError(
                "DiscoveredContent must have at least one of raw_bytes or text"
            )


class ContentExtractor(ABC):
    """Abstract interface for extracting DiscoveredContent from a
    selected candidate.

    Concrete implementations are added in a later milestone.
    """

    @abstractmethod
    def extract(self, candidate_id: str, source_reference: str) -> DiscoveredContent:
        """Retrieve and package the content for a given candidate.

        Raises:
            ContentExtractionError: if the content cannot be retrieved
                (e.g. removed, inaccessible, malformed).
        """
        raise NotImplementedError
