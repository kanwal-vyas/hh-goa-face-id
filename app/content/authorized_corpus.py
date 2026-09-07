"""Content extraction for records in the bounded authorized corpus."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from app.content.extractor import (
    ContentExtractionError,
    ContentExtractor,
    DiscoveredContent,
)
from app.search.authorized_corpus import AuthorizedCorpusRecord


class AuthorizedCorpusContentExtractor(ContentExtractor):
    """Extract the exact local content for an authorized corpus record.

    This implementation is deliberately limited to already-validated records
    supplied by :class:`AuthorizedCorpusSearchProvider`. It performs no
    network retrieval and cannot discover content outside that corpus.
    """

    def __init__(self, records: Iterable[AuthorizedCorpusRecord]) -> None:
        self._records = {record.candidate_id: record for record in records}

    def extract(self, candidate_id: str, source_reference: str) -> DiscoveredContent:
        record = self._records.get(candidate_id)
        if record is None:
            raise ContentExtractionError(
                f"Candidate {candidate_id!r} is not in the authorized corpus."
            )
        if record.source_reference != source_reference:
            raise ContentExtractionError(
                "Candidate source reference does not match the authorized corpus record."
            )

        try:
            raw_bytes = record.image_path.read_bytes()
        except OSError as exc:
            raise ContentExtractionError(
                f"Could not read authorized candidate content: {exc}"
            ) from exc

        return DiscoveredContent(
            candidate_id=record.candidate_id,
            raw_bytes=raw_bytes,
            text=record.text,
            metadata=dict(record.metadata),
            source_reference=record.source_reference,
            retrieved_at=datetime.now(timezone.utc),
        )
