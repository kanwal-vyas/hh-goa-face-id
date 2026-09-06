"""Bounded, local ``SearchProvider`` for an authorized demo corpus.

This module deliberately has no network client and does not inspect face
embeddings or reference images. It performs deterministic text, tag, dataset,
and source-reference retrieval over the JSON corpus supplied by the operator.
Face processing and similarity banding remain downstream responsibilities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.search.interface import (
    ContentType,
    SearchCandidate,
    SearchProvider,
    SearchQuery,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUPPORTED_HINTS = frozenset({"dataset_id", "source_reference", "tag", "keywords"})
DEFAULT_DEMO_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "authorized_demo_corpus.json"
)


class CorpusValidationError(ValueError):
    """Raised when an authorized corpus is missing or does not meet schema v1."""


@dataclass(frozen=True)
class AuthorizedCorpusRecord:
    """Validated local content record loaded from an authorized corpus.

    ``image_path`` is retained on the local record for the later content
    extraction/face-processing stages. Search itself never opens or compares
    the image bytes.
    """

    candidate_id: str
    source_reference: str
    text: str
    metadata: dict[str, str]
    image_path: Path
    tags: tuple[str, ...]

    @property
    def content_type(self) -> ContentType:
        return ContentType.MIXED if self.text else ContentType.IMAGE

    @property
    def preview_available(self) -> bool:
        return self.image_path.is_file()

    @property
    def searchable_text(self) -> str:
        return " ".join(
            (
                self.candidate_id,
                self.source_reference,
                self.text,
                *self.metadata.values(),
                *self.tags,
            )
        ).lower()


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _required_string(record: dict[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CorpusValidationError(f"Record {index} requires a non-empty {field!r} string")
    return value


def _load_record(raw: Any, index: int, corpus_directory: Path) -> AuthorizedCorpusRecord:
    if not isinstance(raw, dict):
        raise CorpusValidationError(f"Record {index} must be an object")

    candidate_id = _required_string(raw, "candidate_id", index)
    source_reference = _required_string(raw, "source_reference", index)
    text = _required_string(raw, "text", index)
    image_path_value = _required_string(raw, "image_path", index)
    if "://" not in source_reference:
        raise CorpusValidationError(
            f"Record {index} source_reference must be a URL-like reference"
        )

    metadata = raw.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise CorpusValidationError(f"Record {index} requires a non-empty metadata object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise CorpusValidationError(f"Record {index} metadata values must be strings")

    raw_tags = raw.get("tags", [])
    if not isinstance(raw_tags, list) or not all(
        isinstance(tag, str) and tag.strip() for tag in raw_tags
    ):
        raise CorpusValidationError(f"Record {index} tags must be a list of non-empty strings")

    image_path = (corpus_directory / image_path_value).resolve()
    if not image_path.is_file():
        raise CorpusValidationError(
            f"Record {index} image_path does not exist: {image_path_value!r}"
        )

    return AuthorizedCorpusRecord(
        candidate_id=candidate_id,
        source_reference=source_reference,
        text=text,
        metadata=dict(metadata),
        image_path=image_path,
        tags=tuple(raw_tags),
    )


class AuthorizedCorpusSearchProvider(SearchProvider):
    """Search only a local, operator-supplied authorized JSON corpus.

    ``dataset_id``, ``source_reference``, ``tag``, and ``keywords`` are the
    supported query hints. Unknown hints are ignored. ``face_embedding_ref``
    is intentionally not read: this provider has no biometric lookup index and
    never maps a supplied reference image or embedding to a candidate.
    """

    def __init__(self, corpus_path: str | Path | None = None) -> None:
        """Load ``corpus_path`` or the bundled synthetic demo corpus."""
        selected_path = DEFAULT_DEMO_CORPUS_PATH if corpus_path is None else corpus_path
        self.corpus_path = Path(selected_path).resolve()
        self._records = self._load_corpus(self.corpus_path)

    @property
    def records(self) -> tuple[AuthorizedCorpusRecord, ...]:
        """The validated local records, useful to a later extraction stage."""
        return self._records

    @property
    def candidate_image_paths(self) -> dict[str, Path]:
        """Local image paths for records this provider is authorized to return.

        A copy is returned so callers can pass it to a matcher without
        mutating the provider's validated corpus state.
        """
        return {record.candidate_id: record.image_path for record in self._records}

    @staticmethod
    def _load_corpus(corpus_path: Path) -> tuple[AuthorizedCorpusRecord, ...]:
        if not corpus_path.is_file():
            raise CorpusValidationError(f"Authorized corpus not found: {corpus_path}")
        try:
            payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorpusValidationError(f"Authorized corpus is not valid JSON: {corpus_path}") from exc
        except OSError as exc:
            raise CorpusValidationError(f"Unable to read authorized corpus: {corpus_path}") from exc

        if not isinstance(payload, dict) or payload.get("schema_version") != "v1":
            raise CorpusValidationError("Authorized corpus must declare schema_version 'v1'")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise CorpusValidationError("Authorized corpus must contain a non-empty records list")

        records = tuple(
            _load_record(raw, index, corpus_path.parent)
            for index, raw in enumerate(raw_records, start=1)
        )
        candidate_ids = [record.candidate_id for record in records]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise CorpusValidationError("Authorized corpus candidate_id values must be unique")
        return records

    def search(self, query: SearchQuery) -> list[SearchCandidate]:
        """Retrieve and rank corpus candidates without reading biometric input.

        Text matches are scored as query-token coverage. Dataset, source, and
        tag filters contribute a score of one when they match. Results are
        sorted by score then candidate id, making repeated queries deterministic.
        """
        hints = {
            key: value.strip()
            for key, value in query.query_hints.items()
            if key in _SUPPORTED_HINTS and isinstance(value, str) and value.strip()
        }
        keyword_tokens = _tokens(hints.get("keywords", ""))
        requested_tags = _tokens(hints.get("tag", ""))
        ranked: list[tuple[float | None, AuthorizedCorpusRecord]] = []

        for record in self._records:
            if hints.get("dataset_id") and record.metadata.get("dataset_id") != hints["dataset_id"]:
                continue
            if hints.get("source_reference") and hints["source_reference"].lower() not in record.source_reference.lower():
                continue
            record_tags = _tokens(" ".join(record.tags))
            if requested_tags and not requested_tags.issubset(record_tags):
                continue

            score_parts: list[float] = []
            if keyword_tokens:
                overlap = len(keyword_tokens.intersection(_tokens(record.searchable_text)))
                if overlap == 0:
                    continue
                score_parts.append(overlap / len(keyword_tokens))
            if hints.get("dataset_id"):
                score_parts.append(1.0)
            if hints.get("source_reference"):
                score_parts.append(1.0)
            if requested_tags:
                score_parts.append(1.0)
            ranked.append((sum(score_parts) / len(score_parts) if score_parts else None, record))

        ranked.sort(key=lambda item: (-(item[0] or 0.0), item[1].candidate_id))
        return [
            SearchCandidate(
                candidate_id=record.candidate_id,
                source_reference=record.source_reference,
                content_type=record.content_type,
                preview_available=record.preview_available,
                provider_confidence=score,
            )
            for score, record in ranked[: query.max_results]
        ]
