import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.content.authorized_corpus import AuthorizedCorpusContentExtractor
from app.search.authorized_corpus import (
    AuthorizedCorpusSearchProvider,
    CorpusValidationError,
)
from app.search.interface import SearchProvider, SearchQuery

CORPUS_PATH = Path(__file__).parents[2] / "examples" / "authorized_demo_corpus.json"


def _query(**hints: str) -> SearchQuery:
    return SearchQuery(
        reference_id="test-run",
        face_embedding_ref="opaque-embedding-handle",
        query_hints=hints,
        max_results=5,
    )


def test_provider_initializes_authorized_demo_corpus():
    provider = AuthorizedCorpusSearchProvider()

    assert isinstance(provider, SearchProvider)
    assert provider.corpus_path == CORPUS_PATH.resolve()
    assert len(provider.records) == 3
    assert all(record.image_path.is_file() for record in provider.records)


def test_authorized_content_extractor_returns_selected_record_content():
    provider = AuthorizedCorpusSearchProvider()
    record = provider.records[0]

    extracted = AuthorizedCorpusContentExtractor(provider.records).extract(
        record.candidate_id, record.source_reference
    )

    assert extracted.raw_bytes == record.image_path.read_bytes()
    assert extracted.text == record.text
    assert extracted.metadata == record.metadata
    assert extracted.source_reference == record.source_reference


def test_bundled_corpus_uses_opencv_decodable_raster_candidate_images():
    provider = AuthorizedCorpusSearchProvider()

    for record in provider.records:
        assert record.image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        image = cv2.imdecode(
            np.frombuffer(record.image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert image is not None, record.image_path


def test_successful_keyword_search_returns_content_candidate():
    provider = AuthorizedCorpusSearchProvider(CORPUS_PATH)

    results = provider.search(_query(keywords="harbor cleanup"))

    assert [candidate.candidate_id for candidate in results] == ["demo-post-harbor-01"]
    assert results[0].provider_confidence == 1.0
    assert results[0].source_reference.startswith("demo://authorized-social/")


def test_tag_search_returns_multiple_deterministically_ranked_candidates():
    provider = AuthorizedCorpusSearchProvider(CORPUS_PATH)

    results = provider.search(_query(tag="workshop"))

    assert [candidate.candidate_id for candidate in results] == [
        "demo-post-coast-02",
        "demo-post-harbor-01",
    ]


def test_search_returns_no_results_when_filter_does_not_match():
    provider = AuthorizedCorpusSearchProvider(CORPUS_PATH)

    assert provider.search(_query(keywords="unlisted astronomy")) == []


@pytest.mark.parametrize(
    "payload",
    [
        "not valid json",
        json.dumps({"schema_version": "v1", "records": [{}]}),
    ],
)
def test_provider_rejects_missing_or_malformed_corpus(tmp_path, payload):
    malformed = tmp_path / "corpus.json"
    malformed.write_text(payload, encoding="utf-8")

    with pytest.raises(CorpusValidationError):
        AuthorizedCorpusSearchProvider(malformed)


def test_provider_rejects_missing_corpus(tmp_path):
    with pytest.raises(CorpusValidationError):
        AuthorizedCorpusSearchProvider(tmp_path / "missing.json")


def test_search_results_are_deterministic():
    provider = AuthorizedCorpusSearchProvider(CORPUS_PATH)
    query = _query(dataset_id="synthetic-coastal-club-2026")

    assert provider.search(query) == provider.search(query)


def test_search_does_not_use_reference_embedding_handle_to_choose_candidate():
    provider = AuthorizedCorpusSearchProvider(CORPUS_PATH)
    first = SearchQuery(
        reference_id="run-a",
        face_embedding_ref="candidate-harbor-image",
        query_hints={"tag": "workshop"},
    )
    second = SearchQuery(
        reference_id="run-b",
        face_embedding_ref="candidate-garden-image",
        query_hints={"tag": "workshop"},
    )

    assert provider.search(first) == provider.search(second)


def test_search_provider_contract_returns_search_candidates_and_honors_limit():
    provider = AuthorizedCorpusSearchProvider(CORPUS_PATH)
    query = SearchQuery(
        reference_id="contract-run",
        face_embedding_ref="opaque-handle",
        query_hints={"dataset_id": "synthetic-coastal-club-2026"},
        max_results=1,
    )

    results = provider.search(query)

    assert len(results) == 1
    assert results[0].candidate_id == "demo-post-coast-02"
    assert results[0].provider_confidence == 1.0
