import pytest

from app.blockchain.interface import BlockchainProvider, OnChainPayload
from app.search.interface import (
    ContentType,
    SearchCandidate,
    SearchProvider,
    SearchQuery,
)


def test_search_provider_is_abstract():
    with pytest.raises(TypeError):
        SearchProvider()  # type: ignore[abstract]


def test_blockchain_provider_is_abstract():
    with pytest.raises(TypeError):
        BlockchainProvider()  # type: ignore[abstract]


def test_search_query_constructs():
    query = SearchQuery(
        reference_id="run-1",
        face_embedding_ref="embed-ref-1",
        query_hints={"known_url": "https://example.org/authorized-post"},
        max_results=3,
    )
    assert query.max_results == 3
    assert query.query_hints["known_url"].startswith("https://")


def test_search_query_rejects_non_positive_max_results():
    with pytest.raises(ValueError):
        SearchQuery(reference_id="r", face_embedding_ref="e", max_results=0)


def test_search_candidate_constructs():
    candidate = SearchCandidate(
        candidate_id="c1",
        source_reference="https://example.org/authorized-post",
        content_type=ContentType.IMAGE,
        preview_available=True,
        provider_confidence=0.42,
    )
    assert candidate.content_type == ContentType.IMAGE


def test_search_candidate_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        SearchCandidate(
            candidate_id="c1",
            source_reference="ref",
            content_type=ContentType.TEXT,
            preview_available=False,
            provider_confidence=1.7,
        )


def test_on_chain_payload_constructs():
    payload = OnChainPayload(
        content_hash="a" * 64,
        algorithm="SHA-256",
        version="v1",
    )
    assert payload.algorithm == "SHA-256"


def test_search_interface_module_has_no_concrete_provider():
    """Guard against accidentally shipping a concrete SearchProvider in
    this milestone — only the abstract interface should be defined."""
    import app.search.interface as mod

    concrete_subclasses = [
        obj
        for name, obj in vars(mod).items()
        if isinstance(obj, type)
        and issubclass(obj, SearchProvider)
        and obj is not SearchProvider
    ]
    assert concrete_subclasses == []
