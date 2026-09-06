from datetime import datetime, timezone

import pytest

from app.blockchain.interface import BlockchainProvider
from app.content.canonicalize import CANONICALIZATION_VERSION, CanonicalContent
from app.content.extractor import DiscoveredContent
from app.matching.candidate_matcher import MatchResult
from app.search.interface import ContentType, SearchCandidate
from app.verification.verifier import VerificationResult, VerificationStatus


def test_discovered_content_requires_bytes_or_text():
    with pytest.raises(ValueError):
        DiscoveredContent(candidate_id="c1", raw_bytes=None, text=None)


def test_discovered_content_constructs_with_text():
    content = DiscoveredContent(candidate_id="c1", raw_bytes=None, text="hello")
    assert content.text == "hello"


def test_canonical_content_rejects_empty_bytes():
    with pytest.raises(ValueError):
        CanonicalContent(canonical_bytes=b"", canonicalization_version="v1")


def test_canonical_content_constructs():
    cc = CanonicalContent(
        canonical_bytes=b"deterministic-bytes",
        canonicalization_version=CANONICALIZATION_VERSION,
        included_fields=("text",),
        excluded_fields=("retrieved_at",),
    )
    assert cc.canonicalization_version == "v1"


def test_match_result_constructs_without_fabricated_scores():
    candidate = SearchCandidate(
        candidate_id="c1",
        source_reference="ref",
        content_type=ContentType.IMAGE,
        preview_available=True,
    )
    result = MatchResult(
        candidate=candidate,
        face_similarity=None,
        content_relevance=None,
        is_selected=False,
    )
    assert result.face_similarity is None
    assert result.content_relevance is None


def test_verification_result_status_must_match_hash_comparison():
    now = datetime.now(timezone.utc)
    # Consistent case: equal hashes -> VERIFIED
    result = VerificationResult(
        local_hash="abc",
        on_chain_hash="abc",
        status=VerificationStatus.VERIFIED,
        tx_reference="0x123",
        checked_at=now,
    )
    assert result.match is True

    # Inconsistent case must raise
    with pytest.raises(ValueError):
        VerificationResult(
            local_hash="abc",
            on_chain_hash="def",
            status=VerificationStatus.VERIFIED,  # wrong on purpose
            tx_reference="0x123",
            checked_at=now,
        )


def test_verification_result_tamper_detected_case():
    now = datetime.now(timezone.utc)
    result = VerificationResult(
        local_hash="abc",
        on_chain_hash="xyz",
        status=VerificationStatus.TAMPER_DETECTED,
        tx_reference="0x123",
        checked_at=now,
    )
    assert result.match is False


def test_blockchain_provider_abstract_methods_present():
    assert hasattr(BlockchainProvider, "register")
    assert hasattr(BlockchainProvider, "retrieve")
