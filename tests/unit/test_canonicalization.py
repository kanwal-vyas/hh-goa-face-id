"""
Tests for app.content.canonicalize and app.content.fingerprint.

Covers the required cases A-L from the Milestone 3 spec plus an
explicit tamper-detection scenario (the basis for Milestone 5's
verification demo).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.content.canonicalize import DeterministicCanonicalizer
from app.content.extractor import DiscoveredContent
from app.content.fingerprint import (
    ContentFingerprint,
    hash_canonical_content,
    hash_source_reference,
)

CANONICALIZER = DeterministicCanonicalizer()


def _content(**overrides) -> DiscoveredContent:
    defaults = {
        "candidate_id": "c1",
        "raw_bytes": None,
        "text": "hello world",
        "metadata": {"caption": "a caption"},
        "source_reference": "https://example.org/authorized-post",
        "retrieved_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return DiscoveredContent(**defaults)


# --- A. Determinism ---


def test_same_content_processed_twice_yields_identical_bytes_and_hash():
    content = _content()
    canon1 = CANONICALIZER.canonicalize(content)
    canon2 = CANONICALIZER.canonicalize(content)
    assert canon1.canonical_bytes == canon2.canonical_bytes

    fp1 = hash_canonical_content(canon1)
    fp2 = hash_canonical_content(canon2)
    assert fp1.hash == fp2.hash


# --- B. Retrieval timestamp independence ---


def test_different_retrieved_at_yields_identical_hash():
    content_a = _content(retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    content_b = _content(retrieved_at=datetime(2030, 6, 15, tzinfo=timezone.utc))

    canon_a = CANONICALIZER.canonicalize(content_a)
    canon_b = CANONICALIZER.canonicalize(content_b)

    assert canon_a.canonical_bytes == canon_b.canonical_bytes
    assert hash_canonical_content(canon_a).hash == hash_canonical_content(canon_b).hash


# --- C. Metadata ordering ---


def test_metadata_key_order_does_not_affect_hash():
    content_a = _content(metadata={"a": "1", "b": "2", "c": "3"})
    content_b = _content(metadata={"c": "3", "a": "1", "b": "2"})

    canon_a = CANONICALIZER.canonicalize(content_a)
    canon_b = CANONICALIZER.canonicalize(content_b)

    assert canon_a.canonical_bytes == canon_b.canonical_bytes
    assert hash_canonical_content(canon_a).hash == hash_canonical_content(canon_b).hash


# --- D. Meaningful content modification ---


def test_one_character_text_change_changes_hash():
    content_a = _content(text="hello world")
    content_b = _content(text="hello worle")  # one character different

    hash_a = hash_canonical_content(CANONICALIZER.canonicalize(content_a)).hash
    hash_b = hash_canonical_content(CANONICALIZER.canonicalize(content_b)).hash
    assert hash_a != hash_b


# --- E. Binary modification ---


def test_one_byte_binary_change_changes_hash():
    content_a = _content(text=None, raw_bytes=b"\x00\x01\x02\x03")
    content_b = _content(text=None, raw_bytes=b"\x00\x01\x02\x04")

    hash_a = hash_canonical_content(CANONICALIZER.canonicalize(content_a)).hash
    hash_b = hash_canonical_content(CANONICALIZER.canonicalize(content_b)).hash
    assert hash_a != hash_b


# --- F. Unicode ---


def test_unicode_text_canonicalizes_deterministically():
    content = _content(text="héllo wörld — 世界 🌍")
    canon1 = CANONICALIZER.canonicalize(content)
    canon2 = CANONICALIZER.canonicalize(content)
    assert canon1.canonical_bytes == canon2.canonical_bytes
    assert (
        hash_canonical_content(canon1).hash == hash_canonical_content(canon2).hash
    )


def test_unicode_nfc_normalization_unifies_equivalent_forms():
    # "é" as a single precomposed codepoint vs "e" + combining acute accent
    precomposed = "café"
    decomposed = "cafe\u0301"
    assert precomposed != decomposed  # different raw strings…

    content_a = _content(text=precomposed)
    content_b = _content(text=decomposed)
    hash_a = hash_canonical_content(CANONICALIZER.canonicalize(content_a)).hash
    hash_b = hash_canonical_content(CANONICALIZER.canonicalize(content_b)).hash
    # …but NFC normalization means they hash identically.
    assert hash_a == hash_b


def test_newline_styles_are_normalized():
    content_a = _content(text="line1\r\nline2")
    content_b = _content(text="line1\nline2")
    hash_a = hash_canonical_content(CANONICALIZER.canonicalize(content_a)).hash
    hash_b = hash_canonical_content(CANONICALIZER.canonicalize(content_b)).hash
    assert hash_a == hash_b


# --- G. Empty content ---


def test_empty_text_is_valid_and_distinguishable_from_missing():
    empty_text_content = _content(text="", raw_bytes=None)
    canon = CANONICALIZER.canonicalize(empty_text_content)
    fp = hash_canonical_content(canon)
    assert len(fp.hash) == 64  # produced a real hash, didn't crash/skip


def test_empty_bytes_is_valid_and_distinguishable_from_missing():
    empty_bytes_content = _content(text=None, raw_bytes=b"")
    canon = CANONICALIZER.canonicalize(empty_bytes_content)
    fp = hash_canonical_content(canon)
    assert len(fp.hash) == 64


def test_empty_text_and_empty_bytes_hash_differently():
    empty_text = _content(text="", raw_bytes=None)
    empty_bytes = _content(text=None, raw_bytes=b"")
    hash_a = hash_canonical_content(CANONICALIZER.canonicalize(empty_text)).hash
    hash_b = hash_canonical_content(CANONICALIZER.canonicalize(empty_bytes)).hash
    assert hash_a != hash_b


def test_completely_missing_content_raises_not_silently_hashed():
    """DiscoveredContent itself refuses to be constructed with neither
    raw_bytes nor text -- this is intentional defense against silently
    hashing None. Verified here at the canonicalization boundary."""
    with pytest.raises(ValueError):
        DiscoveredContent(
            candidate_id="c1", raw_bytes=None, text=None, source_reference="ref"
        )


# --- H. Version ---


def test_canonical_content_and_fingerprint_record_version():
    canon = CANONICALIZER.canonicalize(_content())
    assert canon.canonicalization_version == "v1"
    fp = hash_canonical_content(canon)
    assert fp.canonicalization_version == "v1"


def test_canonical_content_documents_included_and_excluded_fields():
    canon = CANONICALIZER.canonicalize(_content())
    assert "retrieved_at" in canon.excluded_fields
    assert "text" in canon.included_fields
    assert "metadata" in canon.included_fields


# --- I. Algorithm ---


def test_fingerprint_algorithm_is_sha256():
    fp = hash_canonical_content(CANONICALIZER.canonicalize(_content()))
    assert fp.algorithm == "SHA-256"


def test_fingerprint_rejects_wrong_algorithm_label():
    with pytest.raises(ValueError):
        ContentFingerprint(
            hash="a" * 64,
            algorithm="MD5",
            canonicalization_version="v1",
            generated_at=datetime.now(timezone.utc),
        )


# --- J. Generated timestamp ---


def test_generated_at_does_not_affect_hash():
    canon = CANONICALIZER.canonicalize(_content())
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = t1 + timedelta(days=365)

    fp1 = hash_canonical_content(canon, generated_at=t1)
    fp2 = hash_canonical_content(canon, generated_at=t2)

    assert fp1.hash == fp2.hash
    assert fp1.generated_at != fp2.generated_at


# --- K. Hash format ---


def test_hash_is_64_lowercase_hex_characters():
    fp = hash_canonical_content(CANONICALIZER.canonicalize(_content()))
    assert len(fp.hash) == 64
    assert all(c in "0123456789abcdef" for c in fp.hash)


def test_fingerprint_rejects_malformed_hash():
    with pytest.raises(ValueError):
        ContentFingerprint(
            hash="not-hex-and-wrong-length",
            algorithm="SHA-256",
            canonicalization_version="v1",
            generated_at=datetime.now(timezone.utc),
        )


# --- L. No network dependency ---


def test_canonicalization_and_hashing_work_offline(monkeypatch):
    import socket

    def _blocked_connect(*args, **kwargs):
        raise AssertionError("Unexpected network connection during canonicalization/hashing")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    canon = CANONICALIZER.canonicalize(_content())
    hash_canonical_content(canon)


# --- Corrected fingerprint semantics (Part A/B): source/candidate independence ---


def test_different_source_reference_yields_identical_hash():
    content_a = _content(source_reference="https://example.org/post-x")
    content_b = _content(source_reference="https://example.org/post-y")

    canon_a = CANONICALIZER.canonicalize(content_a)
    canon_b = CANONICALIZER.canonicalize(content_b)

    assert canon_a.canonical_bytes == canon_b.canonical_bytes
    assert hash_canonical_content(canon_a).hash == hash_canonical_content(canon_b).hash


def test_different_candidate_id_yields_identical_hash():
    content_a = _content(candidate_id="candidate-1")
    content_b = _content(candidate_id="candidate-2")

    canon_a = CANONICALIZER.canonicalize(content_a)
    canon_b = CANONICALIZER.canonicalize(content_b)

    assert canon_a.canonical_bytes == canon_b.canonical_bytes
    assert hash_canonical_content(canon_a).hash == hash_canonical_content(canon_b).hash


def test_different_source_and_candidate_but_same_content_yields_identical_hash():
    """The example from the corrected spec: the exact same content
    found via a different candidate at a different source must hash
    identically."""
    content_a = _content(
        candidate_id="candidate-x",
        source_reference="https://example.org/post-x",
        text="identical content",
    )
    content_b = _content(
        candidate_id="candidate-y",
        source_reference="https://example.org/post-y",
        text="identical content",
    )
    hash_a = hash_canonical_content(CANONICALIZER.canonicalize(content_a)).hash
    hash_b = hash_canonical_content(CANONICALIZER.canonicalize(content_b)).hash
    assert hash_a == hash_b


def test_canonical_payload_does_not_contain_raw_source_reference_or_candidate_id():
    sentinel_source = "https://example.org/UNIQUE-SENTINEL-SOURCE-REF-3f9a2b"
    sentinel_candidate = "UNIQUE-SENTINEL-CANDIDATE-ID-7c1d"
    content = _content(
        source_reference=sentinel_source, candidate_id=sentinel_candidate
    )
    canon = CANONICALIZER.canonicalize(content)
    assert sentinel_source.encode("utf-8") not in canon.canonical_bytes
    assert sentinel_candidate.encode("utf-8") not in canon.canonical_bytes


def test_excluded_fields_lists_candidate_id_and_source_reference():
    canon = CANONICALIZER.canonicalize(_content())
    assert "candidate_id" in canon.excluded_fields
    assert "source_reference" in canon.excluded_fields
    assert "candidate_id" not in canon.included_fields
    assert "source_reference" not in canon.included_fields


def test_restoring_content_restores_original_hash():
    original = _content(text="original content", source_reference="ref-a")
    hash_original = hash_canonical_content(
        CANONICALIZER.canonicalize(original)
    ).hash

    modified = _content(text="tampered content", source_reference="ref-a")
    hash_modified = hash_canonical_content(
        CANONICALIZER.canonicalize(modified)
    ).hash
    assert hash_modified != hash_original

    restored = _content(text="original content", source_reference="ref-b")
    hash_restored = hash_canonical_content(
        CANONICALIZER.canonicalize(restored)
    ).hash
    assert hash_restored == hash_original


# --- source_reference hashing (separate from content hashing) ---


def test_hash_source_reference_is_deterministic_and_64_hex_chars():
    h1 = hash_source_reference("https://example.org/authorized-post")
    h2 = hash_source_reference("https://example.org/authorized-post")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_hash_source_reference_differs_for_different_references():
    h1 = hash_source_reference("https://example.org/a")
    h2 = hash_source_reference("https://example.org/b")
    assert h1 != h2


def test_content_hash_and_source_reference_hash_are_independent():
    content = _content(
        text="same content", source_reference="https://example.org/a"
    )
    content_hash = hash_canonical_content(CANONICALIZER.canonicalize(content)).hash
    source_hash = hash_source_reference(content.source_reference)
    assert content_hash != source_hash


# --- Tamper-detection scenario (basis for Milestone 5) ---


def test_tamper_scenario_original_vs_modified_content():
    original = _content(text="The bridge is safe to cross.")
    modified = _content(text="The bridge is NOT safe to cross.")

    hash_original = hash_canonical_content(CANONICALIZER.canonicalize(original)).hash
    hash_modified = hash_canonical_content(CANONICALIZER.canonicalize(modified)).hash

    assert hash_original != hash_modified


def test_tamper_scenario_reconstructing_original_reproduces_same_hash():
    """Simulates a future verification run: re-canonicalizing the
    ORIGINAL (unmodified) content later must reproduce the same hash
    that was recorded earlier."""
    original = _content(text="The bridge is safe to cross.")

    hash_at_registration_time = hash_canonical_content(
        CANONICALIZER.canonicalize(original)
    ).hash

    # Simulate "later" by rebuilding an equivalent DiscoveredContent with
    # a different candidate object identity and a different retrieved_at.
    reconstructed = _content(
        text="The bridge is safe to cross.",
        retrieved_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    hash_at_verification_time = hash_canonical_content(
        CANONICALIZER.canonicalize(reconstructed)
    ).hash

    assert hash_at_registration_time == hash_at_verification_time
