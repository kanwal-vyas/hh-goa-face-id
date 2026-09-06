"""
Cryptographic fingerprinting.

Turns a CanonicalContent's deterministic bytes into a SHA-256
ContentFingerprint. This module is intentionally tiny: hashing itself
needs no cleverness once the input bytes are already deterministic
(that determinism work lives in app.content.canonicalize).

IMPORTANT: `generated_at` is metadata ABOUT the fingerprinting event
(when it happened), not part of what gets hashed. Only
`canonical_content.canonical_bytes` is fed into hashlib.sha256(). This
is why hashing the same content at two different times, or on two
different machines, still produces the identical `hash` value -- only
`generated_at` differs between the two ContentFingerprint objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from app.content.canonicalize import CanonicalContent

ALGORITHM = "SHA-256"


@dataclass(frozen=True)
class ContentFingerprint:
    """A SHA-256 fingerprint of some CanonicalContent.

    Attributes:
        hash: Lowercase hex-encoded SHA-256 digest (64 characters).
        algorithm: Always "SHA-256" in this milestone.
        canonicalization_version: Which canonicalization ruleset
            produced the bytes this hash was computed over (see
            app.content.canonicalize.CANONICALIZATION_VERSION).
        generated_at: When this fingerprint was computed. NOT part of
            the hashed bytes -- see module docstring.
    """

    hash: str
    algorithm: str
    canonicalization_version: str
    generated_at: datetime

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM:
            raise ValueError(f"algorithm must be {ALGORITHM!r}, got {self.algorithm!r}")
        if len(self.hash) != 64:
            raise ValueError(
                f"hash must be exactly 64 hex characters, got {len(self.hash)}"
            )
        try:
            int(self.hash, 16)
        except ValueError as exc:
            raise ValueError(f"hash must be valid hexadecimal, got {self.hash!r}") from exc


def hash_source_reference(source_reference: str) -> str:
    """SHA-256 hash of a source reference (e.g. a URL), independent of
    content hashing.

    `content_hash` and `source_reference_hash` serve different purposes
    and must never be conflated: `content_hash` identifies WHAT the
    content is; `source_reference_hash` identifies WHERE it was found.
    This is a plain SHA-256 over the UTF-8 encoded string -- no
    canonicalization envelope, since a source reference is already a
    single, unambiguous string with no sub-fields to separate.
    """
    return hashlib.sha256(source_reference.encode("utf-8")).hexdigest()


def hash_canonical_content(
    canonical_content: CanonicalContent,
    generated_at: datetime | None = None,
) -> ContentFingerprint:
    """Compute a SHA-256 ContentFingerprint from CanonicalContent.

    Pure with respect to the hash value: given the same
    canonical_content.canonical_bytes, the returned `hash` is always
    identical regardless of `generated_at`.

    Args:
        canonical_content: Output of a Canonicalizer.
        generated_at: Optional explicit timestamp (defaults to now,
            UTC). Exposed as a parameter mainly so tests can construct
            two ContentFingerprints with different timestamps and
            assert their `hash` values are still equal.
    """
    digest = hashlib.sha256(canonical_content.canonical_bytes).hexdigest()
    return ContentFingerprint(
        hash=digest,
        algorithm=ALGORITHM,
        canonicalization_version=canonical_content.canonicalization_version,
        generated_at=generated_at if generated_at is not None else datetime.now(timezone.utc),
    )
