"""
Canonicalization.

Canonicalization turns a DiscoveredContent object into a deterministic
byte sequence suitable for hashing. This is the piece that makes the
eventual blockchain verification meaningful: the SAME logical content
must always canonicalize to the SAME bytes, regardless of incidental
variation (fetch timestamp, key ordering, whitespace differences that
are purely platform artifacts, WHERE the content was found, or which
retrieval attempt/candidate object produced it).

WHAT THE CONTENT FINGERPRINT REPRESENTS (v2 semantics)
The fingerprint represents the discovered CONTENT itself -- not its
location and not the retrieval event that found it. Concretely, this
means the exact same content found at two different source references
(or retrieved as two different candidates, or at two different times)
MUST hash identically. Location/provenance is tracked separately (see
"Source reference" below) and is never mixed into the content hash.

    [CORRECTED] An earlier version of this canonicalizer included
    `source_reference` and `candidate_id` in the hashed envelope, which
    made the fingerprint source-dependent (identical content at two
    URLs hashed differently). That was a genuine semantic error for a
    content-integrity fingerprint and has been corrected here: the
    envelope no longer includes `source_reference` or `candidate_id`.

WHY NOT JUST CONCATENATE FIELDS?
Naively hashing something like `text + metadata_str` is ambiguous: two
different (text, metadata) pairs can produce the same concatenated
string (e.g. text="ab", metadata="c" vs text="a", metadata="bc"). This
implementation avoids that by serializing all fields into a single,
explicitly-keyed JSON structure — each field has its own named slot, so
there is no way for content to "leak" across a field boundary.

CANONICALIZATION_VERSION "v1" RULES (implemented below by
DeterministicCanonicalizer):

Included in the hashed envelope:
    - canonicalization_version ("v1")
    - text                     (Unicode NFC-normalized, newlines
                                  normalized to "\n"; otherwise
                                  UNCHANGED -- no stripping, no
                                  lowercasing, no punctuation removal.
                                  `None` if no text content.)
    - metadata                 (the dict as given; key ORDER does not
                                  matter because serialization sorts
                                  keys -- see "Deterministic
                                  serialization" below)
    - raw_content_base64        (base64 encoding of raw_bytes, so binary
                                  content sits in its own explicit JSON
                                  string field rather than being
                                  concatenated with anything else.
                                  `None` if no binary content. Note:
                                  base64 of `b""` is `""`, which is
                                  distinguishable from `None` -- so
                                  "empty bytes" and "no bytes" are not
                                  confused with each other.)

Explicitly EXCLUDED (never affect the hash):
    - candidate_id      -- identifies a retrieval candidate, not content.
    - source_reference  -- WHERE content was found, not what it is. See
                            "Source reference" below for how this is
                            tracked separately instead.
    - retrieved_at      -- WHEN extraction happened, not what was
                            extracted.
    - anything else not listed above (e.g. any future transient/
      logging/runtime field added to DiscoveredContent must be
      excluded here explicitly, not included by default).

METADATA -- A DOCUMENTED, CONSERVATIVE v1 LIMITATION:
`DiscoveredContent.metadata` (`dict[str, str]`) has no schema
distinguishing content-intrinsic fields (e.g. a post's caption) from
volatile retrieval fields (e.g. a request ID or provider-internal
tracking value). This milestone does NOT attempt to guess which is
which -- the entire dict is included in the hash, as-is. This is a
conservative v1 rule, documented here explicitly:

    Callers are responsible for only putting genuinely content-
    intrinsic fields into DiscoveredContent.metadata. Do not put
    request IDs, local file paths, retrieval timestamps, or other
    per-fetch/runtime values in `metadata` -- put them in
    `source_reference`/`retrieved_at` (both excluded from the hash) or
    keep them out of DiscoveredContent entirely. A future
    canonicalization version could introduce an explicit
    content-metadata vs. retrieval-metadata split in the data model
    itself; that is out of scope here.

Source reference:
    `source_reference` is preserved on `DiscoveredContent` and remains
    available off-chain / in logs, but it is not part of the content
    fingerprint. If a source needs to be committed alongside a
    fingerprint later (e.g. in a BlockchainRecord), hash it SEPARATELY
    with `hash_source_reference()` (app/content/fingerprint.py) --
    `content_hash` and `source_reference_hash` are two different
    values serving two different purposes and must never be conflated.

Deterministic serialization:
    The envelope dict is serialized with
    `json.dumps(envelope, sort_keys=True, ensure_ascii=False,
    separators=(",", ":"))` then encoded as UTF-8.
    - `sort_keys=True` makes key insertion order irrelevant, including
      for nested dicts (Python's json module sorts at every level).
    - `separators=(",", ":")` removes whitespace, so pretty-printing
      settings can never change the byte output.
    - `ensure_ascii=False` + explicit `.encode("utf-8")` ensures a
      single, explicit, platform-independent text encoding rather than
      relying on `str.encode()`'s implicit default.
    This is a real, executable serialization (not `repr()`, not
    Python's unordered default dict iteration).

The same logical content, found at a different source, retrieved as a
different candidate, processed at a different time, on a different
machine, with metadata built up in a different key order, MUST produce
byte-for-byte identical canonical_bytes and therefore an identical
SHA-256 hash. See tests/unit/test_canonicalization.py for the tests that
verify this directly.
"""

from __future__ import annotations

import base64
import json
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.content.extractor import DiscoveredContent

CANONICALIZATION_VERSION = "v1"

# Fields from DiscoveredContent that ARE included in canonical_bytes.
INCLUDED_FIELDS: tuple[str, ...] = (
    "text",
    "metadata",
    "raw_bytes",
)

# Fields from DiscoveredContent that are DELIBERATELY excluded because
# they identify location/retrieval, not content, or are otherwise
# volatile.
EXCLUDED_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "source_reference",
    "retrieved_at",
)


@dataclass(frozen=True)
class CanonicalContent:
    """Deterministic byte representation of a DiscoveredContent, ready
    for hashing.

    Attributes:
        canonical_bytes: The exact bytes that will be hashed.
        canonicalization_version: Which canonicalization ruleset produced
            these bytes (see CANONICALIZATION_VERSION). Verification runs
            must use the same version to reproduce a matching hash.
        included_fields: Names of fields from the source DiscoveredContent
            that were included in canonical_bytes. Documented explicitly
            so it's clear what is and isn't covered by the fingerprint.
        excluded_fields: Names of fields that were deliberately excluded
            (e.g. retrieved_at) and why.
    """

    canonical_bytes: bytes
    canonicalization_version: str
    included_fields: tuple[str, ...] = field(default_factory=tuple)
    excluded_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.canonical_bytes:
            raise ValueError("canonical_bytes must not be empty")


class Canonicalizer(ABC):
    """Abstract interface for turning DiscoveredContent into
    CanonicalContent.

    Any implementation MUST be deterministic: canonicalizing the same
    logical content twice must always produce identical
    canonical_bytes.
    """

    @abstractmethod
    def canonicalize(self, content: DiscoveredContent) -> CanonicalContent:
        raise NotImplementedError


def _normalize_text(text: str) -> str:
    """Apply the v1 text normalization rule: newline normalization plus
    Unicode NFC normalization. Deliberately does NOT strip whitespace,
    lowercase, or remove punctuation -- the goal is content integrity,
    not semantic similarity, so a meaningful single-character change
    must still change the resulting hash.
    """
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines)


def _build_envelope(content: DiscoveredContent) -> dict:
    text = _normalize_text(content.text) if content.text is not None else None
    raw_content_base64 = (
        base64.b64encode(content.raw_bytes).decode("ascii")
        if content.raw_bytes is not None
        else None
    )
    return {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "text": text,
        "metadata": dict(content.metadata),
        "raw_content_base64": raw_content_base64,
    }


def _serialize_envelope(envelope: dict) -> bytes:
    return json.dumps(
        envelope, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


class DeterministicCanonicalizer(Canonicalizer):
    """Concrete, deterministic Canonicalizer implementing the v1 rules
    documented in this module's docstring.
    """

    def canonicalize(self, content: DiscoveredContent) -> CanonicalContent:
        if content.raw_bytes is None and content.text is None:
            # Defense in depth: DiscoveredContent's own __post_init__
            # already rejects this at construction time, but a
            # canonicalizer must never silently hash a missing-content
            # object regardless of how it was constructed.
            raise ValueError(
                "Cannot canonicalize DiscoveredContent with neither "
                "raw_bytes nor text set."
            )

        envelope = _build_envelope(content)
        canonical_bytes = _serialize_envelope(envelope)

        return CanonicalContent(
            canonical_bytes=canonical_bytes,
            canonicalization_version=CANONICALIZATION_VERSION,
            included_fields=INCLUDED_FIELDS,
            excluded_fields=EXCLUDED_FIELDS,
        )
