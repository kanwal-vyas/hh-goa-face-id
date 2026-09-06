#!/usr/bin/env python3
"""
HH Goa 2026 — Face Identification & Blockchain Verification
CLI entry point.

Individual local stages are implemented, but they are not yet wired
together as one end-to-end CLI pipeline. The CLI reports stages that
are not part of its current flow and never fabricates a result.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.blockchain.interface import OnChainPayload
from app.blockchain.local_provider import (
    BlockchainConnectionError,
    BlockchainRegistrationError,
    LocalBlockchainProvider,
)
from app.config.settings import ConfigError, load_settings
from app.content.canonicalize import DeterministicCanonicalizer
from app.content.extractor import DiscoveredContent
from app.content.fingerprint import hash_canonical_content, hash_source_reference
from app.face.compare import CosineFaceComparator
from app.face.opencv_processor import OpenCVFaceProcessor, require_single_face
from app.face.processor import FaceProcessingError
from app.pipeline.runner import STAGE_NAMES


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _configure_output_encoding() -> None:
    """Emit CLI text as UTF-8 even when Windows defaults to a legacy code page."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "HH Goa 2026 Task 3 — Face Identification & Blockchain "
            "Verification pipeline (scaffold stage)."
        ),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to a local reference image (explicitly supplied by you).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help=(
            "Path to a second local image to compare against --reference "
            "using real face detection + similarity scoring."
        ),
    )
    parser.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        help=(
            "Path to a local text file to canonicalize and SHA-256 "
            "fingerprint (Milestone 3 demonstration; independent of "
            "--reference/--compare)."
        ),
    )
    parser.add_argument(
        "--blockchain-demo",
        type=str,
        default=None,
        help=(
            "Path to a local text file to canonicalize, fingerprint, "
            "register on the local blockchain, and verify by retrieval "
            "(Milestone 4 demonstration; independent of --reference/"
            "--compare/--fingerprint). Requires a local EVM development "
            "node running at BLOCKCHAIN_RPC_URL."
        ),
    )
    return parser


def _print_pending_stages(start_index: int) -> None:
    total = len(STAGE_NAMES)
    print("\nRemaining pipeline stages are not wired into this CLI flow:")
    for i in range(start_index, total + 1):
        print(f"  · [{i}/{total}] {STAGE_NAMES[i - 1]}: not wired into this CLI flow")


def _run_fingerprint_demo(text_file_path: str) -> int:
    """Milestone 3 demonstration: canonicalize + SHA-256 fingerprint a
    local text file. Independent of the face-processing flow.
    """
    path = Path(text_file_path)
    if not path.is_file():
        print(f"Error: Fingerprint input file not found: {path}", file=sys.stderr)
        return 1

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(
            f"Error: {path} is not valid UTF-8 text. --fingerprint expects "
            "a text file in this milestone.",
            file=sys.stderr,
        )
        return 1

    content = DiscoveredContent(
        candidate_id="local-fingerprint-demo",
        raw_bytes=None,
        text=text,
        source_reference=str(path),
    )
    canonicalizer = DeterministicCanonicalizer()
    canonical = canonicalizer.canonicalize(content)
    fingerprint = hash_canonical_content(canonical)

    print(f"Canonicalization version: {fingerprint.canonicalization_version}")
    print(f"Algorithm: {fingerprint.algorithm}")
    print(f"Fingerprint: {fingerprint.hash}")
    return 0


def _verification_status(local_hash: str, on_chain_hash: str) -> str:
    """Real equality check between a locally computed content hash and
    an on-chain content hash. Pure function so the FAIL path can be
    unit-tested directly without needing a live blockchain.
    """
    return "PASS" if local_hash == on_chain_hash else "FAIL"


def _run_blockchain_demo(text_file_path: str, settings) -> int:
    """Milestone 4 demonstration: canonicalize + fingerprint a local text
    file, register it on the local blockchain, retrieve it back, and
    verify the on-chain hash matches the local hash. Independent of the
    face-processing and --fingerprint flows.
    """
    path = Path(text_file_path)
    if not path.is_file():
        print(f"Error: Blockchain demo input file not found: {path}", file=sys.stderr)
        return 1

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(
            f"Error: {path} is not valid UTF-8 text. --blockchain-demo "
            "expects a text file in this milestone.",
            file=sys.stderr,
        )
        return 1

    # source_reference is provenance only (the local file path here) —
    # it is never fed into content hashing; see hash_source_reference()
    # below for how it is hashed separately.
    content = DiscoveredContent(
        candidate_id="local-blockchain-demo",
        raw_bytes=None,
        text=text,
        source_reference=str(path),
    )

    print("[1/5] Canonicalizing content... ", end="")
    canonical = DeterministicCanonicalizer().canonicalize(content)
    print("\u2713")

    print("[2/5] Computing SHA-256 fingerprint... ", end="")
    fingerprint = hash_canonical_content(canonical)
    print("\u2713")
    print(f"    Content hash: {fingerprint.hash}\n")

    print("[3/5] Computing source-reference hash... ", end="")
    source_hash = hash_source_reference(content.source_reference)
    print("\u2713")
    print(f"    Source hash: {source_hash}\n")

    print("[4/5] Registering on local blockchain... ", end="")
    try:
        provider = LocalBlockchainProvider(
            rpc_url=settings.blockchain_rpc_url,
            private_key=settings.blockchain_private_key,
            contract_address=settings.blockchain_contract_address,
        )
    except BlockchainConnectionError as exc:
        print("\u2717")
        print(f"\nError: {exc}", file=sys.stderr)
        print(
            "\nThe local blockchain demo requires a local EVM development "
            "node running and reachable at "
            f"BLOCKCHAIN_RPC_URL={settings.blockchain_rpc_url}. Start one "
            "with `npx hardhat node` and retry.",
            file=sys.stderr,
        )
        return 1

    payload = OnChainPayload(
        content_hash=fingerprint.hash,
        algorithm=fingerprint.algorithm,
        version=fingerprint.canonicalization_version,
        source_reference_hash=source_hash,
    )

    try:
        record = provider.register(payload)
    except BlockchainRegistrationError as exc:
        print("\u2717")
        print(f"\nError: Registration failed: {exc}", file=sys.stderr)
        return 1

    print("\u2713")
    print(f"    Transaction: {record.tx_hash}")
    print(f"    Block: {record.block_number}\n")

    print("[5/5] Retrieving and verifying blockchain record... ", end="")
    retrieved = provider.retrieve(record.tx_hash)
    if retrieved is None:
        print("\u2717")
        print(
            "\nError: Could not retrieve the just-registered record from "
            f"the blockchain (tx_hash={record.tx_hash}). This should not "
            "happen against a healthy node — check node logs.",
            file=sys.stderr,
        )
        return 1
    print("\u2713\n")

    status = _verification_status(fingerprint.hash, retrieved.on_chain_payload.content_hash)
    print(f"    Local hash:    {fingerprint.hash}")
    print(f"    On-chain hash: {retrieved.on_chain_payload.content_hash}")
    print(f"\n    VERIFICATION: {status}")

    return 0 if status == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    _configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    _configure_logging(settings.log_level)

    if args.blockchain_demo is not None:
        return _run_blockchain_demo(args.blockchain_demo, settings)

    if args.fingerprint is not None:
        return _run_fingerprint_demo(args.fingerprint)

    if args.reference is None:
        parser.print_help()
        return 0

    total_stages = len(STAGE_NAMES)

    ref_path = Path(args.reference)
    if not ref_path.is_file():
        print(f"Error: Reference image not found: {ref_path}", file=sys.stderr)
        print(
            "Provide a valid path to a local image with --reference, "
            "e.g.: python main.py --reference examples/reference.jpg",
            file=sys.stderr,
        )
        return 1

    try:
        processor = OpenCVFaceProcessor()
        reference_embedding = require_single_face(
            processor.process(ref_path.read_bytes()), context="reference image"
        )
    except FaceProcessingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n[1/{total_stages}] Face processing")
    print("    Reference face detected")
    print(f"    Model: {reference_embedding.model_version}")
    print(f"    Embedding dimension: {len(reference_embedding.vector)}")
    if reference_embedding.quality_score is not None:
        print(
            "    Quality heuristic (blur-based, not a model confidence): "
            f"{reference_embedding.quality_score:.3f}"
        )

    if args.compare is None:
        _print_pending_stages(start_index=2)
        print("\nNote: a similarity score is not proof of real-world identity.")
        return 0

    compare_path = Path(args.compare)
    if not compare_path.is_file():
        print(f"Error: Comparison image not found: {compare_path}", file=sys.stderr)
        return 1

    try:
        candidate_embedding = require_single_face(
            processor.process(compare_path.read_bytes()), context="comparison image"
        )
    except FaceProcessingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    comparator = CosineFaceComparator(match_threshold=settings.match_threshold)
    result = comparator.compare(reference_embedding, candidate_embedding)

    print(f"\n[2/{total_stages}] Face comparison")
    print(f"    Similarity: {result.similarity_score:.4f}")
    print(f"    Result: {result.band.value.upper()}")
    print(f"    (match_threshold={settings.match_threshold})")
    print("\nNote: this is a similarity score, not proof of real-world identity.")

    _print_pending_stages(start_index=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
