#!/usr/bin/env python3
"""Run real candidate matching against the bundled authorized corpus only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import load_settings
from app.face.compare import CosineFaceComparator
from app.face.opencv_processor import OpenCVFaceProcessor, require_single_face
from app.face.processor import FaceProcessingError
from app.matching.candidate_matcher import AuthorizedCorpusCandidateMatcher
from app.search.authorized_corpus import AuthorizedCorpusSearchProvider
from app.search.interface import SearchQuery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local authorized-corpus candidate-matching demo."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to an explicitly supplied local reference image.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_path = Path(args.reference)
    if not reference_path.is_file():
        print(f"Error: Reference image not found: {reference_path}", file=sys.stderr)
        return 1

    processor = OpenCVFaceProcessor()
    try:
        reference_embedding = require_single_face(
            processor.process(reference_path.read_bytes()), context="reference image"
        )
    except FaceProcessingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    provider = AuthorizedCorpusSearchProvider()
    candidates = provider.search(
        SearchQuery(
            reference_id="local-authorized-demo",
            face_embedding_ref="ephemeral-reference-embedding",
            max_results=len(provider.records),
        )
    )
    matcher = AuthorizedCorpusCandidateMatcher(
        reference_embedding=reference_embedding,
        face_processor=processor,
        face_comparator=CosineFaceComparator(load_settings().match_threshold),
        candidate_image_paths=provider.candidate_image_paths,
    )
    results = matcher.match(candidates)

    print("[1/3] Face processing")
    print("    Reference face detected")
    print("[2/3] Authorized corpus search")
    print(f"    Candidates discovered: {len(candidates)}")
    print("[3/3] Candidate matching")
    for result in results:
        print(f"    Candidate: {result.candidate_id}")
        if not result.usable:
            print(f"      Rejected: {result.rejection_reason}")
            continue
        print("      Face detected: exactly one")
        print(f"      Similarity: {result.similarity_score:.6f}")
        print(f"      Result: {result.match_band.value.upper()}")
        print(f"      Selected: {result.is_selected}")
    print("Note: scores are similarity signals, not proof of real-world identity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
