#!/usr/bin/env python3
"""
OPTIONAL upgrade path — NOT required to run Milestone 2.

The default face-processing implementation (app/face/opencv_processor.py)
uses OpenCV's bundled Haar cascade + a classical LBP embedding, which
requires no model download at all.

This script fetches the official OpenCV Zoo DNN weights for YuNet
(detection) and SFace (embedding) — a meaningfully more accurate but
heavier alternative — IF you want to build a DNN-based FaceProcessor
implementation later. It downloads nothing on its own initiative unless
you run it explicitly, and every step is logged.

KNOWN CONSTRAINT: these weights are distributed via Git LFS on GitHub.
Some restricted/offline network environments (this project's dev
sandbox included) cannot reach the LFS media host
(media.githubusercontent.com) even though github.com itself is
reachable, so this script may fail in such environments — that failure
is expected there, not a bug in this script. It has NOT been verified
to succeed end-to-end from within this project's sandboxed development
environment; run it on a machine with normal internet access.

Usage:
    python scripts/download_face_models.py --out models/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print(
        "This script requires 'requests'. Install it with:\n"
        "    pip install requests\n"
        "(It is intentionally not a core dependency of the main "
        "pipeline, since the default face pipeline does not need it.)",
        file=sys.stderr,
    )
    sys.exit(1)

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for filename, info in MODELS.items():
        dest = out_dir / filename
        print(f"Downloading {filename} ...")
        try:
            resp = requests.get(info["url"], timeout=60, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        dest.write_bytes(resp.content)
        actual_hash = _sha256_of(dest)
        if actual_hash != info["sha256"]:
            print(
                f"  FAILED: SHA-256 mismatch for {filename}.\n"
                f"    expected: {info['sha256']}\n"
                f"    actual:   {actual_hash}\n"
                "  This usually means an LFS pointer file was downloaded "
                "instead of the real binary (seen in restricted network "
                "environments), or the upstream file changed.",
                file=sys.stderr,
            )
            dest.unlink(missing_ok=True)
            failures += 1
            continue

        print(f"  OK -> {dest} (sha256 verified)")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="models", help="Output directory")
    args = parser.parse_args(argv)

    failures = download(Path(args.out))
    if failures:
        print(
            f"\n{failures} file(s) failed to download. See messages above. "
            "This is a documented, non-default upgrade path — the core "
            "pipeline does not require these files.",
            file=sys.stderr,
        )
        return 1

    print("\nAll model files downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
