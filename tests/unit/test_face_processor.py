"""
Tests for OpenCVFaceProcessor.

Some tests here need an actual photograph containing a detectable face
to exercise real detection (as opposed to validation logic, which is
fully covered without one). Per the milestone's constraint against
downloading or fabricating photographs of people, those specific tests
are gated behind an optional local fixture directory:

    tests/fixtures/authorized_faces/single_face.jpg
    tests/fixtures/authorized_faces/two_faces.jpg

This directory is NOT committed with any image content (see .gitignore
note below) and no such image was available in this project's
development environment, so those specific tests are marked skipped
here and were NOT executed as part of this milestone's validation --
see the final report for an explicit statement of this limitation.

To activate them locally: drop in your own explicitly authorized/
consented photos at the paths above and re-run pytest.
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from app.face.opencv_processor import (
    MODEL_VERSION,
    OpenCVFaceProcessor,
    require_single_face,
)
from app.face.processor import FaceProcessingError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "authorized_faces"
SINGLE_FACE_IMAGE = FIXTURE_DIR / "single_face.jpg"
TWO_FACE_IMAGE = FIXTURE_DIR / "two_faces.jpg"

_no_single_face_fixture = pytest.mark.skipif(
    not SINGLE_FACE_IMAGE.exists(),
    reason=(
        "No authorized single-face test image supplied at "
        f"{SINGLE_FACE_IMAGE} - see module docstring. Not fabricated or "
        "downloaded per project constraints."
    ),
)
_no_two_face_fixture = pytest.mark.skipif(
    not TWO_FACE_IMAGE.exists(),
    reason=(
        "No authorized two-face test image supplied at "
        f"{TWO_FACE_IMAGE} - see module docstring. Not fabricated or "
        "downloaded per project constraints."
    ),
)


def _random_noise_jpeg_bytes() -> bytes:
    import cv2

    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", noise)
    assert ok
    return buf.tobytes()


def test_processor_initializes_without_error():
    OpenCVFaceProcessor()


def test_process_rejects_empty_bytes():
    processor = OpenCVFaceProcessor()
    with pytest.raises(FaceProcessingError):
        processor.process(b"")


def test_process_rejects_corrupted_image():
    processor = OpenCVFaceProcessor()
    with pytest.raises(FaceProcessingError):
        processor.process(b"this-is-not-a-real-image-file")


def test_process_returns_zero_faces_on_random_noise():
    """Genuinely exercised (no fixture needed): a random-noise image
    reliably produces zero face detections with a Haar cascade."""
    processor = OpenCVFaceProcessor()
    embeddings = processor.process(_random_noise_jpeg_bytes())
    assert embeddings == []


def test_require_single_face_raises_on_zero_faces():
    with pytest.raises(FaceProcessingError, match="No face detected"):
        require_single_face([], context="reference image")


def test_require_single_face_raises_on_multiple_faces():
    from app.face.processor import BoundingBox, FaceEmbedding

    fake = FaceEmbedding(
        vector=(0.1, 0.2),
        model_version=MODEL_VERSION,
        face_bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )
    with pytest.raises(FaceProcessingError, match="Multiple faces detected"):
        require_single_face([fake, fake], context="reference image")


def test_require_single_face_returns_the_only_embedding():
    from app.face.processor import BoundingBox, FaceEmbedding

    fake = FaceEmbedding(
        vector=(0.1, 0.2),
        model_version=MODEL_VERSION,
        face_bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )
    assert require_single_face([fake]) is fake


def test_no_network_call_during_face_processing(monkeypatch):
    def _blocked_connect(*args, **kwargs):
        raise AssertionError("Unexpected network connection during face processing")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    processor = OpenCVFaceProcessor()
    # Should complete without touching the network at all.
    processor.process(_random_noise_jpeg_bytes())


@_no_single_face_fixture
def test_process_detects_exactly_one_face_on_authorized_single_face_image():
    processor = OpenCVFaceProcessor()
    embeddings = processor.process(SINGLE_FACE_IMAGE.read_bytes())
    assert len(embeddings) == 1
    assert embeddings[0].model_version == MODEL_VERSION


@_no_two_face_fixture
def test_process_detects_multiple_faces_on_authorized_two_face_image():
    processor = OpenCVFaceProcessor()
    embeddings = processor.process(TWO_FACE_IMAGE.read_bytes())
    assert len(embeddings) >= 2
    with pytest.raises(FaceProcessingError, match="Multiple faces detected"):
        require_single_face(embeddings)
