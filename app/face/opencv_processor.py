"""
Concrete FaceProcessor implementation.

TECHNOLOGY CHOICE (see README "Face processing" section for full rationale):

Detection: OpenCV Haar cascade (haarcascade_frontalface_default.xml),
which ships INSIDE the opencv-python wheel — zero additional download,
zero network dependency, works identically on Linux/macOS/Windows the
moment `pip install opencv-python` succeeds.

Embedding: a classical, hand-rolled grid-based Local Binary Pattern
(LBP) histogram descriptor (Ahonen, Hadid & Pietikäinen, 2006), computed
with OpenCV + NumPy only. Also zero additional download and fully
deterministic.

WHY NOT A DNN EMBEDDING MODEL (e.g. OpenCV's YuNet/SFace)?
That was the first choice evaluated, and cv2 (already installed here)
does support `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF` directly.
However, the official pretrained weights for those models are
distributed via Git LFS on GitHub (opencv/opencv_zoo), and LFS objects
are served from `media.githubusercontent.com`, which was NOT reachable
in this project's sandboxed development environment (only
`github.com` / `raw.githubusercontent.com` were reachable, and those
return LFS pointer files, not the actual binary, for this repo). This
is a CONFIRMED constraint of the dev sandbox this milestone was built
in — not necessarily true of your own machine.

Because of that, this milestone ships the zero-download Haar+LBP path
as the default so the whole test suite and CLI are runnable and
verifiable right now, without any external fetch. `cv2.FaceDetectorYN` /
`cv2.FaceRecognizerSF` remain a documented, straightforward upgrade path
(see README) if you can fetch the official weights from your own
network — the `FaceProcessor`/`FaceComparator` abstraction was built
specifically so that swap requires no changes outside this file plus a
new comparator instance.

HONEST TRADEOFF: Haar+LBP is meaningfully less accurate than a modern
deep face-embedding model. It is offered here because it is genuinely
reproducible today, not because it is the best possible accuracy.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.face.processor import (
    BoundingBox,
    FaceEmbedding,
    FaceProcessingError,
    FaceProcessor,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "opencv-haar+lbp-v1"

# Fixed size every detected face crop is resized to before embedding, so
# the embedding dimension never varies with input image size.
_FACE_CROP_SIZE = (128, 128)

# Grid + histogram-bin configuration for the LBP descriptor. Embedding
# dimension is deterministic: grid_rows * grid_cols * bins.
_DEFAULT_GRID = (8, 8)
_DEFAULT_BINS = 8


class ModelUnavailableError(FaceProcessingError):
    """Raised when the face detection backend cannot be initialized."""


def _load_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise ModelUnavailableError(
            "Face detection model unavailable: could not load the bundled "
            f"Haar cascade from '{cascade_path}'. Re-install opencv-python "
            "(`pip install --force-reinstall opencv-python`) and retry."
        )
    return cascade


def compute_lbp_embedding(
    gray_face: np.ndarray,
    grid: tuple[int, int] = _DEFAULT_GRID,
    bins: int = _DEFAULT_BINS,
) -> tuple[float, ...]:
    """Compute a deterministic grid-based LBP histogram descriptor.

    Pure function: operates on any single-channel (grayscale) array of
    any size, resizes internally to a fixed crop size, and always
    returns a vector of length grid[0] * grid[1] * bins. Does not
    require a detector or model file — usable directly in unit tests.

    Args:
        gray_face: 2D grayscale image array (uint8 or convertible).
        grid: (rows, cols) of cells the face crop is divided into.
        bins: number of histogram bins per cell.

    Returns:
        L2-normalized feature vector as a tuple of floats.
    """
    if gray_face.ndim != 2:
        raise ValueError("compute_lbp_embedding expects a 2D grayscale array")

    face = cv2.resize(gray_face, _FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
    face = face.astype(np.uint8)

    # Standard 8-neighbor LBP code for every interior pixel.
    h, w = face.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    center = face[1:-1, 1:-1]
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1),
    ]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = face[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= ((neighbor >= center).astype(np.uint8)) << bit

    rows, cols = grid
    cell_h, cell_w = lbp.shape[0] // rows, lbp.shape[1] // cols
    histogram: list[float] = []
    for r in range(rows):
        for c in range(cols):
            cell = lbp[
                r * cell_h : (r + 1) * cell_h,
                c * cell_w : (c + 1) * cell_w,
            ]
            hist, _ = np.histogram(cell, bins=bins, range=(0, 256))
            histogram.extend(hist.astype(np.float64).tolist())

    vector = np.array(histogram, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return tuple(float(v) for v in vector)


def _blur_quality_score(gray_face: np.ndarray) -> float:
    """Conservative sharpness heuristic: normalized variance of the
    Laplacian. Higher means sharper. This is NOT a model-reported
    confidence value — it is a simple, documented heuristic, clamped to
    [0.0, 1.0] via a fixed cap chosen for typical webcam/photo sharpness
    ranges. It should be treated as a rough signal only.
    """
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    cap = 500.0  # empirical cap; values above this are treated as "fully sharp"
    return float(min(laplacian_var / cap, 1.0))


class OpenCVFaceProcessor(FaceProcessor):
    """Concrete FaceProcessor using OpenCV Haar cascade detection and a
    classical LBP embedding. See module docstring for the technology
    rationale and tradeoffs.

    Model initialization (loading the cascade file) happens here, in
    __init__, not at module import time — importing this module performs
    no I/O and no model loading.
    """

    def __init__(
        self,
        min_face_size: tuple[int, int] = (60, 60),
        grid: tuple[int, int] = _DEFAULT_GRID,
        bins: int = _DEFAULT_BINS,
    ) -> None:
        self._cascade = _load_cascade()
        self._min_face_size = min_face_size
        self._grid = grid
        self._bins = bins

    def process(self, image_bytes: bytes) -> list[FaceEmbedding]:
        if not image_bytes:
            raise FaceProcessingError("Unsupported or corrupted image: no data provided.")

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise FaceProcessingError(
                "Unsupported or corrupted image: could not decode the supplied bytes "
                "as a JPEG/PNG image."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=self._min_face_size,
        )

        embeddings: list[FaceEmbedding] = []
        for (x, y, w, h) in detections:
            face_crop = gray[y : y + h, x : x + w]
            vector = compute_lbp_embedding(face_crop, grid=self._grid, bins=self._bins)
            quality = _blur_quality_score(face_crop)
            embeddings.append(
                FaceEmbedding(
                    vector=vector,
                    model_version=MODEL_VERSION,
                    face_bbox=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    quality_score=quality,
                )
            )

        logger.info("Detected %d face(s)", len(embeddings))
        return embeddings


def require_single_face(
    embeddings: list[FaceEmbedding], context: str = "reference image"
) -> FaceEmbedding:
    """Enforce the "exactly one face" rule used for reference images.

    Args:
        embeddings: result of FaceProcessor.process().
        context: human-readable label for the error message (e.g.
            "reference image").

    Raises:
        FaceProcessingError: with a clear, actionable message if zero or
            more than one face was found.
    """
    if len(embeddings) == 0:
        raise FaceProcessingError(f"No face detected in {context}.")
    if len(embeddings) > 1:
        raise FaceProcessingError(
            f"Multiple faces detected in {context}. Provide an image "
            "containing exactly one face."
        )
    return embeddings[0]
