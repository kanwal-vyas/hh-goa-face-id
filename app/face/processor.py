"""
Face processing scaffold.

Defines the data model and interface for turning a supplied reference
image into a face embedding. No concrete detection/embedding model is
wired in at this milestone — importing this module must not trigger any
model download or network call.

The concrete implementation (added in a later milestone) will operate
only on an image explicitly supplied by the operator (a local file path
or in-memory bytes) — it does not fetch, crawl, or discover images on
its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class FaceProcessingError(Exception):
    """Raised when a supplied image cannot be processed into an embedding."""


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding box of a detected face."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BoundingBox width and height must be positive")


@dataclass(frozen=True)
class FaceEmbedding:
    """A face embedding produced from a single detected face.

    Attributes:
        vector: The embedding vector. Kept in-memory only; this dataclass
            makes no persistence guarantees, in line with the "keep
            biometric data ephemeral" scope constraint.
        model_version: Identifier of the embedding model/version used,
            so future comparisons stay consistent if the model changes.
        face_bbox: Bounding box of the face this embedding was generated
            from, in the source image's pixel coordinates.
        quality_score: Optional heuristic score (e.g. blur/pose quality)
            in [0.0, 1.0], if the concrete implementation produces one.
    """

    vector: tuple[float, ...]
    model_version: str
    face_bbox: BoundingBox
    quality_score: float | None = None

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("FaceEmbedding.vector must not be empty")
        if self.quality_score is not None and not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("quality_score must be between 0.0 and 1.0")


class FaceProcessor(ABC):
    """Abstract interface for turning image bytes into face embedding(s).

    Concrete implementations are added in a later milestone. This class
    must remain abstract at this stage and must not perform any model
    loading, download, or network access on import.
    """

    @abstractmethod
    def process(self, image_bytes: bytes) -> list[FaceEmbedding]:
        """Detect face(s) in the supplied image and return their embeddings.

        Args:
            image_bytes: Raw bytes of an explicitly supplied image (no
                fetching/downloading is performed by this method).

        Returns:
            A list of FaceEmbedding, one per detected face. An empty list
            indicates no face was detected — callers must handle this
            explicitly rather than assuming a match.

        Raises:
            FaceProcessingError: if the image cannot be decoded/processed.
        """
        raise NotImplementedError
