"""Create the bundled, entirely procedural raster portraits.

The images are made from shapes, gradients, and deterministic noise only.
They do not read, sample, transform, or otherwise use a photograph or
biometric input.  Running this file regenerates the three PNGs under
``examples/authorized_demo_images``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "examples" / "authorized_demo_images"
SIZE = 512


def _ellipse_mask(
    center: tuple[int, int], axes: tuple[int, int], angle: float = 0.0
) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1, lineType=cv2.LINE_AA)
    return mask


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    canvas[mask > 0] = color


def _soft_shading(
    canvas: np.ndarray, mask: np.ndarray, seed: int, grain_strength: float
) -> None:
    """Add texture to a procedural face without using source imagery."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:SIZE, :SIZE]
    light = 15 * np.cos((x - 250) / 135) + 9 * np.sin((y - 210) / 115)
    grain = cv2.GaussianBlur(rng.normal(0, 8, (SIZE, SIZE)).astype(np.float32), (0, 0), 2)
    grain += rng.normal(0, grain_strength, (SIZE, SIZE)).astype(np.float32)
    adjustment = (light + grain).astype(np.int16)
    for channel in range(3):
        plane = canvas[:, :, channel].astype(np.int16)
        plane[mask > 0] = np.clip(plane[mask > 0] + adjustment[mask > 0], 0, 255)
        canvas[:, :, channel] = plane.astype(np.uint8)


def _portrait(
    *,
    seed: int,
    background: tuple[int, int, int],
    skin: tuple[int, int, int],
    hair: tuple[int, int, int],
    shirt: tuple[int, int, int],
    eye_spacing: int,
    mouth_width: int,
    face_angle: float,
    grain_strength: float,
) -> np.ndarray:
    """Draw a single non-identifying, front-facing synthetic portrait."""
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE, 3), background, dtype=np.uint8)

    # A soft, deterministic studio backdrop.
    y, x = np.mgrid[:SIZE, :SIZE]
    vignette = ((x - SIZE / 2) ** 2 + (y - SIZE / 2) ** 2) / (SIZE * 3)
    image = np.clip(image.astype(np.float32) - vignette[:, :, None], 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 2)

    # Neck and shoulders, then ears and the oval face.
    cv2.rectangle(image, (217, 331), (295, 398), skin, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (256, 470), (205, 125), 0, 180, 360, shirt, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (158, 254), (31, 51), 0, 0, 360, skin, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (354, 254), (31, 51), 0, 0, 360, skin, -1, lineType=cv2.LINE_AA)
    face_mask = _ellipse_mask((256, 245), (112, 145), face_angle)
    _paint_mask(image, face_mask, skin)
    _soft_shading(image, face_mask, seed, grain_strength)

    # Hair silhouette and a subtle parting; deliberately generated shapes.
    cv2.ellipse(image, (256, 187), (119, 133), face_angle, 170, 375, hair, 25, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (256, 164), (114, 83), face_angle, 175, 365, hair, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (256, 189), (101, 108), face_angle, 8, 172, skin, -1, lineType=cv2.LINE_AA)
    cv2.line(image, (251, 100), (238, 181), tuple(max(c - 17, 0) for c in hair), 5, lineType=cv2.LINE_AA)

    # Brows, eyes, nose, and mouth are shaped and shaded rather than copied
    # from an image. Their high-contrast layout makes this a stable cascade
    # test subject at the portrait scale.
    left, right = 256 - eye_spacing, 256 + eye_spacing
    for eye_x in (left, right):
        cv2.ellipse(image, (eye_x, 224), (29, 12), 0, 190, 350, (48, 39, 33), 5, lineType=cv2.LINE_AA)
        cv2.ellipse(image, (eye_x, 237), (24, 14), 0, 0, 360, (226, 216, 204), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(image, (eye_x, 237), (8, 10), 0, 0, 360, (32, 29, 27), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (eye_x - 3, 233), 2, (245, 245, 245), -1, lineType=cv2.LINE_AA)
    cv2.line(image, (256, 242), (243, 286), tuple(max(c - 32, 0) for c in skin), 4, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (250, 291), (15, 7), 0, 10, 170, tuple(max(c - 38, 0) for c in skin), 3, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (256, 319), (mouth_width, 17), 0, 5, 175, (71, 51, 63), 5, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (256, 320), (mouth_width - 5, 10), 0, 5, 175, (189, 103, 111), 2, lineType=cv2.LINE_AA)

    # Freckles add local texture independently of any person or reference.
    for x_pos, y_pos in rng.integers([180, 265], [333, 305], size=(30, 2)):
        cv2.circle(image, (int(x_pos), int(y_pos)), 1, tuple(max(c - 25, 0) for c in skin), -1)

    return cv2.GaussianBlur(image, (0, 0), 0.35)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    portraits = {
        "harbor-avatar.png": _portrait(
            seed=11,
            background=(210, 185, 132),
            skin=(164, 185, 210),
            hair=(37, 46, 58),
            shirt=(118, 89, 49),
            eye_spacing=42,
            mouth_width=35,
            face_angle=0,
            grain_strength=20,
        ),
        "coast-avatar.png": _portrait(
            seed=29,
            background=(168, 205, 220),
            skin=(191, 205, 224),
            hair=(45, 58, 72),
            shirt=(73, 112, 74),
            eye_spacing=47,
            mouth_width=46,
            face_angle=-3,
            grain_strength=0,
        ),
        "garden-avatar.png": _portrait(
            seed=47,
            background=(153, 184, 140),
            skin=(114, 154, 185),
            hair=(28, 42, 36),
            shirt=(69, 81, 148),
            eye_spacing=35,
            mouth_width=25,
            face_angle=5,
            grain_strength=40,
        ),
    }
    for name, image in portraits.items():
        success = cv2.imwrite(str(OUTPUT_DIRECTORY / name), image)
        if not success:
            raise RuntimeError(f"Could not write {name}")


if __name__ == "__main__":
    main()
