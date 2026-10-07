"""Cryptographic and perceptual image hashes."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy.fft import dctn


def difference_hash(image: Image.Image) -> str:
    """Return a 64-bit difference hash for near-duplicate grouping."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return "".join("1" if bit else "0" for bit in bits.ravel())


def perceptual_hash(image: Image.Image) -> str:
    """Return a compact DCT perceptual hash."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(gray, dtype=np.float32), norm="ortho")[:8, :8]
    values = coefficients.ravel()[1:]
    median = float(np.median(values))
    bits = coefficients.ravel() > median
    return "".join("1" if bit else "0" for bit in bits)


def mean_rgb(image: Image.Image) -> tuple[float, float, float]:
    """Return a compact color signature used to reduce grayscale pHash collisions."""
    sample = image.convert("RGB").resize((8, 8), Image.Resampling.LANCZOS)
    means = np.asarray(sample, dtype=np.float32).mean(axis=(0, 1))
    return tuple(float(value) for value in means)


def hashes_for_bytes(data: bytes) -> dict[str, str]:
    """Compute perceptual hashes without retaining the input."""
    with Image.open(BytesIO(data)) as image:
        image.load()
        return {"dhash": difference_hash(image), "phash": perceptual_hash(image)}
