"""Deterministic image statistics used by the trainable baseline classifier."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import fft2, fftshift

FEATURE_NAMES = [
    "log_width",
    "log_height",
    "aspect_ratio",
    "mean_luma",
    "std_luma",
    "mean_saturation",
    "std_saturation",
    "red_std",
    "green_std",
    "blue_std",
    "histogram_entropy",
    "edge_density",
    "gradient_mean",
    "gradient_std",
    "laplacian_variance",
    "residual_std",
    "residual_tile_cv",
    "high_frequency_ratio",
    "spectral_entropy",
    "resampling_periodicity",
    "jpeg_quant_mean",
    "jpeg_quant_std",
    "jpeg_quant_count",
    "texture_tile_variance",
]


def _entropy(histogram: np.ndarray) -> float:
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(probabilities.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _downsample(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    result = image.convert("RGB")
    result.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    if min(result.size) < 2:
        result = result.resize(
            (max(2, result.width), max(2, result.height)), Image.Resampling.LANCZOS
        )
    return result


def extract_features(image: Image.Image, max_dimension: int = 1024) -> dict[str, float]:
    """Calculate reproducible content, noise, frequency and compression features."""
    original_width, original_height = image.size
    rgb_image = _downsample(image, max_dimension)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32
    )
    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.float32) / 255.0

    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    gradient = np.hypot(gx, gy)
    laplacian = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    blurred = (
        np.asarray(
            rgb_image.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32
        )
        / 255.0
    )
    residual = rgb - blurred

    height, width = gray.shape
    tile_values: list[float] = []
    residual_values: list[float] = []
    for y in range(0, height, max(8, height // 8)):
        for x in range(0, width, max(8, width // 8)):
            patch = gray[y : y + max(8, height // 8), x : x + max(8, width // 8)]
            if patch.size:
                tile_values.append(float(np.var(patch)))
                residual_values.append(
                    float(
                        np.std(residual[y : y + patch.shape[0], x : x + patch.shape[1]])
                    )
                )
    residual_mean = float(np.mean(residual_values)) if residual_values else 0.0

    spectrum = np.abs(fftshift(fft2(gray - float(gray.mean()))))
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    spectral_energy = spectrum**2
    total_spectrum = float(np.sum(spectral_energy)) + 1e-12
    high_ratio = float(
        np.sum(spectral_energy[radius > min(height, width) * 0.25]) / total_spectrum
    )
    spectral_entropy = _entropy(np.histogram(np.log1p(spectrum), bins=64)[0]) / 6.0

    periodicity: list[float] = []
    for gradient_axis in (gx, gy):
        normalized = gradient_axis - float(gradient_axis.mean())
        denominator = float(np.mean(normalized**2)) + 1e-12
        for period in range(2, 9):
            shifted = np.roll(normalized, period, axis=1 if gradient_axis is gx else 0)
            periodicity.append(float(np.mean(normalized * shifted) / denominator))

    quant_values: list[float] = []
    quantization = getattr(image, "quantization", None)
    if quantization:
        quant_values = [
            float(value) for table in quantization.values() for value in table
        ]
    q_mean = float(np.mean(quant_values)) if quant_values else 0.0
    q_std = float(np.std(quant_values)) if quant_values else 0.0
    histogram = np.histogram(gray, bins=64, range=(0, 1))[0]
    tile_variance = float(np.mean(tile_values)) if tile_values else 0.0
    result = {
        "log_width": math.log1p(original_width),
        "log_height": math.log1p(original_height),
        "aspect_ratio": float(original_width / max(original_height, 1)),
        "mean_luma": float(gray.mean()),
        "std_luma": float(gray.std()),
        "mean_saturation": float(hsv[..., 1].mean()),
        "std_saturation": float(hsv[..., 1].std()),
        "red_std": float(rgb[..., 0].std()),
        "green_std": float(rgb[..., 1].std()),
        "blue_std": float(rgb[..., 2].std()),
        "histogram_entropy": _entropy(histogram) / 6.0,
        "edge_density": float(np.mean(gradient > 0.08)),
        "gradient_mean": float(gradient.mean()),
        "gradient_std": float(gradient.std()),
        "laplacian_variance": float(np.var(laplacian)),
        "residual_std": float(np.std(residual)),
        "residual_tile_cv": float(np.std(residual_values) / (residual_mean + 1e-9)),
        "high_frequency_ratio": high_ratio,
        "spectral_entropy": spectral_entropy,
        "resampling_periodicity": max(periodicity) if periodicity else 0.0,
        "jpeg_quant_mean": q_mean,
        "jpeg_quant_std": q_std,
        "jpeg_quant_count": float(len(quant_values)),
        "texture_tile_variance": tile_variance,
    }
    return {
        key: float(value) if np.isfinite(value) else 0.0
        for key, value in result.items()
    }
