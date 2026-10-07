"""Forensic feature grouping and conservative findings."""

from __future__ import annotations


def summarize_forensics(features: dict[str, float], image_format: str) -> dict:
    """Expose measured statistics without treating them as proof of origin."""
    jpeg = {
        "format": image_format,
        "quantization_mean": features["jpeg_quant_mean"] or None,
        "quantization_std": features["jpeg_quant_std"] or None,
        "quantization_values": int(features["jpeg_quant_count"]),
        "note": "Pillow exposes quantization tables for JPEG input; a single quality estimate is not reliable across encoders.",
    }
    noise = {
        "high_pass_residual_std": features["residual_std"],
        "spatial_variation_cv": features["residual_tile_cv"],
        "note": "Residual statistics are affected by denoising, resizing and compression; they do not establish camera origin.",
    }
    frequency = {
        "high_frequency_energy_ratio": features["high_frequency_ratio"],
        "spectral_entropy_normalized": features["spectral_entropy"],
        "note": "Frequency measurements are descriptive and generator dependent.",
    }
    resampling = {
        "periodicity_score": features["resampling_periodicity"],
        "note": "Periodic gradients can have many causes; this baseline does not infer a definitive resize history.",
    }
    texture = {
        "edge_density": features["edge_density"],
        "gradient_mean": features["gradient_mean"],
        "gradient_std": features["gradient_std"],
        "local_variance_mean": features["texture_tile_variance"],
    }
    color = {
        "mean_luminance": features["mean_luma"],
        "luminance_std": features["std_luma"],
        "mean_saturation": features["mean_saturation"],
        "saturation_std": features["std_saturation"],
        "histogram_entropy_normalized": features["histogram_entropy"],
    }
    return {
        "jpeg": jpeg,
        "noise": noise,
        "frequency": frequency,
        "resampling": resampling,
        "texture": texture,
        "color": color,
    }
