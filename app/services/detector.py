"""Top-level analysis orchestration."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.ml.inference import predict_ai_probability
from app.ml.nonescape import predict as predict_nonescape
from app.ml.sightengine import SightengineError, predict_ai_score
from app.services.ensemble import classify
from app.services.explainability import build_evidence
from app.services.filename import analyze_filename
from app.services.forensics import summarize_forensics
from app.services.image_features import extract_features
from app.services.metadata import extract_metadata
from app.services.report import build_report
from app.utils.hashing import difference_hash, perceptual_hash


def analyze_image(
    data: bytes, validated, supplied_content_type: str | None = None
) -> dict:
    """Run metadata, filename and image feature analysis on decoded bytes."""
    with Image.open(__import__("io").BytesIO(data)) as image:
        image.load()
        model_image = image.copy()
        metadata = extract_metadata(
            image, validated.safe_filename, validated.mime_type, len(data), data
        )
        features = extract_features(image, settings.max_inference_dimension)
        image_hashes = {
            "dhash": difference_hash(image),
            "phash": perceptual_hash(image),
        }
    filename_analysis = analyze_filename(
        validated.safe_filename, validated.image_format, validated.extension_mismatch
    )
    warnings: list[str] = []
    if validated.extension_mismatch:
        warnings.append("Filename extension does not match the decoded image format.")
    if supplied_content_type and supplied_content_type not in {
        validated.mime_type,
        "application/octet-stream",
    }:
        warnings.append(
            "Upload Content-Type header does not match the decoded image format."
        )
    model = None
    if settings.sightengine_enabled:
        warnings.append(
            "This image was sent to Sightengine for AI-generated image analysis."
        )
        try:
            model = predict_ai_score(
                data,
                validated.safe_filename,
                validated.mime_type,
                settings.sightengine_api_user,
                settings.sightengine_api_secret,
            )
        except SightengineError as exc:
            warnings.append(
                f"Sightengine analysis failed ({exc}); local detection was used instead."
            )
    elif settings.sightengine_partially_configured:
        warnings.append(
            "Sightengine credentials are incomplete; local detection was used instead."
        )

    if model is None:
        model = predict_nonescape(model_image, Path(settings.nonescape_model_path))
    if model is None:
        model = predict_ai_probability(features, Path(settings.model_path))
    if model.get("probability") is not None and not model.get("calibrated"):
        warnings.append(
            "The installed model has not been calibrated; classification remains inconclusive."
        )
    classification = classify(model, metadata["findings"])
    forensic_report = summarize_forensics(features, validated.image_format)
    evidence = build_evidence(model, metadata, filename_analysis, warnings)
    report = build_report(
        validated.safe_filename,
        len(data),
        validated,
        metadata,
        filename_analysis,
        forensic_report,
        model,
        classification,
        evidence,
        warnings,
    )
    report["file"].update(image_hashes)
    report["forensics"]["feature_vector"] = features
    return report
