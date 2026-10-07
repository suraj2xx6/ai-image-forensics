"""Stable report assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

DISCLAIMER = "This analysis provides probabilistic forensic evidence and does not constitute absolute proof of image origin."


def build_report(
    filename: str,
    byte_size: int,
    validated,
    metadata: dict,
    filename_analysis: dict,
    forensics: dict,
    model: dict,
    classification: dict,
    evidence: list[dict],
    warnings: list[str],
) -> dict:
    """Create the versioned JSON API and persistence report."""
    analysis_id = str(uuid4())
    return {
        "analysis_id": analysis_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "file": {
            "size": byte_size,
            "mime_type": validated.mime_type,
            "format": validated.image_format,
            "width": validated.width,
            "height": validated.height,
            "sha256": validated.sha256,
        },
        "classification": classification,
        "metadata": metadata,
        "filename_analysis": filename_analysis,
        "forensics": forensics,
        "evidence": evidence,
        "model": {
            key: model.get(key)
            for key in (
                "name",
                "version",
                "calibrated",
                "dataset_version",
                "available",
                "decision_threshold",
                "external",
            )
        },
        "warnings": warnings,
        "limitations": [
            DISCLAIMER,
            (
                "The model score is not calibrated on this application's own held-out dataset."
                if not model.get("calibrated")
                else "Model calibration does not guarantee accuracy on unseen sources."
            ),
            "Manipulated-image classification is not implemented in this binary detector.",
            "Forensic features and metadata can be altered or degraded by image processing.",
            *(
                ["Uploaded image bytes were sent to Sightengine for analysis."]
                if model.get("external")
                else []
            ),
        ],
    }
