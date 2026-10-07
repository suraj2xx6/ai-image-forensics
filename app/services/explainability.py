"""Human-readable evidence descriptions."""

from __future__ import annotations


def build_evidence(
    model: dict, metadata: dict, filename: dict, validation_warnings: list[str]
) -> list[dict]:
    evidence: list[dict] = []
    score = model.get("probability")
    if score is None:
        score = model.get("raw_probability")
    if score is None:
        evidence.append(
            {
                "category": "ML",
                "severity": "LOW",
                "finding": "No image-origin classifier is available",
                "explanation": "The forensic measurements are available, but no local or hosted classifier returned a score for this image.",
            }
        )
    else:
        p = score
        severity = (
            "HIGH" if model.get("calibrated") and (p >= 0.75 or p <= 0.25) else "LOW"
        )
        direction = "AI-generated" if p >= 0.5 else "genuine"
        score_label = (
            "calibrated AI probability"
            if model.get("calibrated")
            else "uncalibrated raw model score"
        )
        evidence.append(
            {
                "category": "ML",
                "severity": severity,
                "finding": f"Classifier result leans toward {direction} imagery",
                "explanation": f"The {model.get('name')} detector returned an {score_label} of {p:.3f}. This score is not a validated confidence measure unless calibrated; results depend on the provider's training data.",
            }
        )
    evidence.extend(metadata.get("findings", []))
    evidence.extend(filename.get("findings", []))
    for warning in validation_warnings:
        evidence.append(
            {
                "category": "FILE",
                "severity": "MEDIUM",
                "finding": warning,
                "explanation": "The decoded content was analyzed; the inconsistency is included for review.",
            }
        )
    return evidence
