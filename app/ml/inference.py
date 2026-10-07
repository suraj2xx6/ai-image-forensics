"""Baseline classifier inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ml.model_loader import get_model
from app.ml.preprocessing import vectorize


def predict_ai_probability(
    features: dict[str, float], model_path: Path
) -> dict[str, Any]:
    """Predict AI probability if a trained artifact is installed."""
    artifact = get_model(model_path)
    if artifact is None:
        return {
            "available": False,
            "probability": None,
            "name": "forensic-feature-baseline",
            "version": "untrained",
            "calibrated": False,
        }
    classifier = artifact["classifier"]
    vector = vectorize(features, artifact["feature_names"])
    probabilities = classifier.predict_proba(vector)[0]
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("Model does not contain the required AI class label 1.")
    probability = float(probabilities[classes.index(1)])
    raw_probability = probability
    if artifact.get("calibrated"):
        raw_classifier = getattr(classifier, "classifier", None)
        if raw_classifier is not None:
            raw_probabilities = raw_classifier.predict_proba(vector)[0]
            raw_classes = list(raw_classifier.classes_)
            if 1 in raw_classes:
                raw_probability = float(raw_probabilities[raw_classes.index(1)])
    return {
        "available": True,
        "probability": min(1.0, max(0.0, probability)),
        "raw_probability": min(1.0, max(0.0, raw_probability)),
        "name": artifact.get("name", "forensic-feature-baseline"),
        "version": artifact.get("version", "unknown"),
        "calibrated": bool(artifact.get("calibrated", False)),
        "dataset_version": artifact.get("dataset_version", "unspecified"),
        "decision_threshold": float(artifact.get("decision_threshold", 0.5)),
    }
