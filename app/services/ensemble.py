"""Conservative classification policy for available model evidence."""

from __future__ import annotations


def classify(model: dict, metadata_findings: list[dict]) -> dict:
    """Use calibrated bands or report the installed model's likely top class."""
    calibrated = bool(model.get("calibrated", False))
    probability = model.get("probability") if calibrated else None
    raw_probability = model.get("raw_probability", model.get("probability"))
    decision_threshold = float(model.get("decision_threshold", 0.5))
    ai_metadata = any(
        item.get("severity") == "HIGH" and item.get("score_contribution", 0) >= 0.3
        for item in metadata_findings
    )
    if probability is None and raw_probability is not None:
        label = (
            "LIKELY_AI_GENERATED"
            if raw_probability >= decision_threshold
            else "LIKELY_AUTHENTIC"
        )
        confidence = None
        reason = "uncalibrated_model_top_class"
    elif probability is None:
        label = "INCONCLUSIVE"
        confidence = None
        reason = "calibrated_model_unavailable"
    elif ai_metadata and probability < decision_threshold:
        label = "INCONCLUSIVE"
        confidence = None
        reason = "metadata_model_disagreement"
    elif probability >= decision_threshold and probability >= max(
        0.95, decision_threshold + 0.20
    ):
        label = "AI_GENERATED"
        confidence = probability
        reason = "calibrated_model_strong_ai_signal"
    elif probability >= decision_threshold and probability >= max(
        0.75, decision_threshold
    ):
        label = "LIKELY_AI_GENERATED"
        confidence = probability
        reason = "calibrated_model_ai_signal"
    elif probability < decision_threshold and probability <= min(
        0.05, decision_threshold - 0.20
    ):
        label = "AUTHENTIC"
        confidence = 1.0 - probability
        reason = "calibrated_model_strong_real_signal"
    elif probability < decision_threshold and probability <= min(
        0.25, decision_threshold
    ):
        label = "LIKELY_AUTHENTIC"
        confidence = 1.0 - probability
        reason = "calibrated_model_real_signal"
    else:
        label = "INCONCLUSIVE"
        confidence = None
        reason = "calibrated_model_inside_inconclusive_band"
    return {
        "label": label,
        "ai_probability": probability,
        "raw_model_score": raw_probability,
        "confidence": confidence,
        "risk_level": "HIGH"
        if label in {"AI_GENERATED", "LIKELY_AI_GENERATED"}
        else ("LOW" if label in {"AUTHENTIC", "LIKELY_AUTHENTIC"} else "UNKNOWN"),
        "score_source": "calibrated_model"
        if calibrated
        else (
            "uncalibrated_raw_score" if raw_probability is not None else "unavailable"
        ),
        "decision_reason": reason,
        "decision_threshold": decision_threshold,
        "manipulation_assessment": "not_supported_by_binary_detector",
    }
