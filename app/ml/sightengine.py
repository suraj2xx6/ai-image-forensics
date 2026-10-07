"""Sightengine AI-generated image detection API client."""

from __future__ import annotations

import math

import httpx


class SightengineError(RuntimeError):
    """Raised when Sightengine cannot return an AI image score."""


def predict_ai_score(
    image_bytes: bytes,
    filename: str,
    mime_type: str,
    api_user: str,
    api_secret: str,
) -> dict:
    """Upload an image to Sightengine's genai model and normalize its result."""
    try:
        response = httpx.post(
            "https://api.sightengine.com/1.0/check.json",
            data={
                "models": "genai",
                "api_user": api_user,
                "api_secret": api_secret,
            },
            files={"media": (filename, image_bytes, mime_type)},
            timeout=httpx.Timeout(45.0, connect=10.0),
        )
    except httpx.RequestError as exc:
        raise SightengineError("Sightengine could not be reached.") from exc

    if response.is_error:
        raise SightengineError(f"Sightengine returned HTTP {response.status_code}.")
    try:
        payload = response.json()
        score = float(payload["type"]["ai_generated"])
    except (ValueError, TypeError, KeyError) as exc:
        raise SightengineError("Sightengine returned an invalid response.") from exc

    if (
        payload.get("status") != "success"
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise SightengineError("Sightengine returned an invalid AI score.")

    return {
        "available": True,
        "probability": None,
        "raw_probability": score,
        "name": "Sightengine genai",
        "version": "provider-managed",
        "calibrated": False,
        "dataset_version": "Sightengine hosted model",
        "decision_threshold": 0.5,
        "external": True,
    }
