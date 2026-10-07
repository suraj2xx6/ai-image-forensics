"""Health and version route helpers."""

from fastapi import APIRouter

from app.core.config import settings
from app.ml.model_loader import get_model
from app.ml.nonescape import get_model as get_nonescape_model

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    """Basic liveness and model readiness status."""
    try:
        model_ready = (
            settings.sightengine_enabled
            or get_nonescape_model(settings.nonescape_model_path) is not None
            or get_model(settings.model_path) is not None
        )
        model_error = None
    except Exception as exc:  # noqa: BLE001 - health must remain available for corrupt model artifacts.
        model_ready = False
        model_error = type(exc).__name__
    return {"status": "ok", "model_ready": model_ready, "model_error": model_error}


@router.get("/api/version")
def version() -> dict:
    """Return application and detector version information."""
    if settings.sightengine_enabled:
        return {
            "app_version": settings.app_version,
            "detector": "sightengine-genai",
            "model_version": "provider-managed",
            "model_available": True,
            "sightengine_enabled": True,
            "sightengine_setup_incomplete": False,
        }
    nonescape_model = None
    try:
        nonescape_model = get_nonescape_model(settings.nonescape_model_path)
        model = get_model(settings.model_path) if nonescape_model is None else None
    except Exception:  # noqa: BLE001 - report the service version even if a model artifact is corrupt.
        model = None
    return {
        "app_version": settings.app_version,
        "detector": "nonescape-mini"
        if nonescape_model is not None
        else "forensic-feature-baseline",
        "model_version": "nonescape-mini-v0"
        if nonescape_model is not None
        else (model.get("version") if model else "untrained"),
        "model_available": nonescape_model is not None or model is not None,
        "sightengine_enabled": False,
        "sightengine_setup_incomplete": settings.sightengine_partially_configured,
    }
