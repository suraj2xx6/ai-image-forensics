"""Image analysis and report retrieval endpoints."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.schemas import DetectionReport
from app.core.config import settings
from app.core.security import InvalidImageError, validate_image
from app.services.detector import analyze_image
from app.services.repository import fetch_report, save_report

router = APIRouter(prefix="/api")
logger = logging.getLogger("image_forensics.api")
_analysis_slots = asyncio.Semaphore(1)
_UPLOAD_FILE = File(...)


@router.post("/analyze", response_model=DetectionReport)
async def analyze(request: Request, file: UploadFile = _UPLOAD_FILE) -> dict:
    """Validate and analyze one image; uploaded bytes are never persisted."""
    started = time.perf_counter()
    request_id = getattr(request.state, "request_id", "unknown")
    try:
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
            )
        try:
            validated = validate_image(data, file.filename, settings.max_image_pixels)
        except InvalidImageError as exc:
            status = 415 if str(exc).startswith(("Use a", "Unsupported image")) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        async with _analysis_slots:
            report = await run_in_threadpool(
                analyze_image, data, validated, file.content_type
            )
        await run_in_threadpool(save_report, report)
        logger.info(
            "analysis complete",
            extra={
                "request_id": request_id,
                "analysis_id": report["analysis_id"],
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "status_code": 200,
            },
        )
        return report
    except HTTPException as exc:
        logger.info(
            "request rejected",
            extra={
                "request_id": request_id,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "status_code": exc.status_code,
            },
        )
        raise
    except Exception as exc:
        logger.exception(
            "analysis failed",
            extra={
                "request_id": request_id,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "status_code": 500,
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Check server logs using the request ID.",
        ) from exc
    finally:
        await file.close()


@router.get("/report/{analysis_id}", response_model=DetectionReport)
async def report(analysis_id: str) -> dict:
    """Fetch a stored JSON report without access to the original upload."""
    result = await run_in_threadpool(fetch_report, analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis report not found.")
    return result
