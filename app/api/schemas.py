"""Pydantic response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DetectionReport(BaseModel):
    analysis_id: str
    timestamp: str
    filename: str
    file: dict[str, Any]
    classification: dict[str, Any]
    metadata: dict[str, Any]
    filename_analysis: dict[str, Any]
    forensics: dict[str, Any]
    evidence: list[dict[str, Any]]
    model: dict[str, Any]
    warnings: list[str]
    limitations: list[str]
