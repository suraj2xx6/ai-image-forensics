"""Filename consistency checks; names are weak evidence by design."""

from __future__ import annotations

import re
from pathlib import PurePath

GENERIC_NAMES = {"image", "generated", "output", "render", "img", "screenshot", "photo"}
TOOL_PATTERN = re.compile(
    r"(?:midjourney|stable.?diffusion|dall.?e|firefly|comfyui|flux)", re.IGNORECASE
)
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.IGNORECASE)


def analyze_filename(
    filename: str, image_format: str, extension_mismatch: bool
) -> dict:
    """Compare filename clues with decoded content without treating names as proof."""
    stem = PurePath(filename).stem
    folded = stem.casefold().strip()
    findings: list[dict] = []
    if extension_mismatch:
        findings.append(
            {
                "category": "FILENAME",
                "severity": "MEDIUM",
                "finding": "Filename extension does not match the decoded image format",
                "evidence": f"Extension {PurePath(filename).suffix.lower()}; content {image_format}",
                "score_contribution": 0.0,
                "explanation": "This is a file-label inconsistency and does not indicate image origin.",
            }
        )
    if folded in GENERIC_NAMES:
        findings.append(
            {
                "category": "FILENAME",
                "severity": "LOW",
                "finding": "Generic filename",
                "evidence": stem,
                "score_contribution": 0.01,
                "explanation": "Generic names occur for camera images and exported files.",
            }
        )
    if TOOL_PATTERN.search(stem):
        findings.append(
            {
                "category": "FILENAME",
                "severity": "LOW",
                "finding": "Filename contains a known image-tool name",
                "evidence": stem,
                "score_contribution": 0.01,
                "explanation": "A name can be changed and is not proof that the tool created the image.",
            }
        )
    if UUID_PATTERN.match(folded) or re.fullmatch(r"[0-9a-f]{24,}", folded):
        findings.append(
            {
                "category": "FILENAME",
                "severity": "LOW",
                "finding": "Filename resembles an opaque generated identifier",
                "evidence": stem[:80],
                "score_contribution": 0.01,
                "explanation": "Opaque identifiers are common in downloads, exports and camera workflows.",
            }
        )
    if len(PurePath(filename).suffixes) > 1:
        findings.append(
            {
                "category": "FILENAME",
                "severity": "LOW",
                "finding": "Multiple filename extensions",
                "evidence": filename,
                "score_contribution": 0.0,
                "explanation": "Multiple suffixes can be an accidental rename and do not establish image origin.",
            }
        )
    status = (
        "MISMATCH"
        if extension_mismatch
        else ("SUSPICIOUS" if findings else "CONSISTENT")
    )
    risk = "HIGH" if extension_mismatch else "LOW"
    return {
        "status": status,
        "risk": risk,
        "findings": findings,
        "note": "Filename patterns are weak clues and do not establish image origin.",
    }
