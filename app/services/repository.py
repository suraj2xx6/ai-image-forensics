"""SQLite result persistence; image bytes are never stored."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.core.config import settings


def _database_path() -> Path:
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        raise ValueError(
            "This build supports SQLite; the repository module is the replacement point for a PostgreSQL adapter."
        )
    raw_path = url[len("sqlite:///") :]
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_database() -> None:
    """Create the analysis table if needed."""
    with _connect() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                classification TEXT NOT NULL,
                ai_probability REAL,
                confidence REAL,
                model_version TEXT NOT NULL,
                report_json TEXT NOT NULL
            )"""
        )


def save_report(report: dict) -> None:
    """Persist analysis results only."""
    classification = report["classification"]
    with _connect() as connection:
        connection.execute(
            "INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report["analysis_id"],
                report["timestamp"],
                report["filename"],
                report["file"]["sha256"],
                classification["label"],
                classification["ai_probability"],
                classification["confidence"],
                report["model"].get("version") or "unknown",
                json.dumps(report, ensure_ascii=False),
            ),
        )


def fetch_report(analysis_id: str) -> dict | None:
    """Read a stored analysis report by ID."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT report_json FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None
