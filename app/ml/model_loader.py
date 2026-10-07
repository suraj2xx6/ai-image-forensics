"""Cached, replaceable scikit-learn model loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


@lru_cache(maxsize=4)
def load_model(path_string: str, modified_ns: int) -> dict[str, Any] | None:
    """Load a trusted local model once per path/version; never fetch weights."""
    path = Path(path_string)
    if not path.is_file():
        return None
    payload = joblib.load(path)
    if (
        not isinstance(payload, dict)
        or "classifier" not in payload
        or "feature_names" not in payload
    ):
        raise ValueError("Model artifact has an unsupported format.")
    return payload


def get_model(path: Path) -> dict[str, Any] | None:
    """Return a cached artifact and refresh the cache when it changes."""
    if not path.is_file():
        return None
    return load_model(str(path.resolve()), path.stat().st_mtime_ns)
