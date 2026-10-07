"""Stable feature ordering for baseline model inputs."""

from __future__ import annotations

import numpy as np


def vectorize(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    """Convert named feature values into the model's expected matrix shape."""
    return np.asarray(
        [[float(features.get(name, 0.0)) for name in feature_names]], dtype=np.float64
    )
