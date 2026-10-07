"""Local inference adapter for the Nonescape Mini image classifier."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import models
from torchvision.transforms import v2 as transforms


@lru_cache(maxsize=1)
def _preprocessing_pipeline() -> Any:
    """Build the published transform chain once."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.JPEG(quality=100),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _preprocess(image: Image.Image) -> torch.Tensor:
    """Apply the preprocessing published with the Nonescape model."""
    return _preprocessing_pipeline()(image.convert("RGB"))


@lru_cache(maxsize=2)
def _load_model(path_string: str, modified_ns: int) -> Any:
    """Load safe local weights once, refreshing when the file changes."""
    from torch import nn

    class NonescapeMini(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.efficientnet_v2_s(
                weights=None, num_classes=1024, dropout=0.2
            )
            self.head = nn.Linear(1024, 2)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            embedding = self.backbone(image)
            return torch.softmax(self.head(embedding), dim=-1)

    model = NonescapeMini()
    model.load_state_dict(load_file(path_string, device="cpu"))
    model.eval()
    return model


def get_model(path: Path) -> Any | None:
    """Return a cached local model or None if its weights are not installed."""
    if not path.is_file():
        return None
    return _load_model(str(path.resolve()), path.stat().st_mtime_ns)


def predict(image: Image.Image, path: Path) -> dict[str, Any] | None:
    """Return the model's AI-class score, explicitly marked uncalibrated."""
    model = get_model(path)
    if model is None:
        return None

    with torch.inference_mode():
        scores = model(_preprocess(image).unsqueeze(0))[0].cpu().numpy()
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ValueError("Nonescape Mini returned invalid class scores.")

    # The official inference example maps index 0 to authentic and index 1 to AI.
    return {
        "available": True,
        "probability": None,
        "raw_probability": float(scores[1]),
        "name": "Nonescape Mini",
        "version": "nonescape-mini-v0",
        "calibrated": False,
        "dataset_version": "third-party model; not independently evaluated here",
        "decision_threshold": 0.5,
    }
