"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings with conservative defaults."""

    max_upload_mb: int = 15
    max_image_pixels: int = 20_000_000
    model_path: Path = Path("models/baseline.joblib")
    nonescape_model_path: Path = Path("models/nonescape-mini-v0.safetensors")
    sightengine_api_user: str = ""
    sightengine_api_secret: str = ""
    database_url: str = "sqlite:///reports/analyses.sqlite3"
    log_level: str = "INFO"
    app_version: str = "0.1.0"
    max_inference_dimension: int = 1024

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def sightengine_enabled(self) -> bool:
        return bool(self.sightengine_api_user and self.sightengine_api_secret)

    @property
    def sightengine_partially_configured(self) -> bool:
        return bool(self.sightengine_api_user) != bool(self.sightengine_api_secret)


def get_settings() -> Settings:
    """Read settings from environment variables."""
    return Settings(
        max_upload_mb=max(1, int(os.getenv("MAX_UPLOAD_MB", "15"))),
        max_image_pixels=max(1_000_000, int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))),
        model_path=Path(os.getenv("MODEL_PATH", "models/baseline.joblib")),
        nonescape_model_path=Path(
            os.getenv("NONESCAPE_MODEL_PATH", "models/nonescape-mini-v0.safetensors")
        ),
        sightengine_api_user=os.getenv("SIGHTENGINE_API_USER", "").strip(),
        sightengine_api_secret=os.getenv("SIGHTENGINE_API_SECRET", "").strip(),
        database_url=os.getenv("DATABASE_URL", "sqlite:///reports/analyses.sqlite3"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
    )


settings = get_settings()
