"""Upload validation and safe image decoding."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

from PIL import Image, ImageFile, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
FORMAT_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
}
FORMAT_EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "TIFF": {".tif", ".tiff"},
}
ImageFile.LOAD_TRUNCATED_IMAGES = False


class InvalidImageError(ValueError):
    """Raised when bytes cannot be safely decoded as a supported image."""


@dataclass
class ValidatedImage:
    image_format: str
    mime_type: str
    width: int
    height: int
    sha256: str
    extension: str
    safe_filename: str
    extension_mismatch: bool


def sanitize_filename(filename: str | None) -> str:
    """Keep only a bounded, display-safe basename."""
    value = unicodedata.normalize("NFKC", filename or "upload")
    value = value.replace("\\", "/").split("/")[-1]
    value = re.sub(r"[\x00-\x1f\x7f]", "", value).strip(" .")
    value = re.sub(r"[^\w. ()-]", "_", value, flags=re.UNICODE)
    return value[:180] or "upload"


def validate_image(
    data: bytes, filename: str | None, max_pixels: int
) -> ValidatedImage:
    """Verify the file by decoding its contents; caller enforces byte limit."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    safe_name = sanitize_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidImageError("Use a JPEG, PNG, WebP, or TIFF filename extension.")

    try:
        with Image.open(BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in FORMAT_MIME:
                raise InvalidImageError("Unsupported image format.")
            width, height = probe.size
            if width < 1 or height < 1 or width * height > max_pixels:
                raise InvalidImageError(
                    "Image dimensions exceed the configured safety limit."
                )
            probe.verify()
        image = Image.open(BytesIO(data))
        image.seek(0)
        image.load()
    except InvalidImageError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImageError("The file is not a valid, complete image.") from exc

    actual_format = (image.format or image_format).upper()
    if actual_format not in FORMAT_MIME:
        raise InvalidImageError("Unsupported image format.")
    image.close()
    return ValidatedImage(
        image_format=actual_format,
        mime_type=FORMAT_MIME[actual_format],
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        extension=extension,
        safe_filename=safe_name,
        extension_mismatch=extension not in FORMAT_EXTENSIONS[actual_format],
    )
