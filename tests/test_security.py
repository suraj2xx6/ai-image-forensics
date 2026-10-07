import pytest

from app.core.security import InvalidImageError, sanitize_filename, validate_image


def test_valid_jpeg_and_sha256(jpeg_bytes):
    result = validate_image(jpeg_bytes, "photo.jpg", 10000)
    assert result.image_format == "JPEG"
    assert result.mime_type == "image/jpeg"
    assert len(result.sha256) == 64


def test_valid_png(png_bytes):
    result = validate_image(png_bytes, "sample.png", 10000)
    assert result.image_format == "PNG"
    assert result.width == 48


@pytest.mark.parametrize(
    "name", ["payload.svg", "page.html", "photo.jpg.exe", "script.py"]
)
def test_reject_disallowed_extension(jpeg_bytes, name):
    with pytest.raises(InvalidImageError):
        validate_image(jpeg_bytes, name, 10000)


def test_reject_corrupted_image():
    with pytest.raises(InvalidImageError):
        validate_image(b"not an image", "broken.png", 10000)


def test_reject_mime_extension_mismatch(jpeg_bytes):
    result = validate_image(jpeg_bytes, "renamed.png", 10000)
    assert result.extension_mismatch is True


def test_reject_decompression_bomb_dimensions(png_bytes):
    with pytest.raises(InvalidImageError):
        validate_image(png_bytes, "sample.png", 100)


def test_sanitizes_path_and_control_characters():
    assert sanitize_filename("../../a\x00b.png") == "ab.png"
