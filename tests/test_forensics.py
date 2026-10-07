import math

from PIL import Image

from app.services.forensics import summarize_forensics
from app.services.image_features import FEATURE_NAMES, extract_features
from app.utils.hashing import difference_hash, perceptual_hash


def test_features_are_finite_and_named(png_bytes):
    import io

    with Image.open(io.BytesIO(png_bytes)) as image:
        features = extract_features(image)
    assert set(features) == set(FEATURE_NAMES)
    assert all(
        math.isfinite(value) and abs(value) < 1e10 for value in features.values()
    )
    assert features["residual_std"] >= 0


def test_hashes_are_stable(png_bytes):
    import io

    with Image.open(io.BytesIO(png_bytes)) as image:
        assert len(difference_hash(image)) == 64
        assert len(perceptual_hash(image)) == 64


def test_jpeg_features_are_labeled_as_measurements(png_bytes):
    import io

    with Image.open(io.BytesIO(png_bytes)) as image:
        features = extract_features(image)
    result = summarize_forensics(features, "PNG")
    assert result["jpeg"]["quantization_values"] == 0
    assert "not reliable" in result["jpeg"]["note"]
