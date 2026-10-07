from PIL import Image

from app.services.metadata import extract_metadata


def test_missing_metadata_is_reported_as_weak_evidence(png_bytes):
    image = Image.open(__import__("io").BytesIO(png_bytes))
    metadata = extract_metadata(image, "sample.png", "image/png", len(png_bytes))
    assert metadata["file"]["width"] == 48
    assert metadata["exif"] == {}
    assert metadata["findings"][0]["severity"] == "LOW"
    assert metadata["findings"][0]["score_contribution"] <= 0.02


def test_software_metadata_does_not_prove_ai(png_bytes):
    image = Image.open(__import__("io").BytesIO(png_bytes))
    image.info["xmp"] = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description CreatorTool="Adobe Photoshop" /></rdf:RDF></x:xmpmeta>'
    )
    result = extract_metadata(image, "edited.png", "image/png", len(png_bytes))
    assert not any(item["severity"] == "HIGH" for item in result["findings"])


def test_gps_coordinates_not_exposed(png_bytes):
    image = Image.open(__import__("io").BytesIO(png_bytes))
    result = extract_metadata(image, "sample.png", "image/png", len(png_bytes))
    assert "GPSInfo" not in result["exif"]


def test_exiftool_absence_falls_back_to_pillow(monkeypatch, png_bytes):
    from app.services import metadata as service

    monkeypatch.setattr(service.shutil, "which", lambda _: None)
    image = Image.open(__import__("io").BytesIO(png_bytes))
    result = extract_metadata(
        image, "sample.png", "image/png", len(png_bytes), png_bytes
    )
    assert result["exiftool"] == {"available": False, "used": False}
