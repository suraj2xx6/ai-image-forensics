from app.services.filename import analyze_filename


def test_extension_mismatch_is_reported():
    result = analyze_filename("photo.png", "JPEG", True)
    assert result["status"] == "MISMATCH"
    assert result["findings"]


def test_name_patterns_are_weak_evidence():
    result = analyze_filename("midjourney-render.png", "PNG", False)
    assert result["status"] == "SUSPICIOUS"
    assert "weak" in result["note"]


def test_normal_name_consistent():
    assert analyze_filename("family-trip.jpg", "JPEG", False)["status"] == "CONSISTENT"
