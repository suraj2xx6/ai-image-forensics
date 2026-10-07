from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_version(monkeypatch):
    from app.api import health

    monkeypatch.setattr(
        health,
        "settings",
        replace(health.settings, sightengine_api_user="", sightengine_api_secret=""),
    )
    monkeypatch.setattr(health, "get_nonescape_model", lambda _: None)
    monkeypatch.setattr(health, "get_model", lambda _: None)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert (
            client.get("/api/version").json()["detector"] == "forensic-feature-baseline"
        )
        assert client.get("/").status_code == 200


def test_valid_upload_returns_report(jpeg_bytes, monkeypatch):
    from app.services import detector

    monkeypatch.setattr(
        detector,
        "settings",
        replace(detector.settings, sightengine_api_user="", sightengine_api_secret=""),
    )
    monkeypatch.setattr(
        detector,
        "predict_ai_probability",
        lambda *_: {
            "available": False,
            "probability": None,
            "name": "forensic-feature-baseline",
            "version": "untrained",
            "calibrated": False,
        },
    )
    monkeypatch.setattr(detector, "predict_nonescape", lambda *_: None)
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        report = response.json()
        assert report["classification"]["label"] == "INCONCLUSIVE"
        assert report["file"]["sha256"]
        assert len(report["file"]["dhash"]) == 64
        assert len(report["file"]["phash"]) == 64
        assert report["forensics"]["noise"]
        assert client.get(f"/api/report/{report['analysis_id']}").status_code == 200


def test_invalid_upload_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", files={"file": ("bad.png", b"no", "image/png")}
        )
    assert response.status_code == 400


def test_content_type_mismatch_is_reported(jpeg_bytes, monkeypatch):
    from app.services import detector

    monkeypatch.setattr(
        detector,
        "settings",
        replace(detector.settings, sightengine_api_user="", sightengine_api_secret=""),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", files={"file": ("photo.jpg", jpeg_bytes, "image/png")}
        )
    assert response.status_code == 200
    assert any("Content-Type" in warning for warning in response.json()["warnings"])


def test_request_body_limit_rejects_oversized_multipart(monkeypatch):
    monkeypatch.setattr(app, "limit", 1)
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", files={"file": ("large.jpg", b"x" * 140_000, "image/jpeg")}
        )
    assert response.status_code == 413


def test_missing_upload_is_422():
    with TestClient(app) as client:
        response = client.post("/api/analyze")
    assert response.status_code == 422
