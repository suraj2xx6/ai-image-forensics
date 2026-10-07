import httpx

from app.ml import sightengine


def test_predict_ai_score_sends_multipart_and_normalizes_response(monkeypatch):
    captured = {}
    request = httpx.Request("POST", "https://api.sightengine.com/1.0/check.json")
    response = httpx.Response(
        200,
        json={"status": "success", "type": {"ai_generated": 0.91}},
        request=request,
    )

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(sightengine.httpx, "post", fake_post)
    result = sightengine.predict_ai_score(
        b"image bytes", "sample.png", "image/png", "test-user", "test-secret"
    )

    assert captured["data"] == {
        "models": "genai",
        "api_user": "test-user",
        "api_secret": "test-secret",
    }
    assert captured["files"]["media"] == ("sample.png", b"image bytes", "image/png")
    assert result["raw_probability"] == 0.91
    assert result["external"] is True
    assert result["calibrated"] is False


def test_predict_ai_score_rejects_invalid_provider_score(monkeypatch):
    request = httpx.Request("POST", "https://api.sightengine.com/1.0/check.json")
    response = httpx.Response(
        200,
        json={"status": "success", "type": {"ai_generated": 1.5}},
        request=request,
    )
    monkeypatch.setattr(sightengine.httpx, "post", lambda *_args, **_kwargs: response)

    try:
        sightengine.predict_ai_score(b"image", "sample.png", "image/png", "u", "s")
    except sightengine.SightengineError as exc:
        assert "invalid AI score" in str(exc)
    else:
        raise AssertionError("Expected invalid provider score to be rejected")
