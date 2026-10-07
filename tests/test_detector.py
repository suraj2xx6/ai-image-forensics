from app.ml.inference import predict_ai_probability
from app.ml.preprocessing import vectorize
from app.services.ensemble import classify
from app.services.image_features import FEATURE_NAMES
from scripts.train import select_decision_threshold


def test_no_model_means_inconclusive():
    result = classify({"probability": None}, [])
    assert result["label"] == "INCONCLUSIVE"
    assert result["confidence"] is None


def test_classifier_only_uses_model_output():
    result = classify({"probability": 0.82, "calibrated": True}, [])
    assert result["label"] == "LIKELY_AI_GENERATED"
    assert result["ai_probability"] == 0.82


def test_uncalibrated_model_reports_its_top_class_without_confidence():
    result = classify(
        {"probability": 0.99, "raw_probability": 0.99, "calibrated": False}, []
    )
    assert result["label"] == "LIKELY_AI_GENERATED"
    assert result["confidence"] is None
    assert result["ai_probability"] is None
    assert result["raw_model_score"] == 0.99


def test_ai_metadata_disagreement_keeps_result_inconclusive():
    metadata = [{"severity": "HIGH", "score_contribution": 0.35}]
    result = classify({"probability": 0.05, "calibrated": True}, metadata)
    assert result["label"] == "INCONCLUSIVE"
    assert result["decision_reason"] == "metadata_model_disagreement"


def test_validation_threshold_selects_balanced_separation():
    threshold, balanced_accuracy = select_decision_threshold(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    )
    assert threshold == 0.5
    assert balanced_accuracy == 1.0


def test_inference_returns_raw_and_calibrated_scores_separately(monkeypatch, tmp_path):
    class RawClassifier:
        classes_ = (0, 1)

        @staticmethod
        def predict_proba(_):
            return [[0.8, 0.2]]

    class CalibratedClassifier:
        classes_ = (0, 1)
        classifier = RawClassifier()

        @staticmethod
        def predict_proba(_):
            return [[0.4, 0.6]]

    artifact = {
        "classifier": CalibratedClassifier(),
        "feature_names": ["sample"],
        "calibrated": True,
    }
    monkeypatch.setattr("app.ml.inference.get_model", lambda _: artifact)
    result = predict_ai_probability({"sample": 1.0}, tmp_path / "model.joblib")
    assert result["probability"] == 0.6
    assert result["raw_probability"] == 0.2


def test_feature_vector_has_expected_order():
    result = vectorize({FEATURE_NAMES[0]: 4.0}, FEATURE_NAMES)
    assert result.shape == (1, len(FEATURE_NAMES))
    assert result[0, 0] == 4.0
    assert result[0, 1] == 0.0
