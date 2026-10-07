"""Train a feature-based AI-vs-real baseline on pre-separated source groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.calibration import CalibratedBinaryClassifier
from app.services.image_features import FEATURE_NAMES, extract_features
from scripts.dataset_utils import (
    assert_no_cross_split_duplicates,
    dataset_version,
    find_samples,
)


def load_matrix(rows):
    features, labels, splits = [], [], []
    for path, split, label in rows:
        with Image.open(path) as image:
            image.load()
            values = extract_features(image)
        features.append([values[name] for name in FEATURE_NAMES])
        labels.append(label)
        splits.append(split)
    return np.asarray(features), np.asarray(labels), np.asarray(splits)


def select_decision_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float | None]:
    """Choose the validation threshold with the best balanced accuracy."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.5, None

    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = probabilities >= threshold
        sensitivity = float(np.mean(predicted[labels == 1]))
        specificity = float(np.mean(~predicted[labels == 0]))
        score = (sensitivity + specificity) / 2
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_score = score
    return round(best_threshold, 4), best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset root with train/real, train/ai, etc.",
    )
    parser.add_argument("--output", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--model-version", default="feature-baseline-1.0")
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    assert_no_cross_split_duplicates(args.dataset)
    rows = find_samples(args.dataset)
    matrix, labels, splits = load_matrix(rows)
    train_mask, validation_mask = splits == "train", splits == "validation"
    if len(set(labels[train_mask])) != 2:
        raise SystemExit("Training data needs both train/real and train/ai examples.")
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.max_iter, class_weight="balanced", random_state=17
        ),
    )
    base.fit(matrix[train_mask], labels[train_mask])
    classifier = base
    calibrated = False
    validation_labels = labels[validation_mask]
    class_counts = {label: int(np.sum(validation_labels == label)) for label in (0, 1)}
    if len(validation_labels) >= 20 and min(class_counts.values()) >= 5:
        scores = base.decision_function(matrix[validation_mask]).reshape(-1, 1)
        calibrator = LogisticRegression(C=1_000_000, max_iter=1000, random_state=17)
        calibrator.fit(scores, validation_labels)
        classifier = CalibratedBinaryClassifier(base, calibrator)
        calibrated = True

    if len(validation_labels):
        validation_probabilities = classifier.predict_proba(matrix[validation_mask])[
            :, 1
        ]
        decision_threshold, validation_balanced_accuracy = select_decision_threshold(
            validation_labels, validation_probabilities
        )
    else:
        decision_threshold, validation_balanced_accuracy = 0.5, None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "classifier": classifier,
        "feature_names": FEATURE_NAMES,
        "name": "forensic-feature-baseline",
        "version": args.model_version,
        "dataset_version": dataset_version(args.dataset),
        "calibrated": calibrated,
        "decision_threshold": decision_threshold,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "threshold_metric": "balanced_accuracy",
        "training_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
    }
    joblib.dump(artifact, args.output)
    print(
        f"Saved {args.output} ({artifact['training_samples']} train, {artifact['validation_samples']} validation; calibrated={calibrated})"
    )
    print("Test data was not used for fitting or calibration.")


if __name__ == "__main__":
    main()
