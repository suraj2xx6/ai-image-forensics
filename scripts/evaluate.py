"""Evaluate a saved model against the isolated test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.services.image_features import FEATURE_NAMES, extract_features
from scripts.dataset_utils import (
    assert_no_cross_split_duplicates,
    dataset_version,
    find_samples,
)


def metrics_at_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_ai": float(precision_score(labels, predicted, zero_division=0)),
        "recall_ai": float(recall_score(labels, predicted, zero_division=0)),
        "f1_ai": float(f1_score(labels, predicted, zero_division=0)),
        "precision_real": float(tn / (tn + fn)) if tn + fn else 0.0,
        "recall_real": float(tn / (tn + fp)) if tn + fp else None,
        "f1_real": float(2 * tn / (2 * tn + fp + fn)) if 2 * tn + fp + fn else 0.0,
        "specificity_real": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def save_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    """Create a dependency-light confusion matrix image."""
    image = Image.new("RGB", (720, 430), "#0c1520")
    draw = ImageDraw.Draw(image)
    draw.text((32, 25), "Held-out test confusion matrix", fill="#edf4fb")
    draw.text((260, 90), "Predicted real", fill="#b9c8d8")
    draw.text((475, 90), "Predicted AI", fill="#b9c8d8")
    draw.text((65, 165), "Actual real", fill="#b9c8d8")
    draw.text((65, 270), "Actual AI", fill="#b9c8d8")
    for row in range(2):
        for col in range(2):
            x, y = 230 + col * 200, 140 + row * 100
            draw.rounded_rectangle(
                (x, y, x + 160, y + 78),
                radius=12,
                fill="#132538",
                outline="#37617f",
                width=2,
            )
            draw.text((x + 64, y + 25), str(matrix[row][col]), fill="#62e0bd")
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation"))
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the model threshold selected on validation",
    )
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    assert_no_cross_split_duplicates(args.dataset)
    artifact = joblib.load(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(artifact.get("decision_threshold", 0.5))
    )
    rows = [row for row in find_samples(args.dataset, ("test",)) if row[1] == "test"]
    if not rows:
        raise SystemExit("No held-out test images found.")
    matrix, labels = [], []
    for path, _, label in rows:
        with Image.open(path) as image:
            image.load()
            values = extract_features(image)
        matrix.append(
            [values[name] for name in artifact.get("feature_names", FEATURE_NAMES)]
        )
        labels.append(label)
    probabilities = artifact["classifier"].predict_proba(np.asarray(matrix))[:, 1]
    selected = metrics_at_threshold(np.asarray(labels), probabilities, threshold)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = None
    rng = np.random.default_rng(23)
    boot = [
        accuracy_score(
            np.asarray(labels)[indices],
            (probabilities[indices] >= threshold).astype(int),
        )
        for indices in (rng.integers(0, len(labels), len(labels)) for _ in range(2000))
    ]
    selected["accuracy_95_percentile_interval"] = [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]
    selected["roc_auc"] = auc
    selected["sample_count"] = len(labels)
    selected["per_class"] = {
        "real": {
            "count": int(np.sum(np.asarray(labels) == 0)),
            "precision": selected["precision_real"],
            "recall": selected["recall_real"],
            "f1": selected["f1_real"],
            "specificity": selected["specificity_real"],
        },
        "ai_generated": {
            "count": int(np.sum(np.asarray(labels) == 1)),
            "precision": selected["precision_ai"],
            "recall": selected["recall_ai"],
            "f1": selected["f1_ai"],
        },
    }
    thresholds = [
        metrics_at_threshold(np.asarray(labels), probabilities, value)
        for value in (0.25, 0.35, 0.5, 0.65, 0.75)
    ]
    result = {
        "model_version": artifact.get("version", "unknown"),
        "dataset_version": dataset_version(args.dataset),
        "test_split": "test",
        "split_leakage_check": "passed_exact_and_phash_distance_4",
        "selected_metrics": selected,
        "threshold_metrics": thresholds,
        "decision_threshold": threshold,
        "decision_threshold_source": "cli_override"
        if args.threshold is not None
        else "validation_selected_model",
        "advertise_90_percent_accuracy": bool(selected["accuracy"] >= 0.9),
        "note": "Metrics describe only this held-out test set. Do not generalize beyond its sources and transformations.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(
        selected["confusion_matrix"], args.output / "confusion_matrix.png"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
