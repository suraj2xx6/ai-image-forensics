"""Measure held-out performance by category and common benign transforms."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.services.image_features import extract_features
from scripts.dataset_utils import EXTENSIONS, dataset_version


def measure(
    model: dict, images: list[Image.Image], labels: list[int], threshold: float
) -> tuple[dict, np.ndarray]:
    """Run the same trained feature pipeline for an image group."""
    matrix = []
    for image in images:
        values = extract_features(image)
        matrix.append([values[name] for name in model["feature_names"]])
    probabilities = model["classifier"].predict_proba(np.asarray(matrix))[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    result = {
        "sample_count": len(labels),
        "class_counts": {
            "real": int(sum(label == 0 for label in labels)),
            "ai": int(sum(label == 1 for label in labels)),
        },
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_ai": float(precision_score(labels, predicted, zero_division=0)),
        "recall_ai": float(recall_score(labels, predicted, zero_division=0)),
        "f1_ai": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if len(set(labels)) == 2
        else None,
    }
    return result, probabilities


def transformed(image: Image.Image) -> dict[str, Image.Image]:
    """Create deterministic, ordinary processing variants for robustness checks."""
    source = image.convert("RGB")
    width, height = source.size
    variants: dict[str, Image.Image] = {}
    for quality, name in ((75, "jpeg_recompressed_q75"), (45, "jpeg_recompressed_q45")):
        buffer = BytesIO()
        source.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            variants[name] = encoded.convert("RGB").copy()

    smaller = source.resize(
        (max(2, round(width * 0.7)), max(2, round(height * 0.7))),
        Image.Resampling.LANCZOS,
    )
    variants["resized_roundtrip_70pct"] = smaller.resize(
        (width, height), Image.Resampling.LANCZOS
    )
    left, top = int(width * 0.05), int(height * 0.05)
    right, bottom = max(left + 1, int(width * 0.95)), max(top + 1, int(height * 0.95))
    variants["center_crop_90pct"] = source.crop((left, top, right, bottom)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    variants["mild_color_adjustment"] = ImageEnhance.Brightness(
        ImageEnhance.Color(source).enhance(1.05)
    ).enhance(1.02)
    variants["mild_blur"] = source.filter(ImageFilter.GaussianBlur(radius=0.5))
    buffer = BytesIO()
    source.save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as stripped:
        variants["metadata_stripped_reencode"] = stripped.convert("RGB").copy()

    social_width = min(width, 2048)
    social_height = max(2, round(height * social_width / max(width, 1)))
    social = source.resize((social_width, social_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    social.save(buffer, format="JPEG", quality=65, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        variants["social_media_style_compression"] = encoded.convert("RGB").copy()
    return variants


def load_group(folder: Path) -> tuple[list[Image.Image], list[int]]:
    """Decode raster samples from a real/ai category folder."""
    images: list[Image.Image] = []
    labels: list[int] = []
    for label_name, label in (("real", 0), ("ai", 1)):
        class_folder = folder / label_name
        if not class_folder.exists():
            continue
        for path in sorted(class_folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    images.append(image.copy())
                    labels.append(label)
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Could not decode benchmark image: {path}") from exc
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Root containing test/<category>/{real,ai}",
    )
    parser.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the model threshold selected on validation",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Generate benign transformations of test/real and test/ai samples",
    )
    parser.add_argument(
        "--unseen-generator", help="Optional folder name under test/unseen_generators"
    )
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    artifact = joblib.load(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(artifact.get("decision_threshold", 0.5))
    )
    test_root = args.dataset / "test"
    if not test_root.is_dir():
        raise SystemExit(f"Missing held-out benchmark directory: {test_root}")
    results: dict[str, dict] = {}

    for category in sorted(
        path
        for path in test_root.iterdir()
        if path.is_dir() and path.name != "unseen_generators"
    ):
        images, labels = load_group(category)
        if images:
            results[category.name], _ = measure(artifact, images, labels, threshold)

    if args.robustness:
        originals, labels = load_group(test_root)
        if not originals:
            raise SystemExit("--robustness needs test/real and test/ai source folders.")
        variant_groups: dict[str, list[Image.Image]] = {}
        for source in originals:
            for name, variant in transformed(source).items():
                variant_groups.setdefault(name, []).append(variant)
        for name, images in variant_groups.items():
            results[name], _ = measure(artifact, images, labels, threshold)
        if "screenshots" not in results:
            results["screenshots"] = {
                "sample_count": 0,
                "note": "Provide genuine screenshot captures in test/screenshots/{real,ai}; screenshots are not simulated by this script.",
            }

    if args.unseen_generator:
        unseen_dir = test_root / "unseen_generators" / args.unseen_generator
        if not unseen_dir.is_dir():
            raise SystemExit(f"Unseen-generator folder not found: {unseen_dir}")
        paths = [
            path
            for path in sorted(unseen_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]
        if not paths:
            raise SystemExit("Unseen-generator directory contains no supported images.")
        images = []
        for path in paths:
            with Image.open(path) as image:
                image.load()
                images.append(image.copy())
        _, probabilities = measure(artifact, images, [1] * len(images), threshold)
        results[f"unseen_generator:{args.unseen_generator}"] = {
            "sample_count": len(images),
            "ai_sensitivity": float(np.mean(probabilities >= threshold)),
            "mean_ai_probability": float(probabilities.mean()),
            "note": "AI-only unseen-source sensitivity; include matched real controls before treating as a comparable classifier metric.",
        }

    output = {
        "model_version": artifact.get("version"),
        "dataset_version": dataset_version(args.dataset),
        "test_split": "test",
        "threshold": threshold,
        "threshold_source": "cli_override"
        if args.threshold is not None
        else "validation_selected_model",
        "categories": results,
        "note": "All generated variants derive from held-out test sources. Screenshot performance requires supplied screenshot captures.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
