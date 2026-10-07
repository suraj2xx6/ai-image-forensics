"""Export image features to a CSV for inspection or research."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from app.services.image_features import FEATURE_NAMES, extract_features
from scripts.dataset_utils import find_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/features.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "split", "label", *FEATURE_NAMES]
        )
        writer.writeheader()
        for path, split, label in find_samples(args.dataset):
            with Image.open(path) as image:
                image.load()
                features = extract_features(image)
            writer.writerow(
                {"path": str(path), "split": split, "label": label, **features}
            )
    print(f"Wrote features for {args.output}")


if __name__ == "__main__":
    main()
