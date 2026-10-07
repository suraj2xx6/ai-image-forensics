"""Create train/validation/test folders without separating near duplicates."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.utils.hashing import mean_rgb, perceptual_hash
from scripts.dataset_utils import EXTENSIONS

SPLIT_NAMES = ("train", "validation", "test")


def grouped_samples(
    source: Path, phash_distance: int = 4
) -> dict[int, list[list[Path]]]:
    """Cluster exact and close perceptual duplicates within each class."""
    paths_by_label = {
        label: sorted(
            path
            for path in (source / label).rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        for label in ("real", "ai")
    }
    if not all(paths_by_label.values()):
        raise ValueError("Input requires non-empty real and ai folders.")

    all_paths = [
        (path, label) for label, paths in paths_by_label.items() for path in paths
    ]
    parent = list(range(len(all_paths)))
    rank = [0] * len(all_paths)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    exact_buckets: dict[str, list[int]] = defaultdict(list)
    band_buckets: list[dict[str, list[int]]] = [
        defaultdict(list) for _ in range(phash_distance + 1)
    ]
    fingerprints: list[tuple[str, str, tuple[float, float, float]]] = []
    edges = [
        round(index * 64 / (phash_distance + 1)) for index in range(phash_distance + 2)
    ]
    for index, (path, label) in enumerate(all_paths):
        exact = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with Image.open(path) as image:
                image.load()
                phash = perceptual_hash(image)
                color = mean_rgb(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not decode dataset image: {path}") from exc
        candidate_indices = set(exact_buckets[exact])
        for band, buckets in enumerate(band_buckets):
            candidate_indices.update(buckets[phash[edges[band] : edges[band + 1]]])
        for prior in candidate_indices:
            _, prior_label = all_paths[prior]
            prior_bytes_hash, prior_phash, prior_color = fingerprints[prior]
            color_distance = (
                sum((a - b) ** 2 for a, b in zip(color, prior_color)) ** 0.5
            )
            match = exact == prior_bytes_hash or (
                sum(a != b for a, b in zip(phash, prior_phash)) <= phash_distance
                and color_distance <= 30
            )
            if match and label != prior_label:
                raise ValueError(
                    f"Near-duplicate images have conflicting labels: {path} and {all_paths[prior][0]}"
                )
            if match:
                union(index, prior)
        fingerprints.append((exact, phash, color))
        exact_buckets[exact].append(index)
        for band, buckets in enumerate(band_buckets):
            buckets[phash[edges[band] : edges[band + 1]]].append(index)

    grouped: dict[int, list[list[Path]]] = {0: [], 1: []}
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(all_paths)):
        clusters[find(index)].append(index)
    for indices in clusters.values():
        labels = {all_paths[index][1] for index in indices}
        if len(labels) != 1:
            raise ValueError("A duplicate cluster contains conflicting labels.")
        label_value = 0 if labels.pop() == "real" else 1
        grouped[label_value].append([all_paths[index][0] for index in indices])
    return grouped


def assign_groups(
    groups: list[list[Path]], ratios: dict[str, float], rng: random.Random
) -> dict[str, list[list[Path]]]:
    """Greedily place largest source clusters in the split with most deficit."""
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(len(group) for group in groups)
    targets = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: 0 for name in ratios}
    result = {name: [] for name in ratios}
    split_names = list(ratios)
    for index, group in enumerate(groups):
        target = (
            split_names[index]
            if index < len(split_names)
            else max(ratios, key=lambda name: targets[name] - counts[name])
        )
        result[target].append(group)
        counts[target] += len(group)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Unsplit directory containing real/ and ai/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for train/validation/test",
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()
    ratios = {"train": args.train, "validation": args.validation, "test": args.test}
    if (
        any(value <= 0 for value in ratios.values())
        or abs(sum(ratios.values()) - 1.0) > 1e-6
    ):
        raise SystemExit("Split ratios must be positive and sum to 1.0.")
    if not 0 <= args.phash_distance <= 16:
        raise SystemExit("--phash-distance must be between 0 and 16.")
    source = args.input.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise SystemExit("Output must be outside the unsplit source directory.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    assignments: dict[str, list[tuple[str, list[Path]]]] = {
        name: [] for name in SPLIT_NAMES
    }
    grouped_by_label = grouped_samples(source, args.phash_distance)
    for class_name, label_value in (("real", 0), ("ai", 1)):
        grouped = grouped_by_label[label_value]
        if len(grouped) < len(SPLIT_NAMES):
            raise SystemExit(
                f"Need at least three independent duplicate groups in {class_name} to populate all splits."
            )
        split_groups = assign_groups(grouped, ratios, rng)
        for split_name, clusters in split_groups.items():
            assignments[split_name].extend(
                (class_name, cluster) for cluster in clusters
            )

    counts: dict[str, dict[str, int]] = {
        name: {"real": 0, "ai": 0} for name in SPLIT_NAMES
    }
    for split_name, class_clusters in assignments.items():
        for class_name, cluster in class_clusters:
            for path in cluster:
                destination = (
                    output
                    / split_name
                    / class_name
                    / path.relative_to(source / class_name)
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                counts[split_name][class_name] += 1
    version_file = source / "dataset_version.txt"
    dataset_id = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.exists()
        else "user-provided-v1"
    )
    (output / "dataset_version.txt").write_text(
        f"{dataset_id[:120]}\n", encoding="utf-8"
    )
    print(
        {
            "seed": args.seed,
            "ratios": ratios,
            "class_counts": counts,
            "duplicate_grouping": f"exact SHA-256 + pHash distance <= {args.phash_distance}",
        }
    )


if __name__ == "__main__":
    main()
