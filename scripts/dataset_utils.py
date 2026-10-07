"""Dataset discovery and cross-split duplicate checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.utils.hashing import mean_rgb, perceptual_hash

SPLITS = ("train", "validation", "test")
LABELS = {"real": 0, "ai": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def find_samples(
    dataset: Path, splits: tuple[str, ...] = SPLITS
) -> list[tuple[Path, str, int]]:
    """Collect real/AI files from the documented split layout."""
    rows: list[tuple[Path, str, int]] = []
    for split in splits:
        for label, target in LABELS.items():
            folder = dataset / split / label
            if folder.exists():
                rows.extend(
                    (path, split, target)
                    for path in sorted(folder.rglob("*"))
                    if path.is_file() and path.suffix.lower() in EXTENSIONS
                )
    if not rows:
        raise ValueError(f"No JPEG/PNG/WebP/TIFF images found under {dataset}.")
    return rows


def cross_split_duplicates(
    rows: list[tuple[Path, str, int]], phash_distance: int = 4
) -> list[tuple[str, str, str]]:
    """Find exact hashes or close pHashes that cross split boundaries."""
    fingerprints: list[tuple[str, str, str, tuple[float, float, float]]] = []
    for path, split, _ in rows:
        data = path.read_bytes()
        exact = hashlib.sha256(data).hexdigest()
        try:
            with Image.open(path) as image:
                image.load()
                phash = perceptual_hash(image)
                color = mean_rgb(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not decode dataset image: {path}") from exc
        fingerprints.append((split, exact, phash, color))

    duplicates: list[tuple[str, str, str]] = []
    candidates: set[tuple[int, int]] = set()
    exact_buckets: dict[str, list[int]] = {}
    band_buckets: list[dict[str, list[int]]] = [{} for _ in range(phash_distance + 1)]
    band_edges = [
        round(index * 64 / (phash_distance + 1)) for index in range(phash_distance + 2)
    ]
    for index, (_, exact_hash, phash, _) in enumerate(fingerprints):
        for prior in exact_buckets.get(exact_hash, []):
            candidates.add((prior, index))
        exact_buckets.setdefault(exact_hash, []).append(index)
        for band, buckets in enumerate(band_buckets):
            key = phash[band_edges[band] : band_edges[band + 1]]
            for prior in buckets.get(key, []):
                candidates.add((prior, index))
            buckets.setdefault(key, []).append(index)

    for index, other_index in sorted(candidates):
        split_a, hash_a, phash_a, color_a = fingerprints[index]
        split_b, hash_b, phash_b, color_b = fingerprints[other_index]
        if split_a == split_b:
            continue
        color_distance = sum((a - b) ** 2 for a, b in zip(color_a, color_b)) ** 0.5
        if hash_a == hash_b or (
            sum(a != b for a, b in zip(phash_a, phash_b)) <= phash_distance
            and color_distance <= 30
        ):
            duplicates.append(
                (
                    str(rows[index][0]),
                    str(rows[other_index][0]),
                    "exact" if hash_a == hash_b else "perceptual",
                )
            )
            if len(duplicates) >= 100:
                break
    return duplicates


def assert_no_cross_split_duplicates(dataset: Path) -> None:
    """Fail closed when source images or close perceptual matches cross splits."""
    rows = find_samples(dataset)
    duplicates = cross_split_duplicates(rows)
    if duplicates:
        preview = "; ".join(f"{a} <> {b} ({kind})" for a, b, kind in duplicates[:3])
        raise ValueError(f"Found near-duplicate images across data splits: {preview}")


def dataset_version(dataset: Path) -> str:
    """Read an optional local version marker."""
    marker = dataset / "dataset_version.txt"
    return (
        marker.read_text(encoding="utf-8").strip()[:120]
        if marker.exists()
        else "unspecified"
    )
