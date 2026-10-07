import random
from pathlib import Path

from PIL import Image

from scripts.dataset_utils import cross_split_duplicates
from scripts.split_dataset import assign_groups, grouped_samples


def test_exact_duplicates_across_splits_detected(tmp_path: Path):
    one = tmp_path / "one.jpg"
    two = tmp_path / "two.jpg"
    Image.new("RGB", (32, 32), (12, 55, 78)).save(one, format="PNG")
    two.write_bytes(one.read_bytes())
    rows = [(one, "train", 0), (two, "test", 0)]
    matches = cross_split_duplicates(rows)
    assert matches == [(str(one), str(two), "exact")]


def test_splitter_keeps_exact_duplicates_together(tmp_path: Path):
    source = tmp_path / "source"
    real = source / "real"
    ai = source / "ai"
    real.mkdir(parents=True)
    ai.mkdir(parents=True)
    image = Image.new("RGB", (32, 32), (12, 55, 78))
    image.save(real / "one.png")
    (real / "two.png").write_bytes((real / "one.png").read_bytes())
    Image.new("RGB", (32, 32), (220, 20, 20)).save(ai / "gen.png")
    Image.new("RGB", (32, 32), (20, 220, 20)).save(ai / "gen2.png")
    groups = grouped_samples(source)
    assert any(len(group) == 2 for group in groups[0])


def test_split_assignment_populates_all_splits():
    groups = [
        [Path(f"{index}-{item}") for item in range(size)]
        for index, size in enumerate((10, 9, 8))
    ]
    assigned = assign_groups(
        groups, {"train": 0.70, "validation": 0.15, "test": 0.15}, random.Random(7)
    )
    assert all(assigned[name] for name in ("train", "validation", "test"))
