"""Grouped, reproducible dataset splitting without changing exported originals."""
import json

import pytest

from balloon_app.config import CHARACTERISTIC_CLASSES
from balloon_app.dataset_split import split_dataset


def dataset(root, drawings=100, pages=2):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for drawing in range(drawings):
        for page in range(1, pages + 1):
            stem = f"drawing-{drawing:03}_p{page}"
            (root / "images" / f"{stem}.png").write_bytes(b"image")
            (root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_deterministic_ratios_pairs_and_class_map(tmp_path):
    source = tmp_path / "source"
    dataset(source)
    first = split_dataset(source, tmp_path / "first")
    second = split_dataset(source, tmp_path / "second")
    assert first == second
    counts = {name: list(first.values()).count(name) for name in ("train", "val", "test")}
    assert counts == {"train": 140, "val": 30, "test": 30}
    all_images = []
    for name in counts:
        images = sorted((tmp_path / "first" / "images" / name).glob("*.png"))
        labels = sorted((tmp_path / "first" / "labels" / name).glob("*.txt"))
        assert [p.stem for p in images] == [p.stem for p in labels]
        all_images.extend(p.name for p in images)
        for image, label in zip(images, labels):
            assert image.read_bytes() == (source / "images" / image.name).read_bytes()
            assert label.read_bytes() == (source / "labels" / label.name).read_bytes()
    assert len(all_images) == len(set(all_images)) == 200
    for drawing in range(100):
        assert first[f"drawing-{drawing:03}_p1.png"] == first[f"drawing-{drawing:03}_p2.png"]
    yaml = (tmp_path / "first" / "ballooniq.yaml").read_text()
    assert "train: images/train" in yaml
    assert "val: images/val" in yaml
    assert "test: images/test" in yaml
    for i, name in enumerate(CHARACTERISTIC_CLASSES):
        assert f"  {i}: {json.dumps(name)}\n" in yaml


def test_project_and_revision_grouping(tmp_path):
    dataset(tmp_path / "source", drawings=12)
    groups = {f"drawing-{i:03}": f"project-{i // 3}" for i in range(12)}
    assignments = split_dataset(tmp_path / "source", tmp_path / "out", group_map=groups)
    for project in set(groups.values()):
        assert len({split for image, split in assignments.items()
                    if groups[image.rsplit("_p", 1)[0]] == project}) == 1


def test_single_group_is_not_split(tmp_path):
    dataset(tmp_path / "source", drawings=1, pages=10)
    assignments = split_dataset(tmp_path / "source", tmp_path / "out")
    assert set(assignments.values()) == {"train"}


def test_missing_pair_fails_before_writing(tmp_path):
    dataset(tmp_path / "source", drawings=1)
    (tmp_path / "source" / "labels" / "drawing-000_p1.txt").unlink()
    with pytest.raises(ValueError, match="label"):
        split_dataset(tmp_path / "source", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_existing_output(tmp_path):
    dataset(tmp_path / "source", drawings=1)
    split_dataset(tmp_path / "source", tmp_path / "out")
    with pytest.raises(FileExistsError):
        split_dataset(tmp_path / "source", tmp_path / "out")
