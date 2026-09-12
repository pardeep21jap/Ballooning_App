"""Copy flat BalloonIQ exports into a grouped 70/15/15 YOLO dataset.

Run: python -m balloon_app.dataset_split datasets --output datasets/split
Optional --groups groups.json maps drawing IDs to shared project/family IDs.
Use the same group for all revisions: current exports do not record those
relationships. Without this map only pages sharing a drawing ID are grouped.
Whole groups are indivisible, so small datasets may have empty splits.
The output directory must not exist, preventing stale split contamination.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from balloon_app.config import CHARACTERISTIC_CLASSES


def split_dataset(source: Path | str, output: Path | str, *, seed: int = 42,
                  group_map: dict[str, str] | None = None) -> dict[str, str]:
    """Return image-name -> split assignments; leave source files untouched."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    group_map = {} if group_map is None else group_map
    if not isinstance(group_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in group_map.items()
    ):
        raise ValueError("Group map must map drawing IDs to nonempty project/family strings")
    class_file = source / "manifests" / "classes.txt"
    if class_file.exists() and class_file.read_text(encoding="utf-8").splitlines() != CHARACTERISTIC_CLASSES:
        raise ValueError("Exported class ordering does not match BalloonIQ's class map")
    images = sorted((source / "images").glob("*.png"))
    if not images:
        raise ValueError("No exported PNG images found")
    groups: dict[tuple[str, str], list[Path]] = {}
    for image in images:
        match = re.fullmatch(r"(.+)_p[1-9][0-9]*", image.stem)
        if not match:
            raise ValueError(f"Cannot identify drawing ID in {image.name}")
        if not (source / "labels" / f"{image.stem}.txt").is_file():
            raise ValueError(f"Missing matching label for {image.name}")
        drawing_id = match[1]
        key = ("group", group_map[drawing_id]) if drawing_id in group_map else ("drawing", drawing_id)
        groups.setdefault(key, []).append(image)

    # Shuffle ties reproducibly, then place larger groups first into the split
    # with the greatest remaining image deficit. Never split a drawing group.
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda key: len(groups[key]), reverse=True)
    ratios = {"train": .70, "val": .15, "test": .15}
    counts = dict.fromkeys(ratios, 0)
    assignments = {}
    for key in ordered:
        split = max(ratios, key=lambda name: len(images) * ratios[name] - counts[name])
        for image in groups[key]:
            assignments[image.name] = split
        counts[split] += len(groups[key])

    output.mkdir(parents=True, exist_ok=False)
    for kind in ("images", "labels"):
        for split in ratios:
            (output / kind / split).mkdir(parents=True)
    for image in images:
        split = assignments[image.name]
        shutil.copy2(image, output / "images" / split / image.name)
        label = source / "labels" / f"{image.stem}.txt"
        shutil.copy2(label, output / "labels" / split / label.name)
    yaml = [f"path: {json.dumps(output.as_posix())}", "train: images/train",
            "val: images/val", "test: images/test", "names:"]
    yaml.extend(f"  {i}: {json.dumps(name)}" for i, name in enumerate(CHARACTERISTIC_CLASSES))
    (output / "ballooniq.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps({
        "seed": seed, "ratios": ratios, "group_map": group_map,
        "counts": counts, "assignments": assignments,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=Path, help="JSON map: drawing ID -> shared project/revision-family ID")
    args = parser.parse_args()
    try:
        groups = json.loads(args.groups.read_text(encoding="utf-8")) if args.groups else None
        output = args.output or args.source / "split"
        assignments = split_dataset(args.source, output, seed=args.seed, group_map=groups)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Copied {len(assignments)} image/label pairs to {output}")
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(value == split for value in assignments.values())}")
    if not groups:
        print("Grouped by drawing ID only; use --groups to keep projects and separate revisions together.")


if __name__ == "__main__":
    main()
