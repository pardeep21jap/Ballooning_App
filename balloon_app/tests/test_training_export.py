"""Invalid annotations must not contaminate YOLO labels or stop export."""
import json

import pymupdf
import pytest

from balloon_app.data_model import Balloon, Drawing, Project
from balloon_app.training_export import CLASS_TO_ID, export_training_dataset


def export(tmp_path, boxes):
    path = tmp_path / "drawing.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=100, height=100)
        doc.save(path)
    project = Project(name="training")
    drawing = Drawing(project_id=project.id, original_path=str(path), file_name=path.name, page_count=1)
    project.drawings.append(drawing)
    for i, (box, char_type, status) in enumerate(boxes):
        b = Balloon(drawing_id=drawing.id, number=i+1, char_type=char_type, status=status)
        if box is not None:
            b.bbox_x0, b.bbox_y0, b.bbox_x1, b.bbox_y1 = box
        project.balloons.append(b)
    result = export_training_dataset(project, tmp_path / "dataset", dpi=72)
    labels = (result.output_root / "labels" / f"{drawing.id}_p1.txt").read_text().splitlines()
    rows = [json.loads(line) for line in result.manifest_jsonl_path.read_text().splitlines()]
    return result, labels, rows


@pytest.mark.parametrize("box,char_type", [
    (None, "diameter"), ((10, 10, 10, 20), "diameter"),
    ((20, 20, 10, 10), "diameter"), ((110, 10, 120, 20), "diameter"),
    ((10, 10, 11, 20), "diameter"), ((float("nan"), 0, 10, 10), "diameter"),
    ((0, 0, float("inf"), 10), "diameter"), (("bad", 0, 10, 10), "diameter"),
    ((0, 0, 10, 10), "unknown_class"),
])
def test_invalid_annotation_skipped_and_reported(tmp_path, box, char_type):
    result, labels, rows = export(tmp_path, [
        (box, char_type, "accepted"), ((30, 30, 40, 40), "diameter", "accepted"),
    ])
    assert len(labels) == result.positive_label_count == 1
    assert len(result.skipped_annotations) == 1
    assert rows[0]["exported_as_positive_label"] is False
    assert rows[0]["skip_reason"]
    assert rows[1]["exported_as_positive_label"] is True


@pytest.mark.parametrize("box,expected", [
    ((10, 20, 30, 40), (.2, .3, .2, .2)),
    ((-10, -20, 120, 140), (.5, .5, 1, 1)),
    ((0, 0, 2, 2), (.01, .01, .02, .02)),
])
def test_valid_and_clipped_normalized_labels(tmp_path, box, expected):
    result, labels, rows = export(tmp_path, [(box, "diameter", "accepted")])
    parts = labels[0].split()
    assert int(parts[0]) == CLASS_TO_ID["diameter"]
    values = list(map(float, parts[1:]))
    assert values == pytest.approx(expected)
    assert all(0 <= value <= 1 for value in values)
    assert not result.skipped_annotations
    assert rows[0]["exported_as_positive_label"] is True


def test_duplicate_and_rejected_labels(tmp_path):
    box = (10, 20, 30, 40)
    result, labels, rows = export(tmp_path, [
        (box, "diameter", "accepted"), (box, "diameter", "accepted"),
        (box, "diameter", "rejected"), (box, "radius", "accepted"),
    ])
    assert len(labels) == 2
    assert len(result.skipped_annotations) == 1
    assert rows[1]["skip_reason"] == "Duplicate label on page"
    assert rows[2]["status"] == "rejected"
    assert rows[2]["exported_as_positive_label"] is False
