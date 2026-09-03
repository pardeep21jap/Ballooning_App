"""Tests for the core data-model dataclasses and their helper logic."""

from __future__ import annotations

from balloon_app.data_model import Balloon, BalloonSource, Drawing, Project, ReviewStatus


def test_next_balloon_number_empty_project():
    project = Project()
    assert project.next_balloon_number() == 1


def test_next_balloon_number_increments():
    project = Project()
    drawing = Drawing(project_id=project.id)
    project.drawings.append(drawing)
    project.balloons.append(Balloon(number=1, drawing_id=drawing.id))
    project.balloons.append(Balloon(number=5, drawing_id=drawing.id))
    assert project.next_balloon_number(drawing.id) == 6


def test_balloons_for_filters_by_drawing_and_page():
    project = Project()
    d1 = Drawing(project_id=project.id)
    d2 = Drawing(project_id=project.id)
    project.drawings.extend([d1, d2])
    project.balloons.append(Balloon(number=1, drawing_id=d1.id, page_number=0))
    project.balloons.append(Balloon(number=2, drawing_id=d1.id, page_number=1))
    project.balloons.append(Balloon(number=1, drawing_id=d2.id, page_number=0))

    assert len(project.balloons_for(d1.id)) == 2
    assert len(project.balloons_for(d1.id, page_number=0)) == 1
    assert len(project.balloons_for(d2.id)) == 1


def test_balloon_snapshot_and_unchanged_detection():
    balloon = Balloon(number=1, char_type="diameter", nominal=10.0, source=BalloonSource.AUTO.value)
    balloon.snapshot_prediction()
    assert balloon.is_unchanged_from_prediction() is True

    balloon.nominal = 12.0
    assert balloon.is_unchanged_from_prediction() is False


def test_balloon_status_and_number_do_not_count_as_edits():
    balloon = Balloon(number=1, char_type="diameter", nominal=10.0, source=BalloonSource.AUTO.value)
    balloon.snapshot_prediction()
    balloon.status = ReviewStatus.ACCEPTED.value
    balloon.number = 7
    assert balloon.is_unchanged_from_prediction() is True


def test_balloon_bbox_helpers():
    balloon = Balloon(bbox_x0=1.0, bbox_y0=2.0, bbox_x1=3.0, bbox_y1=4.0)
    assert balloon.has_bbox() is True
    assert balloon.bbox() == (1.0, 2.0, 3.0, 4.0)

    incomplete = Balloon(bbox_x0=1.0)
    assert incomplete.has_bbox() is False
    assert incomplete.bbox() is None


def test_balloon_to_dict_from_dict_roundtrip():
    balloon = Balloon(number=3, raw_text="Ø25", nominal=25.0, critical=True)
    data = balloon.to_dict()
    restored = Balloon.from_dict(data)
    assert restored.number == 3
    assert restored.raw_text == "Ø25"
    assert restored.nominal == 25.0
    assert restored.critical is True


def test_project_to_dict_from_dict_roundtrip():
    project = Project(name="Widget", part_number="W-1")
    drawing = Drawing(project_id=project.id, file_name="w.pdf", page_count=3)
    project.drawings.append(drawing)
    project.balloons.append(Balloon(number=1, drawing_id=drawing.id))

    data = project.to_dict()
    restored = Project.from_dict(data)
    assert restored.name == "Widget"
    assert len(restored.drawings) == 1
    assert len(restored.balloons) == 1
