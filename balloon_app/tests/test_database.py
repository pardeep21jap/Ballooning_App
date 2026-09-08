"""Tests for SQLite project persistence and JSON portability."""

from __future__ import annotations

from pathlib import Path

import pytest

from balloon_app.data_model import Balloon, BalloonSource, Drawing, Project, ReviewStatus
from balloon_app.database import ProjectDatabase


@pytest.fixture
def populated_project() -> Project:
    project = Project(name="Bracket Assembly", part_number="BR-100", revision="B", customer="Contoso")
    drawing = Drawing(project_id=project.id, file_name="bracket.pdf", original_path="C:/drawings/bracket.pdf", page_count=2)
    project.drawings.append(drawing)
    project.balloons.append(
        Balloon(
            number=1, drawing_id=drawing.id, page_number=0, x=100.5, y=200.25,
            char_type="diameter", raw_text="Ø10 ±0.02", nominal=10.0, tol_plus=0.02, tol_minus=0.02,
            lower_limit=9.98, upper_limit=10.02, source=BalloonSource.AUTO.value,
            status=ReviewStatus.ACCEPTED.value, confidence=0.87, model_version="rules_ocr_v1",
            guessed_symbol_marker="n",
            original_prediction={"char_type": "diameter", "nominal": 10.0},
        )
    )
    project.balloons.append(
        Balloon(
            number=2, drawing_id=drawing.id, page_number=1, x=50.0, y=75.0,
            char_type="note", raw_text="BREAK ALL EDGES", source=BalloonSource.MANUAL.value,
            status=ReviewStatus.ACCEPTED.value, critical=True,
        )
    )
    return project


def test_save_and_load_roundtrip(tmp_path: Path, populated_project: Project):
    db_path = tmp_path / "project.bpdb"
    db = ProjectDatabase(db_path)
    db.connect()
    db.save_project(populated_project)
    db.close()

    db2 = ProjectDatabase(db_path)
    db2.connect()
    loaded = db2.load_project()
    db2.close()

    assert loaded.name == populated_project.name
    assert loaded.part_number == populated_project.part_number
    assert len(loaded.drawings) == 1
    assert loaded.drawings[0].file_name == "bracket.pdf"
    assert loaded.drawings[0].page_count == 2
    assert len(loaded.balloons) == 2

    balloon = next(b for b in loaded.balloons if b.number == 1)
    assert balloon.char_type == "diameter"
    assert balloon.nominal == pytest.approx(10.0)
    assert balloon.tol_plus == pytest.approx(0.02)
    assert balloon.status == ReviewStatus.ACCEPTED.value
    assert balloon.original_prediction == {"char_type": "diameter", "nominal": 10.0}
    assert balloon.guessed_symbol_marker == "n"

    critical_balloon = next(b for b in loaded.balloons if b.number == 2)
    assert critical_balloon.critical is True
    assert critical_balloon.source == BalloonSource.MANUAL.value


def test_save_overwrites_previous_state(tmp_path: Path, populated_project: Project):
    db_path = tmp_path / "project.bpdb"
    db = ProjectDatabase(db_path)
    db.connect()
    db.save_project(populated_project)

    populated_project.balloons.pop()
    db.save_project(populated_project)
    db.close()

    db2 = ProjectDatabase(db_path)
    db2.connect()
    loaded = db2.load_project()
    db2.close()
    assert len(loaded.balloons) == 1


def test_json_export_import_roundtrip(tmp_path: Path, populated_project: Project):
    db_path = tmp_path / "project.bpdb"
    db = ProjectDatabase(db_path)
    db.connect()
    db.save_project(populated_project)

    json_path = tmp_path / "project_export.json"
    db.export_json(populated_project, json_path)
    db.close()

    assert json_path.exists()
    reimported = ProjectDatabase.import_json(json_path)
    assert reimported.name == populated_project.name
    assert len(reimported.balloons) == len(populated_project.balloons)
    assert len(reimported.drawings) == len(populated_project.drawings)
