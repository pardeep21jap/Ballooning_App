"""Tests for the Excel inspection-sheet exporter."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from balloon_app.data_model import Balloon, BalloonSource, Drawing, Project, ReviewStatus
from balloon_app.excel_export import COLUMNS, INSPECTION_SHEET_NAME, PROJECT_INFO_SHEET_NAME, export_excel


@pytest.fixture
def sample_project_and_balloons():
    project = Project(name="Test Project", part_number="PN-123", revision="A", customer="Acme")
    drawing = Drawing(project_id=project.id, file_name="part.pdf", original_path="C:/drawings/part.pdf", page_count=1)
    project.drawings.append(drawing)

    balloons = [
        Balloon(
            number=1, drawing_id=drawing.id, page_number=0, x=10, y=10,
            char_type="linear_dimension", raw_text="50.00 ±0.05", nominal=50.0,
            tol_plus=0.05, tol_minus=0.05, lower_limit=49.95, upper_limit=50.05,
            source=BalloonSource.AUTO.value, status=ReviewStatus.ACCEPTED.value, confidence=0.9,
        ),
        Balloon(
            number=2, drawing_id=drawing.id, page_number=0, x=20, y=20,
            char_type="diameter", raw_text="Ø25", nominal=25.0,
            source=BalloonSource.AUTO.value, status=ReviewStatus.PENDING.value, confidence=0.4,
        ),
        Balloon(
            number=3, drawing_id=drawing.id, page_number=0, x=30, y=30,
            char_type="note", raw_text="REJECTED NOTE",
            source=BalloonSource.AUTO.value, status=ReviewStatus.REJECTED.value, confidence=0.2,
        ),
        Balloon(
            number=4, drawing_id=drawing.id, page_number=0, x=40, y=40,
            char_type="surface_finish", raw_text="Ra 1.6", surface_finish="Ra 1.6",
            source=BalloonSource.MANUAL.value, status=ReviewStatus.ACCEPTED.value, confidence=1.0,
        ),
    ]
    project.balloons.extend(balloons)
    return project, drawing, balloons


def test_headers_match_spec(tmp_path: Path, sample_project_and_balloons):
    project, drawing, balloons = sample_project_and_balloons
    out_path = tmp_path / "inspection.xlsx"
    export_excel(project, drawing, balloons, out_path, include_pending=False)

    wb = openpyxl.load_workbook(out_path)
    assert INSPECTION_SHEET_NAME in wb.sheetnames
    assert PROJECT_INFO_SHEET_NAME in wb.sheetnames

    ws = wb[INSPECTION_SHEET_NAME]
    header_row = [ws.cell(row=1, column=i + 1).value for i in range(len(COLUMNS))]
    expected = [title for _letter, title in COLUMNS]
    assert header_row == expected


def test_default_export_excludes_pending_and_rejected(tmp_path: Path, sample_project_and_balloons):
    project, drawing, balloons = sample_project_and_balloons
    out_path = tmp_path / "inspection.xlsx"
    export_excel(project, drawing, balloons, out_path, include_pending=False)

    wb = openpyxl.load_workbook(out_path)
    ws = wb[INSPECTION_SHEET_NAME]
    # 2 rows expected: balloon #1 (accepted) and #4 (manual); #2 pending and #3 rejected excluded.
    char_numbers = [ws.cell(row=r, column=1).value for r in range(2, ws.max_row + 1) if ws.cell(row=r, column=1).value]
    assert char_numbers == [1, 4]


def test_include_pending_option(tmp_path: Path, sample_project_and_balloons):
    project, drawing, balloons = sample_project_and_balloons
    out_path = tmp_path / "inspection_pending.xlsx"
    export_excel(project, drawing, balloons, out_path, include_pending=True)

    wb = openpyxl.load_workbook(out_path)
    ws = wb[INSPECTION_SHEET_NAME]
    char_numbers = [ws.cell(row=r, column=1).value for r in range(2, ws.max_row + 1) if ws.cell(row=r, column=1).value]
    # Rejected (#3) must never be included, even with include_pending=True.
    assert 3 not in char_numbers
    assert 2 in char_numbers


def test_result_formula_present_and_references_correct_columns(tmp_path: Path, sample_project_and_balloons):
    project, drawing, balloons = sample_project_and_balloons
    out_path = tmp_path / "inspection.xlsx"
    export_excel(project, drawing, balloons, out_path, include_pending=False)

    wb = openpyxl.load_workbook(out_path)
    ws = wb[INSPECTION_SHEET_NAME]
    formula = ws["T2"].value
    assert formula.startswith("=IF(S2=")
    assert "I2" in formula and "J2" in formula
    assert "PASS" in formula and "FAIL" in formula


def test_lower_upper_limits_written(tmp_path: Path, sample_project_and_balloons):
    project, drawing, balloons = sample_project_and_balloons
    out_path = tmp_path / "inspection.xlsx"
    export_excel(project, drawing, balloons, out_path, include_pending=False)

    wb = openpyxl.load_workbook(out_path)
    ws = wb[INSPECTION_SHEET_NAME]
    assert ws["I2"].value == pytest.approx(49.95)
    assert ws["J2"].value == pytest.approx(50.05)


def test_header_frozen_and_autofilter(tmp_path: Path, sample_project_and_balloons):
    project, drawing, balloons = sample_project_and_balloons
    out_path = tmp_path / "inspection.xlsx"
    export_excel(project, drawing, balloons, out_path, include_pending=False)

    wb = openpyxl.load_workbook(out_path)
    ws = wb[INSPECTION_SHEET_NAME]
    assert ws.freeze_panes == "A2"
    assert ws.auto_filter.ref is not None
