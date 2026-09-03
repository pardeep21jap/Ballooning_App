"""Generic Excel inspection-sheet export using openpyxl.

Produces a workbook with:

* ``Inspection Data`` -- one row per exported characteristic, with a live
  PASS/FAIL formula driven by an "Actual" column the inspector fills in by
  hand.
* ``Project Info`` -- project/drawing metadata, including a link back to the
  original source PDF.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.worksheet import Worksheet

from balloon_app.data_model import Balloon, CharacteristicType, Drawing, Project, ReviewStatus
from balloon_app.ocr_parser import compute_limits

logger = logging.getLogger("balloon_app.excel_export")

INSPECTION_SHEET_NAME = "Inspection Data"
PROJECT_INFO_SHEET_NAME = "Project Info"

# Column order matches the spec exactly; letters below are relied upon by the
# generated Result formula, so do not reorder without updating it.
COLUMNS: list[tuple[str, str]] = [
    ("A", "Char #"),
    ("B", "Page"),
    ("C", "Balloon Type"),
    ("D", "Raw Drawing Callout"),
    ("E", "Description"),
    ("F", "Nominal"),
    ("G", "Tol +"),
    ("H", "Tol -"),
    ("I", "Lower Limit"),
    ("J", "Upper Limit"),
    ("K", "GD&T Symbol"),
    ("L", "GD&T Tolerance"),
    ("M", "Material Condition"),
    ("N", "Datums"),
    ("O", "Surface Finish"),
    ("P", "Thread Callout"),
    ("Q", "Critical (Y/N)"),
    ("R", "Inspection Method"),
    ("S", "Actual"),
    ("T", "Result"),
    ("U", "Status"),
]

_COLUMN_WIDTHS: dict[str, float] = {
    "A": 8, "B": 6, "C": 18, "D": 30, "E": 32, "F": 10, "G": 8, "H": 8,
    "I": 12, "J": 12, "K": 14, "L": 14, "M": 14, "N": 10, "O": 13, "P": 15,
    "Q": 10, "R": 18, "S": 10, "T": 9, "U": 11,
}

HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def is_exportable(balloon: Balloon, include_pending: bool) -> bool:
    """Default export rule: accepted, edited, and all manual entries.

    Rejected balloons are never exported. Pending auto-proposals are only
    included when the caller explicitly opts in.
    """
    if balloon.status == ReviewStatus.REJECTED.value:
        return False
    if balloon.status in (ReviewStatus.ACCEPTED.value, ReviewStatus.EDITED.value):
        return True
    if balloon.source == "manual":
        return True
    if include_pending and balloon.status == ReviewStatus.PENDING.value:
        return True
    return False


def _write_headers(ws: Worksheet) -> None:
    for col_letter, title in COLUMNS:
        cell = ws[f"{col_letter}1"]
        cell.value = title
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 28


def _apply_column_widths(ws: Worksheet) -> None:
    for col_letter, width in _COLUMN_WIDTHS.items():
        ws.column_dimensions[col_letter].width = width


def _result_formula(row: int) -> str:
    return (
        f'=IF(S{row}="","",'
        f'IF(OR(I{row}="",J{row}=""),"",'
        f'IF(AND(S{row}>=I{row},S{row}<=J{row}),"PASS","FAIL")))'
    )


def _write_row(ws: Worksheet, row: int, balloon: Balloon) -> None:
    lower_limit = balloon.lower_limit
    upper_limit = balloon.upper_limit
    if lower_limit is None or upper_limit is None:
        computed_lower, computed_upper = compute_limits(balloon.nominal, balloon.tol_plus, balloon.tol_minus)
        lower_limit = lower_limit if lower_limit is not None else computed_lower
        upper_limit = upper_limit if upper_limit is not None else computed_upper

    values = {
        "A": balloon.number,
        "B": balloon.page_number + 1,  # display as 1-based
        "C": CharacteristicType.display_name(balloon.char_type),
        "D": balloon.raw_text,
        "E": balloon.note,
        "F": balloon.nominal,
        "G": balloon.tol_plus,
        "H": balloon.tol_minus,
        "I": lower_limit,
        "J": upper_limit,
        "K": balloon.gdt_symbol,
        "L": balloon.gdt_tolerance,
        "M": balloon.material_condition,
        "N": balloon.datums,
        "O": balloon.surface_finish,
        "P": balloon.thread_callout,
        "Q": "Y" if balloon.critical else "N",
        "R": balloon.inspection_method,
        "S": None,  # Actual -- filled in by the inspector
        "T": None,  # Result -- formula, set below
        "U": balloon.status.capitalize(),
    }
    for col_letter, _title in COLUMNS:
        if col_letter == "T":
            continue
        ws[f"{col_letter}{row}"] = values.get(col_letter)
    ws[f"T{row}"] = _result_formula(row)


def build_inspection_workbook(
    project: Project,
    drawing: Optional[Drawing],
    balloons: Iterable[Balloon],
    include_pending: bool = False,
) -> Workbook:
    """Build (but do not save) the inspection workbook for one drawing."""
    exportable = [b for b in balloons if is_exportable(b, include_pending)]
    exportable.sort(key=lambda b: (b.page_number, b.number))

    wb = Workbook()
    ws = wb.active
    ws.title = INSPECTION_SHEET_NAME
    _write_headers(ws)

    row = 2
    for balloon in exportable:
        _write_row(ws, row, balloon)
        row += 1

    last_row = max(row - 1, 1)
    ws.auto_filter.ref = f"A1:{COLUMNS[-1][0]}{last_row}"
    _apply_column_widths(ws)

    info_ws = wb.create_sheet(PROJECT_INFO_SHEET_NAME)
    _write_project_info(info_ws, project, drawing, exported_count=len(exportable), include_pending=include_pending)

    return wb


def _write_project_info(
    ws: Worksheet, project: Project, drawing: Optional[Drawing], exported_count: int, include_pending: bool
) -> None:
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 60

    rows: list[tuple[str, object]] = [
        ("Project Name", project.name),
        ("Part Number", project.part_number),
        ("Revision", project.revision),
        ("Customer", project.customer),
        ("Notes", project.notes),
        ("Date Created", project.date_created),
        ("Date Modified", project.date_modified),
        ("", ""),
        ("Drawing File Name", drawing.file_name if drawing else ""),
        ("Original PDF Path", drawing.original_path if drawing else ""),
        ("Page Count", drawing.page_count if drawing else ""),
        ("", ""),
        ("Characteristics Exported", exported_count),
        ("Pending Proposals Included", "Yes" if include_pending else "No"),
        ("Generated By", "BalloonApp"),
    ]

    for i, (label, value) in enumerate(rows, start=1):
        label_cell = ws.cell(row=i, column=1, value=label)
        label_cell.font = Font(bold=bool(label))
        ws.cell(row=i, column=2, value=value)

    if drawing and drawing.original_path:
        link_row = 10  # "Original PDF Path" row above
        cell = ws.cell(row=link_row, column=2)
        try:
            uri = Path(drawing.original_path).resolve().as_uri()
            cell.hyperlink = uri
            cell.font = Font(color="0563C1", underline="single")
        except (ValueError, OSError):
            pass  # leave as plain text if the path can't form a valid URI


def export_excel(
    project: Project,
    drawing: Optional[Drawing],
    balloons: Iterable[Balloon],
    output_path: Path | str,
    include_pending: bool = False,
) -> Path:
    """Build and atomically save the inspection workbook to ``output_path``."""
    output_path = Path(output_path)
    wb = build_inspection_workbook(project, drawing, balloons, include_pending=include_pending)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp_", suffix=".xlsx")
    os.close(fd)
    try:
        wb.save(tmp_name)
        os.replace(tmp_name, output_path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        logger.exception("Failed to export Excel workbook to %s", output_path)
        raise
    logger.info("Exported Excel inspection sheet to %s", output_path)
    return output_path
