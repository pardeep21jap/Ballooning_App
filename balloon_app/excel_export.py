"""AS9102 Form 3 (First Article Inspection Report) Excel export using openpyxl.

Produces a single-sheet workbook laid out like the standard AS9102 "Form 3:
Characteristic Accountability, Verification and Compatibility Evaluation":
a title block, a part/order info grid, then one row per ballooned
characteristic under the Char No. / Characteristic Designator / Requirement /
UoM / Upper Limit / Lower Limit / Results / Gauge / NonConformance Number /
Notes column set.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

from balloon_app.data_model import Balloon, CharacteristicType, Drawing, Project, ReviewStatus

logger = logging.getLogger("balloon_app.excel_export")

INSPECTION_SHEET_NAME = "Inspection Data"

FORM_NUMBER = "QF1439"
FORM_REV = "-"

# Column order matches AS9102 Form 3 field numbering.
COLUMNS: list[tuple[str, str]] = [
    ("A", "7.\nChar No."),
    ("B", "7a.\nCharacteristic Designator"),
    ("C", "8.\nRequirement"),
    ("D", "8a.\nUoM"),
    ("E", "±"),
    ("F", "8b.\nUpper Limit"),
    ("G", "8c.\nLower Limit"),
    ("H", "9.\nResults"),
    ("I", "10.\nGauge"),
    ("J", "11.\nNonConformance Number"),
    ("K", "12.\nNotes"),
]

_COLUMN_WIDTHS: dict[str, float] = {
    "A": 8, "B": 24, "C": 14, "D": 8, "E": 5, "F": 12, "G": 12,
    "H": 10, "I": 16, "J": 18, "K": 20,
}

_LAST_COL = COLUMNS[-1][0]

HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
GROUP_FILL = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
LABEL_FILL = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
THIN_BORDER = Border(*(Side(style="thin", color="808080"),) * 4)


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


def _fmt_number(value: Optional[float], strip_leading_zero: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text in ("", "-", "-0"):
        text = "0"
    if strip_leading_zero:
        negative = text.startswith("-")
        body = text[1:] if negative else text
        if body.startswith("0.") and len(body) > 2:
            body = body[1:]
        text = ("-" if negative else "") + body
    return text


def _designator(balloon: Balloon) -> str:
    if balloon.note.strip():
        return balloon.note.strip()
    return CharacteristicType.display_name(balloon.char_type, balloon.gdt_symbol)


def _requirement(balloon: Balloon, strip_leading_zero: bool) -> str:
    if balloon.char_type == CharacteristicType.THREAD.value and balloon.thread_callout:
        return balloon.thread_callout.strip()
    if balloon.char_type == CharacteristicType.NOTE.value:
        return balloon.raw_text.strip() or balloon.note.strip()
    if balloon.nominal is not None:
        return _fmt_number(balloon.nominal, strip_leading_zero=strip_leading_zero) or ""
    return balloon.raw_text.strip()


def _uom(balloon: Balloon, project_unit: str) -> str:
    if balloon.char_type == CharacteristicType.ANGLE.value:
        return "deg"
    if balloon.nominal is not None:
        return project_unit
    return ""


def _tol_deltas(balloon: Balloon) -> tuple[Optional[float], Optional[float]]:
    """Return (upper_delta, lower_delta) -- signed offsets from nominal, not absolute limits."""
    upper_delta = balloon.tol_plus
    if upper_delta is None and balloon.nominal is not None and balloon.upper_limit is not None:
        upper_delta = balloon.upper_limit - balloon.nominal
    lower_delta = -balloon.tol_minus if balloon.tol_minus is not None else None
    if lower_delta is None and balloon.nominal is not None and balloon.lower_limit is not None:
        lower_delta = balloon.lower_limit - balloon.nominal
    return upper_delta, lower_delta


def _write_row(ws: Worksheet, row: int, balloon: Balloon, project_unit: str) -> None:
    strip_zero = project_unit == "in"
    upper_delta, lower_delta = _tol_deltas(balloon)

    values = {
        "A": balloon.number,
        "B": _designator(balloon),
        "C": _requirement(balloon, strip_leading_zero=strip_zero),
        "D": _uom(balloon, project_unit),
        "E": "+" if (upper_delta is not None or lower_delta is not None) else "",
        "F": _fmt_number(upper_delta),
        "G": _fmt_number(lower_delta),
        "H": None,  # Results -- filled in by the inspector
        "I": balloon.inspection_method,
        "J": None,  # NonConformance Number -- filled in by the inspector
        "K": None,  # Notes -- filled in by the inspector
    }
    for col_letter, _title in COLUMNS:
        cell = ws[f"{col_letter}{row}"]
        cell.value = values.get(col_letter)
        cell.border = THIN_BORDER
        cell.alignment = Alignment(
            horizontal="center" if col_letter in ("A", "D", "E", "F", "G", "H") else "left",
            vertical="center",
            wrap_text=col_letter in ("B", "C", "I", "K"),
        )


def _write_title_block(ws: Worksheet) -> int:
    """Write the title rows and return the next free row number."""
    ws.merge_cells(f"A1:{_LAST_COL}1")
    ws["A1"] = "Sheet 1 of 1"
    ws["A1"].alignment = Alignment(horizontal="right")
    ws["A1"].font = Font(size=9, color="808080")

    ws.merge_cells(f"A2:{_LAST_COL}2")
    ws["A2"] = "First Article Inspection Report"
    ws["A2"].font = Font(size=14, bold=True)
    ws["A2"].alignment = Alignment(horizontal="center")
    ws.row_dimensions[2].height = 22

    ws.merge_cells(f"A3:{_LAST_COL}3")
    ws["A3"] = "Form 3: Characteristic Accountability, Verification and Compatibility Evaluation"
    ws["A3"].font = Font(size=11, bold=True)
    ws["A3"].alignment = Alignment(horizontal="center")
    return 4


def _label_value(ws: Worksheet, row: int, col_letter: str, label: str, value: object) -> None:
    label_cell = ws[f"{col_letter}{row}"]
    label_cell.value = label
    label_cell.font = Font(bold=True, size=9)
    label_cell.fill = LABEL_FILL
    label_cell.border = THIN_BORDER
    label_cell.alignment = Alignment(vertical="center")

    value_col = chr(ord(col_letter) + 1)
    value_cell = ws[f"{value_col}{row}"]
    value_cell.value = value
    value_cell.border = THIN_BORDER
    value_cell.alignment = Alignment(vertical="center")


def _write_info_grid(ws: Worksheet, start_row: int, project: Project) -> int:
    row = start_row

    _label_value(ws, row, "A", "1. Part Number", project.part_number)
    _label_value(ws, row, "C", "2. Part Name", project.part_name or project.name)
    _label_value(ws, row, "E", "3. Part Rev", project.revision)

    ws.row_dimensions[row].height = 18
    return row + 2  # one blank spacer row


def _write_group_headers(ws: Worksheet, row: int) -> None:
    groups = [
        ("A", "G", "Characteristic Accountability"),
        ("H", "I", "Inspection / Test Results"),
        ("J", "K", "Other Fields"),
    ]
    for start, end, title in groups:
        ws.merge_cells(f"{start}{row}:{end}{row}")
        cell = ws[f"{start}{row}"]
        cell.value = title
        cell.fill = GROUP_FILL
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = THIN_BORDER


def _write_column_headers(ws: Worksheet, row: int) -> None:
    for col_letter, title in COLUMNS:
        cell = ws[f"{col_letter}{row}"]
        cell.value = title
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER
    ws.row_dimensions[row].height = 30


def _apply_column_widths(ws: Worksheet) -> None:
    for col_letter, width in _COLUMN_WIDTHS.items():
        ws.column_dimensions[col_letter].width = width


def build_inspection_workbook(
    project: Project,
    drawing: Optional[Drawing],
    balloons: Iterable[Balloon],
    include_pending: bool = False,
) -> Workbook:
    """Build (but do not save) the AS9102 Form 3 inspection workbook for one drawing."""
    exportable = [b for b in balloons if is_exportable(b, include_pending)]
    exportable.sort(key=lambda b: (b.page_number, b.number))

    wb = Workbook()
    ws = wb.active
    ws.title = INSPECTION_SHEET_NAME
    _apply_column_widths(ws)

    next_row = _write_title_block(ws)
    next_row = _write_info_grid(ws, next_row, project)

    group_row = next_row
    header_row = next_row + 1
    _write_group_headers(ws, group_row)
    _write_column_headers(ws, header_row)
    ws.freeze_panes = f"A{header_row + 1}"

    row = header_row + 1
    for balloon in exportable:
        _write_row(ws, row, balloon, project.unit or "in")
        row += 1

    last_data_row = max(row - 1, header_row)
    ws.auto_filter.ref = f"A{header_row}:{_LAST_COL}{last_data_row}"

    footer_row = row + 1
    ws.merge_cells(f"I{footer_row}:{_LAST_COL}{footer_row}")
    footer_cell = ws[f"I{footer_row}"]
    footer_cell.value = f"Form {FORM_NUMBER} Rev {FORM_REV}"
    footer_cell.font = Font(size=9, italic=True, color="808080")
    footer_cell.alignment = Alignment(horizontal="right")

    return wb


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
