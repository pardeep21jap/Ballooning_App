"""AS9102 Form 3 (First Article Inspection Report) Excel export using openpyxl.

Produces a single-sheet workbook laid out like the standard AS9102 "Form 3:
Characteristic Accountability, Verification and Compatibility Evaluation":
a title block, a part/order info grid, then one row per ballooned
characteristic under the Char No. / Reference Location / Characteristic Designator / Requirement /
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

# Column A is left blank as a margin (matches the standard AS9102 Form 3
# layout); the form fields start at B, matching AS9102 field numbering.
COLUMNS: list[tuple[str, str]] = [
    ("B", "7. Char No."),
    ("C", "6. Reference Location"),
    ("D", "7a. Characteristic Designator"),
    ("E", "8. Requirement"),
    ("F", "8a. UoM"),
    ("G", ""),
    ("H", "8b. Upper Limit"),
    ("I", "8c. Lower Limit"),
    ("J", "9. Results"),
    ("K", "10. Gauge"),
    ("L", "11. Non-Conformance Number"),
    ("M", "12. Notes"),
]

_COLUMN_WIDTHS: dict[str, float] = {
    "A": 3, "B": 8, "C": 18, "D": 24, "E": 14, "F": 8, "G": 5,
    "H": 12, "I": 12, "J": 10, "K": 16, "L": 18, "M": 18, "N": 18,
}

_LAST_COL = "N"

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
    if balloon.char_type in (CharacteristicType.ANGLE.value, CharacteristicType.CHAMFER_ANGLE.value):
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
        "B": balloon.number,
        "C": None,  # Reference location is not stored by the app
        "D": _designator(balloon),
        "E": _requirement(balloon, strip_leading_zero=strip_zero),
        "F": _uom(balloon, project_unit),
        "G": "+" if (upper_delta is not None or lower_delta is not None) else "",
        "H": _fmt_number(upper_delta),
        "I": _fmt_number(lower_delta),
        "J": None,  # Results -- filled in by the inspector
        "K": balloon.inspection_method,
        "L": None,  # NonConformance Number -- filled in by the inspector
        "M": None,  # Notes -- filled in by the inspector
    }
    for col_letter, _title in COLUMNS:
        cell = ws[f"{col_letter}{row}"]
        cell.value = values.get(col_letter)
        cell.border = THIN_BORDER
        cell.alignment = Alignment(
            horizontal="center" if col_letter in ("B", "F", "G", "H", "I", "J") else "left",
            vertical="center",
            wrap_text=col_letter in ("D", "E", "K", "M"),
        )
    ws.merge_cells(f"M{row}:N{row}")


def _write_title_block(ws: Worksheet) -> int:
    """Write the title rows and return the next free row number."""
    for row, title in (
        (1, "First Article Inspection Report"),
        (2, "Characteristic Accountability, Verification and Compatibility Evaluation"),
    ):
        ws.merge_cells(f"B{row}:{_LAST_COL}{row}")
        ws[f"B{row}"] = title
        ws[f"B{row}"].font = Font(size=14 if row == 1 else 11, bold=True)
        ws[f"B{row}"].alignment = Alignment(horizontal="center")
        ws.row_dimensions[row].height = 22
    return 3


def _write_info_grid(ws: Worksheet, start_row: int, project: Project) -> int:
    fields = [
        (start_row, "B", "D", "1. Part Number", project.part_number),
        (start_row, "E", "L", "2. Part Name", project.part_name or project.name),
        (start_row, "M", "M", "3. Serial/Lot Number", None),
        (start_row, "N", "N", "4. FAI Report", None),
        (start_row + 2, "B", "D", "5. Part Rev", project.revision),
        (start_row + 2, "E", "I", "6. PO Number:", None),
        (start_row + 2, "J", "L", "6a. Mfg WO#:", None),
    ]
    for row, first, last, label, value in fields:
        for offset, text in enumerate((label, value)):
            if first != last:
                ws.merge_cells(f"{first}{row + offset}:{last}{row + offset}")
            cell = ws[f"{first}{row + offset}"]
            cell.value = text
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if offset == 0:
                cell.font = Font(bold=True, size=9)
                cell.fill = LABEL_FILL
            ws.row_dimensions[row + offset].height = 28
    return start_row + 4


def _write_group_headers(ws: Worksheet, row: int) -> None:
    groups = [
        ("B", "I", "Characteristic Accountability"),
        ("J", "L", "Inspection / Test Results"),
        ("M", "N", "Other Fields"),
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
        end_col = "N" if col_letter == "M" else col_letter
        ws.merge_cells(f"{col_letter}{row}:{end_col}{row + 1}")
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
    ws.freeze_panes = f"A{header_row + 2}"

    row = header_row + 2
    for balloon in exportable:
        _write_row(ws, row, balloon, project.unit or "in")
        row += 1

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
