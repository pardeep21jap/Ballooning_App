"""Export a ballooned copy of the original PDF drawing.

The normal path preserves the source PDF as vector content and simply draws
balloon circles, numbers, and leader lines on top of it with PyMuPDF -- the
underlying drawing stays crisp and selectable/searchable. If that fails for
a particular file (e.g. a damaged or unusual PDF structure), a clearly
identified raster fallback re-renders each page to an image first and draws
the same overlay on top of that instead, so the export always succeeds.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF (import name predates the "pymupdf" package rename)

from balloon_app.config import BALLOON_RADIUS_PDF_POINTS, status_color
from balloon_app.data_model import Balloon, Drawing, ReviewStatus

logger = logging.getLogger("balloon_app.pdf_export")


class PdfExportError(Exception):
    """Raised when the ballooned PDF cannot be produced at all."""


@dataclass
class PdfExportResult:
    output_path: Path
    used_raster_fallback: bool
    balloon_count: int
    message: str = ""


def resolve_source_path(drawing: Drawing) -> Optional[Path]:
    """Find a usable path to the drawing's original PDF.

    Checks the relinked path first (``last_known_good_path``), since it is
    only set when the original moved and the user confirmed a new location.
    """
    for candidate in (drawing.last_known_good_path, drawing.original_path):
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    return None


def _include_in_export(balloon: Balloon, include_pending: bool, include_rejected: bool) -> bool:
    if balloon.status == ReviewStatus.REJECTED.value:
        return include_rejected
    if balloon.status == ReviewStatus.PENDING.value:
        return include_pending
    return True  # accepted, edited, manual


def _rgba_unit(rgba: tuple[int, int, int, int]) -> tuple[float, float, float]:
    return (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0)


def _draw_balloons_on_doc(doc: fitz.Document, by_page: dict[int, list[Balloon]]) -> None:
    radius = BALLOON_RADIUS_PDF_POINTS
    for page_number, page_balloons in by_page.items():
        if page_number < 0 or page_number >= doc.page_count:
            logger.warning("Skipping %d balloon(s) for out-of-range page %d", len(page_balloons), page_number)
            continue
        page = doc[page_number]
        for balloon in page_balloons:
            color = _rgba_unit(status_color(balloon.source, balloon.status))
            center = fitz.Point(balloon.x, balloon.y)

            leader_start: Optional[fitz.Point] = None
            if balloon.leader_x is not None and balloon.leader_y is not None:
                leader_start = fitz.Point(balloon.leader_x, balloon.leader_y)
            elif balloon.has_bbox():
                bx0, by0, bx1, by1 = balloon.bbox()  # type: ignore[misc]
                leader_start = fitz.Point((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)

            if leader_start is not None and leader_start.distance_to(center) > radius:
                page.draw_line(leader_start, center, color=color, width=0.75)

            page.draw_circle(center, radius, color=color, fill=color, width=1.0, fill_opacity=0.85)

            text_rect = fitz.Rect(center.x - radius, center.y - radius * 0.75, center.x + radius, center.y + radius * 0.75)
            page.insert_textbox(
                text_rect,
                str(balloon.number),
                fontsize=max(6.0, radius * 1.05),
                fontname="helv",
                color=(1, 1, 1),
                align=1,
            )


def _build_raster_fallback_doc(source_path: Path, dpi: float) -> fitz.Document:
    src = fitz.open(str(source_path))
    out = fitz.open()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in src:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, pixmap=pix)
    finally:
        src.close()
    return out


def export_ballooned_pdf(
    drawing: Drawing,
    balloons: list[Balloon],
    output_path: Path | str,
    include_pending: bool = True,
    include_rejected: bool = False,
    raster_fallback_dpi: float = 300.0,
) -> PdfExportResult:
    """Draw balloons onto a copy of the source PDF and save it to ``output_path``."""
    source_path = resolve_source_path(drawing)
    if source_path is None:
        raise PdfExportError(
            f"Cannot locate the source PDF for drawing '{drawing.file_name}'. "
            "Relink the drawing to its file and try again."
        )

    filtered = [b for b in balloons if _include_in_export(b, include_pending, include_rejected)]
    by_page: dict[int, list[Balloon]] = {}
    for b in filtered:
        by_page.setdefault(b.page_number, []).append(b)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    used_raster_fallback = False
    message = ""
    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(str(source_path))
        _draw_balloons_on_doc(doc, by_page)
        fd, tmp_name = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp_", suffix=".pdf")
        os.close(fd)
        doc.save(tmp_name, garbage=3, deflate=True)
    except Exception as exc:
        logger.warning("Vector ballooned-PDF export failed (%s); using raster fallback", exc)
        if doc is not None:
            doc.close()
        try:
            doc = _build_raster_fallback_doc(source_path, raster_fallback_dpi)
            _draw_balloons_on_doc(doc, by_page)
            fd, tmp_name = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp_", suffix=".pdf")
            os.close(fd)
            doc.save(tmp_name)
            used_raster_fallback = True
            message = (
                "The original PDF could not be preserved as vector content, so this export "
                "is a high-resolution raster fallback with balloons overlaid."
            )
        except Exception as fallback_exc:
            raise PdfExportError(f"Failed to export ballooned PDF: {fallback_exc}") from fallback_exc
    finally:
        if doc is not None:
            doc.close()

    try:
        os.replace(tmp_name, output_path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise

    logger.info(
        "Exported ballooned PDF to %s (%d balloons, raster_fallback=%s)",
        output_path, len(filtered), used_raster_fallback,
    )
    return PdfExportResult(
        output_path=output_path,
        used_raster_fallback=used_raster_fallback,
        balloon_count=len(filtered),
        message=message,
    )
