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

from balloon_app.config import (
    BALLOON_RADIUS_PDF_POINTS,
    STAMP_CORNER_RADIUS_PERCENT,
    STAMP_FONT_SIZE_PDF_POINTS,
    STAMP_MARGIN_PDF_POINTS,
    STAMP_TEXT,
    status_color,
)
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


def _vertical_ink_center_offset(font: fitz.Font, text: str, fontsize: float) -> float:
    """Distance below a top-aligned insert_textbox line's top edge at which
    ``text``'s actual glyph ink -- not the font's full ascender-to-descender
    box -- is vertically centered.

    insert_textbox lays a line out top-down, reserving room for the font's
    full ascender and descender: headroom for accents above and descenders
    below the baseline that all-caps labels and bare digits never use. Left
    at the box's un-shifted top, that reserved-but-unused space ends up
    entirely below the text, making it look like it floats above the
    center of whatever box it's placed in instead of sitting in the middle
    of it. Shifting a box up by this offset (instead of half its height)
    lands the ink itself in the middle.
    """
    tops = [font.glyph_bbox(ord(ch)).y1 for ch in set(text) if not ch.isspace()]
    bottoms = [font.glyph_bbox(ord(ch)).y0 for ch in set(text) if not ch.isspace()]
    if not tops:
        return font.ascender * fontsize / 2.0
    ink_center_em = (max(tops) + min(bottoms)) / 2.0
    return fontsize * (font.ascender - ink_center_em)


def _draw_balloons_on_doc(
    doc: fitz.Document,
    by_page: dict[int, list[Balloon]],
    balloon_size_percent: int = 100,
    stamp_size_percent: int = 100,
    show_stamp: bool = True,
) -> None:
    radius = BALLOON_RADIUS_PDF_POINTS * max(50, min(200, balloon_size_percent)) / 100
    number_font = fitz.Font(fontname="helv")
    for page_number, page_balloons in by_page.items():
        if page_number < 0 or page_number >= doc.page_count:
            logger.warning("Skipping %d balloon(s) for out-of-range page %d", len(page_balloons), page_number)
            continue
        page = doc[page_number]
        # Balloon x/y are stored in the app's display convention (page.rect,
        # i.e. already rotated -- see pdf_engine.py). PyMuPDF's drawing/text
        # APIs instead expect raw/mediabox-space coordinates, so every point
        # and rect has to be mapped back with derotation_matrix before being
        # handed to draw_circle/draw_line/insert_textbox on a rotated page.
        # insert_textbox additionally needs `rotate=page.rotation` so the
        # glyphs themselves are rotated to read upright once PyMuPDF applies
        # the page's own rotation for display.
        derotation_matrix = page.derotation_matrix
        for balloon in page_balloons:
            color = _rgba_unit(status_color(balloon.source, balloon.status))
            center = fitz.Point(balloon.x, balloon.y)

            leader_start: Optional[fitz.Point] = None
            if balloon.leader_x is not None and balloon.leader_y is not None:
                leader_start = fitz.Point(balloon.leader_x, balloon.leader_y)
            elif balloon.has_bbox():
                bx0, by0, bx1, by1 = balloon.bbox()  # type: ignore[misc]
                leader_start = fitz.Point((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)

            if leader_start is not None:
                distance = leader_start.distance_to(center)
                if distance > radius:
                    # Stop the line at the circle's edge, not its center --
                    # otherwise it reads as pointing into the balloon rather
                    # than terminating at it (visible through the circle's
                    # fill_opacity, which isn't fully opaque).
                    edge = center + (leader_start - center) * (radius / distance)
                    page.draw_line(leader_start * derotation_matrix, edge * derotation_matrix, color=color, width=0.75)

            page.draw_circle(center * derotation_matrix, radius, color=color, fill=color, width=1.0, fill_opacity=0.85)

            # Square, circle-diameter box: insert_textbox's internal fit
            # check needs noticeably more headroom than the font's nominal
            # size, so a shorter box (previously radius * 0.75 tall) caused
            # it to silently draw nothing -- on every export, regardless of
            # page rotation -- because the number never "fit".
            fontsize = radius * 1.05
            number_text = str(balloon.number)
            ink_center_offset = _vertical_ink_center_offset(number_font, number_text, fontsize)
            text_rect = fitz.Rect(
                center.x - radius, center.y - ink_center_offset,
                center.x + radius, center.y - ink_center_offset + 2 * radius,
            )
            text_rect = text_rect * derotation_matrix
            text_rect.normalize()
            page.insert_textbox(
                text_rect,
                number_text,
                fontsize=fontsize,
                fontname="helv",
                color=(1, 1, 1),
                align=1,
                rotate=page.rotation,
            )

    if show_stamp:
        for page in doc:
            _stamp_page(page, stamp_size_percent)


def _stamp_page(page: fitz.Page, stamp_size_percent: int = 100) -> None:
    """Stamp "Ballooned Drawing" in the page's visual top-left corner.

    Positioned in page.rect space (the app's display convention, top-left
    origin) and mapped through derotation_matrix -- same approach as the
    balloon overlay in _draw_balloons_on_doc -- so it lands in the visual
    top-left corner and reads upright regardless of the page's /Rotate.
    ``stamp_size_percent`` scales the whole badge (font + box) uniformly,
    the same way balloon_size_percent scales balloons.
    """
    scale = max(50, min(200, stamp_size_percent)) / 100
    fontsize = STAMP_FONT_SIZE_PDF_POINTS * scale
    margin = STAMP_MARGIN_PDF_POINTS * scale

    derotation_matrix = page.derotation_matrix
    rect = page.rect
    font = fitz.Font(fontname="hebo")
    text_width = font.text_length(STAMP_TEXT, fontsize=fontsize)
    # Snug the pill to the actual text width plus a little breathing room,
    # rather than a fixed box width -- STAMP_TEXT is much narrower than that
    # fixed width, which left a wide dead gap on either side of it.
    horizontal_padding = fontsize * 1.0
    box_width = min(text_width + 2 * horizontal_padding, rect.width - 2 * margin)
    # insert_textbox's internal fit check needs noticeably more headroom
    # than the font's nominal size (see the balloon-number box above) --
    # anything under ~2x fontsize silently fails to fit and draws nothing.
    box_height = fontsize * 2.0
    box_x0, box_y0 = rect.x0 + margin, rect.y0 + margin
    box = fitz.Rect(box_x0, box_y0, box_x0 + box_width, box_y0 + box_height)

    # The visible pill border stays exactly this box, but insert_textbox's
    # top-down layout would leave the text's actual ink sitting above its
    # center (see _vertical_ink_center_offset) -- so the text is placed in
    # a same-size box shifted to land the ink in the middle of `box`
    # instead, independent of the border's own position.
    ink_center_offset = _vertical_ink_center_offset(font, STAMP_TEXT, fontsize)
    text_box_y0 = box_y0 + box_height / 2.0 - ink_center_offset
    text_box = fitz.Rect(box.x0, text_box_y0, box.x1, text_box_y0 + box_height)

    box = box * derotation_matrix
    box.normalize()
    text_box = text_box * derotation_matrix
    text_box.normalize()

    stamp_color = (0.75, 0, 0)
    page.draw_rect(box, color=stamp_color, width=max(0.5, 1.1 * scale), radius=STAMP_CORNER_RADIUS_PERCENT)
    page.insert_textbox(
        text_box,
        STAMP_TEXT,
        fontsize=fontsize,
        fontname="hebo",
        color=stamp_color,
        align=1,
        rotate=page.rotation,
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
    balloon_size_percent: int = 100,
    stamp_size_percent: int = 100,
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

    # The "Ballooned Drawing" stamp certifies the drawing as fully reviewed,
    # so it only belongs on the export once every balloon on it -- not just
    # the ones this export happens to include -- is Accepted. A drawing
    # with no balloons at all isn't "ballooned" either.
    reviewed_statuses = (ReviewStatus.ACCEPTED.value, ReviewStatus.EDITED.value)
    show_stamp = bool(balloons) and all(b.status in reviewed_statuses for b in balloons)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    used_raster_fallback = False
    message = ""
    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(str(source_path))
        _draw_balloons_on_doc(doc, by_page, balloon_size_percent, stamp_size_percent, show_stamp)
        fd, tmp_name = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp_", suffix=".pdf")
        os.close(fd)
        doc.save(tmp_name, garbage=3, deflate=True)
    except Exception as exc:
        logger.warning("Vector ballooned-PDF export failed (%s); using raster fallback", exc)
        if doc is not None:
            doc.close()
        try:
            doc = _build_raster_fallback_doc(source_path, raster_fallback_dpi)
            _draw_balloons_on_doc(doc, by_page, balloon_size_percent, stamp_size_percent, show_stamp)
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
