"""PyMuPDF (fitz) wrapper: document loading, page rendering, text extraction,
and pure coordinate-conversion helpers.

Coordinate convention
----------------------
* **PDF coordinates**: points, origin top-left of the page (matches PyMuPDF's
  ``page.rect`` convention, i.e. already flipped from the PDF-spec bottom-left
  origin). All balloon positions are stored in this space so exports line up
  with the source drawing regardless of on-screen zoom level.
* **Pixmap/image coordinates**: pixels, origin top-left, at a given DPI.
* **Scene coordinates**: pixels in the QGraphicsScene, which mirrors pixmap
  coordinates at the current render DPI (pdf_view.py owns this mapping).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF (import name predates the "pymupdf" package rename)

logger = logging.getLogger("balloon_app.pdf_engine")

POINTS_PER_INCH = 72.0


def dpi_to_zoom(dpi: float) -> float:
    """Convert a target DPI to the zoom factor PyMuPDF's Matrix expects."""
    return dpi / POINTS_PER_INCH


def pdf_to_pixel(x: float, y: float, dpi: float) -> tuple[float, float]:
    """Convert PDF-space point coordinates to pixel coordinates at ``dpi``."""
    scale = dpi_to_zoom(dpi)
    return x * scale, y * scale


def pixel_to_pdf(px: float, py: float, dpi: float) -> tuple[float, float]:
    """Convert pixel coordinates at ``dpi`` back to PDF-space points."""
    scale = dpi_to_zoom(dpi)
    if scale == 0:
        return 0.0, 0.0
    return px / scale, py / scale


def rect_pdf_to_pixel(
    rect: tuple[float, float, float, float], dpi: float
) -> tuple[float, float, float, float]:
    x0, y0 = pdf_to_pixel(rect[0], rect[1], dpi)
    x1, y1 = pdf_to_pixel(rect[2], rect[3], dpi)
    return x0, y0, x1, y1


def rect_pixel_to_pdf(
    rect: tuple[float, float, float, float], dpi: float
) -> tuple[float, float, float, float]:
    x0, y0 = pixel_to_pdf(rect[0], rect[1], dpi)
    x1, y1 = pixel_to_pdf(rect[2], rect[3], dpi)
    return x0, y0, x1, y1


@dataclass
class TextBlock:
    """A text block/span extracted from a PDF page's native text layer."""

    text: str
    bbox: tuple[float, float, float, float]  # PDF coords: x0, y0, x1, y1


class PdfLoadError(Exception):
    """Raised when a PDF cannot be opened or read."""


class PdfDocument:
    """Thin, defensive wrapper around a ``fitz.Document``."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._doc: Optional[fitz.Document] = None

    def open(self) -> None:
        if not self.path.exists():
            raise PdfLoadError(f"PDF file not found: {self.path}")
        try:
            self._doc = fitz.open(str(self.path))
            if self._doc.is_encrypted:
                if not self._doc.authenticate(""):
                    raise PdfLoadError(f"PDF is password-protected: {self.path}")
        except PdfLoadError:
            raise
        except Exception as exc:  # fitz raises generic exceptions
            raise PdfLoadError(f"Failed to open PDF '{self.path}': {exc}") from exc

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def __enter__(self) -> "PdfDocument":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def doc(self) -> fitz.Document:
        if self._doc is None:
            self.open()
        assert self._doc is not None
        return self._doc

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def page_size_pdf(self, page_number: int) -> tuple[float, float]:
        """Return (width, height) of a page in PDF points."""
        page = self.doc[page_number]
        rect = page.rect
        return rect.width, rect.height

    def render_page_rgb(self, page_number: int, dpi: float) -> tuple[bytes, int, int]:
        """Render a page to raw RGB bytes. Returns (data, width, height)."""
        page = self.doc[page_number]
        zoom = dpi_to_zoom(dpi)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
        return pixmap.samples, pixmap.width, pixmap.height

    def extract_text_blocks(self, page_number: int, min_chars: int = 1) -> list[TextBlock]:
        """Extract native (vector) text spans with their PDF-space bounding boxes.

        Uses ``page.get_text("dict")`` and returns one :class:`TextBlock` per
        *span* (not per block) so nearby but distinct callouts are not
        merged into one large box.
        """
        page = self.doc[page_number]
        result: list[TextBlock] = []
        try:
            raw = page.get_text("dict")
        except Exception:
            logger.exception("Failed extracting text on page %d of %s", page_number, self.path)
            return result

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if len(text) < min_chars:
                        continue
                    bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                    result.append(TextBlock(text=text, bbox=bbox))  # type: ignore[arg-type]
        return result

    def has_native_text(self, page_number: int) -> bool:
        return len(self.extract_text_blocks(page_number)) > 0

    def page_has_images_only(self, page_number: int) -> bool:
        """Heuristic: page has no usable text layer but does have image content."""
        if self.has_native_text(page_number):
            return False
        page = self.doc[page_number]
        try:
            return len(page.get_images(full=True)) > 0
        except Exception:
            return False
