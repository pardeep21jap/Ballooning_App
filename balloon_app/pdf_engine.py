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

# Heuristic thresholds for find_vector_diameter_symbol (see its docstring):
# a traced-circle diameter glyph is made of many short segments, is roughly
# as wide as it is tall, and (being sized like a text character) spans a
# few PDF points -- not the much larger geometry of an actual drawn hole,
# a leader line, or an arrowhead (2-3 segments).
_DIAMETER_SYMBOL_MIN_SEGMENTS = 4
_DIAMETER_SYMBOL_MIN_ASPECT = 0.4
_DIAMETER_SYMBOL_MAX_ASPECT = 2.2
_DIAMETER_SYMBOL_MIN_SIZE_PT = 2.0
_DIAMETER_SYMBOL_MAX_SIZE_PT = 14.0
_DIAMETER_SYMBOL_TOUCH_EPS = 0.8


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


def _drawing_item_local_bbox(item: tuple) -> Optional[tuple[float, float, float, float]]:
    """The local bounding box of one ``page.get_drawings()`` path item
    (a line, curve, rect, or quad), or ``None`` if it carries no points."""
    points: list[tuple[float, float]] = []
    for arg in item[1:]:
        if hasattr(arg, "x") and hasattr(arg, "y"):
            points.append((arg.x, arg.y))
        elif hasattr(arg, "x0"):  # fitz.Rect or fitz.Quad-like corner pair
            points.append((arg.x0, arg.y0))
            points.append((arg.x1, arg.y1))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _rects_touch(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], eps: float
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + eps < bx0 or bx1 + eps < ax0 or ay1 + eps < by0 or by1 + eps < ay0)


def _cluster_touching_rects(
    rects: list[tuple[float, float, float, float]], eps: float
) -> list[list[tuple[float, float, float, float]]]:
    """Group rects into connected components (two rects in the same group if
    they touch/overlap, directly or transitively, within ``eps``). Used to
    separate one drawn shape (many touching segments) from unrelated nearby
    ink that happens to fall in the same search area."""
    n = len(rects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _rects_touch(rects[i], rects[j], eps):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[tuple[float, float, float, float]]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(rects[i])
    return list(groups.values())


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
        self._drawing_items_cache: dict[int, list[tuple[float, float, float, float]]] = {}

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

        # ``get_text`` reports spans in the page's raw/mediabox space, which
        # differs from ``page.rect`` (and therefore from ``render_page_rgb``'s
        # pixel output) whenever the page has a /Rotate entry. Apply the same
        # rotation PyMuPDF bakes into rendering so text bboxes land on the
        # visible ink instead of the pre-rotation blank area.
        rotation_matrix = page.rotation_matrix
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if len(text) < min_chars:
                        continue
                    rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0))) * rotation_matrix
                    rect.normalize()
                    result.append(TextBlock(text=text, bbox=tuple(rect)))  # type: ignore[arg-type]
        return result

    def has_native_text(self, page_number: int) -> bool:
        return len(self.extract_text_blocks(page_number)) > 0

    def _drawing_item_bboxes(self, page_number: int) -> list[tuple[float, float, float, float]]:
        """All vector-path item bounding boxes on a page (PDF/text coordinate
        space, rotation-corrected to match :meth:`extract_text_blocks`),
        cached per page since a page can have thousands of tiny path items
        and callers probe this once per candidate dimension."""
        cached = self._drawing_items_cache.get(page_number)
        if cached is not None:
            return cached

        page = self.doc[page_number]
        rotation_matrix = page.rotation_matrix
        bboxes: list[tuple[float, float, float, float]] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            logger.exception("Failed extracting vector drawings on page %d of %s", page_number, self.path)
            drawings = []
        for path in drawings:
            for item in path.get("items", []):
                local = _drawing_item_local_bbox(item)
                if local is None:
                    continue
                rect = fitz.Rect(local) * rotation_matrix
                rect.normalize()
                bboxes.append(tuple(rect))  # type: ignore[arg-type]

        self._drawing_items_cache[page_number] = bboxes
        return bboxes

    def find_vector_diameter_symbol(
        self, page_number: int, bbox: tuple[float, float, float, float]
    ) -> bool:
        """Best-effort detection of a diameter (Ø) glyph drawn as vector
        line art immediately to the left of ``bbox``.

        Some CAD PDF exporters draw GD&T symbols like Ø as traced vector
        line segments rather than a font character, so the glyph never
        appears in :meth:`extract_text_blocks`' output at all -- not even as
        a mangled substitute character. A real diameter glyph traced this
        way is many short segments approximating a circle, sized like a
        text character; this looks for such a cluster (spatially isolated
        from unrelated ink -- leader lines, arrowheads, the part's own hole
        geometry -- via connected-component clustering) just left of
        ``bbox``. Returns ``False`` (never raises) if the page has no usable
        vector-drawing data.
        """
        x0, y0, x1, y1 = bbox
        height = y1 - y0
        if height <= 0:
            return False
        search = fitz.Rect(
            x0 - height * 1.8, y0 - height * 0.8, x0 + height * 0.15, y1 + height * 0.8
        )

        try:
            nearby = [r for r in self._drawing_item_bboxes(page_number) if search.intersects(fitz.Rect(r))]
        except Exception:
            logger.exception("Failed diameter-symbol geometry check on page %d of %s", page_number, self.path)
            return False

        for cluster in _cluster_touching_rects(nearby, _DIAMETER_SYMBOL_TOUCH_EPS):
            if len(cluster) < _DIAMETER_SYMBOL_MIN_SEGMENTS:
                continue
            cx0 = min(r[0] for r in cluster)
            cx1 = max(r[2] for r in cluster)
            cy0 = min(r[1] for r in cluster)
            cy1 = max(r[3] for r in cluster)
            width, cheight = cx1 - cx0, cy1 - cy0
            if cheight <= 0 or not (_DIAMETER_SYMBOL_MIN_ASPECT <= width / cheight <= _DIAMETER_SYMBOL_MAX_ASPECT):
                continue
            if (
                _DIAMETER_SYMBOL_MIN_SIZE_PT <= width <= _DIAMETER_SYMBOL_MAX_SIZE_PT
                and _DIAMETER_SYMBOL_MIN_SIZE_PT <= cheight <= _DIAMETER_SYMBOL_MAX_SIZE_PT
            ):
                return True
        return False

    def page_has_images_only(self, page_number: int) -> bool:
        """Heuristic: page has no usable text layer but does have image content."""
        if self.has_native_text(page_number):
            return False
        page = self.doc[page_number]
        try:
            return len(page.get_images(full=True)) > 0
        except Exception:
            return False
