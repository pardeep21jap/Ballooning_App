"""Tests for pure coordinate-conversion helpers in pdf_engine."""

from __future__ import annotations

import math

import pytest

from balloon_app.pdf_engine import (
    PdfDocument,
    dpi_to_zoom,
    pdf_to_pixel,
    pixel_to_pdf,
    rect_pdf_to_pixel,
    rect_pixel_to_pdf,
)

fitz = pytest.importorskip("pymupdf")


def test_dpi_to_zoom():
    assert math.isclose(dpi_to_zoom(72.0), 1.0)
    assert math.isclose(dpi_to_zoom(144.0), 2.0)
    assert math.isclose(dpi_to_zoom(36.0), 0.5)


def test_pdf_to_pixel_and_back_roundtrip():
    dpi = 150.0
    original = (100.25, 200.75)
    px = pdf_to_pixel(*original, dpi)
    back = pixel_to_pdf(*px, dpi)
    assert math.isclose(back[0], original[0], abs_tol=1e-9)
    assert math.isclose(back[1], original[1], abs_tol=1e-9)


def test_pdf_to_pixel_scaling():
    # At 72 DPI (1:1), pixel coords should equal PDF point coords.
    px, py = pdf_to_pixel(50.0, 60.0, 72.0)
    assert math.isclose(px, 50.0)
    assert math.isclose(py, 60.0)

    # At 144 DPI (2x), pixel coords should double.
    px2, py2 = pdf_to_pixel(50.0, 60.0, 144.0)
    assert math.isclose(px2, 100.0)
    assert math.isclose(py2, 120.0)


def test_rect_roundtrip():
    dpi = 200.0
    rect = (10.0, 20.0, 110.0, 220.0)
    px_rect = rect_pdf_to_pixel(rect, dpi)
    back = rect_pixel_to_pdf(px_rect, dpi)
    for a, b in zip(rect, back):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_pixel_to_pdf_zero_dpi_safe():
    # Defensive: must not raise a ZeroDivisionError.
    assert pixel_to_pdf(10.0, 10.0, 0.0) == (0.0, 0.0)


def test_extract_text_blocks_on_rotated_page_matches_rendered_ink(tmp_path):
    """A page with a /Rotate 90 entry (common for landscape CAD drawings
    printed to a portrait mediabox) must report text bboxes in the same
    top-left-origin, ``page.rect`` display space that ``render_page_rgb``
    renders into -- not the pre-rotation mediabox space ``get_text``
    returns natively. Otherwise every bbox lands on the wrong part of the
    rendered raster (see auto_balloon.py's ink-plausibility check).
    """
    pdf_path = tmp_path / "rotated.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=400)
    page.insert_text((20, 40), "0.500", fontsize=12)
    page.set_rotation(90)
    doc.save(str(pdf_path))
    doc.close()

    # Ground truth, computed independently of extract_text_blocks(): PyMuPDF
    # itself documents that get_text() bboxes live in the raw/mediabox space,
    # and that rotation_matrix maps that space into the rotated page.rect
    # ("display") space render_page_rgb() renders into.
    ref_doc = fitz.open(str(pdf_path))
    ref_page = ref_doc[0]
    raw_span = next(
        span
        for block in ref_page.get_text("dict")["blocks"]
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip() == "0.500"
    )
    expected = fitz.Rect(raw_span["bbox"]) * ref_page.rotation_matrix
    expected.normalize()
    ref_doc.close()

    pdf_doc = PdfDocument(pdf_path)
    pdf_doc.open()
    blocks = pdf_doc.extract_text_blocks(0)
    page_width, page_height = pdf_doc.page_size_pdf(0)
    pdf_doc.close()

    assert (page_width, page_height) == (400, 200)  # rotated display size
    match = next(b for b in blocks if b.text == "0.500")
    for got, want in zip(match.bbox, (expected.x0, expected.y0, expected.x1, expected.y1)):
        assert math.isclose(got, want, abs_tol=1e-6)


class TestFindVectorDiameterSymbol:
    """Some CAD PDF exporters draw the Ø glyph as vector line art (a
    traced circle) rather than a font character, so it never appears in
    extract_text_blocks' output at all. find_vector_diameter_symbol is the
    fallback: look for a small, roughly-circular cluster of vector-path
    segments immediately left of a dimension's text bbox.
    """

    def _text_bbox(self, path, text):
        doc = fitz.open(str(path))
        page = doc[0]
        span = next(
            span
            for block in page.get_text("dict")["blocks"]
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip() == text
        )
        doc.close()
        return tuple(span["bbox"])

    def test_detects_small_circle_immediately_left_of_text(self, tmp_path):
        pdf_path = tmp_path / "circle.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "8", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        x0, y0, _x1, y1 = bbox
        cy = (y0 + y1) / 2.0
        shape = page.new_shape()
        shape.draw_circle((x0 - 4.5, cy), 3.0)
        shape.finish()
        shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_diameter_symbol(0, bbox) is True
        pdf_doc.close()

    def test_no_nearby_shape_returns_false(self, tmp_path):
        pdf_path = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "8", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        bbox = self._text_bbox(pdf_path, "8")
        assert pdf_doc.find_vector_diameter_symbol(0, bbox) is False
        pdf_doc.close()

    def test_distant_large_circle_is_not_mistaken_for_the_glyph(self, tmp_path):
        """A circle far larger than a text character (e.g. the drawing's own
        hole geometry happening to sit in the search zone) must not count."""
        pdf_path = tmp_path / "big_circle.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "8", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        x0, y0, _x1, y1 = bbox
        cy = (y0 + y1) / 2.0
        shape = page.new_shape()
        shape.draw_circle((x0 - 20, cy), 25.0)  # far too big to be the glyph
        shape.finish()
        shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_diameter_symbol(0, bbox) is False
        pdf_doc.close()

    def test_result_is_cached_per_page(self, tmp_path):
        pdf_path = tmp_path / "cache.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "8", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert 0 not in pdf_doc._drawing_items_cache
        pdf_doc.find_vector_diameter_symbol(0, bbox)
        assert 0 in pdf_doc._drawing_items_cache
        cached_items = pdf_doc._drawing_items_cache[0]
        pdf_doc.find_vector_diameter_symbol(0, bbox)
        assert pdf_doc._drawing_items_cache[0] is cached_items  # reused, not recomputed
        pdf_doc.close()


class TestFindVectorGdtFrame:
    """A GD&T feature control frame's symbol (flatness, straightness, etc.)
    is almost always drawn as vector line art, contributing no character at
    all to the extracted text. find_vector_gdt_frame looks for the frame's
    boxed-compartment structure -- not the specific symbol -- immediately
    left of the tolerance value's own text bbox.
    """

    def _draw_frame(self, page, bbox, divider: bool = True, symbol_cell: bool = True):
        x0, y0, x1, y1 = bbox
        height = y1 - y0
        padding = height * 0.15
        top, bottom = y0 - padding, y1 + padding
        divider_x = x0 - padding
        outer_left_x = divider_x - height * 1.2 if symbol_cell else divider_x
        right_x = x1 + padding

        shape = page.new_shape()
        shape.draw_line((outer_left_x, top), (right_x, top))
        shape.draw_line((outer_left_x, bottom), (right_x, bottom))
        shape.draw_line((outer_left_x, top), (outer_left_x, bottom))
        if divider and symbol_cell:
            shape.draw_line((divider_x, top), (divider_x, bottom))
        shape.draw_line((right_x, top), (right_x, bottom))
        shape.finish()
        shape.commit()

    def test_detects_boxed_compartment_immediately_left_of_value(self, tmp_path):
        pdf_path = tmp_path / "gdt_frame.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        self._draw_frame(page, bbox)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_gdt_frame(0, bbox) is True
        pdf_doc.close()

    def test_no_nearby_geometry_returns_false(self, tmp_path):
        pdf_path = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_gdt_frame(0, bbox) is False
        pdf_doc.close()

    def test_snug_box_with_no_divider_is_not_mistaken_for_a_frame(self, tmp_path):
        # A single enclosing box with no internal divider (e.g. a "basic
        # dimension" box) must not be read as a GD&T frame -- there's only
        # one compartment, not a separate symbol cell next to this value.
        pdf_path = tmp_path / "basic_dim.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        self._draw_frame(page, bbox, divider=False, symbol_cell=False)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_gdt_frame(0, bbox) is False
        pdf_doc.close()

    def test_distant_leader_line_is_not_mistaken_for_a_divider(self, tmp_path):
        # A leader/extension line kept at a visible distance and far taller
        # than the text -- not a snug, adjacent divider -- must not count.
        pdf_path = tmp_path / "leader.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        x0, y0, x1, y1 = bbox
        shape = page.new_shape()
        shape.draw_line((x0 - 40, y0 - 50), (x0 - 40, y1 + 50))  # far left, far taller
        shape.finish()
        shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_gdt_frame(0, bbox) is False
        pdf_doc.close()


class TestFindVectorCircleAroundText:
    """An assembly drawing's item-reference "balloon" (a bare 1-2 digit
    number matching a Parts List row, with a leader line to the part it
    identifies) is textually indistinguishable from a real whole-number
    dimension -- only the circle drawn around it, again vector line art,
    tells them apart. find_vector_circle_around_text looks for that circle.
    """

    @staticmethod
    def _enclosing_radius(bbox, margin: float = 2.0) -> float:
        # Circumscribe the (possibly narrow, e.g. a "1") text bbox's
        # diagonal with a comfortable margin, so the circle clears it on
        # every side regardless of the glyph's own aspect ratio.
        x0, y0, x1, y1 = bbox
        return math.hypot((x1 - x0) / 2.0, (y1 - y0) / 2.0) + margin

    def test_detects_circle_enclosing_text(self, tmp_path):
        pdf_path = tmp_path / "item_balloon.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "1", fontsize=10)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        x0, y0, x1, y1 = bbox
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        page.draw_circle(center, self._enclosing_radius(bbox), color=(0, 0, 0), width=0.75)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_circle_around_text(0, bbox) is True
        pdf_doc.close()

    def test_no_nearby_circle_returns_false(self, tmp_path):
        pdf_path = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "1", fontsize=10)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_circle_around_text(0, bbox) is False
        pdf_doc.close()

    def test_distant_large_circle_is_not_mistaken_for_a_balloon(self, tmp_path):
        """A circle far larger than the text (e.g. the part's own hole or
        shaft geometry happening to sit near a dimension) must not count."""
        pdf_path = tmp_path / "big_circle.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "1", fontsize=10)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        x0, y0, x1, y1 = bbox
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        page.draw_circle(center, (x1 - x0) * 12.0, color=(0, 0, 0), width=0.75)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_circle_around_text(0, bbox) is False
        pdf_doc.close()

    def test_long_attached_leader_line_does_not_spoil_detection(self, tmp_path):
        """A leader line touching the circle at one end and running far
        away must not drag the cluster's bounding box out with it and
        spoil the aspect-ratio/size checks -- each candidate segment is
        pre-filtered to a short span before clustering for exactly this.
        """
        pdf_path = tmp_path / "with_leader.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "1", fontsize=10)
        bbox = tuple(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"])
        x0, y0, x1, y1 = bbox
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        radius = self._enclosing_radius(bbox)
        page.draw_circle(center, radius, color=(0, 0, 0), width=0.75)
        edge = (center[0] + radius * 0.7, center[1] + radius * 0.7)
        page.draw_line(edge, (edge[0] + 150, edge[1] + 150), color=(0, 0, 0), width=0.5)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        assert pdf_doc.find_vector_circle_around_text(0, bbox) is True
        pdf_doc.close()
