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
