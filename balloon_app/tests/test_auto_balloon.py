"""Tests for the OpenCV-based ink-plausibility check and the full
auto-balloon pipeline's ability to reject stray/invisible text artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from balloon_app.auto_balloon import _bbox_has_ink, auto_balloon_page
from balloon_app.data_model import CharacteristicType
from balloon_app.pdf_engine import PdfDocument

fitz = pytest.importorskip("pymupdf")


def _blank_white_image(width=200, height=200) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _image_with_dark_square(width=200, height=200) -> np.ndarray:
    img = np.full((height, width), 255, dtype=np.uint8)
    img[50:70, 50:100] = 0  # a block of black "ink"
    return img


class TestBboxHasInk:
    def test_blank_region_has_no_ink(self):
        img = _blank_white_image()
        assert _bbox_has_ink(img, (10, 10, 40, 30)) is False

    def test_region_with_ink_detected(self):
        img = _image_with_dark_square()
        assert _bbox_has_ink(img, (50, 50, 100, 70)) is True

    def test_out_of_bounds_bbox_is_safe(self):
        img = _blank_white_image()
        assert _bbox_has_ink(img, (-50, -50, -10, -10)) is False
        assert _bbox_has_ink(img, (1000, 1000, 1010, 1010)) is False


class TestAutoBalloonRejectsInvisibleText:
    def test_invisible_text_layer_is_not_ballooned(self, tmp_path):
        """A text span with no corresponding rendered ink (e.g. a stray
        invisible/misaligned OCR artifact) must not produce a balloon, even
        though it passes the regex pre-filter on its own.
        """
        pdf_path = tmp_path / "invisible_text.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        # render_mode=3 draws nothing (invisible text) but IS still returned
        # by get_text(), simulating a stray/misaligned text-layer artifact.
        page.insert_text((50, 50), "19.43", fontsize=10, render_mode=3)
        # One real, visible dimension so the page isn't entirely empty.
        page.insert_text((50, 150), "0.75", fontsize=10)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        raw_texts = [b.raw_text for b in result.balloons]
        assert "19.43" not in raw_texts
        assert "0.75" in raw_texts

    def test_dimensions_on_rotated_page_are_not_discarded(self, tmp_path):
        """Regression test: a page with /Rotate 90 (typical for a landscape
        CAD drawing exported to a portrait mediabox) must not have its real,
        visible dimensions discarded by the ink-plausibility check. Before
        the pdf_engine.py fix, extract_text_blocks() returned bboxes in the
        pre-rotation mediabox space while the rendered raster used to check
        for ink was in the rotated display space, so every dimension's bbox
        landed on a blank part of the page and got silently dropped.
        """
        pdf_path = tmp_path / "rotated_drawing.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=800)
        page.insert_text((50, 200), "0.500", fontsize=14)
        page.insert_text((50, 400), "1.500", fontsize=14)
        page.set_rotation(90)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        raw_texts = [b.raw_text for b in result.balloons]
        assert "0.500" in raw_texts
        assert "1.500" in raw_texts


class TestCompoundThreadDepthCallout:
    def test_tapped_hole_with_depth_produces_two_balloons(self, tmp_path):
        """A tapped-hole callout like '8-32 UNC-2B 0.500 DEEP' packs two
        independently-inspected requirements (thread class + depth) into
        one piece of drawing text, so it must produce two separate,
        consecutively-numbered balloons rather than a single Thread balloon
        that silently drops the depth value.

        Uses the ASCII "DEEP" depth keyword rather than the ▼ symbol here
        since the base Helvetica font PyMuPDF uses for synthetic test PDFs
        has no glyph for U+25BC (it round-trips as a replacement
        character); the ▼-symbol regex path itself is covered directly in
        test_ocr_parser.py without going through PDF text rendering.
        """
        pdf_path = tmp_path / "tapped_hole.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 200), "8-32 UNC-2B 0.500 DEEP", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 2
        thread, depth = result.balloons
        assert thread.char_type == CharacteristicType.THREAD.value
        assert thread.thread_callout == "8-32 UNC"
        assert thread.number == 1
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert depth.nominal == pytest.approx(0.5)
        assert depth.number == 2
        # Both balloons trace back to the same source text, but are placed
        # at distinct points so neither balloon hides the other.
        assert thread.raw_text == depth.raw_text
        assert (thread.x, thread.y) != (depth.x, depth.y)

    def test_tapped_hole_callout_split_across_two_lines_produces_three_balloons(self, tmp_path):
        """A tapped-hole callout is commonly drawn as two lines: the hole's
        diameter and depth on one line, its thread spec on the next --
        e.g. "2X Ø0.089 0.500 DEEP" / "4-40 UNC-2B". That must produce
        three balloons total: Diameter, Depth, and Thread.
        """
        pdf_path = tmp_path / "tapped_hole_two_lines.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 190), "2X Ø0.089 0.500 DEEP", fontsize=12)
        page.insert_text((50, 210), "4-40 UNC-2B", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        char_types = sorted(b.char_type for b in result.balloons)
        assert char_types == sorted([
            CharacteristicType.DIAMETER.value,
            CharacteristicType.DEPTH.value,
            CharacteristicType.THREAD.value,
        ])
