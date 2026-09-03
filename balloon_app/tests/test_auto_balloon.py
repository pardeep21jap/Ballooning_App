"""Tests for the OpenCV-based ink-plausibility check and the full
auto-balloon pipeline's ability to reject stray/invisible text artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from balloon_app.auto_balloon import _bbox_has_ink, auto_balloon_page
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
