"""Tests for the OpenCV-based ink-plausibility check and the full
auto-balloon pipeline's ability to reject stray/invisible text artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from balloon_app.auto_balloon import _bbox_has_ink, auto_balloon_page
from balloon_app.data_model import CharacteristicType
from balloon_app.ocr_parser import DefaultTolerances
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


class TestDefaultTolerances:
    def test_default_tolerance_backfills_dimension_without_explicit_tolerance(self, tmp_path):
        """A dimension with no tolerance of its own (e.g. "9X Ø0.250 THRU",
        relying on the drawing's title-block "X.XXX: ±0.005" note) must get
        that default tolerance and matching limits when one is supplied.
        """
        pdf_path = tmp_path / "no_explicit_tolerance.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 200), "9X Ø0.250 THRU", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        defaults = DefaultTolerances(three_decimal=0.005)
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200, default_tolerances=defaults)
        pdf_doc.close()

        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.char_type == CharacteristicType.DIAMETER.value
        assert balloon.nominal == pytest.approx(0.25)
        assert balloon.tol_plus == pytest.approx(0.005)
        assert balloon.tol_minus == pytest.approx(0.005)
        assert balloon.lower_limit == pytest.approx(0.245)
        assert balloon.upper_limit == pytest.approx(0.255)

    def test_no_default_tolerances_leaves_dimension_untoleranced(self, tmp_path):
        pdf_path = tmp_path / "no_defaults_passed.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 200), "9X Ø0.250 THRU", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert result.balloons[0].tol_plus is None


class TestCountersinkDiameterAngleCallout:
    def test_countersink_diameter_and_angle_produce_two_balloons(self, tmp_path):
        """A countersink callout packs a diameter and an included angle into
        one line, e.g. "CSK Ø0.507 X 82°" -- must produce two balloons
        (Countersink diameter, Angle) instead of dropping the angle.
        """
        pdf_path = tmp_path / "countersink.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 190), "CSK Ø0.507 X 82°", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 2
        diameter, angle = result.balloons
        assert diameter.char_type == CharacteristicType.COUNTERSINK.value
        assert diameter.nominal == pytest.approx(0.507)
        assert angle.char_type == CharacteristicType.ANGLE.value
        assert angle.nominal == pytest.approx(82)

    def test_both_shape_symbols_mangled_still_splits_and_picks_diameter_default(self, tmp_path):
        """Real-world case: the countersink symbol extracted as "w" and the
        diameter symbol as "n" (both unrecognizable), leaving raw text
        "w n 0.507 X 82°". Must still produce two balloons (a value typed
        as Countersink, and an Angle), with raw_text rewritten to the
        canonical symbols -- and, critically, the diameter must pick up a
        *decimal-place* default tolerance, not the angular one (the bug
        that produced a nonsensical ±0.5 on a 0.507 diameter).
        """
        pdf_path = tmp_path / "mangled_countersink.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 190), "w n 0.507 X 82°", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        defaults = DefaultTolerances(three_decimal=0.005, angular=0.5)
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200, default_tolerances=defaults)
        pdf_doc.close()

        assert len(result.balloons) == 2
        value, angle = result.balloons
        assert value.char_type == CharacteristicType.COUNTERSINK.value
        assert value.nominal == pytest.approx(0.507)
        assert value.raw_text == "⌵⌀0.507 X 82°"
        assert value.tol_plus == pytest.approx(0.005)  # decimal-place default, not angular
        assert angle.char_type == CharacteristicType.ANGLE.value
        assert angle.nominal == pytest.approx(82)


class TestTitleBlockExclusion:
    def test_title_block_note_is_not_ballooned(self, tmp_path):
        """A drawing's title block (numeric tolerance table, fractional
        callouts, etc.) must never be auto-ballooned, even though its text
        would otherwise look exactly like real dimensions/tolerances.
        """
        pdf_path = tmp_path / "with_title_block.pdf"
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        # A real dimension, well above the title block.
        page.insert_text((50, 200), "0.750 ±0.005", fontsize=12)
        # The title block: bottom strip of the sheet, ANSI-style.
        page.insert_text((50, 700), "TOLERANCES UNLESS OTHERWISE NOTED", fontsize=10)
        page.insert_text((50, 715), "FRACTIONAL: 1/64  DECIMAL: X.XX: 0.0100", fontsize=10)
        page.insert_text((50, 730), "ANGLES: 0.5  FINISH: 125 MICRO INCHES", fontsize=10)
        page.insert_text((50, 745), "DRAWN: vernon  DATE: 2/20/2004", fontsize=10)
        page.insert_text((400, 745), "A2048", fontsize=10)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        raw_texts = [b.raw_text for b in result.balloons]
        assert any("0.750" in t for t in raw_texts)
        assert not any("TOLERANCES" in t or "FRACTIONAL" in t or "A2048" in t for t in raw_texts)

    def test_no_title_block_keywords_leaves_page_unaffected(self, tmp_path):
        pdf_path = tmp_path / "no_title_block.pdf"
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 200), "0.750 ±0.005", fontsize=12)
        page.insert_text((50, 700), "0.500 ±0.010", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        raw_texts = [b.raw_text for b in result.balloons]
        assert any("0.750" in t for t in raw_texts)
        assert any("0.500" in t for t in raw_texts)
