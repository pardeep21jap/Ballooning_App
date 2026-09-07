"""Tests for the OpenCV-based ink-plausibility check and the full
auto-balloon pipeline's ability to reject stray/invisible text artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from balloon_app.auto_balloon import _bbox_has_ink, _merge_stacked_tolerance_fragments, auto_balloon_page
from balloon_app.data_model import CharacteristicType
from balloon_app.ocr_parser import DefaultTolerances
from balloon_app.pdf_engine import PdfDocument, TextBlock

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
        defaults = DefaultTolerances(by_decimal_places={3: 0.005})
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
        defaults = DefaultTolerances(by_decimal_places={3: 0.005}, angular=0.5)
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


class TestStackedToleranceFragments:
    def test_plus_above_and_minus_below_merge_into_one(self):
        # A common drawing convention: the nominal on its own line, with
        # the asymmetric tolerance stacked above/below it as separate lines.
        #     +0.005
        # Ø6.38
        #     -0.010
        blocks = [
            TextBlock(text="+0.005", bbox=(100, 180, 130, 190)),
            TextBlock(text="Ø6.38", bbox=(80, 195, 130, 208)),
            TextBlock(text="-0.010", bbox=(100, 210, 130, 220)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 1
        assert merged[0].text == "Ø6.38 +0.005 -0.010"

    def test_only_one_sign_fragment_still_merges(self):
        blocks = [
            TextBlock(text="6.38", bbox=(80, 195, 110, 208)),
            TextBlock(text="-0.010", bbox=(100, 210, 130, 220)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 1
        assert merged[0].text == "6.38 -0.010"

    def test_datum_letter_does_not_steal_a_nearby_fragment(self):
        # A bare datum-reference letter ("B") happens to sit closer (in
        # raw distance) to a "+0.005" fragment than the real numeric
        # dimension does. Since "B" has no digit it must never be treated
        # as an anchor -- the fragment has to reach past it to the real
        # dimension regardless of which one is geometrically nearest.
        blocks = [
            TextBlock(text="B", bbox=(115, 175, 123, 190)),
            TextBlock(text="+0.005", bbox=(120, 180, 150, 190)),
            TextBlock(text="Ø6.38 -0.010", bbox=(80, 195, 160, 208)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        texts = {b.text for b in merged}
        assert "Ø6.38 +0.005 -0.010" in texts
        assert "B" in texts

    def test_fragment_goes_to_the_closer_of_two_eligible_anchors(self):
        # Two numeric dimensions are both technically "in range" of one
        # fragment; it must go to whichever is actually closer.
        blocks = [
            TextBlock(text="+0.005", bbox=(120, 180, 150, 190)),
            TextBlock(text="Ø6.38", bbox=(80, 195, 130, 208)),  # closer
            TextBlock(text="12.70", bbox=(80, 100, 130, 113)),  # farther
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        texts = {b.text for b in merged}
        assert "Ø6.38 +0.005" in texts
        assert "12.70" in texts

    def test_unilateral_tolerance_bare_zero_fills_the_missing_side(self):
        # A unilateral tolerance ("0 / +0.05"): the unsigned "0" -- already
        # same-line merged with the nominal into "Ø8 0" -- means "no
        # negative deviation", conventionally written without a sign since
        # +0 and -0 are equivalent. It must become tol_minus=0, not get
        # left in the nominal text or dropped.
        blocks = [
            TextBlock(text="+0.05", bbox=(190, 295, 220, 305)),
            TextBlock(text="Ø8 0", bbox=(180, 305, 230, 318)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 1
        assert merged[0].text == "Ø8 +0.05 -0"

    def test_bare_zero_not_stripped_without_a_signed_fragment_nearby(self):
        # No external +/- fragment in range -- must not touch the block at
        # all (a bare "Ø8 0" with nothing else is left for the normal
        # single-value path to handle, whatever it decides).
        blocks = [TextBlock(text="Ø8 0", bbox=(180, 305, 230, 318))]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert merged == blocks

    def test_minus_already_same_line_merged_with_nominal_reorders_correctly(self):
        # Real-world layout: "Ø6.38 -0.010" sit on the same line and were
        # already combined by the same-line merge; "+0.005" is a separate
        # line above. Appending "+0.005" naively would produce "Ø6.38
        # -0.010 +0.005", which the asymmetric-tolerance parser (which
        # requires + before -) fails to recognize as a tolerance at all --
        # the embedded "-0.010" must be pulled out and re-emitted after
        # the plus, in canonical order.
        blocks = [
            TextBlock(text="+0.005", bbox=(120, 180, 150, 190)),
            TextBlock(text="Ø6.38 -0.010", bbox=(80, 195, 160, 208)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 1
        assert merged[0].text == "Ø6.38 +0.005 -0.010"

    def test_unrelated_bare_number_far_away_does_not_merge(self):
        blocks = [
            TextBlock(text="6.38", bbox=(80, 195, 110, 208)),
            TextBlock(text="+0.010", bbox=(80, 500, 110, 512)),  # far below -- unrelated
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 2

    def test_double_positive_fragments_both_merge_not_just_the_first(self):
        # A less common but valid style: both stacked deviations share the
        # same sign ("+.3" over "+.1", not the usual "+X above / -Y
        # below"). Previously only the first same-sign fragment encountered
        # was bucketed and the second was silently left behind as its own
        # stray candidate/balloon.
        blocks = [
            TextBlock(text="+.3", bbox=(100, 180, 115, 190)),
            TextBlock(text="Ø12.0", bbox=(80, 195, 130, 208)),
            TextBlock(text="+.1", bbox=(100, 210, 115, 220)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 1
        assert merged[0].text == "Ø12.0 +.3 +.1"

    def test_tolerance_inserted_after_the_value_not_at_the_tail_of_a_compound_anchor(self):
        # The anchor block may already be a compound "Ø12.0 ↧ 100.0" --
        # same-line-merged with a trailing depth clause *before* this
        # stacked-tolerance pass ever runs. Appending the tolerance at the
        # tail of that (the old behavior) would misattach it past the
        # depth instead of to the diameter it actually modifies.
        blocks = [
            TextBlock(text="+.3", bbox=(133, 180, 148, 190)),
            TextBlock(text="Ø12.0 ↧ 100.0", bbox=(80, 195, 230, 208)),
            TextBlock(text="+.1", bbox=(133, 210, 148, 220)),
        ]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert len(merged) == 1
        assert merged[0].text == "Ø12.0 +.3 +.1 ↧ 100.0"

    def test_no_sign_fragments_leaves_blocks_untouched(self):
        blocks = [TextBlock(text="6.38", bbox=(80, 195, 110, 208))]
        merged = _merge_stacked_tolerance_fragments(blocks)
        assert merged == blocks

    def test_end_to_end_nominal_and_minus_same_line_plus_above(self, tmp_path):
        """Exact real-world layout: "Ø6.38 -0.010" on one line, "+0.005" on
        a separate line above it. Must still produce one balloon with the
        correct +0.005/-0.010 tolerance, not a bare nominal with none.
        """
        pdf_path = tmp_path / "same_line_minus.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((120, 180), "+0.005", fontsize=10)
        page.insert_text((80, 195), "Ø6.38 -0.010", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.nominal == pytest.approx(6.38)
        assert balloon.tol_plus == pytest.approx(0.005)
        assert balloon.tol_minus == pytest.approx(0.010)

    def test_end_to_end_stacked_tolerance_produces_one_toleranced_balloon(self, tmp_path):
        """Full pipeline: a diameter with its +/- tolerance stacked as
        separate lines must produce ONE balloon with the correct tolerance,
        not two (or three) meaningless fragment balloons.
        """
        pdf_path = tmp_path / "stacked_tolerance.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 180), "+0.005", fontsize=10)
        page.insert_text((80, 195), "Ø6.38", fontsize=12)
        page.insert_text((100, 210), "-0.010", fontsize=10)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.char_type == CharacteristicType.DIAMETER.value
        assert balloon.nominal == pytest.approx(6.38)
        assert balloon.tol_plus == pytest.approx(0.005)
        assert balloon.tol_minus == pytest.approx(0.010)

    def test_end_to_end_unilateral_tolerance(self, tmp_path):
        """Full pipeline for a unilateral tolerance drawn as "Ø8 0" on one
        line and "+0.05" stacked above it -- must produce one balloon with
        tol_plus=0.05, tol_minus=0 (limits [8.0, 8.05]), not a bare nominal
        with no tolerance at all.
        """
        pdf_path = tmp_path / "unilateral_tolerance.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((190, 180), "+0.05", fontsize=10)
        page.insert_text((180, 195), "Ø8   0", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.nominal == pytest.approx(8.0)
        assert balloon.tol_plus == pytest.approx(0.05)
        assert balloon.tol_minus == pytest.approx(0.0)
        assert balloon.lower_limit == pytest.approx(8.0)
        assert balloon.upper_limit == pytest.approx(8.05)


class TestZoneMarginWholeNumberDimensions:
    def test_whole_number_dimension_ballooned_but_zone_markers_are_not(self, tmp_path):
        """Whole-number metric dimensions ("75", "19") inside the drawing
        body must be ballooned like any other dimension, while the sheet's
        zone/grid reference numbers ("1"-"4" along the top margin) -- which
        look identical as bare text -- must still be excluded.
        """
        pdf_path = tmp_path / "whole_number_dims.pdf"
        doc = fitz.open()
        page = doc.new_page(width=1000, height=800)
        # Zone markers: thin strip just inside the top edge.
        for i, x in enumerate([100, 350, 600, 850], start=1):
            page.insert_text((x, 10), str(i), fontsize=8)
        # Real whole-number dimensions, well inside the drawing body.
        page.insert_text((400, 300), "75", fontsize=12)
        page.insert_text((400, 400), "19", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        raw_texts = {b.raw_text for b in result.balloons}
        assert "75" in raw_texts
        assert "19" in raw_texts
        assert not ({"1", "2", "3", "4"} & raw_texts)


class TestRerunSkipsAlreadyBalloonedRegions:
    def test_second_run_does_not_reballoon_existing_regions(self, tmp_path):
        """Re-running auto-balloon on a page that's already been ballooned
        (manual re-run, or picking up where "Auto-Balloon Entire Drawing"
        left off) must not propose duplicates of characteristics that are
        already ballooned there."""
        pdf_path = tmp_path / "rerun.pdf"
        doc = fitz.open()
        page = doc.new_page(width=600, height=600)
        page.insert_text((200, 200), "50.00 ±0.05", fontsize=12)
        page.insert_text((200, 300), "Ø25.0", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        first = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        assert len(first.balloons) == 2

        second = auto_balloon_page(pdf_doc, "drawing-1", 0, first.balloons, 3, dpi=200)
        pdf_doc.close()

        assert second.balloons == []

    def test_new_region_still_ballooned_alongside_existing(self, tmp_path):
        """A genuinely new characteristic added after the first pass must
        still be proposed, even though other regions on the page are
        already ballooned."""
        pdf_path = tmp_path / "rerun_partial.pdf"
        doc = fitz.open()
        page = doc.new_page(width=600, height=600)
        page.insert_text((200, 200), "50.00 ±0.05", fontsize=12)
        page.insert_text((200, 300), "Ø25.0", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        first = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        only_first = first.balloons[:1]

        second = auto_balloon_page(pdf_doc, "drawing-1", 0, only_first, 2, dpi=200)
        pdf_doc.close()

        assert len(second.balloons) == 1
        assert second.balloons[0].raw_text != only_first[0].raw_text


def _draw_small_circle(page, near_bbox, radius=3.0, gap=1.5):
    """Draw a small vector circle just left of ``near_bbox``, mimicking a
    CAD PDF export that draws the Ø glyph as line art instead of text."""
    x0, y0, _x1, y1 = near_bbox
    cy = (y0 + y1) / 2.0
    cx = x0 - gap - radius
    shape = page.new_shape()
    shape.draw_circle((cx, cy), radius)
    shape.finish()
    shape.commit()


def _draw_small_triangle(page, near_bbox, offset=4.0):
    """Draw a small 3-segment triangle just left of ``near_bbox``, mimicking
    a dimension-line arrowhead that must not be mistaken for a Ø glyph."""
    x0, y0, _x1, y1 = near_bbox
    cy = (y0 + y1) / 2.0
    cx = x0 - offset
    shape = page.new_shape()
    shape.draw_line((cx - 2, cy - 3), (cx + 2, cy))
    shape.draw_line((cx + 2, cy), (cx - 2, cy + 3))
    shape.draw_line((cx - 2, cy + 3), (cx - 2, cy - 3))
    shape.finish()
    shape.commit()


class TestVectorDrawnDiameterSymbol:
    """Some CAD PDF exporters draw the Ø glyph as vector line art (a traced
    circle) instead of a font character, so it never appears in the page's
    extracted text at all -- unlike the countersink/depth glyph mangling
    handled elsewhere in this file, there is no substitute character to
    recover here. The app must still recognize these as diameters by
    noticing the small circular vector shape next to the bare number.
    """

    def test_bare_number_with_adjacent_vector_circle_becomes_diameter(self, tmp_path):
        pdf_path = tmp_path / "vector_diameter.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "8", fontsize=12)
        text_bbox = page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"]
        _draw_small_circle(page, text_bbox)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.char_type == CharacteristicType.DIAMETER.value
        assert balloon.nominal == pytest.approx(8.0)
        assert balloon.raw_text == "⌀8"

    def test_bare_number_without_nearby_shape_stays_linear_dimension(self, tmp_path):
        pdf_path = tmp_path / "no_symbol.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "8", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        assert result.balloons[0].char_type == CharacteristicType.LINEAR_DIMENSION.value
        assert result.balloons[0].raw_text == "8"

    def test_nearby_arrowhead_triangle_is_not_mistaken_for_diameter_symbol(self, tmp_path):
        """A dimension-line arrowhead is a compact few-segment shape too, but
        must not false-positive as a diameter glyph."""
        pdf_path = tmp_path / "arrowhead.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((60, 200), "13", fontsize=12)
        text_bbox = page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"]
        _draw_small_triangle(page, text_bbox)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        assert result.balloons[0].char_type == CharacteristicType.LINEAR_DIMENSION.value


def _draw_gdt_frame(page, near_bbox, symbol_cell=True, divider=True):
    """Draw a boxed-compartment frame just left of ``near_bbox``, mimicking
    a CAD PDF export that draws a GD&T feature control frame's symbol as
    line art rather than a font character."""
    x0, y0, x1, y1 = near_bbox
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


class TestVectorDrawnGdtFrame:
    """A GD&T feature control frame's symbol (flatness, straightness, etc.)
    is almost always drawn as vector line art -- unlike the countersink/
    depth glyph mangling handled elsewhere in this file, there is no
    substitute character to recover here at all, only the frame's boxed-
    compartment structure. The app must at least recognize these as a GD&T
    frame (symbol left blank for review) instead of a plain dimension.
    """

    @pytest.mark.parametrize("symbol", ["flatness", "rectangle", "open"])
    @pytest.mark.parametrize("leader", [False, True])
    def test_vector_flatness(self, tmp_path, symbol, leader):
        pdf_path = tmp_path / "flatness.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        bbox = page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"]
        _draw_gdt_frame(page, bbox)
        x0, y0, _, y1 = bbox
        h = y1 - y0
        left = x0 - h * 1.15
        top, bottom = y0 + h * 0.25, y0 + h * 0.75
        shift = h * 0.25 if symbol != "rectangle" else 0
        points = [(left + shift, top), (left + shift + h * 0.5, top),
                  (left + h * 0.5, bottom), (left, bottom)]
        shape = page.new_shape()
        shape.draw_polyline(points if symbol == "open" else points + points[:1])
        shape.finish(closePath=False)
        shape.commit()
        if leader:
            # The leader joins a long extension line, as in the reported image.
            edge = x0 - h * 1.35
            middle = (y0 + y1) / 2
            page.draw_line((edge, middle), (edge - h, middle))
            page.draw_line((edge - h, middle - h * 0.2), (edge - h, middle + h * 4))
        doc.save(str(pdf_path))
        doc.close()
        pdf_doc = PdfDocument(pdf_path)
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()
        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.char_type == CharacteristicType.GDT_FRAME.value
        assert balloon.gdt_symbol == ("Flatness" if symbol == "flatness" else None)
        assert balloon.gdt_tolerance == "0.01"
        # A feature control frame's stated value is the maximum allowed
        # variation, zero being the best case: nominal is the box value,
        # upper limit equals it, lower limit is 0.0.
        assert balloon.nominal == pytest.approx(0.01)
        assert balloon.lower_limit == pytest.approx(0.0)
        assert balloon.upper_limit == pytest.approx(0.01)

    def test_bare_value_with_adjacent_frame_becomes_gdt_frame(self, tmp_path):
        pdf_path = tmp_path / "vector_gdt_frame.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        text_bbox = page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"]
        _draw_gdt_frame(page, text_bbox)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        balloon = result.balloons[0]
        assert balloon.char_type == CharacteristicType.GDT_FRAME.value
        assert balloon.gdt_tolerance == "0.01"

    def test_bare_value_without_nearby_frame_stays_linear_dimension(self, tmp_path):
        pdf_path = tmp_path / "no_frame.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((100, 200), "0.01", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        pdf_doc = PdfDocument(pdf_path)
        pdf_doc.open()
        result = auto_balloon_page(pdf_doc, "drawing-1", 0, [], 1, dpi=200)
        pdf_doc.close()

        assert len(result.balloons) == 1
        assert result.balloons[0].char_type == CharacteristicType.LINEAR_DIMENSION.value
