"""Tests for the rule-based dimension/tolerance/GD&T text parser."""

from __future__ import annotations

import math

from balloon_app.data_model import CharacteristicType
from balloon_app.ocr_parser import compute_limits, parse_characteristic, parse_characteristics


def _close(a, b, tol=1e-6):
    return a is not None and b is not None and math.isclose(a, b, abs_tol=tol)


class TestBilateralSymmetric:
    def test_basic(self):
        result = parse_characteristic("50.00 ±0.05")
        assert result.char_type == CharacteristicType.LINEAR_DIMENSION.value
        assert _close(result.nominal, 50.00)
        assert _close(result.tol_plus, 0.05)
        assert _close(result.tol_minus, 0.05)
        assert _close(result.lower_limit, 49.95)
        assert _close(result.upper_limit, 50.05)

    def test_spaced(self):
        result = parse_characteristic("50 ± 0.1")
        assert _close(result.nominal, 50)
        assert _close(result.tol_plus, 0.1)
        assert _close(result.tol_minus, 0.1)


class TestBilateralAsymmetric:
    def test_slash_form(self):
        result = parse_characteristic("25.00 +0.02/-0.01")
        assert _close(result.nominal, 25.00)
        assert _close(result.tol_plus, 0.02)
        assert _close(result.tol_minus, 0.01)
        assert _close(result.lower_limit, 24.99)
        assert _close(result.upper_limit, 25.02)

    def test_space_form_leading_dot(self):
        result = parse_characteristic("25.00 +.02 -.01")
        assert _close(result.nominal, 25.00)
        assert _close(result.tol_plus, 0.02)
        assert _close(result.tol_minus, 0.01)


class TestLimitDimensions:
    def test_basic_limits(self):
        result = parse_characteristic("10.0 - 10.2")
        assert _close(result.lower_limit, 10.0)
        assert _close(result.upper_limit, 10.2)
        assert result.nominal is None


class TestDiameterAndRadius:
    def test_diameter_symbol_unicode(self):
        result = parse_characteristic("Ø25")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 25)

    def test_diameter_alt_symbol(self):
        result = parse_characteristic("⌀25")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 25)

    def test_diameter_with_tolerance(self):
        result = parse_characteristic("Ø25 ±0.05")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 25)
        assert _close(result.tol_plus, 0.05)

    def test_radius(self):
        result = parse_characteristic("R5")
        assert result.char_type == CharacteristicType.RADIUS.value
        assert _close(result.nominal, 5)

    def test_radius_not_confused_with_ra(self):
        result = parse_characteristic("Ra 1.6")
        assert result.char_type == CharacteristicType.SURFACE_FINISH.value


class TestHoleFeatureModifiers:
    def test_depth_symbol(self):
        result = parse_characteristic("▼0.500")
        assert result.char_type == CharacteristicType.DEPTH.value
        assert _close(result.nominal, 0.5)

    def test_counterbore_symbol_with_diameter(self):
        result = parse_characteristic("⌴⌀0.750")
        assert result.char_type == CharacteristicType.COUNTERBORE.value
        assert _close(result.nominal, 0.75)

    def test_counterbore_keyword(self):
        result = parse_characteristic("C'BORE 0.750")
        assert result.char_type == CharacteristicType.COUNTERBORE.value

    def test_countersink_symbol_with_diameter(self):
        result = parse_characteristic("⌵⌀0.500 X 82°")
        assert result.char_type == CharacteristicType.COUNTERSINK.value

    def test_countersink_keyword(self):
        result = parse_characteristic("CSK 0.500 X 82")
        assert result.char_type == CharacteristicType.COUNTERSINK.value

    def test_square_symbol(self):
        result = parse_characteristic("□1.250")
        assert result.char_type == CharacteristicType.SQUARE.value
        assert _close(result.nominal, 1.25)

    def test_square_keyword(self):
        result = parse_characteristic("SQ 1.250")
        assert result.char_type == CharacteristicType.SQUARE.value

    def test_diameter_with_depth_splits_into_two(self):
        # "2X Ø0.089 ▼0.500" -- a hole diameter and its depth, checked with
        # different gauges, packed into one line of drawing text.
        results = parse_characteristics("2X Ø0.089 ▼0.500")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 0.089)
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 0.5)

    def test_diameter_without_depth_is_a_single_characteristic(self):
        results = parse_characteristics("2X Ø0.089")
        assert len(results) == 1
        assert results[0].char_type == CharacteristicType.DIAMETER.value

    def test_square_with_depth_splits_into_two(self):
        results = parse_characteristics("SQ 1.250 ▼0.375")
        assert len(results) == 2
        assert results[0].char_type == CharacteristicType.SQUARE.value
        assert _close(results[0].nominal, 1.25)
        assert results[1].char_type == CharacteristicType.DEPTH.value
        assert _close(results[1].nominal, 0.375)

    def test_mangled_symbols_still_split_via_qty_structural_fallback(self):
        # Both the diameter symbol (Ø) and the depth symbol (▼) were mangled
        # into unrelated letters ("n", "x") by this drawing's CAD PDF export
        # font -- no recognizable symbol at all, so this must fall back to
        # the quantity-prefix structural heuristic rather than misreading
        # the "2" in "2X" as the nominal.
        results = parse_characteristics("2X n 0.089 x 0.500")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 0.089)
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 0.5)

    def test_qty_prefix_does_not_contaminate_single_value_nominal(self):
        # "9X n0.250 THRU" -- diameter symbol mangled and no second numeric
        # value, so no split, but the "9" instance count must still not be
        # picked up as the nominal.
        result = parse_characteristic("9X n0.250 THRU")
        assert _close(result.nominal, 0.25)

    def test_qty_prefix_with_real_tolerance_is_not_mistaken_for_two_values(self):
        # "2X 0.500 ±0.010" is one dimension with a symmetric tolerance --
        # must not be split into two characteristics.
        results = parse_characteristics("2X 0.500 ±0.010")
        assert len(results) == 1
        assert _close(results[0].nominal, 0.5)
        assert _close(results[0].tol_plus, 0.01)


class TestAngle:
    def test_angle_with_tolerance(self):
        result = parse_characteristic("45° ±1°")
        assert result.char_type == CharacteristicType.ANGLE.value
        assert _close(result.nominal, 45)
        assert _close(result.tol_plus, 1)
        assert _close(result.tol_minus, 1)


class TestThread:
    def test_metric_thread(self):
        result = parse_characteristic("M6 x 1.0")
        assert result.char_type == CharacteristicType.THREAD.value
        assert result.thread_callout == "M6 x 1.0"

    def test_unified_thread(self):
        result = parse_characteristic("1/4-20 UNC")
        assert result.char_type == CharacteristicType.THREAD.value
        assert "UNC" in result.thread_callout
        assert "1/4-20" in result.thread_callout

    def test_thread_without_depth_is_a_single_characteristic(self):
        results = parse_characteristics("1/4-20 UNC")
        assert len(results) == 1

    def test_thread_with_triangle_depth_symbol_splits_into_two(self):
        results = parse_characteristics("8-32 UNC-2B ▼0.500")
        assert len(results) == 2
        thread, depth = results
        assert thread.char_type == CharacteristicType.THREAD.value
        assert thread.thread_callout == "8-32 UNC"
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 0.5)

    def test_thread_with_deep_keyword_splits_into_two(self):
        results = parse_characteristics("M6 x 1.0 0.500 DEEP")
        assert len(results) == 2
        assert _close(results[1].nominal, 0.5)

    def test_thread_with_mis_extracted_depth_glyph_still_splits(self):
        # Real-world CAD PDF exports render the "depth" glyph via a custom
        # symbol font; without a proper ToUnicode mapping it can extract as
        # an unrelated ASCII character (observed: "x") instead of an arrow.
        results = parse_characteristics("8-32 UNC - 2B x 0.500")
        assert len(results) == 2
        thread, depth = results
        assert thread.thread_callout == "8-32 UNC"
        assert _close(depth.nominal, 0.5)

    def test_thread_class_code_alone_is_not_mistaken_for_depth(self):
        # "2B" is a bare integer glued to a letter (a thread class code),
        # not a decimal -- must not be picked up as a depth value.
        results = parse_characteristics("1/4-20 UNC-2B")
        assert len(results) == 1


class TestSurfaceFinish:
    def test_prefix_form(self):
        result = parse_characteristic("Ra 1.6")
        assert result.char_type == CharacteristicType.SURFACE_FINISH.value
        assert "1.6" in result.surface_finish

    def test_suffix_form(self):
        result = parse_characteristic("3.2 Ra")
        assert result.char_type == CharacteristicType.SURFACE_FINISH.value
        assert "3.2" in result.surface_finish


class TestGdt:
    def test_symbol_with_tolerance_and_datums(self):
        result = parse_characteristic("⌖ ⌀0.1 Ⓜ A B C")
        assert result.char_type == CharacteristicType.GDT_FRAME.value
        assert result.gdt_symbol == "Position"
        assert result.material_condition == "MMC"
        assert result.datums == "A, B, C"

    def test_pipe_fallback_without_symbol(self):
        result = parse_characteristic("0.05|A|B|C")
        assert result.char_type == CharacteristicType.GDT_FRAME.value
        assert result.datums == "A, B, C"


class TestGeneralToleranceAndFallback:
    def test_general_tolerance_note(self):
        result = parse_characteristic("UNLESS OTHERWISE SPECIFIED ±0.1")
        assert result.char_type == CharacteristicType.GENERAL_TOLERANCE.value
        assert _close(result.tol_plus, 0.1)

    def test_unrecognized_text_preserved(self):
        text = "SEE NOTE 3 FOR DETAILS"
        result = parse_characteristic(text)
        assert result.raw_text == text
        assert result.confidence < 0.5

    def test_empty_text(self):
        result = parse_characteristic("")
        assert result.confidence == 0.0


class TestAutoBalloonPreFilter:
    """Regression coverage for balloon_app.auto_balloon._looks_like_characteristic.

    These specific strings were observed causing false-positive balloons on
    a real title-blocked drawing: bare sheet zone markers, dates, QTY
    counts, drawing numbers, and material temper codes.
    """

    def test_rejects_bare_zone_markers(self):
        from balloon_app.auto_balloon import _looks_like_characteristic

        for text in ["1", "2", "3", "4", "(1)", "23"]:
            assert _looks_like_characteristic(text) is False, text

    def test_rejects_title_block_noise(self):
        from balloon_app.auto_balloon import _looks_like_characteristic

        for text in ["3/23/2004", "QTY: 4", "A1539", "6061 T6"]:
            assert _looks_like_characteristic(text) is False, text

    def test_accepts_real_dimensions(self):
        from balloon_app.auto_balloon import _looks_like_characteristic

        for text in ["19.43", "0.25", "0.063", "0.75", "R5", "Ø25", "M6 x 1.0", "1/4-20 UNC", "45°"]:
            assert _looks_like_characteristic(text) is True, text


class TestComputeLimits:
    def test_symmetric(self):
        lower, upper = compute_limits(10.0, 0.1, 0.1)
        assert _close(lower, 9.9)
        assert _close(upper, 10.1)

    def test_missing_nominal(self):
        lower, upper = compute_limits(None, 0.1, 0.1)
        assert lower is None
        assert upper is None
