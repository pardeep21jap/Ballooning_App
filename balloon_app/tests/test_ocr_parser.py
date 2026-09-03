"""Tests for the rule-based dimension/tolerance/GD&T text parser."""

from __future__ import annotations

import math

from balloon_app.data_model import CharacteristicType
from balloon_app.ocr_parser import compute_limits, parse_characteristic


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
