"""Tests for the rule-based dimension/tolerance/GD&T text parser."""

from __future__ import annotations

import math

from balloon_app.data_model import CharacteristicType
from balloon_app.ocr_parser import (
    DefaultTolerances,
    ParsedCharacteristic,
    apply_default_tolerance,
    compute_limits,
    parse_characteristic,
    parse_characteristics,
    parse_default_tolerances,
)


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

    def test_bare_mangled_diameter_letter(self):
        # A CAD PDF export's custom font had no ToUnicode mapping for the
        # Ø glyph, so text extraction substituted an unrelated letter
        # ("n") -- with no other structure around the value (contrast the
        # compound cases in TestHoleFeatureModifiers that split into two
        # characteristics), a lone letter glued to a decimal value is read
        # as a mangled diameter symbol and rewritten to its canonical form.
        result = parse_characteristic("n 0.551")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 0.551)
        assert result.raw_text == "⌀0.551"

    def test_bare_mangled_diameter_letter_no_space(self):
        result = parse_characteristic("n0.551")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 0.551)

    def test_mangled_diameter_with_mangled_depth_splits_into_two(self):
        # Real-world case: both the diameter and depth glyphs were mangled
        # by the same CAD PDF export's custom font ("n" for Ø, "x" for
        # ▽) -- "n.130 x.50 MAX" (Ø.130 ▽.50 MAX). Unlike the bare
        # single-value fallback (which requires nothing else in the text),
        # a genuine trailing depth here must still split into two
        # characteristics rather than losing the depth value entirely.
        results = parse_characteristics("n .130 x .50 MAX")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 0.130)
        assert diameter.raw_text == "⌀.130"
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 0.50)

    def test_double_mangled_shape_symbol_with_tolerance_is_typed_from_first_marker(self):
        # Real-world case: a counterbore whose diameter symbol is *also*
        # mangled separately from the counterbore symbol itself -- "v
        # n.159 +.002 -.000" (⌴Ø.159 +.002/-.000). No depth trails this
        # one, so the secondary-value fallback declines it, but the
        # generic classification path still resolves the type from the
        # *first* (more specific) marker rather than defaulting to a bare
        # linear dimension, and keeps the tolerance the value already
        # carried.
        result = parse_characteristic("v n .159 +.002 -.000")
        assert _close(result.nominal, 0.159)
        assert _close(result.tol_plus, 0.002)
        assert _close(result.tol_minus, 0.0)
        assert result.guessed_symbol_marker == "v"
        assert result.raw_text == "⌀.159 +.002 -.000"

    def test_mangled_diameter_fallback_never_shadows_radius(self):
        # "R" is never a mangled substitute -- it's already an
        # unambiguous, real radius symbol in its own right.
        result = parse_characteristic("R0.551")
        assert result.char_type == CharacteristicType.RADIUS.value
        assert _close(result.nominal, 0.551)

    def test_mangled_diameter_fallback_requires_a_decimal_value(self):
        # A bare whole number glued to a letter (e.g. a drawing/revision
        # code like "A2048") must never be mistaken for a diameter -- a
        # real diameter is essentially always given to several decimal
        # places, an alphanumeric code never is.
        result = parse_characteristic("A2048")
        assert result.char_type != CharacteristicType.DIAMETER.value

    def test_bare_mangled_symbol_records_marker_for_learning(self):
        # The bare default guess (no learned_symbols given) still records
        # which marker letter it guessed from, so a later correction can
        # teach the app what "n" actually means for this drawing's font.
        result = parse_characteristic("n 0.551")
        assert result.guessed_symbol_marker == "n"

    def test_learned_marker_overrides_default_diameter_guess(self):
        # Once the user has taught the app that "w" means Countersink (not
        # the bare default assumption of Diameter) for this font, the same
        # marker on a different value must resolve the same way -- this is
        # exactly the generalization a per-exact-text memory can't provide.
        result = parse_characteristic("w 0.375", learned_symbols={"w": CharacteristicType.COUNTERSINK.value})
        assert result.char_type == CharacteristicType.COUNTERSINK.value
        assert _close(result.nominal, 0.375)
        assert result.guessed_symbol_marker == "w"
        assert result.raw_text == "⌵⌀0.375"

    def test_learned_marker_is_case_and_space_insensitive(self):
        result = parse_characteristic("W 0.375", learned_symbols={"w": CharacteristicType.SQUARE.value})
        assert result.char_type == CharacteristicType.SQUARE.value

    def test_unlearned_marker_still_falls_back_to_diameter_default(self):
        # A learned_symbols map that doesn't mention this particular
        # marker must not change the existing default behavior.
        result = parse_characteristic("n 0.551", learned_symbols={"w": CharacteristicType.SQUARE.value})
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert result.guessed_symbol_marker == "n"

    def test_learned_marker_applies_to_qty_prefixed_mangled_diameter(self):
        result = parse_characteristic(
            "9X w0.250 THRU", learned_symbols={"w": CharacteristicType.DEPTH.value}
        )
        assert result.char_type == CharacteristicType.DEPTH.value
        assert _close(result.nominal, 0.25)
        assert result.guessed_symbol_marker == "w"
        assert result.raw_text == "9X ▼0.250 THRU"


class TestHoleFeatureModifiers:
    def test_depth_symbol(self):
        result = parse_characteristic("▼0.500")
        assert result.char_type == CharacteristicType.DEPTH.value
        assert _close(result.nominal, 0.5)

    def test_hollow_triangle_depth_symbol(self):
        # "▽" -- some CAD tools/fonts render the depth glyph unfilled
        # rather than solid ("▼"); must be recognized the same way.
        result = parse_characteristic("▽0.500")
        assert result.char_type == CharacteristicType.DEPTH.value
        assert _close(result.nominal, 0.5)

    def test_depth_hint_classifies_bare_number_as_depth(self):
        # depth_hint mirrors diameter_hint: the caller found a depth glyph
        # drawn as vector line art (no character at all in the text, see
        # PdfDocument.find_vector_depth_symbol) immediately before a bare
        # number that would otherwise default to a plain linear dimension.
        result = parse_characteristic("12.0", depth_hint=True)
        assert result.char_type == CharacteristicType.DEPTH.value
        assert _close(result.nominal, 12.0)
        assert result.raw_text == "▼12.0"

    def test_depth_hint_is_redundant_but_harmless_when_symbol_already_present(self):
        result = parse_characteristic("▼0.500", depth_hint=True)
        assert result.char_type == CharacteristicType.DEPTH.value
        assert result.raw_text == "▼0.500"  # not double-prefixed

    def test_depth_hint_does_not_apply_once_a_more_specific_pattern_already_matched(self):
        # depth_hint only affects the generic bare-value fallback -- a
        # callout that already matches a more specific earlier pattern
        # (here, a thread) must be classified as that, not read as a depth
        # just because the caller also passed depth_hint=True.
        result = parse_characteristic("M4-6H", depth_hint=True)
        assert result.char_type == CharacteristicType.THREAD.value

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

    def test_countersink_diameter_with_angle_splits_into_two(self):
        # "⌵ Ø0.507 X 82°" -- a countersink's diameter and its included
        # angle, checked with different gauges, packed into one line.
        results = parse_characteristics("⌵Ø0.507 X 82°")
        assert len(results) == 2
        diameter, angle = results
        assert diameter.char_type == CharacteristicType.COUNTERSINK.value
        assert _close(diameter.nominal, 0.507)
        assert angle.char_type == CharacteristicType.ANGLE.value
        assert _close(angle.nominal, 82)

    def test_counterbore_diameter_with_angle_splits_into_two(self):
        results = parse_characteristics("⌴Ø0.750 X 90°")
        assert len(results) == 2
        assert results[0].char_type == CharacteristicType.COUNTERBORE.value
        assert results[1].char_type == CharacteristicType.ANGLE.value
        assert _close(results[1].nominal, 90)

    def test_bare_diameter_with_trailing_angle_splits_into_two(self):
        results = parse_characteristics("Ø0.500 X 82°")
        assert len(results) == 2
        assert results[0].char_type == CharacteristicType.DIAMETER.value
        assert results[1].char_type == CharacteristicType.ANGLE.value

    def test_angle_alone_is_not_mistaken_for_a_compound_callout(self):
        # No preceding shape symbol -- a plain angle dimension must stay one.
        results = parse_characteristics("45° ±1°")
        assert len(results) == 1
        assert results[0].char_type == CharacteristicType.ANGLE.value

    def test_both_shape_symbols_mangled_still_splits_value_and_angle(self):
        # Real-world case: the countersink symbol extracted as "w" and the
        # diameter symbol as "n" -- neither recognizable -- but "0.507 X
        # 82°" is still an unambiguous value-times-angle structural shape,
        # which is specifically the countersink notation. raw_text is
        # rewritten to the canonical symbols since the mangled letters
        # carry no real information.
        results = parse_characteristics("w n 0.507 X 82°")
        assert len(results) == 2
        value, angle = results
        assert value.char_type == CharacteristicType.COUNTERSINK.value
        assert _close(value.nominal, 0.507)
        assert value.raw_text == "⌵⌀0.507 X 82°"
        assert angle.char_type == CharacteristicType.ANGLE.value
        assert _close(angle.nominal, 82)
        assert angle.raw_text == "⌵⌀0.507 X 82°"

    def test_qty_prefixed_tolerance_dimension_not_mistaken_for_value_angle(self):
        # No degree sign at all -- must not trigger the value-then-angle fallback.
        results = parse_characteristics("2X 0.500 ±0.010")
        assert len(results) == 1

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

    def test_diameter_with_downwards_arrow_from_bar_depth_symbol_splits_into_two(self):
        # "6 x Ø3.3 ↧ 12.0" -- ↧ (U+21A7) is ASME's actual "depth" symbol,
        # distinct from the mangled/ambiguous glyphs handled elsewhere.
        results = parse_characteristics("6 x Ø3.3 ↧ 12.0")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 3.3)
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 12.0)

    def test_diameter_with_vector_drawn_depth_glyph_still_splits(self):
        # Real-world case: this drawing's CAD PDF export draws the depth
        # glyph as vector line art rather than a font character, so it
        # contributes no character at all to the extracted text -- not
        # even a mangled one. "6 x Ø3.3 12.0" (diameter then a bare
        # trailing number, no symbol of any kind between them) must still
        # split into diameter + depth, since nothing between the two
        # numbers looks like tolerance markup.
        results = parse_characteristics("6 x Ø3.3 12.0")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 3.3)
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 12.0)

    def test_diameter_with_asymmetric_tolerance_is_not_mistaken_for_depth(self):
        # A genuine tolerance on the diameter itself -- the "+"/"-" markup
        # between the nominal and the next number must block the new
        # symbol-agnostic depth fallback from misreading it as a depth.
        results = parse_characteristics("Ø6.38 +0.005 -0.010")
        assert len(results) == 1
        result = results[0]
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 6.38)
        assert _close(result.tol_plus, 0.005)
        assert _close(result.tol_minus, 0.010)

    def test_diameter_with_symmetric_tolerance_is_not_mistaken_for_depth(self):
        results = parse_characteristics("Ø0.250 ±0.005")
        assert len(results) == 1
        assert _close(results[0].tol_plus, 0.005)

    def test_diameter_with_double_positive_tolerance_and_depth_splits_into_two(self):
        # "Ø12.0 +.3 +.1 ↧ 100.0" -- a double-positive stacked tolerance
        # ("+.3" over "+.1", both on the same side of nominal) immediately
        # followed by a depth clause. Must split into exactly two
        # characteristics -- not three (the tolerance fragment must not
        # become its own stray balloon) -- and the tolerance must attach to
        # the diameter rather than being misread as the depth value.
        results = parse_characteristics("Ø12.0 +.3 +.1 ↧ 100.0")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 12.0)
        assert _close(diameter.lower_limit, 12.1)
        assert _close(diameter.upper_limit, 12.3)
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 100.0)

    def test_diameter_with_countersink_glyph_used_as_depth_splits_into_two(self):
        # "6 x Ø3.3 ⌵ 12.0" -- a drilled hole's diameter and depth, where
        # this drawing's CAD export happens to render the depth glyph as
        # "⌵" (ASME's countersink symbol) rather than "▼"/"↓". Since ⌵ only
        # means countersink when it immediately precedes a diameter symbol
        # ("⌵⌀..."), here -- trailing an already-stated diameter -- it must
        # be read as depth instead, not silently drop the 12.0.
        results = parse_characteristics("6 x Ø3.3 ⌵ 12.0")
        assert len(results) == 2
        diameter, depth = results
        assert diameter.char_type == CharacteristicType.DIAMETER.value
        assert _close(diameter.nominal, 3.3)
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 12.0)

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
        assert diameter.raw_text == "2X ⌀0.089 ▼0.500"
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 0.5)
        assert depth.raw_text == "2X ⌀0.089 ▼0.500"

    def test_qty_prefix_does_not_contaminate_single_value_nominal(self):
        # "9X n0.250 THRU" -- diameter symbol mangled and no second numeric
        # value, so no split, but the "9" instance count must still not be
        # picked up as the nominal. The mangled letter must still resolve
        # to a Diameter, not fall through to a plain linear dimension --
        # "THRU" is a note, not a second dimension to split on.
        result = parse_characteristic("9X n0.250 THRU")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 0.25)
        assert result.raw_text == "9X ⌀0.250 THRU"

    def test_qty_prefix_mangled_diameter_with_space_before_value(self):
        result = parse_characteristic("8X n 0.157 THRU")
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert _close(result.nominal, 0.157)
        assert result.raw_text == "8X ⌀0.157 THRU"

    def test_qty_prefix_mangled_diameter_fallback_never_shadows_radius(self):
        # "R" is never a mangled substitute -- it's already an
        # unambiguous, real radius symbol in its own right.
        result = parse_characteristic("9X R0.250 TYP")
        assert result.char_type == CharacteristicType.RADIUS.value
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

    def test_metric_thread_class_without_pitch(self):
        # "M4-6H" -- a tapped hole's thread size and internal tolerance
        # class, coarse pitch implied. Previously unmatched by any thread
        # pattern (no "x pitch", no UNC/UNF suffix), so it fell through to
        # being misread as a numeric range "4 to 6".
        result = parse_characteristic("M4-6H")
        assert result.char_type == CharacteristicType.THREAD.value
        assert result.thread_callout == "M4-6H"

    def test_metric_thread_class_preserves_case(self):
        # Case is meaningful: uppercase (6H) is an internal thread class,
        # lowercase (6g) an external one -- must never be normalized.
        result = parse_characteristic("M6-6g")
        assert result.thread_callout == "M6-6g"

    def test_metric_thread_class_with_explicit_pitch(self):
        result = parse_characteristic("M10 x 1.5-6H")
        assert result.char_type == CharacteristicType.THREAD.value
        assert result.thread_callout == "M10 x 1.5-6H"

    def test_metric_thread_class_with_depth_splits_into_two(self):
        # "M4 - 6H ⌵ 8.0" -- a tapped hole's thread class and its tapped
        # depth, packed into one line, with the ambiguous "⌵" depth glyph
        # (see the countersink/depth split above) sitting between them.
        results = parse_characteristics("M4 - 6H ⌵ 8.0")
        assert len(results) == 2
        thread, depth = results
        assert thread.char_type == CharacteristicType.THREAD.value
        assert thread.thread_callout == "M4-6H"
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 8.0)

    def test_metric_thread_class_with_downwards_arrow_depth_splits_into_two(self):
        results = parse_characteristics("M4 - 6H ↧ 8.0")
        assert len(results) == 2
        thread, depth = results
        assert thread.thread_callout == "M4-6H"
        assert _close(depth.nominal, 8.0)

    def test_metric_thread_class_with_en_dash_separator(self):
        # "M4 – 6H" -- word-processing "smart typography" (and CAD
        # annotation editors built on it) silently converts a spaced hyphen
        # into an en dash as it's typed, so a drawing authored as "M4 - 6H"
        # can end up with this character in its extracted PDF text. Must
        # still be recognized as a thread, not fall through to being
        # misread as a bogus linear dimension ("4").
        results = parse_characteristics("M4 – 6H ▽ 8.0")
        assert len(results) == 2
        thread, depth = results
        assert thread.char_type == CharacteristicType.THREAD.value
        assert thread.thread_callout == "M4-6H"
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 8.0)

    def test_pipe_thread_with_class_and_depth_splits_into_two(self):
        # "G1/2\" - 6H" -- an ISO 228 parallel pipe thread (BSPP), common on
        # hydraulic/pneumatic fittings. Previously unmatched by any thread
        # pattern (no "M" prefix, no UNC/UNF suffix), so the leading "1" in
        # the "1/2" fraction was misread as a bogus depth/nominal value.
        results = parse_characteristics('G1/2" - 6H ↧ 18.0')
        assert len(results) == 2
        thread, depth = results
        assert thread.char_type == CharacteristicType.THREAD.value
        assert thread.thread_callout == 'G1/2"-6H'
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 18.0)

    def test_pipe_thread_with_en_dash_class_separator(self):
        # Same en-dash substitution as the metric-class case above, on a
        # pipe thread's class separator.
        results = parse_characteristics('G1/2" – 6H ▽ 18.0')
        assert len(results) == 2
        thread, depth = results
        assert thread.char_type == CharacteristicType.THREAD.value
        assert thread.thread_callout == 'G1/2"-6H'
        assert depth.char_type == CharacteristicType.DEPTH.value
        assert _close(depth.nominal, 18.0)

    def test_pipe_thread_without_class_is_a_single_characteristic(self):
        result = parse_characteristic('G3/8"')
        assert result.char_type == CharacteristicType.THREAD.value
        assert result.thread_callout == 'G3/8"'

    def test_whole_number_material_code_is_not_mistaken_for_pipe_thread(self):
        # "G10" (a fiberglass-laminate material code) must not be read as a
        # pipe thread -- only a fractional size ("G1/2") is unambiguous.
        result = parse_characteristic("G10 FIBERGLASS")
        assert result.char_type != CharacteristicType.THREAD.value


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

    def test_tolerance_value_is_unilateral_zero_to_value(self):
        # A feature control frame's stated value (e.g. flatness 0.01) is the
        # maximum allowed variation, with zero as the best case -- so nominal
        # is the box value, upper limit equals it, and lower limit is 0.0.
        result = parse_characteristic("⏥ 0.01")
        assert result.char_type == CharacteristicType.GDT_FRAME.value
        assert _close(result.nominal, 0.01)
        assert _close(result.tol_plus, 0.0)
        assert _close(result.tol_minus, 0.01)
        assert _close(result.lower_limit, 0.0)
        assert _close(result.upper_limit, 0.01)


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

    def test_bare_short_integer_without_geometry_stays_rejected(self):
        # Backward-compatible default: without bbox/page_size info, a bare
        # short integer can't be told apart from a zone marker, so it's
        # conservatively rejected exactly as before.
        from balloon_app.auto_balloon import _looks_like_characteristic

        assert _looks_like_characteristic("75") is False

    def test_bare_short_integer_outside_zone_margin_is_accepted(self):
        # A whole-number metric dimension ("75", "19", "Ø11"'s bare "11")
        # sitting well inside the drawing body -- far from every sheet
        # edge -- is not a zone marker and must be accepted once geometry
        # is available.
        from balloon_app.auto_balloon import _looks_like_characteristic

        page_size = (1000.0, 800.0)
        interior_bbox = (400.0, 300.0, 430.0, 315.0)  # nowhere near any edge
        assert _looks_like_characteristic("75", bbox=interior_bbox, page_size=page_size) is True

    def test_bare_short_integer_inside_zone_margin_stays_rejected(self):
        # An actual zone/grid reference number, printed in the thin margin
        # strip just inside the top edge, must still be rejected.
        from balloon_app.auto_balloon import _looks_like_characteristic

        page_size = (1000.0, 800.0)
        top_margin_bbox = (500.0, 2.0, 510.0, 14.0)  # y0 within the top 4% of page height
        assert _looks_like_characteristic("3", bbox=top_margin_bbox, page_size=page_size) is False

    def test_bare_integer_inside_bottom_zone_margin_stays_rejected(self):
        from balloon_app.auto_balloon import _looks_like_characteristic

        page_size = (1000.0, 800.0)
        bottom_margin_bbox = (500.0, 785.0, 510.0, 798.0)
        assert _looks_like_characteristic("3", bbox=bottom_margin_bbox, page_size=page_size) is False


class TestComputeLimits:
    def test_symmetric(self):
        lower, upper = compute_limits(10.0, 0.1, 0.1)
        assert _close(lower, 9.9)
        assert _close(upper, 10.1)

    def test_missing_nominal(self):
        lower, upper = compute_limits(None, 0.1, 0.1)
        assert lower is None
        assert upper is None


class TestParseDefaultTolerances:
    TITLE_BLOCK = (
        "TOLERANCES UNLESS OTHERWISE NOTED\n"
        "FRACTIONAL  0\"-6\": ±1/64   6\"-24\": ±1/32\n"
        "DECIMAL:  X.X: ±0.1000   X.XX: ±0.0100\n"
        "          X.XXX: ±0.0050  X.XXXX: ±0.0001\n"
        "ANGLES: ±0.5°  FINISH: 125 MICRO INCHES\n"
    )

    def test_extracts_all_decimal_places_and_angle(self):
        result = parse_default_tolerances(self.TITLE_BLOCK)
        assert _close(result.by_decimal_places[1], 0.1000)
        assert _close(result.by_decimal_places[2], 0.0100)
        assert _close(result.by_decimal_places[3], 0.0050)
        assert _close(result.by_decimal_places[4], 0.0001)
        assert _close(result.angular, 0.5)

    def test_no_title_block_gives_empty_result(self):
        result = parse_default_tolerances("PIC PULLEY, BALL SCREW\nA1219")
        assert result.is_empty()


class TestApplyDefaultTolerance:
    DEFAULTS = DefaultTolerances(by_decimal_places={1: 0.1, 2: 0.01, 3: 0.005}, angular=0.5)

    def test_three_decimal_diameter_gets_matching_default(self):
        parsed = ParsedCharacteristic(
            char_type=CharacteristicType.DIAMETER.value, nominal=0.25, nominal_text="0.250",
        )
        result = apply_default_tolerance(parsed, self.DEFAULTS)
        assert _close(result.tol_plus, 0.005)
        assert _close(result.tol_minus, 0.005)
        assert _close(result.lower_limit, 0.245)
        assert _close(result.upper_limit, 0.255)

    def test_two_decimal_vs_three_decimal_pick_different_defaults(self):
        two = apply_default_tolerance(
            ParsedCharacteristic(char_type=CharacteristicType.LINEAR_DIMENSION.value, nominal=0.25, nominal_text="0.25"),
            self.DEFAULTS,
        )
        three = apply_default_tolerance(
            ParsedCharacteristic(char_type=CharacteristicType.LINEAR_DIMENSION.value, nominal=0.25, nominal_text="0.250"),
            self.DEFAULTS,
        )
        assert _close(two.tol_plus, 0.01)
        assert _close(three.tol_plus, 0.005)

    def test_angle_uses_angular_default(self):
        parsed = ParsedCharacteristic(char_type=CharacteristicType.ANGLE.value, nominal=82.0, nominal_text="82")
        result = apply_default_tolerance(parsed, self.DEFAULTS)
        assert _close(result.tol_plus, 0.5)

    def test_whole_number_gets_zero_decimal_default_when_configured(self):
        defaults = DefaultTolerances(by_decimal_places={0: 1.0, 1: 0.1}, angular=0.5)
        parsed = ParsedCharacteristic(
            char_type=CharacteristicType.LINEAR_DIMENSION.value, nominal=30.0, nominal_text="30",
        )
        result = apply_default_tolerance(parsed, defaults)
        assert _close(result.tol_plus, 1.0)

    def test_whole_number_without_zero_tier_configured_stays_untoleranced(self):
        parsed = ParsedCharacteristic(
            char_type=CharacteristicType.LINEAR_DIMENSION.value, nominal=30.0, nominal_text="30",
        )
        result = apply_default_tolerance(parsed, self.DEFAULTS)  # no 0-place entry configured
        assert result.tol_plus is None

    def test_explicit_tolerance_is_never_overwritten(self):
        parsed = ParsedCharacteristic(
            char_type=CharacteristicType.DIAMETER.value, nominal=0.25, nominal_text="0.250",
            tol_plus=0.02, tol_minus=0.02,
        )
        result = apply_default_tolerance(parsed, self.DEFAULTS)
        assert _close(result.tol_plus, 0.02)

    def test_note_and_thread_types_are_never_defaulted(self):
        parsed = ParsedCharacteristic(char_type=CharacteristicType.THREAD.value, nominal=None, nominal_text=None)
        result = apply_default_tolerance(parsed, self.DEFAULTS)
        assert result.tol_plus is None

    def test_empty_defaults_leaves_parsed_unchanged(self):
        parsed = ParsedCharacteristic(char_type=CharacteristicType.DIAMETER.value, nominal=0.25, nominal_text="0.250")
        result = apply_default_tolerance(parsed, DefaultTolerances())
        assert result.tol_plus is None


class TestDiameterHint:
    """diameter_hint lets a caller (auto_balloon.py, backed by a vector-shape
    geometry check for CAD exports that draw Ø as line art rather than a
    font character) classify a bare number as a diameter even though no
    Ø symbol/word appears in the text itself.
    """

    def test_bare_number_with_hint_is_classified_as_diameter(self):
        result = parse_characteristic("8", diameter_hint=True)
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert result.nominal == 8.0
        assert result.raw_text == "⌀8"  # rewritten since the symbol wasn't in the text

    def test_bare_number_without_hint_stays_linear_dimension(self):
        result = parse_characteristic("8", diameter_hint=False)
        assert result.char_type == CharacteristicType.LINEAR_DIMENSION.value
        assert result.raw_text == "8"

    def test_hint_does_not_override_explicit_depth_symbol(self):
        """A more specific hole-feature symbol (here, depth) still wins over
        a diameter hint, exactly as it wins over a real Ø in the text."""
        result = parse_characteristic("▼0.500", diameter_hint=True)
        assert result.char_type == CharacteristicType.DEPTH.value

    def test_hint_is_redundant_but_harmless_when_symbol_already_present(self):
        result = parse_characteristic("⌀8", diameter_hint=True)
        assert result.char_type == CharacteristicType.DIAMETER.value
        assert result.raw_text == "⌀8"  # not double-prefixed
