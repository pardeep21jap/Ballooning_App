"""Rule-based parsing of dimension / tolerance / GD&T / thread / surface-finish
text into structured characteristic fields.

This module is intentionally regex-based rather than ML-based: it is meant
to be a transparent, fast, CPU-only baseline that handles the common
notations found on mechanical drawings. It will not perfectly understand
every possible GD&T notation or OCR artifact -- it is an assistive first
pass. Anything it cannot classify confidently is preserved verbatim in
``raw_text`` with a low confidence score so the user can correct it in the
review panel.

The main entry point is :func:`parse_characteristic`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from balloon_app.data_model import CharacteristicType

# ---------------------------------------------------------------------------
# Shared numeric building blocks
# ---------------------------------------------------------------------------
NUM = r"(?:\d+\.\d+|\.\d+|\d+)"

# A thread's size-class separator ("M4-6H", "G1/2\" - 6H") is sometimes not
# a plain ASCII hyphen-minus in the drawing's extracted text: word-processing
# "smart typography" (and some CAD annotation editors built on it) silently
# converts a space-hyphen-space sequence into an en dash as it's typed, so a
# drawing authored with visible spaces around the dash ("M4 - 6H") commonly
# ends up as "M4 \u2013 6H" in the actual PDF text. Every thread regex below
# that uses "-" as a literal separator matches this whole class instead, so
# that substitution doesn't silently make the thread unrecognizable.
# Not prefixed with "_" -- also imported by auto_balloon.py's characteristic
# pre-filter, which needs to recognize the same thread shapes this module parses.
THREAD_DASH_CHARS = "\\-\u2010\u2011\u2012\u2013\u2014\u2212"

_DIAMETER_SYMBOLS = "⌀ØΦ∅"
_GDT_SYMBOLS: dict[str, str] = {
    "⏤": "Straightness",
    "⏥": "Flatness",
    "○": "Circularity",
    "⌭": "Cylindricity",
    "⌒": "Profile of a Line",
    "⌓": "Profile of a Surface",
    "⟂": "Perpendicularity",
    "∠": "Angularity",
    "∥": "Parallelism",
    "⌯": "Symmetry",
    "⌖": "Position",
    "◎": "Concentricity",
    "↗": "Circular Runout",
    "⌰": "Circular Runout",
    "⌇": "Total Runout",
}

_THREAD_UNIFIED_RE = re.compile(
    rf"\b(\d+(?:/\d+)?|#\d+)\s*[{THREAD_DASH_CHARS}]\s*(\d+)\s*(UNC|UNF|UNEF|UN|NPT|NPTF)\b",
    re.IGNORECASE,
)
_THREAD_METRIC_RE = re.compile(rf"\bM\s?(\d+\.?\d*)\s*[xX×]\s*({NUM})\b")
# A metric thread's tolerance-class designation, e.g. "M4-6H" (a tapped
# hole's internal thread class, no pitch given -- coarse pitch is implied)
# or "M10 x 1.5-6H" (class given alongside an explicit pitch). Case is
# preserved in the captured class ("6H" vs "6g"): uppercase denotes an
# internal thread, lowercase an external one -- meaningfully different, so
# it must never be normalized.
_THREAD_METRIC_CLASS_RE = re.compile(
    rf"\bM\s?(\d+\.?\d*)\s*(?:[xX×]\s*({NUM})\s*)?[{THREAD_DASH_CHARS}]\s*(\d{{1,2}}[A-Za-z](?:\d{{1,2}}[A-Za-z])?)\b"
)
# A parallel pipe thread (ISO 228 "G" designation, e.g. "G1/2\" - 6H"),
# common on hydraulic/pneumatic fittings. Deliberately restricted to a
# fractional size (a literal "/"): a bare "G" + whole number, e.g. "G10",
# is indistinguishable from common fiberglass-laminate material codes
# (G10, G11) and would false-positive as a thread on a material note.
# Case-sensitive ("G" only, not "g") for the same reason "M" isn't matched
# case-insensitively elsewhere -- a lowercase "g" is too common a unit
# abbreviation (grams) to treat as a thread prefix.
_THREAD_PIPE_RE = re.compile(rf"\bG(\d+/\d+)(\s*[\"'])?(?:\s*[{THREAD_DASH_CHARS}]\s*([A-Za-z0-9]{{1,3}})\b)?")

# A depth callout trailing a thread spec, e.g. "8-32 UNC-2B ▼0.500" (tapped
# hole depth). CAD PDF exports draw the "depth" glyph from a custom
# GD&T/dingbat font, so the character actually extracted from the PDF for
# that glyph varies by CAD tool/font -- it can be ▼, ⌵, ↓, or (when the font
# has no proper ToUnicode mapping for it) an unrelated ASCII letter like
# "x". Matching specific symbol characters is therefore unreliable; instead
# any decimal number trailing the thread match is treated as the depth,
# since nothing else legitimately follows a thread callout on a drawing.
# Thread *class* codes ("2B", "6H", "3A") are bare integers glued to a
# letter, never decimals, so they don't false-match here.
_TRAILING_DECIMAL_RE = re.compile(r"(\d+\.\d+|\.\d+)")

_SURFACE_FINISH_RE_PREFIX = re.compile(rf"\bRa\s*({NUM})\s*(µm|um|μm)?\b", re.IGNORECASE)
_SURFACE_FINISH_RE_SUFFIX = re.compile(rf"({NUM})\s*(µm|um|μm)?\s*Ra\b", re.IGNORECASE)

_GENERAL_TOL_KEYWORDS_RE = re.compile(
    r"(UNLESS OTHERWISE SPECIFIED|GENERAL TOLERANCE|DEFAULT TOLERANCE|U\.?O\.?S\.?)",
    re.IGNORECASE,
)

_DIAMETER_RE = re.compile(rf"[{_DIAMETER_SYMBOLS}]|\bDIA\b", re.IGNORECASE)
_RADIUS_RE = re.compile(r"(?<![A-Za-z])R(?![a-zA-Z])\s*" + NUM)
_ANGLE_HINT_RE = re.compile(r"°")

# ASME Y14.5 dimensioning-modifier symbols for hole features. Matching
# specific Unicode characters alone is unreliable (see _TRAILING_DECIMAL_RE
# above -- CAD PDF exports often draw these from a custom dingbat font with
# no ToUnicode mapping, so the extracted text can be an unrelated ASCII
# character), so each also matches its common textual/abbreviation form.
#
# ⌵ is ASME's countersink symbol, but the same glyph is also one of the
# unreliable extractions a CAD PDF export can produce for the unrelated
# "depth" glyph (see _TRAILING_DECIMAL_RE above) -- e.g. a tapped hole's
# "Ø3.3 ⌵ 12.0" (drill diameter, then depth) versus a countersink's
# "⌵⌀0.500 X 82°" (diameter, then included angle). Position disambiguates:
# only immediately *before* a diameter symbol does ⌵ mean countersink;
# anywhere else (typically trailing a completed value) it's read as depth.
# ↧ (downwards arrow from bar) is ASME's actual "depth" symbol and is
# unambiguous -- unlike ⌵, it has no other meaning, so it's matched outright.
# ▽ (hollow/outline down-pointing triangle) is a further font/CAD-tool
# variant of the same depth glyph as ▼ (filled) -- some drawing tools render
# it unfilled instead, and it carries no other meaning on a mechanical
# drawing, so it's matched the same way.
_DEPTH_HINT_RE = re.compile(r"[▼▽↓⌵↧]|\bDEPTH\b|\bDEEP\b|\bDP\b", re.IGNORECASE)
# Tolerance markup between a shape's value and a later trailing number --
# ("Ø6.38 +0.005 -0.010") rules out reading that trailing number as a depth
# via the symbol-agnostic fallback in _try_shape_with_secondary_value below.
_DEPTH_BLOCKING_RE = re.compile(r"[+\-±]")
_COUNTERBORE_HINT_RE = re.compile(r"[⌴]|C['’]?BORE|\bCOUNTERBORE\b", re.IGNORECASE)
_COUNTERSINK_HINT_RE = re.compile(
    rf"[⌵]\s*[{_DIAMETER_SYMBOLS}]|C['’]?SINK|\bCSK\b|\bCOUNTERSINK\b", re.IGNORECASE
)
_SQUARE_HINT_RE = re.compile(r"[□]|\bSQ\b|\bSQUARE\b", re.IGNORECASE)

# Paired "symbol/keyword + its own number" extractors, used to pull a shape
# value back out of a compound callout that also carries a depth (see
# _try_shape_with_secondary_value below) -- unlike the *_HINT_RE checks above these
# capture the number that belongs to the shape itself, not just detect that
# the shape symbol is present somewhere in the text.
_DIAMETER_VALUE_RE = re.compile(rf"[{_DIAMETER_SYMBOLS}]\s*({NUM})")
_RADIUS_VALUE_RE = re.compile(rf"(?<![A-Za-z])R(?![a-zA-Z])\s*({NUM})")
_SQUARE_VALUE_RE = re.compile(rf"(?:[□]|\bSQ\b|\bSQUARE\b)\s*({NUM})", re.IGNORECASE)

# A leading "instance count" prefix, e.g. "2X " / "9X " ("2 places", "9
# holes"). Not a measurable value itself -- stripped before generic numeric
# extraction so it isn't mistaken for the dimension's nominal (e.g. "9X
# 0.250" must not parse as nominal=9).
_LEADING_QTY_RE = re.compile(r"^\s*\d+\s*[Xx]\s+")

# A quantity-prefixed hole callout with two bare numbers and no explicit
# tolerance markup between them, e.g. "2X n 0.089 x 0.500" -- almost always
# a hole's diameter and its depth, with the actual symbols mangled into
# unrelated letters ("n", "x") by a CAD PDF export's custom dingbat font
# with no ToUnicode mapping (the same root cause _TRAILING_DECIMAL_RE above
# works around for threads). Unlike the symbol-based extractors, this can't
# know *which* letters are meant to be symbols, so it is a lower-confidence,
# purely structural fallback tried only when nothing more specific matched.
_QTY_TWO_VALUE_RE = re.compile(
    rf"^\s*(\d+)\s*[Xx]\s+[A-Za-z]{{0,3}}\s*({NUM})\s+[A-Za-z]{{0,3}}\s*({NUM})\b"
)

# The same font-substitution problem as above, but with only *one* value --
# e.g. "9X n0.250 THRU" (a hole's diameter, no depth given, just a "THRU"
# note that isn't a second dimension at all). Tried only after
# _QTY_TWO_VALUE_RE fails to match, so a real second value still takes
# priority. "R" is excluded from the mangled-letter class since it's never
# mangled -- it's already an unambiguous, real radius symbol on its own.
_QTY_SINGLE_MANGLED_DIAMETER_RE = re.compile(
    rf"^\s*(\d+)\s*[Xx]\s+(?![Rr])([A-Za-z])\s*({NUM})\b"
)

# A number immediately followed by the degree sign, e.g. "82°" in
# "⌵⌀0.507 X 82°" (a countersink's included angle, following its
# diameter). Unlike the diameter/depth glyphs, ° (U+00B0) is a plain
# Latin-1 character that survives CAD PDF font substitution reliably.
_TRAILING_ANGLE_RE = re.compile(rf"({NUM})\s*°")

# Structural fallback for "<value> X <angle>°" (a countersink/chamfer's
# diameter times its included angle) when the shape symbol(s) in front of
# the value are *also* unrecognizable -- e.g. "w n 0.507 X 82°", where both
# the countersink glyph ("w") and the diameter glyph ("n") were mangled by
# the drawing's font. Unlike _TRAILING_ANGLE_RE (used once a real shape
# symbol has already been resolved), this doesn't require recognizing
# anything at all before the value -- the "X ...°" shape alone is signal
# enough, since it isn't produced by any other tolerance/dimension format.
_VALUE_THEN_ANGLE_RE = re.compile(rf"({NUM})\s*[Xx]\s*({NUM})\s*°")

# Structural fallback for a lone diameter (Ø) glyph mangled into an
# unrelated letter with no other structure around it at all, e.g. "n
# 0.551" -- the simplest form the same font-substitution problem takes
# (compare the compound cases above, "n" in "2X n 0.089 x 0.500" and "w n
# 0.507 X 82°", where a second value or trailing angle gives the mangled
# letter away). Thread classes and shape keywords have already been ruled
# out by earlier, more specific checks by the time this runs, so a single
# leftover letter immediately glued to a value has no other explanation on
# a mechanical drawing -- except "R", which is never mangled: it's already
# an unambiguous, real radius symbol in its own right, so it's excluded
# here rather than mistaken for a diameter. Also restricted to a *decimal*
# value -- never a bare whole number -- so a drawing/revision code like
# "A2048" is never mistaken for one; a diameter is essentially always
# given to several decimal places, an alphanumeric code never is.
_BARE_MANGLED_DIAMETER_RE = re.compile(r"^(?![Rr])([A-Za-z])\s*(\d+\.\d+|\.\d+)$")

# What a marker letter resolves to as a canonical display symbol, once its
# char_type is known (whether from a learned mapping or the bare default
# guess) -- used to rewrite raw_text so a garbled font substitution never
# shows through verbatim (see _try_bare_mangled_diameter et al.).
_CANONICAL_SYMBOL_FOR_TYPE: dict[str, str] = {
    CharacteristicType.DIAMETER.value: "⌀",
    CharacteristicType.DEPTH.value: "▼",
    CharacteristicType.COUNTERSINK.value: "⌵⌀",
    CharacteristicType.COUNTERBORE.value: "⌴⌀",
    CharacteristicType.SQUARE.value: "□",
}


def _resolve_learned_marker(marker: str, learned_symbols: Optional[dict[str, str]]) -> tuple[str, float]:
    """The (char_type, confidence) a mangled marker letter should resolve
    to: a previously-learned mapping if one exists (the user confirmed it,
    so it's treated as confident), otherwise the bare default assumption --
    diameter, by far the most commonly mangled symbol on a mechanical
    drawing -- at the same low confidence these fallbacks have always used.
    """
    if learned_symbols:
        learned = learned_symbols.get(marker.strip().lower())
        if learned:
            return learned, 0.75
    return CharacteristicType.DIAMETER.value, 0.55

# Numeric tolerance extraction patterns, tried in priority order.
_ASYM_SLASH_RE = re.compile(rf"({NUM})\s*\+\s*({NUM})\s*/\s*-\s*({NUM})")
_ASYM_SPACE_RE = re.compile(rf"({NUM})\s*\+\s*({NUM})\s+-\s*({NUM})")
# A less common but valid style: both stacked deviations share the same
# sign (e.g. "+.3" over "+.1" -- a hole enlarged by somewhere between 0.1
# and 0.3, never undersized), rather than the usual "+X above / -Y below".
# ASME lists the numerically larger (less restrictive) deviation on top
# regardless of sign, but since the merge step that produces this text
# (see _merge_stacked_tolerance_fragments) doesn't guarantee that order,
# the two values are sorted here rather than assumed positional.
_DOUBLE_PLUS_RE = re.compile(rf"({NUM})\s*\+\s*({NUM})\s+\+\s*({NUM})")
_DOUBLE_MINUS_RE = re.compile(rf"({NUM})\s*-\s*({NUM})\s+-\s*({NUM})")
# Same patterns, minus the leading nominal capture -- used to recognize a
# tolerance immediately following a shape value already matched elsewhere
# (see _extract_leading_tolerance below), via re.Pattern.match(text, pos)
# which anchors at ``pos`` without needing a "^" in the pattern itself.
_TAIL_ASYM_SLASH_RE = re.compile(rf"\s*\+\s*({NUM})\s*/\s*-\s*({NUM})")
_TAIL_ASYM_SPACE_RE = re.compile(rf"\s*\+\s*({NUM})\s+-\s*({NUM})")
_TAIL_SYMMETRIC_RE = re.compile(rf"\s*±\s*({NUM})")
_TAIL_DOUBLE_PLUS_RE = re.compile(rf"\s*\+\s*({NUM})\s+\+\s*({NUM})")
_TAIL_DOUBLE_MINUS_RE = re.compile(rf"\s*-\s*({NUM})\s+-\s*({NUM})")
_SYMMETRIC_RE = re.compile(rf"({NUM})\s*±\s*({NUM})")
_BARE_SYMMETRIC_TOL_RE = re.compile(rf"±\s*({NUM})")
_LIMITS_RE = re.compile(rf"({NUM})\s*-\s*({NUM})")
_BARE_NUMBER_RE = re.compile(rf"({NUM})")

_GDT_TOL_AFTER_SYMBOL_RE = re.compile(rf"[{_DIAMETER_SYMBOLS}]?\s*({NUM})")
_MATERIAL_CONDITION_PATTERNS = (
    (re.compile(r"Ⓜ|\(M\)|ⓜ"), "MMC"),
    (re.compile(r"Ⓛ|\(L\)|ⓛ"), "LMC"),
    (re.compile(r"Ⓢ|\(S\)|ⓢ"), "RFS"),
)
_SINGLE_UPPER_LETTER_RE = re.compile(r"\b[A-Z]\b")
_PIPE_DATUM_FALLBACK_RE = re.compile(rf"({NUM})\s*\|\s*([A-Z](?:\s*\|\s*[A-Z])*)")


@dataclass
class ParsedCharacteristic:
    """Result of parsing a chunk of drawing text into structured fields."""

    char_type: str = CharacteristicType.OTHER.value
    raw_text: str = ""
    nominal: Optional[float] = None
    # The nominal exactly as printed (e.g. "0.250"), before float conversion
    # loses the distinction between "0.25" and "0.250" -- needed to pick the
    # right entry in a decimal-place-keyed default tolerance table. Only
    # populated where the nominal has no explicit tolerance of its own.
    nominal_text: Optional[str] = None
    tol_plus: Optional[float] = None
    tol_minus: Optional[float] = None
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    gdt_symbol: Optional[str] = None
    gdt_tolerance: Optional[str] = None
    material_condition: Optional[str] = None
    datums: Optional[str] = None
    surface_finish: Optional[str] = None
    thread_callout: Optional[str] = None
    note: str = ""
    confidence: float = 0.3
    extra: dict = field(default_factory=dict)
    # The mangled letter a "font substituted the real symbol" fallback (see
    # _resolve_learned_marker) read the char_type from -- whether resolved
    # via a previously learned mapping or the bare default guess (assume
    # diameter). Preserved even though raw_text is rewritten to the
    # canonical symbol for display, since without it the marker would be
    # unrecoverable if the user later corrects this guess and the app
    # should remember (or update) what it means. None for anything that
    # didn't go through this path -- a real symbol/keyword was actually
    # present in the text.
    guessed_symbol_marker: Optional[str] = None


def _round(value: float) -> float:
    return round(value, 5)


def _extract_numeric_tolerance(text: str) -> Optional[dict]:
    """Try, in priority order, to pull nominal/tolerance/limits out of ``text``.

    Returns ``None`` if no numeric pattern at all is found.
    """
    search_text = _LEADING_QTY_RE.sub("", text.replace("°", " "), count=1)

    m = _ASYM_SLASH_RE.search(search_text)
    if m:
        nominal, plus, minus = (float(g) for g in m.groups())
        return {
            "nominal": _round(nominal),
            "tol_plus": _round(plus),
            "tol_minus": _round(minus),
            "lower_limit": _round(nominal - minus),
            "upper_limit": _round(nominal + plus),
            "confidence": 0.9,
        }

    m = _ASYM_SPACE_RE.search(search_text)
    if m:
        nominal, plus, minus = (float(g) for g in m.groups())
        return {
            "nominal": _round(nominal),
            "tol_plus": _round(plus),
            "tol_minus": _round(minus),
            "lower_limit": _round(nominal - minus),
            "upper_limit": _round(nominal + plus),
            "confidence": 0.9,
        }

    m = _SYMMETRIC_RE.search(search_text)
    if m:
        nominal, tol = (float(g) for g in m.groups())
        return {
            "nominal": _round(nominal),
            "tol_plus": _round(tol),
            "tol_minus": _round(tol),
            "lower_limit": _round(nominal - tol),
            "upper_limit": _round(nominal + tol),
            "confidence": 0.9,
        }

    m = _DOUBLE_PLUS_RE.search(search_text)
    if m:
        nominal, a, b = (float(g) for g in m.groups())
        hi, lo = max(a, b), min(a, b)
        return {
            "nominal": _round(nominal),
            "tol_plus": _round(hi),
            "tol_minus": _round(-lo),
            "lower_limit": _round(nominal + lo),
            "upper_limit": _round(nominal + hi),
            "confidence": 0.85,
        }

    m = _DOUBLE_MINUS_RE.search(search_text)
    if m:
        nominal, a, b = (float(g) for g in m.groups())
        near, far = min(a, b), max(a, b)  # smaller magnitude = less negative = upper
        return {
            "nominal": _round(nominal),
            "tol_plus": _round(-near),
            "tol_minus": _round(far),
            "lower_limit": _round(nominal - far),
            "upper_limit": _round(nominal - near),
            "confidence": 0.85,
        }

    if "±" not in search_text and "+" not in search_text:
        m = _LIMITS_RE.search(search_text)
        if m:
            a, b = (float(g) for g in m.groups())
            lower, upper = (a, b) if a <= b else (b, a)
            return {
                "nominal": None,
                "tol_plus": None,
                "tol_minus": None,
                "lower_limit": _round(lower),
                "upper_limit": _round(upper),
                "confidence": 0.85,
            }

    m = _BARE_NUMBER_RE.search(search_text)
    if m:
        return {
            "nominal": _round(float(m.group(1))),
            "nominal_text": m.group(1),
            "tol_plus": None,
            "tol_minus": None,
            "lower_limit": None,
            "upper_limit": None,
            "confidence": 0.45,
        }

    return None


def _try_thread(text: str) -> Optional[list[ParsedCharacteristic]]:
    m = _THREAD_METRIC_CLASS_RE.search(text)
    if m:
        size, pitch, tol_class = m.group(1), m.group(2), m.group(3)
        callout = f"M{size}" + (f" x {pitch}" if pitch else "") + f"-{tol_class}"
    else:
        m = _THREAD_METRIC_RE.search(text)
        if m:
            callout = f"M{m.group(1)} x {m.group(2)}"
        else:
            m = _THREAD_UNIFIED_RE.search(text)
            if m:
                callout = f"{m.group(1)}-{m.group(2)} {m.group(3).upper()}"
            else:
                m = _THREAD_PIPE_RE.search(text)
                if m:
                    size, inch_mark, tol_class = m.group(1), m.group(2), m.group(3)
                    callout = f"G{size}" + ('"' if inch_mark else "") + (f"-{tol_class}" if tol_class else "")
                else:
                    return None

    results = [
        ParsedCharacteristic(
            char_type=CharacteristicType.THREAD.value,
            raw_text=text,
            thread_callout=callout,
            confidence=0.85,
        )
    ]

    depth_match = _TRAILING_DECIMAL_RE.search(text, m.end())
    if depth_match:
        results.append(
            ParsedCharacteristic(
                char_type=CharacteristicType.DEPTH.value,
                raw_text=text,
                nominal=_round(float(depth_match.group(1))),
                nominal_text=depth_match.group(1),
                confidence=0.8,
            )
        )
    return results


def _resolve_shape_char_type(text: str) -> Optional[str]:
    """Priority-ordered shape/hole-modifier type for ``text``, matching the
    main single-value classification's priority (a counterbore/countersink
    symbol is more specific than the diameter symbol it precedes). Used to
    label the "primary" value in a compound shape+depth or shape+angle
    split -- returns ``None`` when there's no identifiable shape at all.
    """
    if _COUNTERBORE_HINT_RE.search(text):
        return CharacteristicType.COUNTERBORE.value
    if _COUNTERSINK_HINT_RE.search(text):
        return CharacteristicType.COUNTERSINK.value
    if _SQUARE_HINT_RE.search(text):
        return CharacteristicType.SQUARE.value
    if _DIAMETER_RE.search(text):
        return CharacteristicType.DIAMETER.value
    if _RADIUS_RE.search(text):
        return CharacteristicType.RADIUS.value
    return None


def _extract_leading_tolerance(
    text: str, pos: int, nominal: float
) -> Optional[tuple[float, float, float, float, int]]:
    """Tolerance markup immediately (whitespace-only gap) following a value
    already matched ending at ``pos`` -- e.g. the "+.3 +.1" in "Ø12.0 +.3
    +.1 ↧ 100.0", which belongs to the diameter, not to whatever trails
    further along the same compound callout. That trailing content can be
    a depth clause the stacked-tolerance merge step already appended
    *ahead* of the tolerance fragments it found (see
    _merge_stacked_tolerance_fragments), so this must be checked, and
    consumed, before searching further along for a depth/angle.

    Returns ``(tol_plus, tol_minus, lower_limit, upper_limit, new_pos)``
    with ``new_pos`` placed right after the tolerance, or ``None`` if no
    recognizable tolerance starts at ``pos``.
    """
    m = _TAIL_ASYM_SLASH_RE.match(text, pos)
    if m:
        plus, minus = float(m.group(1)), float(m.group(2))
        return (_round(plus), _round(minus), _round(nominal - minus), _round(nominal + plus), m.end())
    m = _TAIL_ASYM_SPACE_RE.match(text, pos)
    if m:
        plus, minus = float(m.group(1)), float(m.group(2))
        return (_round(plus), _round(minus), _round(nominal - minus), _round(nominal + plus), m.end())
    m = _TAIL_SYMMETRIC_RE.match(text, pos)
    if m:
        tol = float(m.group(1))
        return (_round(tol), _round(tol), _round(nominal - tol), _round(nominal + tol), m.end())
    m = _TAIL_DOUBLE_PLUS_RE.match(text, pos)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        hi, lo = max(a, b), min(a, b)
        return (_round(hi), _round(-lo), _round(nominal + lo), _round(nominal + hi), m.end())
    m = _TAIL_DOUBLE_MINUS_RE.match(text, pos)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        near, far = min(a, b), max(a, b)
        return (_round(-near), _round(far), _round(nominal - far), _round(nominal - near), m.end())
    return None


def _try_shape_with_secondary_value(text: str) -> Optional[list[ParsedCharacteristic]]:
    """A shape value paired with a trailing depth *or* angle callout on the
    same line -- each is inspected with a different gauge, so it becomes
    its own characteristic instead of one that silently drops a value:

    * "2X ⌀0.089 ▼0.500" -- a hole's diameter and its depth.
    * "⌵⌀0.507 X 82°" -- a countersink's diameter and its included angle.

    Matches how :func:`_try_thread` already splits a thread callout from
    its trailing depth.
    """
    shape_char_type = _resolve_shape_char_type(text)
    if shape_char_type is None:
        return None

    if shape_char_type == CharacteristicType.SQUARE.value:
        value_re = _SQUARE_VALUE_RE
    elif shape_char_type == CharacteristicType.RADIUS.value:
        value_re = _RADIUS_VALUE_RE
    else:
        # DIAMETER, COUNTERBORE, and COUNTERSINK all precede a diameter callout.
        value_re = _DIAMETER_VALUE_RE

    m = value_re.search(text)
    if not m:
        return None

    nominal = float(m.group(1))
    leading_tol = _extract_leading_tolerance(text, m.end(), nominal)
    search_start = m.end() if leading_tol is None else leading_tol[4]

    has_depth = bool(_DEPTH_HINT_RE.search(text))
    has_degree = "°" in text

    secondary: Optional[ParsedCharacteristic] = None
    depth_match = _TRAILING_DECIMAL_RE.search(text, search_start)
    if depth_match:
        # A recognized depth symbol/keyword always confirms it. Otherwise,
        # still read an isolated trailing number as a depth as long as
        # nothing between the two numbers looks like tolerance markup
        # (+/-/±) -- this drawing's CAD export may draw the depth glyph as
        # vector line art rather than a font character (the same class of
        # problem PdfDocument.find_vector_diameter_symbol works around for
        # Ø), so it can contribute no recognizable character -- or none at
        # all -- to the extracted text, not just an unreliable one.
        gap = text[search_start : depth_match.start()]
        if has_depth or not _DEPTH_BLOCKING_RE.search(gap):
            secondary = ParsedCharacteristic(
                char_type=CharacteristicType.DEPTH.value,
                raw_text=text,
                nominal=_round(float(depth_match.group(1))),
                nominal_text=depth_match.group(1),
                confidence=0.8 if has_depth else 0.6,
            )
    if secondary is None and has_degree:
        angle_match = _TRAILING_ANGLE_RE.search(text, search_start)
        if angle_match:
            secondary = ParsedCharacteristic(
                char_type=CharacteristicType.ANGLE.value,
                raw_text=text,
                nominal=_round(float(angle_match.group(1))),
                nominal_text=angle_match.group(1),
                confidence=0.85,
            )
    if secondary is None:
        return None

    primary = ParsedCharacteristic(
        char_type=shape_char_type,
        raw_text=text,
        nominal=_round(nominal),
        nominal_text=m.group(1),
        tol_plus=leading_tol[0] if leading_tol else None,
        tol_minus=leading_tol[1] if leading_tol else None,
        lower_limit=leading_tol[2] if leading_tol else None,
        upper_limit=leading_tol[3] if leading_tol else None,
        confidence=0.85,
    )
    return [primary, secondary]


# A mangled shape-symbol marker (see _resolve_learned_marker) immediately
# followed by its own value -- optionally preceded by a second mangled
# marker letter (a shape modifier like counterbore mangled separately from
# the diameter symbol it precedes, e.g. the "v" in "v n.159"). Unlike
# _BARE_MANGLED_DIAMETER_RE (which anchors the marker+value as the *entire*
# text), this only anchors the leading marker(s)+value, leaving whatever
# follows -- a tolerance, then a trailing depth/angle -- for
# _try_mangled_shape_with_secondary_value to inspect the same way
# _try_shape_with_secondary_value does for a *real* shape symbol.
_LEADING_MANGLED_MARKERS_RE = re.compile(
    r"^(?![Rr])([A-Za-z])(?:\s+(?![Rr])([A-Za-z])(?!\s*[A-Za-z]))?\s*(\d+\.\d+|\.\d+)"
)


def _try_mangled_shape_with_secondary_value(
    text: str, learned_symbols: Optional[dict[str, str]] = None
) -> Optional[list[ParsedCharacteristic]]:
    """Mirrors _try_shape_with_secondary_value, but for a shape symbol that
    was itself mangled into an unrelated ASCII letter rather than a real
    Ø/⌴/⌵/□/R -- e.g. "n.130 x.50 MAX" (Ø.130 ▽.50 MAX, both symbols
    mangled) or "v n.159 +.002/-.000 x.167" (⌴Ø.159 +.002/-.000 ▽.167, a
    counterbore whose own diameter symbol is *also* mangled).

    Only returns a result when a genuine secondary depth value is found
    trailing the primary -- a bare "marker+value" with nothing trailing is
    left for _try_bare_mangled_diameter (tried after this), since that
    fallback already handles the no-secondary-value case as the *entire*
    text, and re-matching it here would just duplicate that path with a
    different (unwarranted) confidence. A trailing *angle* is deliberately
    not handled here either, even though the shape is superficially
    similar: "<value> X <angle>°" is specifically the countersink notation
    regardless of what (if anything) precedes the value, and
    _try_bare_value_with_angle (tried later) already recognizes that
    structural shape on its own with its own, better-founded confidence --
    matching it here first would instead read the leading marker generically
    (defaulting to diameter absent a learned mapping) and miss that it's
    actually always a countersink.
    """
    m = _LEADING_MANGLED_MARKERS_RE.match(text)
    if not m:
        return None
    marker, value = m.group(1), m.group(3)
    char_type, confidence = _resolve_learned_marker(marker, learned_symbols)
    nominal = float(value)

    leading_tol = _extract_leading_tolerance(text, m.end(), nominal)
    search_start = m.end() if leading_tol is None else leading_tol[4]

    has_depth = bool(_DEPTH_HINT_RE.search(text[search_start:]))

    secondary: Optional[ParsedCharacteristic] = None
    depth_match = _TRAILING_DECIMAL_RE.search(text, search_start)
    if depth_match:
        gap = text[search_start : depth_match.start()]
        if has_depth or not _DEPTH_BLOCKING_RE.search(gap):
            secondary = ParsedCharacteristic(
                char_type=CharacteristicType.DEPTH.value,
                raw_text=text,
                nominal=_round(float(depth_match.group(1))),
                nominal_text=depth_match.group(1),
                confidence=0.75 if has_depth else 0.55,
            )
    if secondary is None:
        return None

    symbol = _CANONICAL_SYMBOL_FOR_TYPE.get(char_type, "⌀")
    primary = ParsedCharacteristic(
        char_type=char_type,
        raw_text=f"{symbol}{value}",
        nominal=_round(nominal),
        nominal_text=value,
        tol_plus=leading_tol[0] if leading_tol else None,
        tol_minus=leading_tol[1] if leading_tol else None,
        lower_limit=leading_tol[2] if leading_tol else None,
        upper_limit=leading_tol[3] if leading_tol else None,
        confidence=max(confidence, 0.6),
        guessed_symbol_marker=marker.strip().lower(),
    )
    return [primary, secondary]


def _try_qty_prefixed_two_values(text: str) -> Optional[list[ParsedCharacteristic]]:
    """Structural fallback for a quantity-prefixed hole callout whose two
    dimension symbols are both unrecognized (see _QTY_TWO_VALUE_RE above).
    Assumes the far more common ordering: diameter first, depth second.

    ``raw_text`` is rewritten from the mangled source ("2X n 0.089 x
    0.500") to its canonical symbol form ("2X ⌀0.089 ▼0.500") -- the
    original letters carry no information (they're an artifact of the
    drawing's font, not real content), so showing them verbatim would only
    confuse review, not aid traceability.
    """
    m = _QTY_TWO_VALUE_RE.match(text)
    if not m:
        return None
    qty, diameter_value, depth_value = m.group(1), m.group(2), m.group(3)
    cleaned_text = f"{qty}X ⌀{diameter_value} ▼{depth_value}"
    return [
        ParsedCharacteristic(
            char_type=CharacteristicType.DIAMETER.value,
            raw_text=cleaned_text,
            nominal=_round(float(diameter_value)),
            nominal_text=diameter_value,
            confidence=0.55,
        ),
        ParsedCharacteristic(
            char_type=CharacteristicType.DEPTH.value,
            raw_text=cleaned_text,
            nominal=_round(float(depth_value)),
            nominal_text=depth_value,
            confidence=0.5,
        ),
    ]


def _try_qty_prefixed_mangled_diameter(
    text: str, learned_symbols: Optional[dict[str, str]] = None
) -> Optional[ParsedCharacteristic]:
    """Structural fallback for a quantity-prefixed diameter whose glyph was
    mangled into an unrelated letter with no second value to split on (see
    _try_qty_prefixed_two_values for when there is one) -- e.g. "9X
    n0.250 THRU", where "THRU" is a note, not a second dimension. Without
    this, such text falls all the way through to a plain linear dimension
    since "n" isn't a recognized diameter symbol and there's no vector
    circle to fall back on either (the glyph *is* a font character here,
    just the wrong one -- see PdfDocument.find_vector_diameter_symbol,
    which only helps when the glyph is missing from the text entirely).

    ``raw_text`` is rewritten to replace the mangled letter with its
    resolved canonical symbol, keeping the quantity prefix and any trailing
    note (e.g. "9X ⌀0.250 THRU") since those are real content, unlike the
    single mangled letter itself.
    """
    m = _QTY_SINGLE_MANGLED_DIAMETER_RE.match(text)
    if not m:
        return None
    qty, marker, value = m.group(1), m.group(2), m.group(3)
    char_type, confidence = _resolve_learned_marker(marker, learned_symbols)
    symbol = _CANONICAL_SYMBOL_FOR_TYPE.get(char_type, "⌀")
    cleaned_text = f"{qty}X {symbol}{value}{text[m.end():]}"
    return ParsedCharacteristic(
        char_type=char_type,
        raw_text=cleaned_text,
        nominal=_round(float(value)),
        nominal_text=value,
        confidence=confidence,
        guessed_symbol_marker=marker.strip().lower(),
    )


def _try_bare_value_with_angle(text: str) -> Optional[list[ParsedCharacteristic]]:
    """Structural fallback for "<value> X <angle>°" when no shape symbol at
    all is recognizable (see _VALUE_THEN_ANGLE_RE above). "A diameter times
    an included angle" is specifically the countersink notation -- a plain
    diameter is never followed by "X <angle>°" -- so unlike the other
    unrecognized-symbol fallbacks, the shape here can be inferred with
    reasonable confidence.

    ``raw_text`` is rewritten from the mangled source ("w n 0.507 X 82°")
    to its canonical symbol form ("⌵⌀0.507 X 82°") -- the original letters
    carry no information (they're an artifact of the drawing's font, not
    real content), so showing them verbatim would only confuse review, not
    aid traceability.
    """
    m = _VALUE_THEN_ANGLE_RE.search(text)
    if not m:
        return None
    diameter_value, angle_value = m.group(1), m.group(2)
    cleaned_text = f"⌵⌀{diameter_value} X {angle_value}°"
    return [
        ParsedCharacteristic(
            char_type=CharacteristicType.COUNTERSINK.value,
            raw_text=cleaned_text,
            nominal=_round(float(diameter_value)),
            nominal_text=diameter_value,
            confidence=0.6,
        ),
        ParsedCharacteristic(
            char_type=CharacteristicType.ANGLE.value,
            raw_text=cleaned_text,
            nominal=_round(float(angle_value)),
            nominal_text=angle_value,
            confidence=0.6,
        ),
    ]


def _try_bare_mangled_diameter(
    text: str, learned_symbols: Optional[dict[str, str]] = None
) -> Optional[ParsedCharacteristic]:
    """Structural fallback for a lone dimensioning-symbol glyph mangled into
    an unrelated letter with nothing else around it, e.g. "n 0.551" (see
    _BARE_MANGLED_DIAMETER_RE above).

    Defaults to assuming a diameter (by far the most commonly mangled
    symbol) unless ``learned_symbols`` says this specific marker letter
    means something else for this font -- see _resolve_learned_marker.

    ``raw_text`` is rewritten from the mangled source ("n 0.551") to its
    resolved canonical symbol form ("⌀0.551") -- the original letter
    carries no information on its own (it's an artifact of the drawing's
    font, not real content), so showing it verbatim would only confuse
    review, not aid traceability.
    """
    m = _BARE_MANGLED_DIAMETER_RE.match(text.strip())
    if not m:
        return None
    marker, value = m.group(1), m.group(2)
    char_type, confidence = _resolve_learned_marker(marker, learned_symbols)
    symbol = _CANONICAL_SYMBOL_FOR_TYPE.get(char_type, "⌀")
    return ParsedCharacteristic(
        char_type=char_type,
        raw_text=f"{symbol}{value}",
        nominal=_round(float(value)),
        nominal_text=value,
        confidence=confidence,
        guessed_symbol_marker=marker.strip().lower(),
    )


def _try_surface_finish(text: str) -> Optional[ParsedCharacteristic]:
    m = _SURFACE_FINISH_RE_PREFIX.search(text) or _SURFACE_FINISH_RE_SUFFIX.search(text)
    if m:
        value = m.group(1)
        return ParsedCharacteristic(
            char_type=CharacteristicType.SURFACE_FINISH.value,
            raw_text=text,
            surface_finish=f"Ra {value}",
            nominal=_round(float(value)),
            confidence=0.85,
        )
    return None


def _try_gdt(text: str) -> Optional[ParsedCharacteristic]:
    for symbol, name in _GDT_SYMBOLS.items():
        if symbol in text:
            remainder = text.replace(symbol, " ")
            tol_match = _GDT_TOL_AFTER_SYMBOL_RE.search(remainder)
            tolerance = None
            if tol_match:
                tolerance = tol_match.group(1)
                remainder = remainder.replace(tol_match.group(0), " ", 1)

            material_condition = None
            for pattern, code in _MATERIAL_CONDITION_PATTERNS:
                if pattern.search(remainder):
                    material_condition = code
                    remainder = pattern.sub(" ", remainder)
                    break

            datum_letters = _SINGLE_UPPER_LETTER_RE.findall(remainder)
            datums = ", ".join(dict.fromkeys(datum_letters)) if datum_letters else None
            limits = _gdt_tolerance_limits(tolerance)

            return ParsedCharacteristic(
                char_type=CharacteristicType.GDT_FRAME.value,
                raw_text=text,
                gdt_symbol=name,
                gdt_tolerance=(f"⌀{tolerance}" if tolerance and _DIAMETER_RE.search(text) else tolerance),
                material_condition=material_condition,
                datums=datums,
                confidence=0.75,
                **limits,
            )

    # Fallback: no recognizable GD&T symbol, but a pipe-delimited
    # tolerance|datum pattern strongly suggests a mangled feature control frame.
    m = _PIPE_DATUM_FALLBACK_RE.search(text)
    if m:
        tolerance = m.group(1)
        datum_letters = [d.strip() for d in m.group(2).split("|")]
        return ParsedCharacteristic(
            char_type=CharacteristicType.GDT_FRAME.value,
            raw_text=text,
            gdt_symbol=None,
            gdt_tolerance=tolerance,
            datums=", ".join(dict.fromkeys(datum_letters)),
            confidence=0.55,
            **_gdt_tolerance_limits(tolerance),
        )
    return None


def _gdt_tolerance_limits(tolerance: Optional[str | float]) -> dict:
    """A GD&T feature control frame's stated value is always the *maximum*
    allowed variation, with zero as the implicit best case -- there is no
    separate plus/minus split the way a dimensional tolerance has one. So a
    flatness callout of 0.01, for example, is treated as nominal 0.01 with a
    unilateral -0.01/-0.0 tolerance: upper limit 0.01, lower limit 0.0.
    """
    if tolerance is None or tolerance == "":
        return {}
    value = _round(float(tolerance))
    return {
        "nominal": value,
        "tol_plus": 0.0,
        "tol_minus": value,
        "lower_limit": 0.0,
        "upper_limit": value,
    }


def _try_general_tolerance(text: str) -> Optional[ParsedCharacteristic]:
    if not _GENERAL_TOL_KEYWORDS_RE.search(text):
        return None
    m = _SYMMETRIC_RE.search(text) or _BARE_SYMMETRIC_TOL_RE.search(text)
    tol_plus = tol_minus = None
    if m:
        tol_plus = tol_minus = _round(float(m.group(m.lastindex)))
    return ParsedCharacteristic(
        char_type=CharacteristicType.GENERAL_TOLERANCE.value,
        raw_text=text,
        tol_plus=tol_plus,
        tol_minus=tol_minus,
        note=text,
        confidence=0.7,
    )


def compute_limits(
    nominal: Optional[float], tol_plus: Optional[float], tol_minus: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """Derive (lower_limit, upper_limit) from nominal +/- tolerance, if possible."""
    if nominal is None:
        return None, None
    lower = _round(nominal - tol_minus) if tol_minus is not None else None
    upper = _round(nominal + tol_plus) if tol_plus is not None else None
    return lower, upper


# ---------------------------------------------------------------------------
# Default/general tolerance table (title-block "TOLERANCES UNLESS OTHERWISE
# NOTED" note), used to backfill tol_plus/tol_minus on dimensions that don't
# carry their own explicit tolerance callout.
# ---------------------------------------------------------------------------
# "X." followed by 1-6 more X's matches any decimal-place tier a title block
# might use (X.X, X.XX, ... up to X.XXXXXX); the run length of X's is the
# decimal-place count, so this one pattern replaces a fixed per-count table.
_DECIMAL_TOL_RE = re.compile(rf"X\.(X{{1,6}})\b\s*[:=]?\s*±?\s*({NUM})")
_ANGULAR_TOL_RE = re.compile(rf"ANGLES?\s*[:=]?\s*±?\s*({NUM})\s*°?", re.IGNORECASE)

_TOLERANCED_DIMENSION_TYPES = {
    CharacteristicType.LINEAR_DIMENSION.value,
    CharacteristicType.DIAMETER.value,
    CharacteristicType.RADIUS.value,
    CharacteristicType.DEPTH.value,
    CharacteristicType.COUNTERBORE.value,
    CharacteristicType.COUNTERSINK.value,
    CharacteristicType.SQUARE.value,
}


@dataclass
class DefaultTolerances:
    """A drawing's general/default tolerance table -- either parsed from its
    title block or entered manually -- keyed by decimal-place count (the
    "X.XX: ±0.0100" convention), plus a separate angular tolerance. The
    table is an open-ended ``{decimal_places: tolerance}`` map rather than a
    fixed set of fields, since drawings vary in how many tiers they define.
    """

    by_decimal_places: dict[int, float] = field(default_factory=dict)
    angular: Optional[float] = None

    def is_empty(self) -> bool:
        return not self.by_decimal_places and self.angular is None

    def for_decimal_places(self, places: int) -> Optional[float]:
        """The configured tolerance for a value with this many decimal
        places (0 for a whole number, e.g. "30"), falling back to the
        next-coarsest configured entry if the table has gaps (e.g. only
        X.XXX is set but a value has 4 places)."""
        if not self.by_decimal_places:
            return None
        candidates = [p for p in self.by_decimal_places if p <= places]
        if not candidates:
            return None
        return self.by_decimal_places[max(candidates)]


def decimal_places(nominal_text: Optional[str]) -> int:
    """Digits after the decimal point in ``nominal_text`` as originally
    printed (e.g. "0.250" -> 3, "0.25" -> 2), or 0 if there's no decimal
    point at all. Used to pick the matching entry in a decimal-place-keyed
    default tolerance table.
    """
    if not nominal_text or "." not in nominal_text:
        return 0
    return len(nominal_text.split(".", 1)[1])


def parse_default_tolerances(page_text: str) -> DefaultTolerances:
    """Best-effort extraction of a drawing's general/default tolerance table
    from its title-block note (commonly headed "TOLERANCES UNLESS OTHERWISE
    NOTED"), e.g. "X.XX: ±0.0100" / "X.XXX: ±0.0050" / "ANGLES: ±0.5°".

    Returns an empty ``DefaultTolerances`` if no such table is found --
    callers should let the user review/fill in the result either way, since
    title-block layouts vary too much for this to be fully reliable.
    """
    result = DefaultTolerances()
    for m in _DECIMAL_TOL_RE.finditer(page_text):
        places = len(m.group(1))
        result.by_decimal_places[places] = _round(float(m.group(2)))
    m = _ANGULAR_TOL_RE.search(page_text)
    if m:
        result.angular = _round(float(m.group(1)))
    return result


def apply_default_tolerance(
    parsed: ParsedCharacteristic, defaults: DefaultTolerances
) -> ParsedCharacteristic:
    """Backfill ``tol_plus``/``tol_minus``/limits on ``parsed`` from
    ``defaults`` when it has a nominal but no explicit tolerance of its own
    (e.g. "9X Ø0.250 THRU" relying on the drawing's general "X.XXX: ±0.005"
    note). Returns ``parsed`` unchanged if it already has an explicit
    tolerance, has no nominal, its decimal precision is unknown, or no
    matching default is configured.
    """
    if parsed.nominal is None or parsed.tol_plus is not None or parsed.tol_minus is not None:
        return parsed
    if defaults.is_empty():
        return parsed

    if parsed.char_type == CharacteristicType.ANGLE.value:
        tol = defaults.angular
    elif parsed.char_type in _TOLERANCED_DIMENSION_TYPES:
        tol = defaults.for_decimal_places(decimal_places(parsed.nominal_text))
    else:
        tol = None

    if tol is None:
        return parsed

    parsed.tol_plus = tol
    parsed.tol_minus = tol
    parsed.lower_limit, parsed.upper_limit = compute_limits(parsed.nominal, tol, tol)
    return parsed


def parse_characteristics(
    text: str,
    diameter_hint: bool = False,
    gdt_frame_hint: bool = False,
    depth_hint: bool = False,
    learned_symbols: Optional[dict[str, str]] = None,
) -> list[ParsedCharacteristic]:
    """Classify and parse a chunk of drawing text into one or more characteristics.

    Usually returns a single item, but a compound callout that packs two
    independently-inspected requirements into one piece of drawing text
    (e.g. a tapped hole's thread class *and* its depth, checked with
    different gauges) is split into separate characteristics here so each
    becomes its own balloon/line item.

    The original text is always preserved in ``raw_text`` regardless of
    whether parsing fully succeeds, so nothing is ever silently lost.

    ``learned_symbols``: marker letter -> char_type, taught by confirming a
    correction in the UI (see AppSettings.learn_symbol). Consulted only by
    the "font mangled a real symbol into an unrelated letter" fallbacks
    (_try_bare_mangled_diameter, _try_qty_prefixed_mangled_diameter), which
    otherwise default to assuming a diameter -- overriding that default is
    the whole point of remembering a correction instead of repeating it.

    ``diameter_hint``: the caller found evidence (outside of ``text`` --
    e.g. a Ø glyph drawn as vector line art rather than a font character,
    see :meth:`balloon_app.pdf_engine.PdfDocument.find_vector_diameter_symbol`)
    that this callout is a diameter even though no diameter symbol/word
    appears in the text itself. Only affects the bare-value fallback below;
    a callout that already matches a more specific pattern (thread, GD&T,
    depth, etc.) is unaffected.

    ``gdt_frame_hint``: the caller found a GD&T feature control frame's
    boxed-compartment structure drawn as vector line art around this text
    (see :meth:`balloon_app.pdf_engine.PdfDocument.find_vector_gdt_frame`).
    A frame's own symbol (flatness, straightness, etc.) is virtually always
    drawn this way, contributing no character at all to the text -- unlike
    ``diameter_hint``, there's no reliable way to tell *which* symbol from
    vector art alone, so this only prevents a bare tolerance value like
    "0.01" from being misread as a plain linear dimension; the specific
    symbol is left for the user to fill in during review.

    ``depth_hint``: like ``diameter_hint``, but for the "depth" glyph (see
    :meth:`balloon_app.pdf_engine.PdfDocument.find_vector_depth_symbol`) --
    a bare number that would otherwise fall through to being read as an
    unrelated plain linear dimension is instead classified as its feature's
    depth. Same scope restriction as ``diameter_hint``: only affects the
    bare-value fallback, not a callout that already matches something more
    specific.
    """
    text = (text or "").strip()
    if not text:
        return [ParsedCharacteristic(char_type=CharacteristicType.OTHER.value, raw_text=text, confidence=0.0)]

    thread_results = _try_thread(text)
    if thread_results is not None:
        return thread_results

    for attempt in (_try_surface_finish, _try_gdt, _try_general_tolerance):
        result = attempt(text)
        if result is not None:
            return [result]

    shape_and_secondary = _try_shape_with_secondary_value(text)
    if shape_and_secondary is not None:
        return shape_and_secondary

    mangled_shape_and_secondary = _try_mangled_shape_with_secondary_value(text, learned_symbols)
    if mangled_shape_and_secondary is not None:
        return mangled_shape_and_secondary

    qty_two_values = _try_qty_prefixed_two_values(text)
    if qty_two_values is not None:
        return qty_two_values

    qty_mangled_diameter = _try_qty_prefixed_mangled_diameter(text, learned_symbols)
    if qty_mangled_diameter is not None:
        return [qty_mangled_diameter]

    value_and_angle = _try_bare_value_with_angle(text)
    if value_and_angle is not None:
        return value_and_angle

    mangled_diameter = _try_bare_mangled_diameter(text, learned_symbols)
    if mangled_diameter is not None:
        return [mangled_diameter]

    has_depth_text = bool(_DEPTH_HINT_RE.search(text))
    is_depth = has_depth_text or depth_hint
    is_counterbore = bool(_COUNTERBORE_HINT_RE.search(text))
    is_countersink = bool(_COUNTERSINK_HINT_RE.search(text))
    is_square = bool(_SQUARE_HINT_RE.search(text))
    has_diameter_text = bool(_DIAMETER_RE.search(text))
    is_diameter = has_diameter_text or diameter_hint
    is_radius = bool(_RADIUS_RE.search(text))
    is_angle = bool(_ANGLE_HINT_RE.search(text))

    numeric = _extract_numeric_tolerance(text)

    # No real symbol/keyword identified anything above -- as a last resort
    # before falling all the way to a plain linear dimension, check whether
    # the text is *shaped* like a mangled marker leading its own value (see
    # _try_mangled_shape_with_secondary_value's docstring), e.g. "v n.159
    # +.002 -.000" (a counterbore diameter with tolerance, both symbols
    # mangled, no depth trailing it -- so the secondary-value fallback
    # above declined it, but the type is still worth guessing at rather
    # than defaulting to a plain dimension).
    mangled_marker_match = None
    if not (is_depth or is_counterbore or is_countersink or is_square or is_diameter or is_radius or is_angle):
        mangled_marker_match = _LEADING_MANGLED_MARKERS_RE.match(text)
    mangled_char_type: Optional[str] = None
    mangled_confidence = 0.0
    guessed_marker: Optional[str] = None
    if mangled_marker_match is not None:
        marker = mangled_marker_match.group(1)
        mangled_char_type, mangled_confidence = _resolve_learned_marker(marker, learned_symbols)
        guessed_marker = marker.strip().lower()

    # A hole-feature modifier symbol (depth/counterbore/countersink/square)
    # is more specific than a bare diameter/radius, so it wins when both are
    # present in the same callout (e.g. a counterbore diameter "⌴⌀.500").
    if is_depth:
        char_type = CharacteristicType.DEPTH.value
    elif is_counterbore:
        char_type = CharacteristicType.COUNTERBORE.value
    elif is_countersink:
        char_type = CharacteristicType.COUNTERSINK.value
    elif is_square:
        char_type = CharacteristicType.SQUARE.value
    elif is_diameter:
        char_type = CharacteristicType.DIAMETER.value
    elif is_radius:
        char_type = CharacteristicType.RADIUS.value
    elif is_angle:
        char_type = CharacteristicType.ANGLE.value
    elif mangled_char_type is not None:
        char_type = mangled_char_type
    elif gdt_frame_hint and numeric is not None:
        return [
            ParsedCharacteristic(
                char_type=CharacteristicType.GDT_FRAME.value,
                raw_text=text,
                gdt_tolerance=numeric.get("nominal_text") or text,
                confidence=0.55,
                **_gdt_tolerance_limits(numeric.get("nominal")),
            )
        ]
    elif numeric is not None:
        char_type = CharacteristicType.LINEAR_DIMENSION.value
    else:
        char_type = CharacteristicType.NOTE.value

    if numeric is not None:
        confidence = numeric["confidence"]
        if is_diameter or is_radius or is_angle or is_depth or is_counterbore or is_countersink or is_square:
            confidence = max(confidence, 0.75) if numeric["confidence"] >= 0.8 else 0.6
        elif mangled_char_type is not None:
            # The type itself is only a guess (a learned mapping, or the
            # bare default) -- cap confidence at that guess's own, however
            # confidently the value/tolerance itself was read.
            confidence = min(confidence, mangled_confidence)
        # diameter_hint/depth_hint fired but the text itself never carried
        # its symbol (it was drawn as vector line art, not a character) --
        # rewrite raw_text so the review table shows what the drawing
        # actually says, using the type's canonical symbol. Likewise for a
        # mangled leading marker resolved above: the mangled letter carries
        # no real information, so showing it verbatim would only confuse
        # review, not aid traceability.
        if char_type == CharacteristicType.DIAMETER.value and not has_diameter_text and guessed_marker is None:
            raw_text_out = f"⌀{text}"
        elif char_type == CharacteristicType.DEPTH.value and not has_depth_text and guessed_marker is None:
            raw_text_out = f"▼{text}"
        elif guessed_marker is not None:
            symbol = _CANONICAL_SYMBOL_FOR_TYPE.get(char_type, "⌀")
            raw_text_out = symbol + text[mangled_marker_match.start(3):]
        else:
            raw_text_out = text
        return [
            ParsedCharacteristic(
                char_type=char_type,
                raw_text=raw_text_out,
                nominal=numeric["nominal"],
                nominal_text=numeric.get("nominal_text"),
                tol_plus=numeric["tol_plus"],
                tol_minus=numeric["tol_minus"],
                lower_limit=numeric["lower_limit"],
                upper_limit=numeric["upper_limit"],
                confidence=confidence,
                guessed_symbol_marker=guessed_marker,
            )
        ]

    # Nothing numeric recognized at all -- preserve as a note for manual review.
    return [
        ParsedCharacteristic(
            char_type=CharacteristicType.NOTE.value,
            raw_text=text,
            note=text,
            confidence=0.2,
        )
    ]


def parse_characteristic(
    text: str,
    diameter_hint: bool = False,
    gdt_frame_hint: bool = False,
    depth_hint: bool = False,
    learned_symbols: Optional[dict[str, str]] = None,
) -> ParsedCharacteristic:
    """Classify and parse a chunk of drawing text into a single characteristic.

    Convenience wrapper around :func:`parse_characteristics` for callers
    that only need one representative result (e.g. estimating a confidence
    label for a detection) rather than every characteristic packed into it.
    """
    return parse_characteristics(
        text, diameter_hint=diameter_hint, gdt_frame_hint=gdt_frame_hint, depth_hint=depth_hint,
        learned_symbols=learned_symbols
    )[0]
