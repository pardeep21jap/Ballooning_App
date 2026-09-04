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
    r"\b(\d+(?:/\d+)?|#\d+)\s*-\s*(\d+)\s*(UNC|UNF|UNEF|UN|NPT|NPTF)\b",
    re.IGNORECASE,
)
_THREAD_METRIC_RE = re.compile(rf"\bM\s?(\d+\.?\d*)\s*[xX×]\s*({NUM})\b")

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
_DEPTH_HINT_RE = re.compile(r"[▼↓]|\bDEPTH\b|\bDEEP\b|\bDP\b", re.IGNORECASE)
_COUNTERBORE_HINT_RE = re.compile(r"[⌴]|C['’]?BORE|\bCOUNTERBORE\b", re.IGNORECASE)
_COUNTERSINK_HINT_RE = re.compile(r"[⌵]|C['’]?SINK|\bCSK\b|\bCOUNTERSINK\b", re.IGNORECASE)
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

# Numeric tolerance extraction patterns, tried in priority order.
_ASYM_SLASH_RE = re.compile(rf"({NUM})\s*\+\s*({NUM})\s*/\s*-\s*({NUM})")
_ASYM_SPACE_RE = re.compile(rf"({NUM})\s*\+\s*({NUM})\s+-\s*({NUM})")
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
    m = _THREAD_METRIC_RE.search(text)
    if m:
        callout = f"M{m.group(1)} x {m.group(2)}"
    else:
        m = _THREAD_UNIFIED_RE.search(text)
        if m:
            callout = f"{m.group(1)}-{m.group(2)} {m.group(3).upper()}"
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


def _try_shape_with_secondary_value(text: str) -> Optional[list[ParsedCharacteristic]]:
    """A shape value paired with a trailing depth *or* angle callout on the
    same line -- each is inspected with a different gauge, so it becomes
    its own characteristic instead of one that silently drops a value:

    * "2X ⌀0.089 ▼0.500" -- a hole's diameter and its depth.
    * "⌵⌀0.507 X 82°" -- a countersink's diameter and its included angle.

    Matches how :func:`_try_thread` already splits a thread callout from
    its trailing depth.
    """
    has_depth = bool(_DEPTH_HINT_RE.search(text))
    has_degree = "°" in text
    if not has_depth and not has_degree:
        return None  # nothing to pair with -- let the single-value path handle it

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

    secondary: Optional[ParsedCharacteristic] = None
    if has_depth:
        depth_match = _TRAILING_DECIMAL_RE.search(text, m.end())
        if depth_match:
            secondary = ParsedCharacteristic(
                char_type=CharacteristicType.DEPTH.value,
                raw_text=text,
                nominal=_round(float(depth_match.group(1))),
                nominal_text=depth_match.group(1),
                confidence=0.8,
            )
    if secondary is None and has_degree:
        angle_match = _TRAILING_ANGLE_RE.search(text, m.end())
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
        nominal=_round(float(m.group(1))),
        nominal_text=m.group(1),
        confidence=0.85,
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

            return ParsedCharacteristic(
                char_type=CharacteristicType.GDT_FRAME.value,
                raw_text=text,
                gdt_symbol=name,
                gdt_tolerance=(f"⌀{tolerance}" if tolerance and _DIAMETER_RE.search(text) else tolerance),
                material_condition=material_condition,
                datums=datums,
                confidence=0.75,
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
        )
    return None


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
_DECIMAL_TOL_RE: dict[int, re.Pattern] = {
    1: re.compile(rf"X\.X\b\s*[:=]?\s*±?\s*({NUM})"),
    2: re.compile(rf"X\.XX\b\s*[:=]?\s*±?\s*({NUM})"),
    3: re.compile(rf"X\.XXX\b\s*[:=]?\s*±?\s*({NUM})"),
    4: re.compile(rf"X\.XXXX\b\s*[:=]?\s*±?\s*({NUM})"),
}
_ANGULAR_TOL_RE = re.compile(rf"ANGLES?\s*[:=]?\s*±?\s*({NUM})\s*°?", re.IGNORECASE)

_DECIMAL_PLACES_FIELD = {1: "one_decimal", 2: "two_decimal", 3: "three_decimal", 4: "four_decimal"}

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
    "X.XX: ±0.0100" convention), plus a separate angular tolerance.
    """

    one_decimal: Optional[float] = None
    two_decimal: Optional[float] = None
    three_decimal: Optional[float] = None
    four_decimal: Optional[float] = None
    angular: Optional[float] = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (self.one_decimal, self.two_decimal, self.three_decimal, self.four_decimal, self.angular)
        )

    def for_decimal_places(self, places: int) -> Optional[float]:
        """The configured tolerance for a value with this many decimal
        places, falling back to the next-coarsest configured entry if the
        table has gaps (e.g. only X.XXX is set but a value has 4 places)."""
        if places <= 0:
            return None
        for p in range(min(places, 4), 0, -1):
            value = getattr(self, _DECIMAL_PLACES_FIELD[p])
            if value is not None:
                return value
        return None


def _decimal_places(nominal_text: Optional[str]) -> int:
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
    for places, pattern in _DECIMAL_TOL_RE.items():
        m = pattern.search(page_text)
        if m:
            setattr(result, _DECIMAL_PLACES_FIELD[places], _round(float(m.group(1))))
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
        tol = defaults.for_decimal_places(_decimal_places(parsed.nominal_text))
    else:
        tol = None

    if tol is None:
        return parsed

    parsed.tol_plus = tol
    parsed.tol_minus = tol
    parsed.lower_limit, parsed.upper_limit = compute_limits(parsed.nominal, tol, tol)
    return parsed


def parse_characteristics(text: str) -> list[ParsedCharacteristic]:
    """Classify and parse a chunk of drawing text into one or more characteristics.

    Usually returns a single item, but a compound callout that packs two
    independently-inspected requirements into one piece of drawing text
    (e.g. a tapped hole's thread class *and* its depth, checked with
    different gauges) is split into separate characteristics here so each
    becomes its own balloon/line item.

    The original text is always preserved in ``raw_text`` regardless of
    whether parsing fully succeeds, so nothing is ever silently lost.
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

    qty_two_values = _try_qty_prefixed_two_values(text)
    if qty_two_values is not None:
        return qty_two_values

    value_and_angle = _try_bare_value_with_angle(text)
    if value_and_angle is not None:
        return value_and_angle

    is_depth = bool(_DEPTH_HINT_RE.search(text))
    is_counterbore = bool(_COUNTERBORE_HINT_RE.search(text))
    is_countersink = bool(_COUNTERSINK_HINT_RE.search(text))
    is_square = bool(_SQUARE_HINT_RE.search(text))
    is_diameter = bool(_DIAMETER_RE.search(text))
    is_radius = bool(_RADIUS_RE.search(text))
    is_angle = bool(_ANGLE_HINT_RE.search(text))

    numeric = _extract_numeric_tolerance(text)

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
    elif numeric is not None:
        char_type = CharacteristicType.LINEAR_DIMENSION.value
    else:
        char_type = CharacteristicType.NOTE.value

    if numeric is not None:
        confidence = numeric["confidence"]
        if is_diameter or is_radius or is_angle or is_depth or is_counterbore or is_countersink or is_square:
            confidence = max(confidence, 0.75) if numeric["confidence"] >= 0.8 else 0.6
        return [
            ParsedCharacteristic(
                char_type=char_type,
                raw_text=text,
                nominal=numeric["nominal"],
                nominal_text=numeric.get("nominal_text"),
                tol_plus=numeric["tol_plus"],
                tol_minus=numeric["tol_minus"],
                lower_limit=numeric["lower_limit"],
                upper_limit=numeric["upper_limit"],
                confidence=confidence,
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


def parse_characteristic(text: str) -> ParsedCharacteristic:
    """Classify and parse a chunk of drawing text into a single characteristic.

    Convenience wrapper around :func:`parse_characteristics` for callers
    that only need one representative result (e.g. estimating a confidence
    label for a detection) rather than every characteristic packed into it.
    """
    return parse_characteristics(text)[0]
