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
# _try_shape_with_depth below) -- unlike the *_HINT_RE checks above these
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
    rf"^\s*\d+\s*[Xx]\s+[A-Za-z]{{0,3}}\s*({NUM})\s+[A-Za-z]{{0,3}}\s*({NUM})\b"
)

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
                confidence=0.8,
            )
        )
    return results


def _try_shape_with_depth(text: str) -> Optional[list[ParsedCharacteristic]]:
    """A shape value paired with a trailing depth callout on the same line,

    e.g. "2X ⌀0.089 ▼0.500" -- a hole's diameter *and* its depth, checked
    with different gauges -- becomes two characteristics (Diameter, Depth)
    instead of one, matching how :func:`_try_thread` already splits a
    thread callout from its trailing depth.
    """
    if not _DEPTH_HINT_RE.search(text):
        return None  # nothing to pair with -- let the single-value path handle it

    for value_re, char_type in (
        (_DIAMETER_VALUE_RE, CharacteristicType.DIAMETER.value),
        (_SQUARE_VALUE_RE, CharacteristicType.SQUARE.value),
        (_RADIUS_VALUE_RE, CharacteristicType.RADIUS.value),
    ):
        m = value_re.search(text)
        if not m:
            continue
        depth_match = _TRAILING_DECIMAL_RE.search(text, m.end())
        if not depth_match:
            continue
        return [
            ParsedCharacteristic(
                char_type=char_type,
                raw_text=text,
                nominal=_round(float(m.group(1))),
                confidence=0.85,
            ),
            ParsedCharacteristic(
                char_type=CharacteristicType.DEPTH.value,
                raw_text=text,
                nominal=_round(float(depth_match.group(1))),
                confidence=0.8,
            ),
        ]
    return None


def _try_qty_prefixed_two_values(text: str) -> Optional[list[ParsedCharacteristic]]:
    """Structural fallback for a quantity-prefixed hole callout whose two
    dimension symbols are both unrecognized (see _QTY_TWO_VALUE_RE above).
    Assumes the far more common ordering: diameter first, depth second.
    """
    m = _QTY_TWO_VALUE_RE.match(text)
    if not m:
        return None
    return [
        ParsedCharacteristic(
            char_type=CharacteristicType.DIAMETER.value,
            raw_text=text,
            nominal=_round(float(m.group(1))),
            confidence=0.55,
        ),
        ParsedCharacteristic(
            char_type=CharacteristicType.DEPTH.value,
            raw_text=text,
            nominal=_round(float(m.group(2))),
            confidence=0.5,
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

    shape_and_depth = _try_shape_with_depth(text)
    if shape_and_depth is not None:
        return shape_and_depth

    qty_two_values = _try_qty_prefixed_two_values(text)
    if qty_two_values is not None:
        return qty_two_values

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
