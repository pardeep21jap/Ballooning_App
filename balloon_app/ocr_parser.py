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

_SURFACE_FINISH_RE_PREFIX = re.compile(rf"\bRa\s*({NUM})\s*(µm|um|μm)?\b", re.IGNORECASE)
_SURFACE_FINISH_RE_SUFFIX = re.compile(rf"({NUM})\s*(µm|um|μm)?\s*Ra\b", re.IGNORECASE)

_GENERAL_TOL_KEYWORDS_RE = re.compile(
    r"(UNLESS OTHERWISE SPECIFIED|GENERAL TOLERANCE|DEFAULT TOLERANCE|U\.?O\.?S\.?)",
    re.IGNORECASE,
)

_DIAMETER_RE = re.compile(rf"[{_DIAMETER_SYMBOLS}]|\bDIA\b", re.IGNORECASE)
_RADIUS_RE = re.compile(r"(?<![A-Za-z])R(?![a-zA-Z])\s*" + NUM)
_ANGLE_HINT_RE = re.compile(r"°")

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
    search_text = text.replace("°", " ")

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


def _try_thread(text: str) -> Optional[ParsedCharacteristic]:
    m = _THREAD_METRIC_RE.search(text)
    if m:
        callout = f"M{m.group(1)} x {m.group(2)}"
        return ParsedCharacteristic(
            char_type=CharacteristicType.THREAD.value,
            raw_text=text,
            thread_callout=callout,
            confidence=0.85,
        )
    m = _THREAD_UNIFIED_RE.search(text)
    if m:
        callout = f"{m.group(1)}-{m.group(2)} {m.group(3).upper()}"
        return ParsedCharacteristic(
            char_type=CharacteristicType.THREAD.value,
            raw_text=text,
            thread_callout=callout,
            confidence=0.85,
        )
    return None


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


def parse_characteristic(text: str) -> ParsedCharacteristic:
    """Classify and parse a chunk of drawing text into a characteristic.

    The original text is always preserved in ``raw_text`` regardless of
    whether parsing fully succeeds, so nothing is ever silently lost.
    """
    text = (text or "").strip()
    if not text:
        return ParsedCharacteristic(char_type=CharacteristicType.OTHER.value, raw_text=text, confidence=0.0)

    for attempt in (_try_thread, _try_surface_finish, _try_gdt, _try_general_tolerance):
        result = attempt(text)
        if result is not None:
            return result

    is_diameter = bool(_DIAMETER_RE.search(text))
    is_radius = bool(_RADIUS_RE.search(text))
    is_angle = bool(_ANGLE_HINT_RE.search(text))

    numeric = _extract_numeric_tolerance(text)

    if is_diameter:
        char_type = CharacteristicType.DIAMETER.value
    elif is_radius:
        char_type = CharacteristicType.RADIUS.value
    elif is_angle:
        char_type = CharacteristicType.ANGLE.value
    elif numeric is not None and numeric["confidence"] >= 0.8:
        char_type = CharacteristicType.LINEAR_DIMENSION.value
    elif numeric is not None:
        char_type = CharacteristicType.LINEAR_DIMENSION.value
    else:
        char_type = CharacteristicType.NOTE.value

    if numeric is not None:
        confidence = numeric["confidence"]
        if is_diameter or is_radius or is_angle:
            confidence = max(confidence, 0.75) if numeric["confidence"] >= 0.8 else 0.6
        return ParsedCharacteristic(
            char_type=char_type,
            raw_text=text,
            nominal=numeric["nominal"],
            tol_plus=numeric["tol_plus"],
            tol_minus=numeric["tol_minus"],
            lower_limit=numeric["lower_limit"],
            upper_limit=numeric["upper_limit"],
            confidence=confidence,
        )

    # Nothing numeric recognized at all -- preserve as a note for manual review.
    return ParsedCharacteristic(
        char_type=CharacteristicType.NOTE.value,
        raw_text=text,
        note=text,
        confidence=0.2,
    )
