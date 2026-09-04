"""Automatic ballooning pipeline: native PDF text first, OCR fallback second.

Pipeline (see README for full description):

1. Render the page and pull native PDF text spans via :mod:`pdf_engine`.
2. If no usable native text exists (e.g. a scanned drawing), fall back to
   Tesseract OCR via :class:`RulesOcrDetector` -- handled gracefully if
   Tesseract is not installed.
3. Each candidate text chunk is classified/parsed by
   :func:`balloon_app.ocr_parser.parse_characteristic`.
4. Proposed balloons are placed near their source text, nudged to avoid
   overlapping already-placed balloons, and returned as ``pending``/``auto``
   :class:`~balloon_app.data_model.Balloon` objects for user review.

This module also defines a small, model-agnostic detector interface
(:class:`Detection`, :class:`BaseDetector`) so a future trained YOLO model
can be dropped in (:class:`YoloDetector`) without changing the rest of the
application.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from balloon_app.config import AUTO_BALLOON_DPI, BALLOON_RADIUS_PDF_POINTS, CHARACTERISTIC_CLASSES, RULES_OCR_MODEL_VERSION
from balloon_app.data_model import Balloon, BalloonSource, CharacteristicType, ReviewStatus
from balloon_app.ocr_parser import (
    NUM,
    DefaultTolerances,
    apply_default_tolerance,
    parse_characteristic,
    parse_characteristics,
)
from balloon_app.pdf_engine import PdfDocument, TextBlock, rect_pdf_to_pixel, rect_pixel_to_pdf

logger = logging.getLogger("balloon_app.auto_balloon")

# A *decimal* number (has a decimal point) is a strong dimension signal.
# A bare integer is not -- it's just as likely to be a sheet zone marker
# ("1" "2" "3" "4" along the border), a date fragment, a QTY count, a
# drawing/part number, or a material temper code ("6061 T6").
_DECIMAL_NUMBER_RE = re.compile(r"\d+\.\d+|\.\d+")
_SYMBOL_HINT_RE = re.compile(
    r"[⌀ØΦ∅]|±|°|\bRa\b|\bDIA\b|"
    r"⏤|⏥|○|⌭|⌒|⌓|⟂|∠|∥|⌯|⌖|◎|↗|⌰|⌇|"
    r"[▼↓⌴⌵□]",
    re.IGNORECASE,
)
_RADIUS_HINT_RE = re.compile(r"(?<![A-Za-z])R(?![a-zA-Z])\s*\d")
_METRIC_THREAD_HINT_RE = re.compile(r"\bM\d+\.?\d*\s*[xX×]\s*\d")
_UNIFIED_THREAD_HINT_RE = re.compile(
    r"\b\d+(?:/\d+)?\s*-\s*\d+\s*(UNC|UNF|UNEF|UN|NPT|NPTF)\b", re.IGNORECASE
)
_PIPE_DATUM_HINT_RE = re.compile(r"\d\s*\|\s*[A-Z]")
# A short, bare integer with nothing else attached: "1", "2", "23", "(4)".
# These are almost always zone/sheet/revision markers, not dimensions.
_BARE_SHORT_INTEGER_RE = re.compile(r"^\(?\d{1,2}\)?$")

# Boilerplate phrases that appear only inside a drawing's title block, never
# as an inspection characteristic in their own right. The title-block note
# often contains real decimal numbers and symbols (a "TOLERANCES UNLESS
# OTHERWISE NOTED" table, "±0.5°", fractional tolerances) that would
# otherwise pass _looks_like_characteristic and get ballooned individually
# -- see _title_block_cutoff_y below, which uses these to exclude the whole
# title block region rather than trying to keyword-match every field in it.
_TITLE_BLOCK_KEYWORDS_RE = re.compile(
    r"TOLERANCES UNLESS OTHERWISE (NOTED|SPECIFIED)|UNLESS OTHERWISE SPECIFIED|"
    r"\bDRAWN\b|\bCHECKED\b|\bDESIGNED\b|\bENGINEER(ED)?\b|\bAPPROVED\b|"
    r"\bTITLE\b|\bSCALE\b|\bSHEET\b|\bQTY\.?\s*:|\bMATERIAL\b|\bFINISH\b|"
    r"\bPROJECT CODE\b|\bDWG\.?\s*NO\.?\b|\bNEXT ASSY\b|\bDO NOT SCALE\b|"
    r"\bPROPRIETARY\b|\bCONFIDENTIAL\b|\bINTERPRET (DRAWING|PER)\b|"
    r"\bBREAK ALL SHARP EDGES\b|\bFRACTIONAL\b",
    re.IGNORECASE,
)


def _in_zone_margin(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    margin_fraction: float = 0.04,
) -> bool:
    """True if ``bbox`` sits within the thin zone/grid-reference margin
    strip just inside a sheet edge (top, left, or right -- the bottom is
    handled separately by the title-block cutoff), where ANSI/ISO Y14.1
    zone letters/numbers ("1 2 3 4", "A B C D") are printed. A real
    dimension is never drawn in that margin.
    """
    x0, y0, x1, y1 = bbox
    margin_x = page_width * margin_fraction
    margin_y = page_height * margin_fraction
    return y0 <= margin_y or x0 <= margin_x or x1 >= page_width - margin_x


def _title_block_cutoff_y(blocks: list[TextBlock], page_height: float) -> Optional[float]:
    """Return a page-y cutoff below which everything is treated as inside
    the title block, or ``None`` if no title-block boilerplate was found.

    Rather than keyword-matching every individual title-block field (drawing
    number, revision, company name/logo, dates -- an open-ended list that
    varies per template), this finds the topmost boilerplate phrase in the
    bottom half of the page and excludes that whole horizontal strip down
    to the bottom edge. Matches ANSI/ISO title blocks, which run the full
    sheet width along the bottom; a title block running the full height
    along one side instead would need a different heuristic.
    """
    matches = [
        b for b in blocks
        if b.bbox[1] > page_height * 0.5 and _TITLE_BLOCK_KEYWORDS_RE.search(b.text)
    ]
    if not matches:
        return None
    return min(b.bbox[1] for b in matches) - 4.0  # small padding above the topmost match


def _looks_like_characteristic(
    text: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    page_size: Optional[tuple[float, float]] = None,
) -> bool:
    """Cheap pre-filter so we don't propose a balloon for every scrap of text.

    A drawing has plenty of text that is never an inspection characteristic:
    title-block boilerplate, drawing/part numbers, dates, QTY counts,
    material codes, and -- notably -- the zone/grid reference numbers
    printed along a drawing border (ANSI/ISO sheet format), which are bare
    1-2 digit integers and are textually indistinguishable from a real
    whole-number dimension ("75", "19", "Ø11", a bare "3" on a radius) in a
    metric drawing. Only position tells them apart: zone markers live in
    the sheet's margin strip (see _in_zone_margin), dimensions don't. When
    ``bbox``/``page_size`` are supplied, a bare short integer outside that
    margin is accepted; without geometry, it's conservatively rejected
    (the old behavior) since it can't be told apart from a zone marker.
    """
    text = text.strip()
    if not text or len(text) > 120:
        return False
    if _BARE_SHORT_INTEGER_RE.match(text):
        if bbox is not None and page_size is not None and not _in_zone_margin(bbox, *page_size):
            return True
        return False
    return bool(
        _DECIMAL_NUMBER_RE.search(text)
        or _SYMBOL_HINT_RE.search(text)
        or _RADIUS_HINT_RE.search(text)
        or _METRIC_THREAD_HINT_RE.search(text)
        or _UNIFIED_THREAD_HINT_RE.search(text)
        or _PIPE_DATUM_HINT_RE.search(text)
    )


@dataclass
class Detection:
    """A minimal, model-agnostic detection result.

    ``bbox`` is in the coordinate space of whatever image was passed to
    ``detect()`` (pixel space). ``raw_text`` is ``None`` when the detector
    has no text-recognition capability (e.g. a plain YOLO box detector).
    """

    bbox: tuple[float, float, float, float]
    label: str
    confidence: float
    raw_text: Optional[str] = None


class BaseDetector(ABC):
    """Common interface for anything that can find candidate regions on a page image."""

    available: bool = False
    error: Optional[str] = None

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections found in ``image`` (an RGB uint8 numpy array)."""
        raise NotImplementedError


class RulesOcrDetector(BaseDetector):
    """OCR-based detector used when a page has no native/selectable text.

    Uses Tesseract (via pytesseract) to find text lines, then classifies
    each line with the same rule-based parser used for native PDF text.
    If Tesseract is not installed or not found, ``available`` is False and
    ``detect()`` returns an empty list instead of raising -- callers must
    check ``available``/``error`` to inform the user.
    """

    def __init__(self, tesseract_path: Optional[str] = None):
        self.available = False
        self.error = None
        self._pytesseract = None
        try:
            import pytesseract  # noqa: WPS433 - intentional lazy/optional import
        except ImportError:
            self.error = "pytesseract is not installed. OCR fallback is unavailable."
            return

        self._pytesseract = pytesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        try:
            pytesseract.get_tesseract_version()
            self.available = True
        except Exception as exc:  # pytesseract raises its own TesseractNotFoundError
            self.error = (
                "Tesseract OCR engine was not found on this system. Install it or set "
                f"TESSERACT_PATH in Settings. ({exc})"
            )

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.available or self._pytesseract is None:
            return []
        try:
            data = self._pytesseract.image_to_data(image, output_type=self._pytesseract.Output.DICT)
        except Exception:
            logger.exception("OCR (image_to_data) failed")
            return []

        lines: dict[tuple[int, int, int], list[int]] = {}
        count = len(data.get("text", []))
        for i in range(count):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(i)

        image_height, image_width = image.shape[:2]
        detections: list[Detection] = []
        for idxs in lines.values():
            words = [data["text"][i] for i in idxs]
            line_text = " ".join(w for w in words if w).strip()
            x0 = min(data["left"][i] for i in idxs)
            y0 = min(data["top"][i] for i in idxs)
            x1 = max(data["left"][i] + data["width"][i] for i in idxs)
            y1 = max(data["top"][i] + data["height"][i] for i in idxs)
            if not _looks_like_characteristic(line_text, bbox=(x0, y0, x1, y1), page_size=(image_width, image_height)):
                continue

            confs = []
            for i in idxs:
                try:
                    c = float(data["conf"][i])
                    if c >= 0:
                        confs.append(c)
                except (TypeError, ValueError):
                    continue
            ocr_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.5

            parsed = parse_characteristic(line_text)
            combined = max(0.05, min(0.99, parsed.confidence * max(ocr_conf, 0.2)))
            detections.append(
                Detection(bbox=(x0, y0, x1, y1), label=parsed.char_type, confidence=combined, raw_text=line_text)
            )
        return detections


class YoloDetector(BaseDetector):
    """Optional future detector backed by a trained Ultralytics YOLO model.

    Disables itself gracefully (``available = False``) if ``ultralytics``
    is not installed or the model file does not exist, so the rest of the
    app can always fall back to :class:`RulesOcrDetector` without special
    casing.
    """

    def __init__(self, model_path: Path | str, confidence_threshold: float = 0.25):
        self.available = False
        self.error = None
        self.model = None
        self.confidence_threshold = confidence_threshold

        model_path = Path(model_path)
        if not model_path.exists():
            self.error = f"YOLO model file not found: {model_path}"
            return
        try:
            from ultralytics import YOLO  # noqa: WPS433 - optional dependency
        except ImportError:
            self.error = "ultralytics package is not installed. Using rule-based detector instead."
            return
        try:
            self.model = YOLO(str(model_path))
            self.available = True
        except Exception as exc:
            self.error = f"Failed to load YOLO model '{model_path}': {exc}"

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.available or self.model is None:
            return []
        try:
            results = self.model.predict(
                source=image, device="cpu", verbose=False, conf=self.confidence_threshold
            )
        except Exception:
            logger.exception("YOLO inference failed")
            return []

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except Exception:
                    continue
                label = (
                    CHARACTERISTIC_CLASSES[cls_id]
                    if 0 <= cls_id < len(CHARACTERISTIC_CLASSES)
                    else CharacteristicType.OTHER.value
                )
                detections.append(Detection(bbox=tuple(xyxy), label=label, confidence=conf, raw_text=None))
        return detections


def _merge_nearby_text_blocks(blocks: list[TextBlock], y_tol: float = 3.0, x_gap: float = 18.0) -> list[TextBlock]:
    """Merge text spans that sit on the same line and are close together.

    PDF text is often split into several spans (e.g. ``"50.00"`` and
    ``"±0.05"`` as separate runs with different fonts). Merging them lets
    the parser see the full callout.
    """
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda b: (round(b.bbox[1] / max(y_tol, 0.1)), b.bbox[0]))
    merged: list[TextBlock] = [ordered[0]]
    for nxt in ordered[1:]:
        current = merged[-1]
        same_line = abs(nxt.bbox[1] - current.bbox[1]) <= y_tol and abs(nxt.bbox[3] - current.bbox[3]) <= y_tol * 3
        gap = nxt.bbox[0] - current.bbox[2]
        if same_line and -2.0 <= gap <= x_gap:
            new_bbox = (
                min(current.bbox[0], nxt.bbox[0]),
                min(current.bbox[1], nxt.bbox[1]),
                max(current.bbox[2], nxt.bbox[2]),
                max(current.bbox[3], nxt.bbox[3]),
            )
            merged[-1] = TextBlock(text=f"{current.text} {nxt.text}", bbox=new_bbox)
        else:
            merged.append(nxt)
    return merged


_BARE_SIGNED_NUM_RE = re.compile(rf"^\s*([+-])\s*({NUM})\s*$")
# A +/- value embedded *within* a larger block's text, e.g. the "-0.010" in
# an already same-line-merged "Ø6.38 -0.010" -- used to pull out and
# reorder a sign that's already part of the anchor block, so it isn't
# duplicated or emitted in the wrong order when merged with an externally
# found fragment (see _merge_stacked_tolerance_fragments).
_EMBEDDED_PLUS_RE = re.compile(rf"\+\s*({NUM})")
_EMBEDDED_MINUS_RE = re.compile(rf"-\s*({NUM})")
# A trailing bare (unsigned) zero after the nominal, e.g. the "0" in "Ø8 0"
# -- the unsigned side of a unilateral tolerance ("0 / +0.05"), where the
# zero side is conventionally written without a +/- sign since +0 and -0
# are equivalent. Requires a preceding token (the nominal itself) so a
# bare "0" is never mistaken for the whole value.
_TRAILING_BARE_ZERO_RE = re.compile(r"(?<=\S)\s+(0(?:\.0+)?)\s*$")


def _bboxes_are_near(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], y_gap: float, x_slack: float
) -> bool:
    a_x0, a_y0, a_x1, a_y1 = a
    b_x0, b_y0, b_x1, b_y1 = b
    vertical_gap = max(a_y0, b_y0) - min(a_y1, b_y1)  # <= 0 when the boxes already overlap in y
    if vertical_gap > y_gap:
        return False
    horizontal_gap = max(a_x0, b_x0) - min(a_x1, b_x1)
    return horizontal_gap <= x_slack


def _union_bbox(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _merge_stacked_tolerance_fragments(
    blocks: list[TextBlock], y_gap: float = 20.0, x_slack: float = 40.0
) -> list[TextBlock]:
    """Merge a nominal value with +/- tolerance fragments stacked above or
    below it as separate lines -- a common drawing convention:

            +0.005
        Ø6.38
            -0.010

    Each line above is its own PDF text span, on a different y-baseline
    than the nominal, so :func:`_merge_nearby_text_blocks` (same-line only)
    leaves them separate. Without this, "+0.005" and "-0.010" would either
    balloon as their own (meaningless) characteristics, or the nominal
    would balloon with no tolerance at all even though one was clearly
    given on the drawing.

    A block only participates as a tolerance fragment if its *entire* text
    is just a signed number -- this is deliberately narrow so it can't
    accidentally swallow an unrelated nearby dimension. Conversely, an
    anchor must actually contain a digit -- a bare datum-reference letter
    ("B") sitting near a fragment is not a dimension and must not steal it.
    When more than one eligible anchor is in range of the same fragment
    (e.g. two dimensions stacked close together), the fragment goes to
    whichever is geometrically *closest*, not just whichever is processed
    first.

    The anchor block itself may already carry one sign (e.g. same-line
    merging already combined "Ø6.38" and "-0.010" into one block since
    they share a line, leaving only "+0.005" -- on a different line --
    still separate). That embedded sign is pulled back out and re-emitted
    in canonical "nominal +plus -minus" order along with anything merged in
    externally, rather than just appended wherever it happened to already
    be -- appending blindly can produce "6.38 -0.010 +0.005", which the
    asymmetric-tolerance parser (which requires + before -) then fails to
    recognize as a tolerance at all.
    """
    is_sign_fragment = [bool(_BARE_SIGNED_NUM_RE.match(b.text)) for b in blocks]
    has_digit = [bool(re.search(r"\d", b.text)) for b in blocks]

    # Each fragment picks its single closest eligible anchor, rather than
    # anchors greedily claiming whatever fragment they encounter first.
    fragment_anchor: dict[int, int] = {}
    for j, frag in enumerate(blocks):
        if not is_sign_fragment[j]:
            continue
        best_i, best_dist = None, None
        for i, cand in enumerate(blocks):
            if i == j or is_sign_fragment[i] or not has_digit[i]:
                continue
            if not _bboxes_are_near(cand.bbox, frag.bbox, y_gap, x_slack):
                continue
            dist = _bbox_center_distance(cand.bbox, frag.bbox)
            if best_dist is None or dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is not None:
            fragment_anchor[j] = best_i

    used: set[int] = set()
    merged: list[TextBlock] = []

    for i, nominal in enumerate(blocks):
        if is_sign_fragment[i] or i in used:
            continue

        assigned = [j for j, anchor in fragment_anchor.items() if anchor == i]
        plus_idx = minus_idx = None
        for j in assigned:
            sign = _BARE_SIGNED_NUM_RE.match(blocks[j].text).group(1)
            if sign == "+" and plus_idx is None:
                plus_idx = j
            elif sign == "-" and minus_idx is None:
                minus_idx = j

        if plus_idx is None and minus_idx is None:
            continue  # no external fragment to merge -- leave this block as-is

        base_text = nominal.text
        plus_val = minus_val = None
        embedded_plus = _EMBEDDED_PLUS_RE.search(base_text)
        if embedded_plus:
            plus_val = embedded_plus.group(1)
            base_text = base_text.replace(embedded_plus.group(0), " ", 1)
        embedded_minus = _EMBEDDED_MINUS_RE.search(base_text)
        if embedded_minus:
            minus_val = embedded_minus.group(1)
            base_text = base_text.replace(embedded_minus.group(0), " ", 1)
        base_text = " ".join(base_text.split())

        combined_bbox = nominal.bbox
        if plus_idx is not None:
            plus_val = _BARE_SIGNED_NUM_RE.match(blocks[plus_idx].text).group(2)
            combined_bbox = _union_bbox(combined_bbox, blocks[plus_idx].bbox)
            used.add(plus_idx)
        if minus_idx is not None:
            minus_val = _BARE_SIGNED_NUM_RE.match(blocks[minus_idx].text).group(2)
            combined_bbox = _union_bbox(combined_bbox, blocks[minus_idx].bbox)
            used.add(minus_idx)

        # A unilateral tolerance ("0 / +0.05") -- only one side has an
        # explicit sign; the other, always exactly 0, is conventionally
        # written bare since +0 and -0 are equivalent. That bare "0" sits
        # right after the nominal (same-line merged already, e.g. "Ø8 0"),
        # so only one of plus_val/minus_val was found above; the trailing
        # bare number fills in the other side.
        if (plus_val is None) != (minus_val is None):
            trailing_zero = _TRAILING_BARE_ZERO_RE.search(base_text)
            if trailing_zero:
                base_text = base_text[: trailing_zero.start()].rstrip()
                if plus_val is None:
                    plus_val = trailing_zero.group(1)
                else:
                    minus_val = trailing_zero.group(1)

        parts = [base_text]
        if plus_val is not None:
            parts.append(f"+{plus_val}")
        if minus_val is not None:
            parts.append(f"-{minus_val}")
        used.add(i)
        merged.append(TextBlock(text=" ".join(parts), bbox=combined_bbox))

    untouched = [b for i, b in enumerate(blocks) if i not in used]
    return untouched + merged


def _bbox_has_ink(gray_image: np.ndarray, bbox_px: tuple[float, float, float, float], padding: float = 2.0, min_dark_fraction: float = 0.01) -> bool:
    """OpenCV-based plausibility check: does this bbox actually cover ink?

    A text span's *reported* bounding box can be wrong for reasons that have
    nothing to do with our regex logic -- a leftover invisible OCR text
    layer from a "searchable PDF" conversion, a misaligned/hidden layer, or
    a bad span union from merging. All of these produce a candidate that
    looks fine on paper (real digits, plausible bbox) but sits over a blank
    part of the rendered page. Cross-checking against actual rendered pixels
    catches this regardless of the underlying cause.
    """
    height, width = gray_image.shape[:2]
    x0 = max(0, int(bbox_px[0] - padding))
    y0 = max(0, int(bbox_px[1] - padding))
    x1 = min(width, int(bbox_px[2] + padding))
    y1 = min(height, int(bbox_px[3] + padding))
    if x1 <= x0 or y1 <= y0:
        return False
    region = gray_image[y0:y1, x0:x1]
    if region.size == 0:
        return False
    dark_pixels = int(np.count_nonzero(region < 200))
    return (dark_pixels / region.size) >= min_dark_fraction


def _placement_point(
    bbox: tuple[float, float, float, float],
    occupied: list[tuple[float, float]],
    radius: float = BALLOON_RADIUS_PDF_POINTS,
    offset: float = 14.0,
) -> tuple[float, float]:
    """Pick a balloon center near ``bbox`` that does not overlap existing balloons."""
    x0, y0, x1, y1 = bbox
    cx = x1 + offset
    cy = (y0 + y1) / 2.0
    min_dist = radius * 2.2
    attempts = 0
    while attempts < 40 and any(math.hypot(cx - ox, cy - oy) < min_dist for ox, oy in occupied):
        cy += min_dist * 0.9
        attempts += 1
    return cx, cy


@dataclass
class AutoBalloonResult:
    """Outcome of an auto-ballooning pass over one page."""

    balloons: list[Balloon]
    used_ocr: bool = False
    ocr_available: bool = True
    message: str = ""


def auto_balloon_page(
    pdf_doc: PdfDocument,
    drawing_id: str,
    page_number: int,
    existing_balloons: list[Balloon],
    start_number: int,
    dpi: float = AUTO_BALLOON_DPI,
    tesseract_path: Optional[str] = None,
    detector: Optional[BaseDetector] = None,
    default_tolerances: Optional[DefaultTolerances] = None,
) -> AutoBalloonResult:
    """Run the full auto-balloon pipeline for a single page.

    ``default_tolerances``, when given, backfills tol_plus/tol_minus (and
    the resulting limits) on any detected dimension that has a nominal but
    no explicit tolerance of its own -- e.g. "9X Ø0.250 THRU" relying on the
    drawing's general "X.XXX: ±0.005" title-block note.

    Returns proposed balloons (status=pending, source=auto) plus a status
    message suitable for display in the UI status bar. Never raises for
    missing OCR/ML dependencies -- it degrades to "no proposals" instead.
    """
    candidates: list[Detection] = []

    # Render once up front: used both as the OCR fallback's input image and
    # to sanity-check (via OpenCV) that every candidate -- native-text or
    # OCR -- actually sits over visible ink, not a blank part of the page.
    gray_image: Optional[np.ndarray] = None
    rendered_image: Optional[np.ndarray] = None
    try:
        rgb_bytes, width, height = pdf_doc.render_page_rgb(page_number, dpi)
        rendered_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
        gray_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2GRAY)
    except Exception:
        logger.exception("Failed to render page %d for auto-balloon", page_number)

    try:
        native_blocks = pdf_doc.extract_text_blocks(page_number)
    except Exception:
        logger.exception("Failed to extract native text on page %d", page_number)
        native_blocks = []

    try:
        page_width, page_height = pdf_doc.page_size_pdf(page_number)
    except Exception:
        page_width = page_height = None
    page_size = (page_width, page_height) if page_width and page_height else None
    title_block_cutoff_y = (
        _title_block_cutoff_y(native_blocks, page_height) if page_height else None
    )

    merged_native_blocks = _merge_stacked_tolerance_fragments(_merge_nearby_text_blocks(native_blocks))

    for block in merged_native_blocks:
        if title_block_cutoff_y is not None and block.bbox[1] >= title_block_cutoff_y:
            continue  # inside the title block -- never a real characteristic
        if not _looks_like_characteristic(block.text, bbox=block.bbox, page_size=page_size):
            continue
        if gray_image is not None:
            bbox_px = rect_pdf_to_pixel(block.bbox, dpi)
            if not _bbox_has_ink(gray_image, bbox_px):
                logger.debug("Discarding candidate with no ink under its bbox: %r", block.text)
                continue
        candidates.append(Detection(bbox=block.bbox, label="", confidence=0.0, raw_text=block.text))

    used_ocr = False
    ocr_available = True
    message = ""

    if not candidates:
        active_detector = detector or RulesOcrDetector(tesseract_path)
        if not active_detector.available:
            ocr_available = False
            message = active_detector.error or (
                "This page has no selectable text and OCR is unavailable. "
                "Add balloons manually for this page."
            )
        elif rendered_image is None:
            message = "Failed to render this page for OCR."
        else:
            used_ocr = True
            raw_detections = active_detector.detect(rendered_image)
            for det in raw_detections:
                # OCR boxes are inherently ink-backed (Tesseract only reports
                # boxes where it found glyphs), so no extra ink check needed.
                bbox_pdf = rect_pixel_to_pdf(det.bbox, dpi)
                if title_block_cutoff_y is not None and bbox_pdf[1] >= title_block_cutoff_y:
                    continue  # inside the title block -- never a real characteristic
                candidates.append(
                    Detection(bbox=bbox_pdf, label=det.label, confidence=det.confidence, raw_text=det.raw_text)
                )
            if not raw_detections:
                message = "OCR ran but found no recognizable dimensions/tolerances on this page."

    candidates = candidates[:300]  # sanity cap against pathological pages
    candidates.sort(key=lambda d: (round(d.bbox[1] / 10.0), d.bbox[0]))

    occupied = [(b.x, b.y) for b in existing_balloons if b.page_number == page_number]
    balloons: list[Balloon] = []
    number = start_number

    for det in candidates:
        raw_text = det.raw_text or ""
        # A single detected text chunk can pack more than one independently
        # inspected requirement (e.g. a tapped hole's thread class *and* its
        # depth) -- each becomes its own balloon, placed near the same text.
        parsed_list = parse_characteristics(raw_text) if raw_text else [None]
        if default_tolerances is not None:
            parsed_list = [
                apply_default_tolerance(p, default_tolerances) if p is not None else None
                for p in parsed_list
            ]

        for parsed in parsed_list:
            if parsed is not None:
                char_type = parsed.char_type
                confidence = det.confidence if det.confidence > 0 else parsed.confidence
            else:
                char_type = det.label or CharacteristicType.OTHER.value
                confidence = det.confidence

            cx, cy = _placement_point(det.bbox, occupied)
            occupied.append((cx, cy))

            balloon = Balloon(
                number=number,
                drawing_id=drawing_id,
                page_number=page_number,
                x=cx,
                y=cy,
                bbox_x0=det.bbox[0],
                bbox_y0=det.bbox[1],
                bbox_x1=det.bbox[2],
                bbox_y1=det.bbox[3],
                char_type=char_type,
                raw_text=parsed.raw_text if parsed is not None else raw_text,
                nominal=parsed.nominal if parsed else None,
                tol_plus=parsed.tol_plus if parsed else None,
                tol_minus=parsed.tol_minus if parsed else None,
                lower_limit=parsed.lower_limit if parsed else None,
                upper_limit=parsed.upper_limit if parsed else None,
                gdt_symbol=parsed.gdt_symbol if parsed else None,
                gdt_tolerance=parsed.gdt_tolerance if parsed else None,
                material_condition=parsed.material_condition if parsed else None,
                datums=parsed.datums if parsed else None,
                surface_finish=parsed.surface_finish if parsed else None,
                thread_callout=parsed.thread_callout if parsed else None,
                note=(parsed.note if parsed else "") or ("Detected by ML model; please fill in details." if not raw_text else ""),
                source=BalloonSource.AUTO.value,
                model_version=RULES_OCR_MODEL_VERSION,
                confidence=round(max(0.0, min(1.0, confidence)), 3),
                status=ReviewStatus.PENDING.value,
            )
            balloon.snapshot_prediction()
            balloons.append(balloon)
            number += 1

    if not balloons and not message:
        message = "No inspection characteristics were automatically detected on this page. Add balloons manually if needed."

    return AutoBalloonResult(balloons=balloons, used_ocr=used_ocr, ocr_available=ocr_available, message=message)
