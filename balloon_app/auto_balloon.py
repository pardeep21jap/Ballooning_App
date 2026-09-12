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
from balloon_app.gdt_vision import SYMBOLS, find_gdt_frames
from balloon_app.ocr_parser import (
    NUM,
    THREAD_DASH_CHARS,
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
    r"[▼▽↓⌴⌵□]",
    re.IGNORECASE,
)
_RADIUS_HINT_RE = re.compile(r"(?<![A-Za-z])R(?![a-zA-Z])\s*\d")
_METRIC_THREAD_HINT_RE = re.compile(r"\bM\d+\.?\d*\s*[xX×]\s*\d")
# A metric thread's tolerance-class form with no pitch given ("M4-6H") has
# no "x"/decimal for _METRIC_THREAD_HINT_RE to key off, and no UNC/UNF-style
# keyword either -- without this, such a callout has nothing recognizable
# to a pre-filter that only knows to look for a decimal number or a known
# symbol, so it's discarded here before ever reaching the actual parser
# (see ocr_parser._THREAD_METRIC_CLASS_RE, which this mirrors).
_METRIC_THREAD_CLASS_HINT_RE = re.compile(
    rf"\bM\s?\d+\.?\d*\s*(?:[xX×]\s*{NUM}\s*)?[{THREAD_DASH_CHARS}]\s*\d{{1,2}}[A-Za-z]\b"
)
_UNIFIED_THREAD_HINT_RE = re.compile(
    rf"\b\d+(?:/\d+)?\s*[{THREAD_DASH_CHARS}]\s*\d+\s*(UNC|UNF|UNEF|UN|NPT|NPTF)\b", re.IGNORECASE
)
# An ISO 228 parallel pipe thread ("G1/2\" - 6H") has the same problem as the
# metric class-only form above -- no decimal, no UNC/UNF keyword -- plus its
# own prefix ("G") isn't covered by any other hint. Restricted to a
# fractional size for the same reason ocr_parser._THREAD_PIPE_RE is: a bare
# "G" + whole number ("G10", "G11") is a common fiberglass-laminate material
# code, not a thread.
_PIPE_THREAD_HINT_RE = re.compile(rf"\bG\d+/\d+\b")
_PIPE_DATUM_HINT_RE = re.compile(r"\d\s*\|\s*[A-Z]")
# A short, bare integer with nothing else attached: "1", "2", "23", "(4)".
# These are almost always zone/sheet/revision markers, not dimensions.
_BARE_SHORT_INTEGER_RE = re.compile(r"^\(?\d{1,2}\)?$")
# Whole-number dimensions in metric drawings are commonly three digits
# (124, 135, 140, ...). Keep the narrower expression above for detecting
# circled BOM item numbers, but allow these as dimension candidates when
# page geometry proves they are not in the zone margin/title block.
_BARE_DIMENSION_INTEGER_RE = re.compile(r"^\(?\d{1,3}\)?$")

# Boilerplate phrases that appear only inside a drawing's title block, never
# as an inspection characteristic in their own right. The title-block note
# often contains real decimal numbers and symbols (a "TOLERANCES UNLESS
# OTHERWISE NOTED" table, "±0.5°", fractional tolerances) that would
# otherwise pass _looks_like_characteristic and get ballooned individually
# -- see _title_block_cutoff_y below, which uses these to exclude the whole
# title block region rather than trying to keyword-match every field in it.
#
# A Parts List / Bill of Materials table is drawn the same way -- its ITEM,
# QTY, and PART NUMBER columns are all bare numbers/codes that otherwise
# pass _looks_like_characteristic just as easily as a real dimension would
# -- and, being assembly-level bookkeeping rather than a feature of the
# part itself, belongs excluded for the same reason. It's usually a
# separate table stacked directly above the title block rather than part
# of it, so its own heading is included here too: the topmost match of
# either sets the cutoff high enough to cover both tables in one strip.
_TITLE_BLOCK_KEYWORDS_RE = re.compile(
    r"TOLERANCES UNLESS OTHERWISE (NOTED|SPECIFIED)|UNLESS OTHERWISE SPECIFIED|"
    r"\bDRAWN\b|\bCHECKED\b|\bDESIGNED\b|\bENGINEER(ED)?\b|\bAPPROVED\b|"
    r"\bTITLE\b|\bSCALE\b|\bSHEET\b|\bQTY\.?\s*:|\bMATERIAL\b|\bFINISH\b|"
    r"\bPROJECT CODE\b|\bDWG\.?\s*NO\.?\b|\bNEXT ASSY\b|\bDO NOT SCALE\b|"
    r"\bPROPRIETARY\b|\bCONFIDENTIAL\b|\bINTERPRET (DRAWING|PER)\b|"
    r"\bBREAK ALL SHARP EDGES\b|\bFRACTIONAL\b|"
    r"\bPARTS? LIST\b|\bBILL OF MATERIALS?\b|\bBOM\b",
    re.IGNORECASE,
)


def _in_zone_margin(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    margin_fraction: float = 0.04,
) -> bool:
    """True if ``bbox`` sits within the thin zone/grid-reference margin
    strip just inside any sheet edge, where ANSI/ISO Y14.1
    zone letters/numbers ("1 2 3 4", "A B C D") are printed. A real
    dimension is never drawn in that margin.
    """
    x0, y0, x1, y1 = bbox
    margin_x = page_width * margin_fraction
    margin_y = page_height * margin_fraction
    return (
        y0 <= margin_y
        or y1 >= page_height - margin_y
        or x0 <= margin_x
        or x1 >= page_width - margin_x
    )


_TITLE_BLOCK_ANCHOR_RE = re.compile(
    r"TOLERANCES UNLESS OTHERWISE (?:NOTED|SPECIFIED)|UNLESS OTHERWISE SPECIFIED|"
    r"\bDRAWN(?: BY)?\b|\bCHECKED(?: BY)?\b|\bDESIGNED(?: BY)?\b|"
    r"\bAPPROVED(?: BY)?\b|\bMODEL FILE NAME\b|\bPART NUMBER\b|"
    r"\bPARTS? LIST\b|\bBILL OF MATERIALS?\b|\bBOM\b|^\s*TITLE\s*$",
    re.IGNORECASE,
)


def _title_block_regions(
    blocks: list[TextBlock],
    page_width: float,
    page_height: float,
    pdf_doc: Optional[PdfDocument] = None,
    page_number: int = 0,
) -> list[tuple[float, float, float, float]]:
    """Infer spatial title-block/BOM regions without discarding a full band.

    Many large CAD sheets place the title block in the lower-right while
    legitimate drawing views continue at the same y coordinates on the
    left.  The former single horizontal cutoff erased those dimensions.
    An anchor in the left quarter still denotes a conventional full-width
    bottom strip; a right-side anchor excludes only the area to its right.
    """
    anchors = [
        block for block in blocks
        if block.bbox[1] > page_height * 0.5 and _TITLE_BLOCK_ANCHOR_RE.search(block.text)
    ]
    regions: list[tuple[float, float, float, float]] = []
    left_anchors = [block for block in anchors if block.bbox[0] <= page_width * 0.25]
    if left_anchors:
        # Conventional full-width title strip.
        regions.append((0.0, min(block.bbox[1] for block in left_anchors) - 4.0, page_width, page_height))

    right_anchors = [block for block in anchors if block.bbox[0] > page_width * 0.25]
    if right_anchors:
        # The anchor text generally sits inside the grid, not at its top-left
        # corner. Expand up and left to cover the full bordered title block.
        # A fixed page-width fraction here badly undershoots on smaller
        # sheets: an A4 title block commonly starts around 35-40% of the
        # page width even though its "UNLESS OTHERWISE SPECIFIED"-style
        # anchor text sits further right, inside the box, around 55-60%.
        # Scaling the left edge off the anchor's own x-position (rather
        # than a page-wide constant) tracks that relationship regardless
        # of sheet size.
        anchor_left = min(block.bbox[0] for block in right_anchors)
        region_left = max(page_width * 0.25, anchor_left * 0.65)
        anchor_top = min(block.bbox[1] for block in right_anchors)
        region_top = max(page_height * 0.5, anchor_top - page_height * 0.10)
        # Title blocks vary too much in height for that fixed 10%-of-page
        # margin to fit them all -- on a dense sheet, a legitimate
        # dimension can sit just above the box's real top border but still
        # within that margin's reach. When the border is actually drawn as
        # a vector line, trust its real position over the generic guess
        # (this can only shrink the excluded band, never grow it, since the
        # search is bounded by the margin's own top).
        if pdf_doc is not None:
            border_y = pdf_doc.find_horizontal_rule(
                page_number, x_range=(region_left, page_width), y_range=(region_top, anchor_top)
            )
            if border_y is not None:
                region_top = border_y
        regions.append((region_left, region_top, page_width, page_height))
    return regions


_REVISION_TABLE_ANCHOR_RE = re.compile(r"\bREVISION\s+HISTORY\b", re.IGNORECASE)
_REVISION_TABLE_HEADER_RE = re.compile(
    r"^(ZONE|REV|VER|VERSION|DESCRIPTION|DATE|APPROVED(?:\s+BY)?|BY)$", re.IGNORECASE
)
_REVISION_TABLE_ROW_MARKER_RE = re.compile(r"^(REV|ZONE|VER|VERSION)$", re.IGNORECASE)
# How far below the "REVISION HISTORY" heading its own column-header row
# (ZONE/REV/VER/...) can plausibly start.
_REVISION_TABLE_HEADER_SEARCH_HEIGHT_PDF_POINTS = 40.0
# A gap this large between successive rows in the narrow marker column (see
# below) marks the end of the table's populated rows.
_REVISION_TABLE_ROW_GAP_PDF_POINTS = 20.0
_REVISION_TABLE_COLUMN_MARGIN_PDF_POINTS = 6.0


def _revision_table_regions(
    blocks: list[TextBlock],
) -> list[tuple[float, float, float, float]]:
    """Infer a revision-history table's bounding region from its own
    "REVISION HISTORY" heading, wherever it sits on the page.

    Unlike the title block (which conventionally runs to a sheet edge, so
    "everything from this anchor to the page edge" is a safe cutoff), a
    revision table is commonly drawn at the *top* of the sheet with real
    drawing content close below and around it, so its extent can't be
    assumed to reach any edge, and a generously wide "grow until a gap"
    search across the table's full (wide) column span is a real hazard: a
    dimension callout sitting just a bit closer below the table than the
    table's own row spacing can get silently folded in, and everything
    below that dimension along with it, snowballing the excluded region
    far past the actual table.

    Instead the table's width comes from its own column-header cells
    (ZONE/REV/VER/DESCRIPTION/DATE/APPROVED[ BY]), found within a small
    window below the heading, and its height is measured only through the
    REV or ZONE column specifically -- a narrow strip holding nothing but
    a short per-row code, so unrelated drawing content essentially never
    coincidentally lands in that exact x-slice the way it can across the
    table's much wider DESCRIPTION column. Without this, every row's
    REV/VER code is a bare 1-3 character value textually indistinguishable
    from a real whole-number dimension.
    """
    anchors = [b for b in blocks if _REVISION_TABLE_ANCHOR_RE.search(b.text)]
    regions: list[tuple[float, float, float, float]] = []
    for anchor in anchors:
        ax0, ay0, ax1, ay1 = anchor.bbox
        headers = [
            b for b in blocks
            if b is not anchor and _REVISION_TABLE_HEADER_RE.match(b.text.strip())
            and ay1 <= b.bbox[1] <= ay1 + _REVISION_TABLE_HEADER_SEARCH_HEIGHT_PDF_POINTS
        ]
        if not headers:
            continue
        table_left = min(min(b.bbox[0] for b in headers), ax0)
        table_right = max(max(b.bbox[2] for b in headers), ax1)
        header_bottom = max(b.bbox[3] for b in headers)

        row_markers = [b for b in headers if _REVISION_TABLE_ROW_MARKER_RE.match(b.text.strip())]
        bottom = header_bottom
        for row_marker in row_markers:
            # A table's ZONE column, in particular, is commonly left blank
            # on every row (no zone applies) -- try each candidate narrow
            # column in turn and keep whichever actually has row data
            # below it, rather than committing to the first one found.
            col_left = row_marker.bbox[0] - _REVISION_TABLE_COLUMN_MARGIN_PDF_POINTS
            col_right = row_marker.bbox[2] + _REVISION_TABLE_COLUMN_MARGIN_PDF_POINTS
            column_cells = sorted(
                (b for b in blocks if b is not anchor and b not in headers
                 and b.bbox[1] >= header_bottom and col_left <= b.bbox[0] and b.bbox[2] <= col_right),
                key=lambda b: b.bbox[1],
            )
            if not column_cells or column_cells[0].bbox[1] - header_bottom > _REVISION_TABLE_ROW_GAP_PDF_POINTS:
                continue
            candidate_bottom, last_bottom = header_bottom, header_bottom
            # The gap tolerance tightens as real rows are found: the very
            # first gap (header row to the first data row) can legitimately
            # be as large as the fixed ceiling, but a mangled dimensioning-
            # symbol character (see ocr_parser's mangled-symbol fallbacks)
            # is just as narrow as a REV/VER code and can coincidentally
            # land in this same column further down the page -- once the
            # table's own row-to-row rhythm is established (consistently
            # much smaller), a gap blowing well past that rhythm is no
            # longer read as "just a slightly tall row".
            max_row_gap = None
            for block in column_cells:
                gap = block.bbox[1] - last_bottom
                limit = _REVISION_TABLE_ROW_GAP_PDF_POINTS if max_row_gap is None else max(max_row_gap * 1.8, 6.0)
                if gap > limit:
                    break
                max_row_gap = gap if max_row_gap is None else max(max_row_gap, gap)
                candidate_bottom = max(candidate_bottom, block.bbox[3])
                last_bottom = block.bbox[3]
            bottom = max(bottom, candidate_bottom)
        # No narrow, reliable column had any data below it -- fall back to
        # just the header row's own extent rather than risk an open-ended
        # guess (bottom already defaults to header_bottom in that case).
        regions.append((table_left - 4.0, ay0 - 4.0, table_right + 4.0, bottom + 4.0))
    return regions


def _bbox_in_regions(
    bbox: tuple[float, float, float, float],
    regions: list[tuple[float, float, float, float]],
) -> bool:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions)


def _title_block_cutoff_y(blocks: list[TextBlock], page_height: float) -> Optional[float]:
    """Return a page-y cutoff below which everything is treated as inside
    the title block (or an adjoining Parts List / Bill of Materials table),
    or ``None`` if no boilerplate was found.

    Rather than keyword-matching every individual title-block field (drawing
    number, revision, company name/logo, dates -- an open-ended list that
    varies per template), this finds the topmost boilerplate phrase in the
    bottom half of the page and excludes that whole horizontal strip down
    to the bottom edge. Matches ANSI/ISO title blocks, which run the full
    sheet width along the bottom; a title block running the full height
    along one side instead would need a different heuristic. A Parts List
    table stacked above the title block (see _TITLE_BLOCK_KEYWORDS_RE) is
    covered the same way, as long as its heading sits no lower than the
    title block's own topmost boilerplate line.
    """
    matches = [b for b in blocks if b.bbox[1] > page_height * 0.5 and _TITLE_BLOCK_ANCHOR_RE.search(b.text)]
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
    if _BARE_DIMENSION_INTEGER_RE.match(text):
        if bbox is not None and page_size is not None and not _in_zone_margin(bbox, *page_size):
            return True
        return False
    return bool(
        _DECIMAL_NUMBER_RE.search(text)
        or _SYMBOL_HINT_RE.search(text)
        or _RADIUS_HINT_RE.search(text)
        or _METRIC_THREAD_HINT_RE.search(text)
        or _METRIC_THREAD_CLASS_HINT_RE.search(text)
        or _UNIFIED_THREAD_HINT_RE.search(text)
        or _PIPE_THREAD_HINT_RE.search(text)
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
    diameter_hint: bool = False
    gdt_frame_hint: bool = False
    flatness_hint: bool = False
    depth_hint: bool = False
    gdt_symbol_hint: Optional[str] = None
    model_version: str = RULES_OCR_MODEL_VERSION


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

    def read_cell(self, image: np.ndarray) -> str:
        """Read one frame compartment after the caller removes its borders."""
        if not self.available or self._pytesseract is None:
            return ""
        padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        try:
            return self._pytesseract.image_to_string(padded, config="--psm 7").strip()
        except Exception:
            logger.exception("OCR of GD&T cell failed")
            return ""

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
        if not model_path.is_file():
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


_BARE_SHORT_ZERO_RE = re.compile(r"^0(\.0{1,2})?$")
_BARE_NUMBER_ONLY_RE = re.compile(rf"^{NUM}$")
# A leading quantity-count prefix ("9X") and/or a short mangled-symbol
# letter (a Ø/⌵/etc. glyph substituted by a CAD font with no ToUnicode
# mapping for it, e.g. the "n" in "9X n0.250") -- neither is a dimension
# value in its own right, so _established_value_count strips at most one
# of each off the front before counting what's left.
_LEADING_MARKER_RE = re.compile(r"^\s*(?:\d+\s*[Xx]\s*)?[A-Za-z]{0,3}\s*")
# A lone, standalone letter (not "R", which is never mangled -- see
# _BARE_MANGLED_DIAMETER_RE) at the very end of what's merged so far, e.g.
# the "x" (mangled depth glyph) in "2X n 0.089 x". It hasn't been given its
# own value yet, even though an earlier value (the diameter) already
# exists further back in the text -- see _merge_gap_is_safe below.
_TRAILING_UNRESOLVED_MARKER_RE = re.compile(r"(?:^|\s)(?![Rr]$)[A-Za-z]$")


def _established_value_count(text: str) -> int:
    """How many real dimension-value numbers ``text`` already has, ignoring
    a leading quantity-count prefix and/or mangled-symbol letter (see
    ``_LEADING_MARKER_RE``)."""
    return len(re.findall(NUM, _LEADING_MARKER_RE.sub("", text, count=1)))


def _merge_gap_is_safe(current_text: str, next_text: str) -> bool:
    """Decide whether ``next_text`` may join ``current_text`` across a
    same-line gap, rather than staying its own independent value.

    Always safe: a tolerance continuation (starts with a sign, or is the
    short bare "0"/"0.0" unilateral-tolerance convention -- see
    ``_TRAILING_BARE_ZERO_RE`` in ocr_parser.py), trailing non-numeric text
    (a note like "THRU"/"TYP" can never be mistaken for its own
    ordinate/chain dimension, since it has no digits at all), or a fresh
    mangled/shape marker with no value of its own yet (see
    ``_TRAILING_UNRESOLVED_MARKER_RE`` -- e.g. a hole's depth glyph
    trailing its own diameter value, as in "2X n 0.089 x 0.500": the
    diameter's value is already established, but the "x" right before
    ``next_text`` is still waiting on its own).

    Otherwise ``next_text`` is a complete standalone number -- exactly the
    shape both a legitimate second value (the actual value following a
    quantity prefix and/or mangled symbol, e.g. "9X" + "0.250") and an
    independent ordinate/chain dimension ("0.550" next to "0.250", each
    with its own extension line) take. The two are told apart by what
    ``current_text`` has accumulated so far: if it doesn't have an
    established value of its own yet (still just a quantity-count and/or a
    mangled-symbol letter), the next number is the value it's missing, not
    a competing dimension -- but once it already has one, a further plain
    number is always read as independent.
    """
    stripped_next = next_text.strip()
    if not stripped_next:
        return False
    if stripped_next[0] in "±+-":
        return True
    if _BARE_SHORT_ZERO_RE.match(stripped_next):
        return True
    if not _BARE_NUMBER_ONLY_RE.match(stripped_next):
        return True  # non-numeric trailing note ("THRU", "TYP", ...)
    if _TRAILING_UNRESOLVED_MARKER_RE.search(current_text):
        return True  # a fresh marker just joined with no value of its own yet
    return _established_value_count(current_text) == 0


def _merge_nearby_text_blocks(blocks: list[TextBlock], y_tol: float = 3.0, x_gap: float = 18.0) -> list[TextBlock]:
    """Merge text spans that sit on the same line and are close together.

    PDF text is often split into several spans (e.g. ``"50.00"`` and
    ``"±0.05"`` as separate runs with different fonts, or "9X" and a
    mangled-symbol value in a different sub-style). Merging them lets the
    parser see the full callout.

    Proximity alone isn't enough of a signal, though: a row of ordinate/
    chain dimensions (several independent whole values -- "0.550 0.250
    0.000 0.283 ...", each with its own extension line -- packed tightly
    along one baseline) looks geometrically identical to a nominal split
    from its tolerance or quantity prefix across two font runs. Merging
    every one of them into a single block would silently keep only the
    first value and drop the rest as one balloon instead of several -- see
    _merge_gap_is_safe for how the two are told apart.

    Rows are found by chaining blocks whose vertical centers lie within
    ``y_tol`` of a neighbor (transitively, like _cluster_touching_rects in
    pdf_engine.py), not by independently rounding each block's position
    into a bucket. A mangled dimensioning-symbol glyph (see ocr_parser's
    mangled-symbol fallbacks) is often drawn from a different, taller
    custom symbol font than the digits beside it, so its box can start a
    couple of points higher even though it's meant to sit on the very same
    visual line -- rounding each block's position independently can then
    put two blocks whose raw centers differ by less than y_tol on opposite
    sides of a rounding boundary, splitting them into different buckets by
    pure chance. Chaining sidesteps that: a diameter's "n" and its own
    value end up in the same row regardless of exactly where either one's
    center rounds to.
    """
    if not blocks:
        return []
    centers = [(b.bbox[1] + b.bbox[3]) / 2.0 for b in blocks]
    by_center = sorted(range(len(blocks)), key=lambda i: centers[i])
    parent = list(range(len(blocks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(by_center, by_center[1:]):
        if centers[b] - centers[a] <= y_tol:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    rows: dict[int, list[int]] = {}
    for i in by_center:
        rows.setdefault(find(i), []).append(i)
    row_order = sorted(rows.values(), key=lambda idxs: sum(centers[i] for i in idxs) / len(idxs))
    ordered = [blocks[i] for idxs in row_order for i in sorted(idxs, key=lambda i: blocks[i].bbox[0])]

    merged: list[TextBlock] = [ordered[0]]
    for nxt in ordered[1:]:
        current = merged[-1]
        same_line = abs(nxt.bbox[1] - current.bbox[1]) <= y_tol and abs(nxt.bbox[3] - current.bbox[3]) <= y_tol * 3
        gap = nxt.bbox[0] - current.bbox[2]
        if same_line and -2.0 <= gap <= x_gap and _merge_gap_is_safe(current.text, nxt.text):
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
_BARE_UNSIGNED_ZERO_RE = re.compile(r"^\s*((?:0?\.0+)|0)\s*$")
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
# The anchor's own value within (possibly already compound) base_text --
# used to insert a stacked tolerance right after it, see the comment where
# this is used in _merge_stacked_tolerance_fragments below.
_FIRST_NUMBER_RE = re.compile(NUM)


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


def _bbox_gap_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Distance between two bboxes' nearest edges (0 if they overlap on an
    axis), not their centers. A wide anchor block's center can sit far from
    a fragment even when the anchor's own edge is right next to it --
    e.g. a long merged thread+depth line whose center happens to fall
    closer to an unrelated tolerance fragment stacked on the line below
    than the fragment's own true anchor (a short value immediately beside
    it) does. Comparing edges instead of centers isn't fooled by an
    anchor's width/height this way.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(ax0 - bx1, bx0 - ax1, 0.0)
    dy = max(ay0 - by1, by0 - ay1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _bbox_center_within(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float], pad: float = 1.0
) -> bool:
    """True if ``inner``'s center falls inside ``outer`` (expanded by
    ``pad``). Used to tell whether a raw (pre-merge) text block -- and
    anything found near it, like a vector-drawn diameter symbol -- ended up
    part of a given merged candidate block."""
    cx, cy = (inner[0] + inner[2]) / 2.0, (inner[1] + inner[3]) / 2.0
    return (outer[0] - pad) <= cx <= (outer[2] + pad) and (outer[1] - pad) <= cy <= (outer[3] + pad)


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
    is_signed_fragment = [bool(_BARE_SIGNED_NUM_RE.match(b.text)) for b in blocks]
    # A zero upper deviation is sometimes written without an explicit plus
    # sign. It is tolerance only when a signed sibling is nearby, so an
    # independent 0.000 ordinate dimension remains independent.
    is_unsigned_zero = [bool(_BARE_UNSIGNED_ZERO_RE.match(b.text)) for b in blocks]
    is_sign_fragment = [
        signed or (is_unsigned_zero[i] and any(
            is_signed_fragment[j] and _bboxes_are_near(b.bbox, blocks[j].bbox, y_gap * 2, x_slack)
            for j in range(len(blocks)) if j != i
        ))
        for i, (b, signed) in enumerate(zip(blocks, is_signed_fragment))
    ]
    has_digit = [bool(re.search(r"\d", b.text)) for b in blocks]

    # Two sign fragments that are each other's mutually-nearest sign
    # fragment are read as one stacked +/- pair (e.g. "+.002" over
    # "-.000") and must resolve to the *same* anchor -- otherwise, decided
    # purely individually, one can end up a hair's-width closer to some
    # other nearby value than to the value its sibling already resolved
    # to, splitting a single tolerance across two different balloons.
    def _nearest_sign_fragment(idx: int) -> Optional[int]:
        best_j, best_d = None, None
        for j2 in range(len(blocks)):
            if j2 == idx or not is_sign_fragment[j2]:
                continue
            if not _bboxes_are_near(blocks[idx].bbox, blocks[j2].bbox, y_gap, x_slack):
                continue
            d = _bbox_gap_distance(blocks[idx].bbox, blocks[j2].bbox)
            if best_d is None or d < best_d:
                best_j, best_d = j2, d
        return best_j

    # Each fragment (or fragment pair) picks its single closest eligible
    # anchor, rather than anchors greedily claiming whatever fragment they
    # encounter first.
    fragment_anchor: dict[int, int] = {}
    for j, frag in enumerate(blocks):
        if not is_sign_fragment[j]:
            continue
        sibling = _nearest_sign_fragment(j)
        query_bbox = frag.bbox
        if sibling is not None and _nearest_sign_fragment(sibling) == j:
            query_bbox = _union_bbox(frag.bbox, blocks[sibling].bbox)
        best_i, best_key = None, None
        for i, cand in enumerate(blocks):
            if i == j or is_sign_fragment[i] or not has_digit[i]:
                continue
            if not _bboxes_are_near(cand.bbox, query_bbox, y_gap, x_slack):
                continue
            dist = _bbox_gap_distance(cand.bbox, query_bbox)
            overlaps_row = min(cand.bbox[3], query_bbox[3]) > max(cand.bbox[1], query_bbox[1])
            # A +/- fragment (or pair) almost always shares some vertical
            # extent with its own value's line -- it's written stacked
            # immediately above/below/beside that value, not offset to a
            # whole other row. Rank an anchor whose row it actually
            # overlaps ahead of one it's merely a hair's-width closer to on
            # a pure gap-distance basis (e.g. a fragment sitting almost
            # exactly between two dimensions can end up fractionally
            # nearer the unrelated one below it than the value it actually
            # belongs to).
            key = (0 if overlaps_row else 1, dist)
            if best_key is None or key < best_key:
                best_i, best_key = i, key
        if best_i is not None:
            fragment_anchor[j] = best_i

    used: set[int] = set()
    merged: list[TextBlock] = []

    for i, nominal in enumerate(blocks):
        if is_sign_fragment[i] or i in used:
            continue

        assigned = [j for j, anchor in fragment_anchor.items() if anchor == i]
        plus_idx = minus_idx = None
        # A second fragment sharing the same sign as one already claimed
        # (e.g. a "+.3 / +.1" double-positive tolerance style, not the more
        # common "+X / -Y") -- kept and merged with its own true sign
        # rather than silently dropped, which previously left it behind as
        # its own stray candidate/balloon.
        extra_idx = extra_sign = None

        def fragment_parts(j: int) -> tuple[str, str]:
            match = _BARE_SIGNED_NUM_RE.match(blocks[j].text)
            if match:
                return match.group(1), match.group(2)
            frag_y = (blocks[j].bbox[1] + blocks[j].bbox[3]) / 2.0
            nominal_y = (nominal.bbox[1] + nominal.bbox[3]) / 2.0
            return ("+" if frag_y < nominal_y else "-"), blocks[j].text.strip()

        for j in assigned:
            sign, _value = fragment_parts(j)
            if sign == "+" and plus_idx is None:
                plus_idx = j
            elif sign == "-" and minus_idx is None:
                minus_idx = j
            elif extra_idx is None:
                extra_idx, extra_sign = j, sign

        if plus_idx is None and minus_idx is None and extra_idx is None:
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
            _sign, plus_val = fragment_parts(plus_idx)
            combined_bbox = _union_bbox(combined_bbox, blocks[plus_idx].bbox)
            used.add(plus_idx)
        if minus_idx is not None:
            _sign, minus_val = fragment_parts(minus_idx)
            combined_bbox = _union_bbox(combined_bbox, blocks[minus_idx].bbox)
            used.add(minus_idx)
        extra_val = None
        if extra_idx is not None:
            _sign, extra_val = fragment_parts(extra_idx)
            combined_bbox = _union_bbox(combined_bbox, blocks[extra_idx].bbox)
            used.add(extra_idx)

        # A unilateral tolerance ("0 / +0.05") -- only one side has an
        # explicit sign; the other, always exactly 0, is conventionally
        # written bare since +0 and -0 are equivalent. That bare "0" sits
        # right after the nominal (same-line merged already, e.g. "Ø8 0"),
        # so only one of plus_val/minus_val was found above; the trailing
        # bare number fills in the other side. Not applicable when a
        # same-sign extra fragment already supplies the second value.
        if extra_val is None and (plus_val is None) != (minus_val is None):
            trailing_zero = _TRAILING_BARE_ZERO_RE.search(base_text)
            if trailing_zero:
                base_text = base_text[: trailing_zero.start()].rstrip()
                if plus_val is None:
                    plus_val = trailing_zero.group(1)
                else:
                    minus_val = trailing_zero.group(1)

        tol_parts = []
        if plus_val is not None:
            tol_parts.append(f"+{plus_val}")
        if minus_val is not None:
            tol_parts.append(f"-{minus_val}")
        if extra_val is not None:
            tol_parts.append(f"{extra_sign}{extra_val}")

        # Insert the tolerance right after the anchor's own value rather
        # than at the tail of base_text -- base_text may already be a
        # compound "Ø12.0 ↧ 100.0" (nominal same-line-merged with a
        # trailing depth clause *before* this stacked-tolerance pass ever
        # runs), and appending there would misattach the tolerance past
        # the depth instead of to the value it actually modifies.
        value_match = _FIRST_NUMBER_RE.search(base_text)
        if value_match:
            head, tail = base_text[: value_match.end()], base_text[value_match.end() :].strip()
        else:
            head, tail = base_text, ""
        parts = [head, *tol_parts] + ([tail] if tail else [])
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
    learned_symbols: Optional[dict[str, str]] = None,
) -> AutoBalloonResult:
    """Run the full auto-balloon pipeline for a single page.

    ``existing_balloons`` does double duty: it keeps new balloon markers
    from being placed on top of ones that already exist (see ``occupied``
    below), and -- since it also carries each existing balloon's source
    bbox -- it's used to skip any newly-found candidate whose source region
    overlaps one that's already ballooned, so re-running auto-balloon on an
    already-ballooned page doesn't propose duplicates of what's there.

    ``default_tolerances``, when given, backfills tol_plus/tol_minus (and
    the resulting limits) on any detected dimension that has a nominal but
    no explicit tolerance of its own -- e.g. "9X Ø0.250 THRU" relying on the
    drawing's general "X.XXX: ±0.005" title-block note.

    ``learned_symbols``, when given, is passed straight through to
    parse_characteristics for the "font mangled a real symbol into an
    unrelated letter" fallbacks -- see AppSettings.learn_symbol.

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
    # A revision-history table is excluded the same way as the title
    # block/parts list below -- never a real characteristic -- but isn't
    # restricted to the bottom half of the page, since it's conventionally
    # drawn at the *top* of the sheet instead.
    excluded_regions = (
        (
            _title_block_regions(native_blocks, page_width, page_height, pdf_doc, page_number)
            if page_width and page_height else []
        )
        + _revision_table_regions(native_blocks)
    )

    # Some CAD PDF exports draw the diameter (Ø) glyph as vector line art
    # instead of a font character, so it never shows up in native_blocks'
    # text at all (see PdfDocument.find_vector_diameter_symbol). Check each
    # raw (pre-merge) block here, while bboxes are still tight to a single
    # span -- merging can stack a value with its +/- tolerance into a much
    # taller box that throws off the "immediately to the left" geometry check.
    diameter_hint_bboxes: list[tuple[float, float, float, float]] = []
    # Likewise, a GD&T feature control frame's symbol (flatness,
    # straightness, etc.) is almost always drawn as vector line art, so it
    # never shows up in the text at all -- not even as a mangled character
    # (see PdfDocument.find_vector_gdt_frame). Checked on the same raw,
    # pre-merge blocks and for the same reason.
    gdt_frame_hint_bboxes: list[tuple[float, float, float, float]] = []
    flatness_hint_bboxes: list[tuple[float, float, float, float]] = []
    # Likewise, the "depth" glyph is often drawn as vector line art rather
    # than a font character (see PdfDocument.find_vector_depth_symbol) --
    # without this, a hole/thread's depth value is textually indistinguishable
    # from an unrelated bare linear dimension, since nothing in the text
    # itself says "depth" at all.
    depth_hint_bboxes: list[tuple[float, float, float, float]] = []
    # A bare 1-2 digit number circled on an assembly drawing is an
    # item-reference "balloon" pointing at a Parts List row, not a
    # dimension -- textually identical to a real whole-number dimension, so
    # only the enclosing circle (again vector line art, see
    # PdfDocument.find_vector_circle_around_text) tells them apart. Only
    # checked for blocks that could plausibly be such a number, since the
    # geometry check is comparatively expensive to run on every block.
    item_balloon_hint_bboxes: list[tuple[float, float, float, float]] = []
    for block in native_blocks:
        if _bbox_in_regions(block.bbox, excluded_regions):
            continue
        if pdf_doc.find_vector_diameter_symbol(page_number, block.bbox):
            diameter_hint_bboxes.append(block.bbox)
        if pdf_doc.find_vector_gdt_frame(page_number, block.bbox):
            gdt_frame_hint_bboxes.append(block.bbox)
            if pdf_doc.find_vector_flatness_symbol(page_number, block.bbox):
                flatness_hint_bboxes.append(block.bbox)
        if pdf_doc.find_vector_depth_symbol(page_number, block.bbox):
            depth_hint_bboxes.append(block.bbox)
        if _BARE_SHORT_INTEGER_RE.match(block.text.strip()) and pdf_doc.find_vector_circle_around_text(
            page_number, block.bbox
        ):
            item_balloon_hint_bboxes.append(block.bbox)

    merged_native_blocks = _merge_stacked_tolerance_fragments(_merge_nearby_text_blocks(native_blocks))

    for block in merged_native_blocks:
        if _bbox_in_regions(block.bbox, excluded_regions):
            continue  # inside the title block -- never a real characteristic
        if any(_bbox_center_within(raw_bbox, block.bbox) for raw_bbox in item_balloon_hint_bboxes):
            continue  # circled item-reference number, not a dimension
        if not _looks_like_characteristic(block.text, bbox=block.bbox, page_size=page_size):
            continue
        if gray_image is not None:
            bbox_px = rect_pdf_to_pixel(block.bbox, dpi)
            if not _bbox_has_ink(gray_image, bbox_px):
                logger.debug("Discarding candidate with no ink under its bbox: %r", block.text)
                continue
        diameter_hint = any(_bbox_center_within(raw_bbox, block.bbox) for raw_bbox in diameter_hint_bboxes)
        gdt_frame_hint = any(_bbox_center_within(raw_bbox, block.bbox) for raw_bbox in gdt_frame_hint_bboxes)
        candidates.append(
            Detection(
                bbox=block.bbox,
                label="",
                confidence=0.0,
                raw_text=block.text,
                diameter_hint=diameter_hint,
                gdt_frame_hint=gdt_frame_hint,
                flatness_hint=any(_bbox_center_within(raw_bbox, block.bbox) for raw_bbox in flatness_hint_bboxes),
                depth_hint=any(_bbox_center_within(raw_bbox, block.bbox) for raw_bbox in depth_hint_bboxes),
            )
        )

    used_ocr = False
    ocr_available = True
    message = ""

    if not candidates:
        active_detector = detector or RulesOcrDetector(tesseract_path)
        if isinstance(active_detector, YoloDetector) and not active_detector.available:
            logger.warning("%s Falling back to rules/OCR.", active_detector.error)
            active_detector = RulesOcrDetector(tesseract_path)
        using_yolo = isinstance(active_detector, YoloDetector)
        if not active_detector.available:
            ocr_available = False
            message = active_detector.error or (
                "This page has no selectable text and OCR is unavailable. "
                "Add balloons manually for this page."
            )
        elif rendered_image is None:
            message = "Failed to render this page for YOLO." if using_yolo else "Failed to render this page for OCR."
        else:
            used_ocr = not using_yolo
            raw_detections = active_detector.detect(rendered_image)
            for det in raw_detections:
                # OCR boxes are inherently ink-backed (Tesseract only reports
                # boxes where it found glyphs), so no extra ink check needed.
                bbox_pdf = rect_pixel_to_pdf(det.bbox, dpi)
                if _bbox_in_regions(bbox_pdf, excluded_regions):
                    continue  # inside the title block -- never a real characteristic
                candidates.append(
                    Detection(
                        bbox=bbox_pdf, label=det.label, confidence=det.confidence, raw_text=det.raw_text,
                        model_version="yolo" if using_yolo else RULES_OCR_MODEL_VERSION,
                    )
                )
            if not raw_detections:
                message = (
                    "YOLO ran but found no candidate regions on this page." if using_yolo else
                    "OCR ran but found no recognizable dimensions/tolerances on this page."
                )

    # Read symbols from the rendered ink as well as the text layer. This
    # covers CAD vector paths, unmapped fonts, and scanned feature frames.
    # OCR each enclosed value separately when selectable text is absent;
    # full-page OCR often discards short text enclosed by table borders.
    frame_ocr = None
    if gray_image is not None:
        for frame in find_gdt_frames(gray_image):
            frame_used_ocr = False
            cell_texts = []
            cell_boxes = [rect_pixel_to_pdf(cell, dpi) for cell in frame.cells]
            if _bbox_in_regions(cell_boxes[0], excluded_regions):
                continue
            for cell, box in zip(frame.cells, cell_boxes):
                spans = [b for b in native_blocks if _bbox_center_within(b.bbox, box)]
                spans.sort(key=lambda b: (round(b.bbox[1] / 5), b.bbox[0]))
                cell_text = " ".join(b.text for b in spans)
                if not cell_text:
                    if frame_ocr is None:
                        frame_ocr = RulesOcrDetector(tesseract_path)
                    if frame_ocr.available:
                        x0, y0, x1, y1 = cell
                        pad = max(2, round((y1-y0) * 0.06))
                        crop = gray_image[y0+pad:y1-pad, x0+pad:x1-pad]
                        cell_text = frame_ocr.read_cell(crop)
                        used_ocr = frame_used_ocr = True
                    else:
                        ocr_available = False
                        message = frame_ocr.error or "GD&T frame values need OCR. Configure Tesseract in Settings."
                cell_texts.append(cell_text.strip())
            # Never invent a tolerance when neither PDF text nor OCR read it.
            if not cell_texts or not re.search(NUM, cell_texts[0]):
                continue
            raw_text = " | ".join(cell_texts)
            candidates = [d for d in candidates
                          if not any(_bbox_center_within(d.bbox, box) for box in cell_boxes)]
            candidates.append(Detection(
                bbox=(cell_boxes[0][0], cell_boxes[0][1], cell_boxes[-1][2], cell_boxes[-1][3]),
                label=CharacteristicType.GDT_FRAME.value,
                confidence=min(frame.confidence, 0.55 if any(not t for t in cell_texts)
                               else 0.75 if frame_used_ocr else 0.9),
                raw_text=raw_text, gdt_frame_hint=True, gdt_symbol_hint=frame.symbol,
            ))
            if message == "OCR ran but found no recognizable dimensions/tolerances on this page.":
                message = ""

    # Re-running auto-balloon on a page that already has balloons (manual
    # re-run, or "Auto-Balloon Entire Drawing" after touching up one page)
    # must not re-propose a characteristic that's already ballooned there --
    # drop any candidate whose source region is essentially the same region
    # an existing balloon on this page was already detected from.
    existing_bboxes = [
        b.bbox() for b in existing_balloons if b.page_number == page_number and b.has_bbox()
    ]
    if existing_bboxes:
        candidates = [
            det for det in candidates
            if not any(
                _bbox_center_within(det.bbox, eb) or _bbox_center_within(eb, det.bbox)
                for eb in existing_bboxes
            )
        ]

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
        parsed_list = (
            parse_characteristics((SYMBOLS[det.gdt_symbol_hint] + " " if det.gdt_symbol_hint
                                   else "⏥ " if det.flatness_hint else "") + raw_text,
                                  diameter_hint=det.diameter_hint, gdt_frame_hint=det.gdt_frame_hint,
                                  depth_hint=det.depth_hint, learned_symbols=learned_symbols)
            if raw_text
            else [None]
        )
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
                guessed_symbol_marker=parsed.guessed_symbol_marker if parsed else None,
                note=(parsed.note if parsed else "") or ("Detected by ML model; please fill in details." if not raw_text else ""),
                source=BalloonSource.AUTO.value,
                model_version=det.model_version,
                confidence=round(max(0.0, min(1.0, confidence)), 3),
                status=ReviewStatus.PENDING.value,
            )
            balloon.snapshot_prediction()
            balloons.append(balloon)
            number += 1

    if not balloons and not message:
        message = "No inspection characteristics were automatically detected on this page. Add balloons manually if needed."

    return AutoBalloonResult(balloons=balloons, used_ocr=used_ocr, ocr_available=ocr_available, message=message)
