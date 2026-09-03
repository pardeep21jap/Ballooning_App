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
from balloon_app.ocr_parser import parse_characteristic
from balloon_app.pdf_engine import PdfDocument, TextBlock, rect_pdf_to_pixel, rect_pixel_to_pdf

logger = logging.getLogger("balloon_app.auto_balloon")

# A *decimal* number (has a decimal point) is a strong dimension signal.
# A bare integer is not -- it's just as likely to be a sheet zone marker
# ("1" "2" "3" "4" along the border), a date fragment, a QTY count, a
# drawing/part number, or a material temper code ("6061 T6").
_DECIMAL_NUMBER_RE = re.compile(r"\d+\.\d+|\.\d+")
_SYMBOL_HINT_RE = re.compile(
    r"[⌀ØΦ∅]|±|°|\bRa\b|\bDIA\b|"
    r"⏤|⏥|○|⌭|⌒|⌓|⟂|∠|∥|⌯|⌖|◎|↗|⌰|⌇",
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


def _looks_like_characteristic(text: str) -> bool:
    """Cheap pre-filter so we don't propose a balloon for every scrap of text.

    A drawing has plenty of text that is never an inspection characteristic:
    title-block boilerplate, drawing/part numbers, dates, QTY counts,
    material codes, and -- notably -- the zone/grid reference numbers
    printed along a drawing border (ANSI/ISO sheet format), which are bare
    1-2 digit integers and would otherwise look exactly like a "candidate".
    We require either a decimal number, a recognized dimension/GD&T/thread
    symbol or keyword, and reject bare short integers outright.
    """
    text = text.strip()
    if not text or len(text) > 120:
        return False
    if _BARE_SHORT_INTEGER_RE.match(text):
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

        detections: list[Detection] = []
        for idxs in lines.values():
            words = [data["text"][i] for i in idxs]
            line_text = " ".join(w for w in words if w).strip()
            if not _looks_like_characteristic(line_text):
                continue
            x0 = min(data["left"][i] for i in idxs)
            y0 = min(data["top"][i] for i in idxs)
            x1 = max(data["left"][i] + data["width"][i] for i in idxs)
            y1 = max(data["top"][i] + data["height"][i] for i in idxs)

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
) -> AutoBalloonResult:
    """Run the full auto-balloon pipeline for a single page.

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

    for block in _merge_nearby_text_blocks(native_blocks):
        if not _looks_like_characteristic(block.text):
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
        parsed = parse_characteristic(raw_text) if raw_text else None
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
            raw_text=raw_text,
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
