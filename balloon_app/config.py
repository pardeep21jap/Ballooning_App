"""Application-wide configuration, paths, and persistent user settings.

All paths are resolved relative to the project root (the directory that
contains ``main.py``) so the application behaves consistently whether it is
run with ``python main.py`` or packaged with PyInstaller.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QSettings

APP_NAME = "BalloonIQ"
APP_ORG = "BalloonIQ"
APP_VERSION = "0.1.0"


def get_base_dir() -> Path:
    """Return the application's base directory.

    When frozen by PyInstaller, resources are unpacked next to the .exe
    (or in ``sys._MEIPASS`` for the one-file build); otherwise it is the
    project root two levels above this file (``BalloonApp/``).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
PROJECTS_DIR = BASE_DIR / "projects"
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_IMAGES_DIR = DATASETS_DIR / "images"
DATASETS_LABELS_DIR = DATASETS_DIR / "labels"
DATASETS_CROPS_DIR = DATASETS_DIR / "crops"
DATASETS_MANIFESTS_DIR = DATASETS_DIR / "manifests"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for _dir in (
    PROJECTS_DIR,
    DATASETS_IMAGES_DIR,
    DATASETS_LABELS_DIR,
    DATASETS_CROPS_DIR,
    DATASETS_MANIFESTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Rendering / detection defaults
# ---------------------------------------------------------------------------
DEFAULT_RENDER_DPI = 200
AUTO_BALLOON_DPI = 300
MIN_ZOOM = 0.1
MAX_ZOOM = 8.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
BALLOON_RADIUS_PDF_POINTS = 9.0

# "Ballooned Drawing" corner stamp -- shared by the on-screen viewer
# (pdf_view.py) and the PDF export (pdf_export.py) so both render it
# identically, scaled by the same stamp_size_percent setting. A simple
# vector rounded-rect "rubber stamp" look, not a raster image, so it stays
# crisp and legible at any stamp size instead of turning to noise when
# scaled down.
STAMP_TEXT = "BALLOONED DRAWING"
STAMP_MARGIN_PDF_POINTS = 8.0
STAMP_FONT_SIZE_PDF_POINTS = 5.0
STAMP_CORNER_RADIUS_PERCENT = 0.35  # rounded-corner radius, as % of the box's shorter side

RULES_OCR_MODEL_VERSION = "rules_ocr_v2_gdt"

# ---------------------------------------------------------------------------
# Balloon status colors (RGBA) - used consistently across the graphics view,
# review table, and PDF export.
# ---------------------------------------------------------------------------
COLOR_PENDING = (255, 165, 0, 255)       # orange - auto-proposed, pending review
COLOR_ACCEPTED = (46, 160, 67, 255)      # green
COLOR_REJECTED = (170, 60, 60, 255)      # muted red
COLOR_EDITED = (46, 160, 67, 255)        # treated visually like accepted
COLOR_SELECTED_OUTLINE = (255, 0, 255, 255)  # magenta selection outline


def status_color(source: str, status: str) -> tuple[int, int, int, int]:
    """Return the RGBA display color for a balloon given its source/status.

    Color reflects review status only -- a manually added balloon that's
    accepted looks identical to an auto-proposal that's accepted, since
    "accepted" is meant to read as one consistent final state regardless of
    how the balloon was created.
    """
    if status == "rejected":
        return COLOR_REJECTED
    if status == "pending":
        return COLOR_PENDING
    if status in ("accepted", "edited"):
        return COLOR_ACCEPTED
    return COLOR_PENDING


# ---------------------------------------------------------------------------
# Characteristic type -> class id mapping for YOLO export. Keep stable once
# any dataset has been exported, since class ids are baked into label files.
# ---------------------------------------------------------------------------
CHARACTERISTIC_CLASSES: list[str] = [
    "linear_dimension",
    "diameter",
    "radius",
    "angle",
    "thread",
    "gdt_frame",
    "surface_finish",
    "note",
    "general_tolerance",
    "other",
    # Appended after the fact -- keep new entries at the end so class ids
    # baked into any already-exported dataset/label files stay stable.
    "depth",
    "counterbore",
    "countersink",
    "square",
]


def _load_learned_symbols(qs: QSettings) -> dict[str, str]:
    raw = qs.value("learned_symbols", "", type=str)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


@dataclass
class AppSettings:
    """User-editable settings, persisted via QSettings (registry-backed)."""

    tesseract_path: str = ""
    default_dpi: int = DEFAULT_RENDER_DPI
    auto_balloon_dpi: int = AUTO_BALLOON_DPI
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    yolo_model_path: str = ""
    use_yolo_if_available: bool = False
    recent_projects: list[str] = field(default_factory=list)
    max_recent_projects: int = 10
    balloon_size_percent: int = 100
    stamp_size_percent: int = 100
    theme: str = "light"
    # A marker letter (a mangled dimensioning-symbol glyph -- see
    # ocr_parser._resolve_learned_marker) -> the characteristic type it
    # actually means, taught by confirming a correction in the UI. Global
    # (not per-project) since the same CAD export tool/font typically
    # produces the same mangling across every drawing from that source.
    learned_symbols: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "AppSettings":
        qs = QSettings(APP_ORG, APP_NAME)
        recent_raw = qs.value("recent_projects", [], type=list)
        recent = [str(p) for p in recent_raw] if recent_raw else []
        return cls(
            tesseract_path=str(qs.value("tesseract_path", "", type=str)),
            default_dpi=int(qs.value("default_dpi", DEFAULT_RENDER_DPI, type=int)),
            auto_balloon_dpi=int(qs.value("auto_balloon_dpi", AUTO_BALLOON_DPI, type=int)),
            confidence_threshold=float(
                qs.value("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD, type=float)
            ),
            yolo_model_path=str(qs.value("yolo_model_path", "", type=str)),
            use_yolo_if_available=bool(
                qs.value("use_yolo_if_available", False, type=bool)
            ),
            recent_projects=recent,
            balloon_size_percent=max(50, min(200, int(qs.value("balloon_size_percent", 100, type=int)))),
            stamp_size_percent=max(50, min(200, int(qs.value("stamp_size_percent", 100, type=int)))),
            theme="dark" if qs.value("theme", "light", type=str) == "dark" else "light",
            learned_symbols=_load_learned_symbols(qs),
        )

    def save(self) -> None:
        qs = QSettings(APP_ORG, APP_NAME)
        qs.setValue("tesseract_path", self.tesseract_path)
        qs.setValue("default_dpi", self.default_dpi)
        qs.setValue("auto_balloon_dpi", self.auto_balloon_dpi)
        qs.setValue("confidence_threshold", self.confidence_threshold)
        qs.setValue("yolo_model_path", self.yolo_model_path)
        qs.setValue("use_yolo_if_available", self.use_yolo_if_available)
        qs.setValue("recent_projects", self.recent_projects)
        qs.setValue("balloon_size_percent", self.balloon_size_percent)
        qs.setValue("stamp_size_percent", self.stamp_size_percent)
        qs.setValue("theme", self.theme)
        qs.setValue("learned_symbols", json.dumps(self.learned_symbols))
        qs.sync()

    def add_recent_project(self, path: str) -> None:
        path = str(path)
        if path in self.recent_projects:
            self.recent_projects.remove(path)
        self.recent_projects.insert(0, path)
        self.recent_projects = self.recent_projects[: self.max_recent_projects]
        self.save()

    def learn_symbol(self, marker: str, char_type: str) -> None:
        """Teach a mangled-symbol marker letter what it actually means,
        persisting immediately so it applies from the very next auto-balloon
        run, in this project and every other one.
        """
        self.learned_symbols[marker.strip().lower()] = char_type
        self.save()

    def forget_symbol(self, marker: str) -> None:
        self.learned_symbols.pop(marker.strip().lower(), None)
        self.save()

    def effective_tesseract_path(self) -> str | None:
        """Resolve the Tesseract executable path.

        Priority: explicit setting -> ``TESSERACT_PATH`` env var -> ``None``
        (meaning "search system PATH", handled by pytesseract itself).
        """
        if self.tesseract_path and Path(self.tesseract_path).exists():
            return self.tesseract_path
        env_path = os.environ.get("TESSERACT_PATH")
        if env_path and Path(env_path).exists():
            return env_path
        return None


def setup_logging() -> logging.Logger:
    """Configure root logging to a rotating local file plus console."""
    log_file = LOGS_DIR / "balloon_app.log"
    logger = logging.getLogger("balloon_app")
    if logger.handlers:
        return logger  # already configured (e.g. re-entrant import)

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger
