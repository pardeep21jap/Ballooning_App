"""Core data model: Project, Drawing, and Balloon.

These are plain dataclasses used throughout the application (UI, database,
exporters). They know how to serialize to/from ``dict`` for both SQLite
storage and portable JSON export/import.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    return str(uuid.uuid4())


class CharacteristicType(str, Enum):
    LINEAR_DIMENSION = "linear_dimension"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ANGLE = "angle"
    THREAD = "thread"
    GDT_FRAME = "gdt_frame"
    SURFACE_FINISH = "surface_finish"
    NOTE = "note"
    GENERAL_TOLERANCE = "general_tolerance"
    DEPTH = "depth"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"
    SQUARE = "square"
    OTHER = "other"

    @classmethod
    def display_name(cls, value: str) -> str:
        names = {
            cls.LINEAR_DIMENSION: "Linear Dimension",
            cls.DIAMETER: "Diameter",
            cls.RADIUS: "Radius",
            cls.ANGLE: "Angle",
            cls.THREAD: "Thread",
            cls.GDT_FRAME: "GD&T Feature Control Frame",
            cls.SURFACE_FINISH: "Surface Finish",
            cls.NOTE: "Note",
            cls.GENERAL_TOLERANCE: "General Tolerance",
            cls.DEPTH: "Depth",
            cls.COUNTERBORE: "Counterbore",
            cls.COUNTERSINK: "Countersink",
            cls.SQUARE: "Square",
            cls.OTHER: "Other",
        }
        try:
            return names[cls(value)]
        except ValueError:
            return value


class ReviewStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class BalloonSource(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


@dataclass
class Balloon:
    """A single inspection characteristic / balloon on a drawing page."""

    id: str = field(default_factory=new_id)
    number: int = 0
    drawing_id: str = ""
    page_number: int = 0  # 0-based internally

    # Position in PDF page coordinates (points, origin top-left, PyMuPDF convention)
    x: float = 0.0
    y: float = 0.0
    leader_x: Optional[float] = None
    leader_y: Optional[float] = None

    # Source region bounding box in PDF coordinates, if known
    bbox_x0: Optional[float] = None
    bbox_y0: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None

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
    inspection_method: str = ""
    critical: bool = False

    source: str = BalloonSource.MANUAL.value
    model_version: Optional[str] = None
    confidence: float = 1.0
    status: str = ReviewStatus.ACCEPTED.value

    created_at: str = field(default_factory=_now_iso)
    modified_at: str = field(default_factory=_now_iso)

    # Snapshot of the original auto-prediction fields, preserved even after
    # the user edits the balloon, so we can compute accepted/edited/rejected
    # feedback statistics and export "predicted vs final" training manifests.
    original_prediction: Optional[dict[str, Any]] = None

    def touch(self) -> None:
        self.modified_at = _now_iso()

    def has_bbox(self) -> bool:
        return None not in (self.bbox_x0, self.bbox_y0, self.bbox_x1, self.bbox_y1)

    def bbox(self) -> Optional[tuple[float, float, float, float]]:
        if not self.has_bbox():
            return None
        return (self.bbox_x0, self.bbox_y0, self.bbox_x1, self.bbox_y1)  # type: ignore[return-value]

    def snapshot_prediction(self) -> None:
        """Capture current field values as the "original prediction" baseline.

        Call this once, immediately after an auto-detector creates the
        balloon, before any user edits happen.
        """
        self.original_prediction = self.to_dict(include_prediction=False)

    def is_unchanged_from_prediction(self) -> bool:
        if self.original_prediction is None:
            return True
        current = self.to_dict(include_prediction=False)
        # Ignore fields that change purely due to bookkeeping.
        ignore = {"status", "modified_at", "number"}
        for key, value in current.items():
            if key in ignore:
                continue
            if self.original_prediction.get(key) != value:
                return False
        return True

    def to_dict(self, include_prediction: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_prediction:
            data.pop("original_prediction", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Balloon":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class Drawing:
    """A single PDF drawing attached to a project (possibly multi-page)."""

    id: str = field(default_factory=new_id)
    project_id: str = ""
    file_name: str = ""
    original_path: str = ""
    page_count: int = 0
    date_added: str = field(default_factory=_now_iso)
    last_known_good_path: str = ""

    # Default/general tolerance table (this drawing's "TOLERANCES UNLESS
    # OTHERWISE NOTED" convention), keyed by decimal-place count, plus a
    # separate angular tolerance -- applied to any auto-detected dimension
    # that has no explicit tolerance of its own. `tolerances_configured`
    # distinguishes "reviewed and left blank on purpose" from "never asked".
    tol_one_decimal: Optional[float] = None
    tol_two_decimal: Optional[float] = None
    tol_three_decimal: Optional[float] = None
    tol_four_decimal: Optional[float] = None
    tol_angular: Optional[float] = None
    tolerances_configured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Drawing":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class Project:
    """Top-level project: metadata plus a collection of drawings/balloons."""

    id: str = field(default_factory=new_id)
    name: str = "Untitled Project"
    part_number: str = ""
    part_name: str = ""
    revision: str = ""
    customer: str = ""
    notes: str = ""
    unit: str = "in"  # "in" or "mm" -- chosen once, at project/ballooning start
    serial_lot_number: str = ""
    fai_report: str = ""
    po_number: str = ""
    mfg_wo: str = ""
    date_created: str = field(default_factory=_now_iso)
    date_modified: str = field(default_factory=_now_iso)
    file_path: str = ""  # path to the .bpdb SQLite file, set after save

    drawings: list[Drawing] = field(default_factory=list)
    balloons: list[Balloon] = field(default_factory=list)

    def touch(self) -> None:
        self.date_modified = _now_iso()

    def balloons_for(self, drawing_id: str, page_number: Optional[int] = None) -> list[Balloon]:
        result = [b for b in self.balloons if b.drawing_id == drawing_id]
        if page_number is not None:
            result = [b for b in result if b.page_number == page_number]
        return result

    def next_balloon_number(self, drawing_id: Optional[str] = None) -> int:
        relevant = self.balloons if drawing_id is None else self.balloons_for(drawing_id)
        if not relevant:
            return 1
        return max(b.number for b in relevant) + 1

    def get_drawing(self, drawing_id: str) -> Optional[Drawing]:
        for d in self.drawings:
            if d.id == drawing_id:
                return d
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "part_number": self.part_number,
            "part_name": self.part_name,
            "revision": self.revision,
            "customer": self.customer,
            "notes": self.notes,
            "unit": self.unit,
            "serial_lot_number": self.serial_lot_number,
            "fai_report": self.fai_report,
            "po_number": self.po_number,
            "mfg_wo": self.mfg_wo,
            "date_created": self.date_created,
            "date_modified": self.date_modified,
            "file_path": self.file_path,
            "drawings": [d.to_dict() for d in self.drawings],
            "balloons": [b.to_dict() for b in self.balloons],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        drawings = [Drawing.from_dict(d) for d in data.get("drawings", [])]
        balloons = [Balloon.from_dict(b) for b in data.get("balloons", [])]
        project = cls(
            id=data.get("id", new_id()),
            name=data.get("name", "Untitled Project"),
            part_number=data.get("part_number", ""),
            part_name=data.get("part_name", ""),
            revision=data.get("revision", ""),
            customer=data.get("customer", ""),
            notes=data.get("notes", ""),
            unit=data.get("unit", "in"),
            serial_lot_number=data.get("serial_lot_number", ""),
            fai_report=data.get("fai_report", ""),
            po_number=data.get("po_number", ""),
            mfg_wo=data.get("mfg_wo", ""),
            date_created=data.get("date_created", _now_iso()),
            date_modified=data.get("date_modified", _now_iso()),
            file_path=data.get("file_path", ""),
            drawings=drawings,
            balloons=balloons,
        )
        return project
