"""SQLite-backed persistence for BalloonApp projects.

Each project is a single portable SQLite file (extension ``.bpdb``) that can
live anywhere, including inside ``projects/<name>/``. In addition to the
SQLite format, :meth:`ProjectDatabase.export_json` and
:meth:`ProjectDatabase.import_json` support a plain-JSON portable format for
sharing a project without a binary file, or for backups.

Saves are atomic: data is written to a temporary file first and only moved
into place on success, so a crash or crashed export never corrupts an
existing project file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

from balloon_app.data_model import Balloon, Drawing, Project

logger = logging.getLogger("balloon_app.database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS project (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    part_number TEXT,
    part_name TEXT,
    revision TEXT,
    customer TEXT,
    notes TEXT,
    unit TEXT,
    serial_lot_number TEXT,
    fai_report TEXT,
    po_number TEXT,
    mfg_wo TEXT,
    date_created TEXT,
    date_modified TEXT
);

CREATE TABLE IF NOT EXISTS drawing (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    file_name TEXT,
    original_path TEXT,
    page_count INTEGER,
    date_added TEXT,
    last_known_good_path TEXT,
    tol_one_decimal REAL,
    tol_two_decimal REAL,
    tol_three_decimal REAL,
    tol_four_decimal REAL,
    tol_angular REAL,
    tolerances_configured INTEGER,
    FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS balloon (
    id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL,
    number INTEGER,
    page_number INTEGER,
    x REAL, y REAL,
    leader_x REAL, leader_y REAL,
    bbox_x0 REAL, bbox_y0 REAL, bbox_x1 REAL, bbox_y1 REAL,
    char_type TEXT,
    raw_text TEXT,
    nominal REAL,
    tol_plus REAL,
    tol_minus REAL,
    lower_limit REAL,
    upper_limit REAL,
    gdt_symbol TEXT,
    gdt_tolerance TEXT,
    material_condition TEXT,
    datums TEXT,
    surface_finish TEXT,
    thread_callout TEXT,
    note TEXT,
    inspection_method TEXT,
    critical INTEGER,
    source TEXT,
    model_version TEXT,
    confidence REAL,
    status TEXT,
    created_at TEXT,
    modified_at TEXT,
    original_prediction TEXT,
    FOREIGN KEY (drawing_id) REFERENCES drawing(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_drawing_project ON drawing(project_id);
CREATE INDEX IF NOT EXISTS idx_balloon_drawing ON balloon(drawing_id);
"""

SCHEMA_VERSION = 1


class ProjectDatabase:
    """Owns a single SQLite connection for one project file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate_project_columns()
        self._migrate_drawing_columns()
        self._conn.commit()

    def _migrate_project_columns(self) -> None:
        """Add project columns introduced after a project file's schema was created."""
        assert self._conn is not None
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(project)")}
        new_columns = {
            "part_name": "TEXT",
            "unit": "TEXT",
            "serial_lot_number": "TEXT",
            "fai_report": "TEXT",
            "po_number": "TEXT",
            "mfg_wo": "TEXT",
        }
        for column, sql_type in new_columns.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE project ADD COLUMN {column} {sql_type}")

    def _migrate_drawing_columns(self) -> None:
        """Add drawing columns introduced after a project file's schema was created."""
        assert self._conn is not None
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(drawing)")}
        new_columns = {
            "tol_one_decimal": "REAL",
            "tol_two_decimal": "REAL",
            "tol_three_decimal": "REAL",
            "tol_four_decimal": "REAL",
            "tol_angular": "REAL",
            "tolerances_configured": "INTEGER",
        }
        for column, sql_type in new_columns.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE drawing ADD COLUMN {column} {sql_type}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    # ------------------------------------------------------------------
    # Save / load a full Project
    # ------------------------------------------------------------------
    def save_project(self, project: Project) -> None:
        """Persist an entire project (replace-all strategy, single transaction)."""
        project.touch()
        conn = self.conn
        try:
            with conn:
                conn.execute("DELETE FROM balloon")
                conn.execute("DELETE FROM drawing")
                conn.execute("DELETE FROM project")

                conn.execute(
                    """INSERT INTO project
                       (id, name, part_number, part_name, revision, customer, notes,
                        unit, serial_lot_number, fai_report, po_number, mfg_wo,
                        date_created, date_modified)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        project.id,
                        project.name,
                        project.part_number,
                        project.part_name,
                        project.revision,
                        project.customer,
                        project.notes,
                        project.unit,
                        project.serial_lot_number,
                        project.fai_report,
                        project.po_number,
                        project.mfg_wo,
                        project.date_created,
                        project.date_modified,
                    ),
                )

                for d in project.drawings:
                    conn.execute(
                        """INSERT INTO drawing
                           (id, project_id, file_name, original_path, page_count,
                            date_added, last_known_good_path,
                            tol_one_decimal, tol_two_decimal, tol_three_decimal,
                            tol_four_decimal, tol_angular, tolerances_configured)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            d.id,
                            project.id,
                            d.file_name,
                            d.original_path,
                            d.page_count,
                            d.date_added,
                            d.last_known_good_path,
                            d.tol_one_decimal,
                            d.tol_two_decimal,
                            d.tol_three_decimal,
                            d.tol_four_decimal,
                            d.tol_angular,
                            int(d.tolerances_configured),
                        ),
                    )

                for b in project.balloons:
                    conn.execute(
                        """INSERT INTO balloon (
                            id, drawing_id, number, page_number, x, y,
                            leader_x, leader_y, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                            char_type, raw_text, nominal, tol_plus, tol_minus,
                            lower_limit, upper_limit, gdt_symbol, gdt_tolerance,
                            material_condition, datums, surface_finish, thread_callout,
                            note, inspection_method, critical, source, model_version,
                            confidence, status, created_at, modified_at, original_prediction
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            b.id, b.drawing_id, b.number, b.page_number, b.x, b.y,
                            b.leader_x, b.leader_y, b.bbox_x0, b.bbox_y0, b.bbox_x1, b.bbox_y1,
                            b.char_type, b.raw_text, b.nominal, b.tol_plus, b.tol_minus,
                            b.lower_limit, b.upper_limit, b.gdt_symbol, b.gdt_tolerance,
                            b.material_condition, b.datums, b.surface_finish, b.thread_callout,
                            b.note, b.inspection_method, int(b.critical), b.source, b.model_version,
                            b.confidence, b.status, b.created_at, b.modified_at,
                            json.dumps(b.original_prediction) if b.original_prediction else None,
                        ),
                    )
            project.file_path = str(self.path)
            logger.info("Saved project '%s' (%d drawings, %d balloons) to %s",
                        project.name, len(project.drawings), len(project.balloons), self.path)
        except sqlite3.Error:
            logger.exception("Failed to save project to %s", self.path)
            raise

    def load_project(self) -> Project:
        conn = self.conn
        row = conn.execute("SELECT * FROM project LIMIT 1").fetchone()
        if row is None:
            raise ValueError(f"No project found in database: {self.path}")

        row_keys = row.keys()
        project = Project(
            id=row["id"],
            name=row["name"],
            part_number=row["part_number"] or "",
            part_name=(row["part_name"] or "") if "part_name" in row_keys else "",
            revision=row["revision"] or "",
            customer=row["customer"] or "",
            notes=row["notes"] or "",
            unit=(row["unit"] or "in") if "unit" in row_keys else "in",
            serial_lot_number=(row["serial_lot_number"] or "") if "serial_lot_number" in row_keys else "",
            fai_report=(row["fai_report"] or "") if "fai_report" in row_keys else "",
            po_number=(row["po_number"] or "") if "po_number" in row_keys else "",
            mfg_wo=(row["mfg_wo"] or "") if "mfg_wo" in row_keys else "",
            date_created=row["date_created"] or "",
            date_modified=row["date_modified"] or "",
            file_path=str(self.path),
        )

        for drow in conn.execute("SELECT * FROM drawing WHERE project_id = ?", (project.id,)):
            project.drawings.append(
                Drawing(
                    id=drow["id"],
                    project_id=drow["project_id"],
                    file_name=drow["file_name"] or "",
                    original_path=drow["original_path"] or "",
                    page_count=drow["page_count"] or 0,
                    date_added=drow["date_added"] or "",
                    last_known_good_path=drow["last_known_good_path"] or "",
                    tol_one_decimal=drow["tol_one_decimal"],
                    tol_two_decimal=drow["tol_two_decimal"],
                    tol_three_decimal=drow["tol_three_decimal"],
                    tol_four_decimal=drow["tol_four_decimal"],
                    tol_angular=drow["tol_angular"],
                    tolerances_configured=bool(drow["tolerances_configured"]),
                )
            )

        drawing_ids = [d.id for d in project.drawings]
        if drawing_ids:
            placeholders = ",".join("?" for _ in drawing_ids)
            query = f"SELECT * FROM balloon WHERE drawing_id IN ({placeholders})"
            for brow in conn.execute(query, drawing_ids):
                original_prediction = None
                if brow["original_prediction"]:
                    try:
                        original_prediction = json.loads(brow["original_prediction"])
                    except json.JSONDecodeError:
                        original_prediction = None
                project.balloons.append(
                    Balloon(
                        id=brow["id"],
                        number=brow["number"],
                        drawing_id=brow["drawing_id"],
                        page_number=brow["page_number"],
                        x=brow["x"], y=brow["y"],
                        leader_x=brow["leader_x"], leader_y=brow["leader_y"],
                        bbox_x0=brow["bbox_x0"], bbox_y0=brow["bbox_y0"],
                        bbox_x1=brow["bbox_x1"], bbox_y1=brow["bbox_y1"],
                        char_type=brow["char_type"] or "other",
                        raw_text=brow["raw_text"] or "",
                        nominal=brow["nominal"],
                        tol_plus=brow["tol_plus"],
                        tol_minus=brow["tol_minus"],
                        lower_limit=brow["lower_limit"],
                        upper_limit=brow["upper_limit"],
                        gdt_symbol=brow["gdt_symbol"],
                        gdt_tolerance=brow["gdt_tolerance"],
                        material_condition=brow["material_condition"],
                        datums=brow["datums"],
                        surface_finish=brow["surface_finish"],
                        thread_callout=brow["thread_callout"],
                        note=brow["note"] or "",
                        inspection_method=brow["inspection_method"] or "",
                        critical=bool(brow["critical"]),
                        source=brow["source"] or "manual",
                        model_version=brow["model_version"],
                        confidence=brow["confidence"] if brow["confidence"] is not None else 1.0,
                        status=brow["status"] or "accepted",
                        created_at=brow["created_at"] or "",
                        modified_at=brow["modified_at"] or "",
                        original_prediction=original_prediction,
                    )
                )
        logger.info("Loaded project '%s' from %s", project.name, self.path)
        return project

    # ------------------------------------------------------------------
    # Portable JSON export / import
    # ------------------------------------------------------------------
    def export_json(self, project: Project, out_path: Path | str) -> None:
        out_path = Path(out_path)
        data = project.to_dict()
        _atomic_write_text(out_path, json.dumps(data, indent=2))
        logger.info("Exported project '%s' to JSON: %s", project.name, out_path)

    @staticmethod
    def import_json(in_path: Path | str) -> Project:
        in_path = Path(in_path)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        project = Project.from_dict(data)
        logger.info("Imported project '%s' from JSON: %s", project.name, in_path)
        return project


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to ``path`` atomically (write to temp file, then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def atomic_copy_db(src: Path, dst: Path) -> None:
    """Copy a SQLite file to a new location atomically (used by Save As)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=".tmp_", suffix=".bpdb")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp_name)
        os.replace(tmp_name, dst)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
