"""Main application window: wires together the PDF viewer, review panel,
project/database persistence, auto-ballooning, and all exports.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from balloon_app.auto_balloon import AutoBalloonResult, auto_balloon_page
from balloon_app.config import AppSettings, PROJECTS_DIR, setup_logging
from balloon_app.data_model import (
    Balloon,
    BalloonSource,
    CharacteristicType,
    Drawing,
    Project,
    ReviewStatus,
    new_id,
)
from balloon_app.database import ProjectDatabase
from balloon_app.dialogs import (
    COMMON_INSPECTION_METHODS,
    AboutDialog,
    BalloonEditDialog,
    ConfidenceThresholdDialog,
    DefaultTolerancesDialog,
    ExportExcelOptionsDialog,
    ExportPdfOptionsDialog,
    NewProjectDialog,
    ProjectPropertiesDialog,
    RenumberDialog,
    SettingsDialog,
    TeachTrainingDialog,
)
from balloon_app.excel_export import export_excel
from balloon_app.ocr_parser import DefaultTolerances, parse_characteristics, parse_default_tolerances
from balloon_app.pdf_engine import PdfDocument, PdfLoadError
from balloon_app.pdf_export import PdfExportResult, export_ballooned_pdf, resolve_source_path
from balloon_app.pdf_view import PdfGraphicsView
from balloon_app.training_export import TrainingExportResult, compute_teach_stats, export_training_dataset

logger = logging.getLogger("balloon_app.app")

STATUS_FILTER_OPTIONS = ["All", "Pending", "Accepted", "Edited", "Rejected", "Manual", "Auto"]
SORT_OPTIONS = ["Page", "Number", "Confidence", "Status"]


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-. ]", "_", name).strip().strip(".")
    return cleaned.replace(" ", "_") or "project"


# ---------------------------------------------------------------------------
# Background task functions (run inside a worker QThread; must not touch
# any fitz.Document shared with the live viewer -- they open their own).
# ---------------------------------------------------------------------------
def _auto_balloon_page_task(
    source_path: Path, drawing_id: str, page_number: int, existing: list[Balloon],
    start_number: int, dpi: float, tesseract_path: Optional[str],
    default_tolerances: Optional[DefaultTolerances] = None,
) -> AutoBalloonResult:
    doc = PdfDocument(source_path)
    doc.open()
    try:
        return auto_balloon_page(
            doc, drawing_id, page_number, existing, start_number, dpi=dpi,
            tesseract_path=tesseract_path, default_tolerances=default_tolerances,
        )
    finally:
        doc.close()


def _auto_balloon_drawing_task(
    source_path: Path, drawing_id: str, page_count: int, existing: list[Balloon],
    start_number: int, dpi: float, tesseract_path: Optional[str],
    default_tolerances: Optional[DefaultTolerances] = None,
) -> list[AutoBalloonResult]:
    doc = PdfDocument(source_path)
    doc.open()
    results: list[AutoBalloonResult] = []
    try:
        number = start_number
        running_existing = list(existing)
        for page_number in range(page_count):
            result = auto_balloon_page(
                doc, drawing_id, page_number, running_existing, number, dpi=dpi,
                tesseract_path=tesseract_path, default_tolerances=default_tolerances,
            )
            results.append(result)
            running_existing = running_existing + result.balloons
            number += len(result.balloons)
    finally:
        doc.close()
    return results


# ---------------------------------------------------------------------------
# Undo/redo commands
# ---------------------------------------------------------------------------
class AddBalloonsCommand(QUndoCommand):
    def __init__(self, project: Project, balloons: list[Balloon], refresh_cb, text: str = "Add Balloon(s)"):
        super().__init__(text)
        self.project = project
        self.balloons = list(balloons)
        self.refresh_cb = refresh_cb

    def redo(self) -> None:
        existing_ids = {b.id for b in self.project.balloons}
        for b in self.balloons:
            if b.id not in existing_ids:
                self.project.balloons.append(b)
        self.refresh_cb()

    def undo(self) -> None:
        ids = {b.id for b in self.balloons}
        self.project.balloons = [b for b in self.project.balloons if b.id not in ids]
        self.refresh_cb()


class DeleteBalloonsCommand(QUndoCommand):
    def __init__(self, project: Project, balloons: list[Balloon], refresh_cb, text: str = "Delete Balloon(s)"):
        super().__init__(text)
        self.project = project
        self.balloons = list(balloons)
        self.refresh_cb = refresh_cb

    def redo(self) -> None:
        ids = {b.id for b in self.balloons}
        self.project.balloons = [b for b in self.project.balloons if b.id not in ids]
        self.refresh_cb()

    def undo(self) -> None:
        existing_ids = {b.id for b in self.project.balloons}
        for b in self.balloons:
            if b.id not in existing_ids:
                self.project.balloons.append(b)
        self.refresh_cb()


class SplitBalloonCommand(QUndoCommand):
    """Replace one balloon with several (from "Split Balloon"), as one undo step."""

    def __init__(self, project: Project, original: Balloon, new_balloons: list[Balloon], refresh_cb, text: str = "Split Balloon"):
        super().__init__(text)
        self.project = project
        self.original = original
        self.new_balloons = list(new_balloons)
        self.refresh_cb = refresh_cb

    def redo(self) -> None:
        self.project.balloons = [b for b in self.project.balloons if b.id != self.original.id]
        existing_ids = {b.id for b in self.project.balloons}
        for b in self.new_balloons:
            if b.id not in existing_ids:
                self.project.balloons.append(b)
        self.refresh_cb()

    def undo(self) -> None:
        new_ids = {b.id for b in self.new_balloons}
        self.project.balloons = [b for b in self.project.balloons if b.id not in new_ids]
        if self.original.id not in {b.id for b in self.project.balloons}:
            self.project.balloons.append(self.original)
        self.refresh_cb()


class BalloonFieldChangeCommand(QUndoCommand):
    """Generic "change some fields on some balloons" command, snapshot-based."""

    MOVE_MERGE_ID_BASE = 5000

    def __init__(
        self,
        project: Project,
        balloon_ids: list[str],
        before: dict[str, dict],
        after: dict[str, dict],
        refresh_cb,
        text: str = "Edit Balloon(s)",
        merge_session_id: Optional[int] = None,
    ):
        super().__init__(text)
        self.project = project
        self.balloon_ids = balloon_ids
        self.before = before
        self.after = after
        self.refresh_cb = refresh_cb
        # A drag gesture fires many intermediate position updates; without
        # merging, each one would become its own undo step. Qt merges
        # consecutive pushes that share an id() via mergeWith(), so the
        # whole gesture collapses into a single undo entry. `merge_session_id`
        # must be a fresh value per drag *gesture* (not just per balloon) --
        # otherwise two separate drags of the same balloon would also merge
        # into one, since Qt only compares id(), not wall-clock time.
        self._merge_session_id = merge_session_id

    def id(self) -> int:  # noqa: A003 (Qt API name)
        if self._merge_session_id is None:
            return -1
        return self.MOVE_MERGE_ID_BASE + self._merge_session_id

    def mergeWith(self, other: "QUndoCommand") -> bool:  # noqa: N802 (Qt API name)
        if not isinstance(other, BalloonFieldChangeCommand) or other._merge_session_id is None:
            return False
        if self._merge_session_id != other._merge_session_id:
            return False
        if self.balloon_ids != other.balloon_ids:
            return False
        self.after = other.after
        return True

    def _apply(self, snapshots: dict[str, dict]) -> None:
        by_id = {b.id: b for b in self.project.balloons}
        for bid, snap in snapshots.items():
            balloon = by_id.get(bid)
            if balloon is None:
                continue
            for key, value in snap.items():
                setattr(balloon, key, value)
        self.refresh_cb()

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


# ---------------------------------------------------------------------------
# Generic background worker
# ---------------------------------------------------------------------------
class _CallableWorker(QThread):
    finished_ok = pyqtSignal(object)
    finished_error = pyqtSignal(str)

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user via dialog
            logger.exception("Background task failed")
            self.finished_error.emit(str(exc))
            return
        self.finished_ok.emit(result)


class ReviewTable(QTableWidget):
    """A :class:`QTableWidget` with drag-and-drop row reordering.

    Qt's built-in ``InternalMove`` drag-drop moves individual *items*, not
    whole rows, which corrupts a multi-column table like this one. Instead,
    this only detects "row A was dropped onto row B" and hands it to a
    callback -- the caller is expected to recompute balloon numbers from
    the intended order and fully repaint the table from data, rather than
    letting Qt attempt the move itself.
    """

    def __init__(self, rows: int, columns: int, on_rows_dropped, parent: Optional[QWidget] = None):
        super().__init__(rows, columns, parent)
        self._on_rows_dropped = on_rows_dropped
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event) -> None:
        if event.source() is not self:
            super().dropEvent(event)
            return
        source_row = self.currentRow()
        target_row = self.indexAt(event.position().toPoint()).row()
        if target_row == -1:
            target_row = self.rowCount() - 1
        event.ignore()  # the table is repainted from data by the callback, not by Qt
        if source_row == -1 or target_row == -1 or source_row == target_row:
            return
        self._on_rows_dropped(source_row, target_row)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BalloonApp")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "resources" / "balloonapp.ico")))
        self.resize(1440, 900)

        self.settings = AppSettings.load()
        self.project: Optional[Project] = None
        self.drawing: Optional[Drawing] = None
        self.pdf_doc: Optional[PdfDocument] = None
        self.current_page = 0
        self._dirty = False
        self._active_worker: Optional[_CallableWorker] = None
        self._drag_start_snapshots: dict[str, dict] = {}
        self._drag_session_ids: dict[str, int] = {}
        self._drag_session_counter = 0
        self._leader_drag_start_snapshots: dict[str, dict] = {}
        self._leader_drag_session_ids: dict[str, int] = {}

        self.undo_stack = QUndoStack(self)
        self.undo_stack.indexChanged.connect(lambda _i: self._update_window_title())

        self._build_ui()
        self._build_actions_and_menus()
        self._update_window_title()
        self._update_page_controls()
        self._refresh_recent_projects_menu()

        self.statusBar().showMessage("Ready. Create or open a project to begin.", 5000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.pdf_view = PdfGraphicsView(self)
        self.pdf_view.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 4000))
        self.pdf_view.balloonSelected.connect(self._on_canvas_balloon_selected)
        self.pdf_view.balloonDoubleClicked.connect(self._edit_balloon)
        self.pdf_view.balloonMoved.connect(self._on_balloon_moved)
        self.pdf_view.balloonDragFinished.connect(self._on_balloon_drag_finished)
        self.pdf_view.newBalloonRequested.connect(self._on_new_balloon_requested)
        self.pdf_view.leaderPointPicked.connect(self._on_leader_point_picked)
        self.pdf_view.leaderHandleMoved.connect(self._on_leader_handle_moved)
        self.pdf_view.leaderHandleDragFinished.connect(self._on_leader_handle_drag_finished)
        self.pdf_view.emptySpaceClicked.connect(self._on_empty_space_clicked)
        self.pdf_view.renderStarted.connect(lambda: self._set_busy(True, "Rendering page..."))
        self.pdf_view.renderFinished.connect(lambda: self._set_busy(False))

        review_panel = self._build_review_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.pdf_view)
        splitter.addWidget(review_panel)
        # Stretch factors alone multiply each widget's size hint. The
        # empty canvas has a tiny hint while the review controls have a
        # large one, so explicitly seed the intended starting widths.
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([900, 540])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_review_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(STATUS_FILTER_OPTIONS)
        self.filter_combo.currentIndexChanged.connect(self._refresh_review_table)
        filter_row.addWidget(self.filter_combo)

        filter_row.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(SORT_OPTIONS)
        self.sort_combo.currentIndexChanged.connect(self._refresh_review_table)
        filter_row.addWidget(self.sort_combo)
        layout.addLayout(filter_row)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Balloon #, text, or nominal...")
        self.search_edit.textChanged.connect(self._refresh_review_table)
        search_row.addWidget(self.search_edit)
        layout.addLayout(search_row)

        self.review_table = ReviewTable(0, 8, self._on_review_rows_dropped)
        self.review_table.setHorizontalHeaderLabels(
            ["Balloon #", "Page", "Type", "Raw Text", "Nominal", "Tolerance", "Inspection Method", "Status"]
        )
        self.review_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.review_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.review_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.review_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.review_table.doubleClicked.connect(lambda _i: self._edit_selected_balloon())
        layout.addWidget(self.review_table, stretch=1)

        reorder_hint = QLabel("Drag a row to renumber balloons to match (Filter must be \"All\").")
        reorder_hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(reorder_hint)

        row1 = QHBoxLayout()
        accept_btn = QPushButton("Accept")
        accept_btn.clicked.connect(self._accept_selected)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_selected_balloon)
        reject_btn = QPushButton("Reject")
        reject_btn.clicked.connect(self._reject_selected)
        add_btn = QPushButton("Add Manual")
        add_btn.clicked.connect(self._add_manual_balloon)
        row1.addWidget(accept_btn)
        row1.addWidget(edit_btn)
        row1.addWidget(reject_btn)
        row1.addWidget(add_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        accept_all_btn = QPushButton("Accept All Above Threshold...")
        accept_all_btn.clicked.connect(self._accept_all_above_threshold)
        reject_all_btn = QPushButton("Reject Selected/Pending")
        reject_all_btn.clicked.connect(self._reject_all_selected_or_pending)
        row2.addWidget(accept_all_btn)
        row2.addWidget(reject_all_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        duplicate_btn = QPushButton("Duplicate")
        duplicate_btn.clicked.connect(self._duplicate_selected_balloon)
        split_btn = QPushButton("Split Balloon")
        split_btn.setToolTip(
            "Re-run automatic detection on this balloon's raw drawing text and, if it now "
            "recognizes more than one characteristic (e.g. a hole's diameter and its depth), "
            "replace this balloon with one properly-typed balloon per characteristic."
        )
        split_btn.clicked.connect(self._split_selected_balloon)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected_balloons)
        renumber_btn = QPushButton("Renumber...")
        renumber_btn.clicked.connect(self._show_renumber_dialog)
        row3.addWidget(duplicate_btn)
        row3.addWidget(split_btn)
        row3.addWidget(delete_btn)
        row3.addWidget(renumber_btn)
        layout.addLayout(row3)

        return panel

    def _build_actions_and_menus(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        new_project_act = QAction("New Project...", self)
        new_project_act.triggered.connect(self._new_project)
        file_menu.addAction(new_project_act)

        open_project_act = QAction("Open Project...", self)
        open_project_act.triggered.connect(self._open_project)
        file_menu.addAction(open_project_act)

        self.recent_menu = file_menu.addMenu("Recent Projects")

        close_project_act = QAction("Close Project", self)
        close_project_act.triggered.connect(self._close_project)
        file_menu.addAction(close_project_act)

        file_menu.addSeparator()
        save_project_act = QAction("Save Project", self)
        save_project_act.setShortcut(QKeySequence("Ctrl+S"))
        save_project_act.triggered.connect(self._save_project)
        file_menu.addAction(save_project_act)

        save_as_act = QAction("Save Project As...", self)
        save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_act.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_act)

        project_props_act = QAction("Project Properties...", self)
        project_props_act.triggered.connect(self._edit_project_properties)
        file_menu.addAction(project_props_act)

        file_menu.addSeparator()
        add_pdf_act = QAction("Add/Open PDF Drawing...", self)
        add_pdf_act.setShortcut(QKeySequence("Ctrl+O"))
        add_pdf_act.triggered.connect(self._add_pdf_drawing)
        file_menu.addAction(add_pdf_act)

        relink_act = QAction("Relink Current Drawing...", self)
        relink_act.triggered.connect(self._relink_drawing)
        file_menu.addAction(relink_act)

        file_menu.addSeparator()
        export_pdf_act = QAction("Export Ballooned PDF...", self)
        export_pdf_act.triggered.connect(self._export_ballooned_pdf)
        file_menu.addAction(export_pdf_act)

        export_excel_act = QAction("Export Excel Inspection Sheet...", self)
        export_excel_act.setShortcut(QKeySequence("Ctrl+E"))
        export_excel_act.triggered.connect(self._export_excel)
        file_menu.addAction(export_excel_act)

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        undo_act = self.undo_stack.createUndoAction(self, "Undo")
        undo_act.setShortcut(QKeySequence("Ctrl+Z"))
        edit_menu.addAction(undo_act)
        redo_act = self.undo_stack.createRedoAction(self, "Redo")
        redo_act.setShortcut(QKeySequence("Ctrl+Y"))
        edit_menu.addAction(redo_act)
        edit_menu.addSeparator()

        self.add_balloon_act = QAction("Add Balloon Mode", self, checkable=True)
        self.add_balloon_act.setShortcut(QKeySequence("Ctrl+B"))
        self.add_balloon_act.toggled.connect(self._toggle_add_balloon_mode)
        edit_menu.addAction(self.add_balloon_act)

        leader_act = QAction("Set Leader Point for Selected Balloon", self)
        leader_act.triggered.connect(self._start_leader_pick)
        edit_menu.addAction(leader_act)

        duplicate_act = QAction("Duplicate Balloon", self)
        duplicate_act.setShortcut(QKeySequence("Ctrl+D"))
        duplicate_act.triggered.connect(self._duplicate_selected_balloon)
        edit_menu.addAction(duplicate_act)

        split_act = QAction("Split Balloon", self)
        split_act.triggered.connect(self._split_selected_balloon)
        edit_menu.addAction(split_act)

        delete_act = QAction("Delete Balloon", self)
        delete_act.setShortcut(QKeySequence("Delete"))
        delete_act.triggered.connect(self._delete_selected_balloons)
        edit_menu.addAction(delete_act)

        renumber_act = QAction("Renumber Balloons...", self)
        renumber_act.triggered.connect(self._show_renumber_dialog)
        edit_menu.addAction(renumber_act)

        # View menu
        view_menu = menu_bar.addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")
        self.theme_actions = QActionGroup(self)
        self.theme_actions.setExclusive(True)
        for theme in ("light", "dark"):
            action = theme_menu.addAction(theme.capitalize())
            action.setCheckable(True)
            action.setChecked(self.settings.theme == theme)
            self.theme_actions.addAction(action)
            action.triggered.connect(lambda checked, choice=theme: self._change_theme(choice))
        view_menu.addSeparator()
        zoom_in_act = QAction("Zoom In", self)
        zoom_in_act.setShortcut(QKeySequence("Ctrl+="))
        zoom_in_act.triggered.connect(self.pdf_view.zoom_in)
        view_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("Zoom Out", self)
        zoom_out_act.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_act.triggered.connect(self.pdf_view.zoom_out)
        view_menu.addAction(zoom_out_act)

        fit_page_act = QAction("Fit Page", self)
        fit_page_act.setShortcut(QKeySequence("Ctrl+0"))
        fit_page_act.triggered.connect(self.pdf_view.fit_page)
        view_menu.addAction(fit_page_act)

        actual_size_act = QAction("Actual Size (100%)", self)
        actual_size_act.triggered.connect(self.pdf_view.actual_size)
        view_menu.addAction(actual_size_act)

        view_menu.addSeparator()
        self.prev_page_action = QAction("Previous Page", self)
        self.prev_page_action.setShortcut(QKeySequence("PgUp"))
        self.prev_page_action.triggered.connect(lambda: self._load_page(self.current_page - 1))
        view_menu.addAction(self.prev_page_action)

        self.next_page_action = QAction("Next Page", self)
        self.next_page_action.setShortcut(QKeySequence("PgDown"))
        self.next_page_action.triggered.connect(lambda: self._load_page(self.current_page + 1))
        view_menu.addAction(self.next_page_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        auto_page_act = QAction("Auto-Balloon Current Page", self)
        auto_page_act.triggered.connect(self._auto_balloon_current_page)
        tools_menu.addAction(auto_page_act)

        auto_drawing_act = QAction("Auto-Balloon Entire Drawing", self)
        auto_drawing_act.triggered.connect(self._auto_balloon_entire_drawing)
        tools_menu.addAction(auto_drawing_act)

        default_tol_act = QAction("Default Tolerances...", self)
        default_tol_act.triggered.connect(self._edit_default_tolerances)
        tools_menu.addAction(default_tol_act)

        tools_menu.addSeparator()
        review_act = QAction("Review", self)
        review_act.triggered.connect(self._focus_review_panel)
        tools_menu.addAction(review_act)

        teach_act = QAction("Teach / Training Data...", self)
        teach_act.triggered.connect(self._show_teach_dialog)
        tools_menu.addAction(teach_act)

        tools_menu.addSeparator()
        settings_act = QAction("Settings...", self)
        settings_act.triggered.connect(self._show_settings_dialog)
        tools_menu.addAction(settings_act)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_act = QAction("About BalloonApp", self)
        about_act.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_act)

        # Toolbar (subset of the most common actions)
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for act in (new_project_act, open_project_act, save_project_act, add_pdf_act):
            toolbar.addAction(act)
        toolbar.addSeparator()
        toolbar.addAction(self.prev_page_action)

        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.valueChanged.connect(lambda v: self._load_page(v - 1))
        toolbar.addWidget(self.page_spin)
        self.page_count_label = QLabel("/ 0")
        toolbar.addWidget(self.page_count_label)
        toolbar.addAction(self.next_page_action)

        toolbar.addSeparator()
        toolbar.addAction(zoom_in_act)
        toolbar.addAction(zoom_out_act)
        toolbar.addAction(fit_page_act)
        toolbar.addSeparator()
        toolbar.addAction(self.add_balloon_act)
        toolbar.addAction(auto_page_act)
        toolbar.addAction(auto_drawing_act)
        toolbar.addSeparator()
        toolbar.addAction(export_pdf_act)
        toolbar.addAction(export_excel_act)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Drawing: "))
        self.drawing_combo = QComboBox()
        self.drawing_combo.setMinimumWidth(160)
        self.drawing_combo.currentIndexChanged.connect(self._on_drawing_combo_changed)
        self.drawing_combo.setVisible(False)
        toolbar.addWidget(self.drawing_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Balloon Size: "))
        self.balloon_size_spin = QSpinBox()
        self.balloon_size_spin.setRange(50, 200)
        self.balloon_size_spin.setSingleStep(10)
        self.balloon_size_spin.setSuffix("%")
        self.balloon_size_spin.setValue(self.settings.balloon_size_percent)
        self.balloon_size_spin.setToolTip("Size of all balloons on screen and in exported PDFs (100% is the default)")
        self.pdf_view.balloon_size_percent = self.settings.balloon_size_percent
        self.balloon_size_spin.valueChanged.connect(self._change_balloon_size)
        toolbar.addWidget(self.balloon_size_spin)

    def _change_balloon_size(self, percent: int) -> None:
        self.settings.balloon_size_percent = percent
        self.settings.save()
        self.pdf_view.set_balloon_size(percent)

    def _change_theme(self, theme: str) -> None:
        from balloon_app.theme import apply_theme

        apply_theme(QApplication.instance(), theme)
        self.settings.theme = theme
        self.settings.save()

    # ------------------------------------------------------------------
    # Busy / progress
    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)
            if message:
                self.statusBar().showMessage(message)
            self.setCursor(Qt.CursorShape.WaitCursor)
        else:
            self.unsetCursor()

    def _run_background(self, fn, on_success, busy_message: str, *args, **kwargs) -> None:
        if self._active_worker is not None and self._active_worker.isRunning():
            QMessageBox.information(self, "Busy", "Please wait for the current operation to finish.")
            return
        self._set_busy(True, busy_message)
        worker = _CallableWorker(fn, args, kwargs)
        worker.finished_ok.connect(lambda result: self._on_worker_success(result, on_success))
        worker.finished_error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_worker_success(self, result, callback) -> None:
        self._set_busy(False)
        callback(result)

    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Operation Failed", message)
        self.statusBar().showMessage(f"Error: {message}", 8000)

    # ------------------------------------------------------------------
    # Dirty tracking / window title
    # ------------------------------------------------------------------
    def _is_dirty(self) -> bool:
        return self._dirty or not self.undo_stack.isClean()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        if self.project:
            star = "*" if self._is_dirty() else ""
            self.setWindowTitle(f"{self.project.name}{star} - BalloonApp")
        else:
            self.setWindowTitle("BalloonApp")

    def _confirm_discard_changes(self) -> bool:
        if self.project is None or not self._is_dirty():
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes", "Save changes to the current project first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_project()
        return reply == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------
    def _new_project(self) -> bool:
        if not self._confirm_discard_changes():
            return False
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        project = Project(**dialog.values())

        folder_name = _sanitize_filename(project.name)
        candidate_dir = PROJECTS_DIR / folder_name
        counter = 1
        while candidate_dir.exists():
            candidate_dir = PROJECTS_DIR / f"{folder_name}_{counter}"
            counter += 1
        candidate_dir.mkdir(parents=True, exist_ok=True)
        db_path = candidate_dir / f"{folder_name}.bpdb"

        try:
            db = ProjectDatabase(db_path)
            db.connect()
            db.save_project(project)
            db.close()
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Create Project", str(exc))
            return False

        self._load_project_object(project, db_path)
        return True

    def _open_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", str(PROJECTS_DIR), "BalloonApp Project (*.bpdb)")
        if not path:
            return
        self._open_project_path(Path(path))

    def _open_project_path(self, path: Path) -> None:
        try:
            db = ProjectDatabase(path)
            db.connect()
            project = db.load_project()
            db.close()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Open Project", str(exc))
            return
        self._load_project_object(project, path)

    def _close_project(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "No Project", "No project is currently open.")
            return
        if not self._confirm_discard_changes():
            return
        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
        self.project = None
        self.drawing = None
        self.pdf_doc = None
        self.current_page = 0
        self.pdf_view.load_document(None)
        self.undo_stack.clear()
        self._dirty = False

        self._refresh_drawing_selector()
        self._refresh_review_table()
        self._update_window_title()
        self._update_page_controls()
        self.statusBar().showMessage("Project closed.", 4000)

    def _load_project_object(self, project: Project, path: Path) -> None:
        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
        self.project = project
        self.project.file_path = str(path)
        self.drawing = None
        self.pdf_doc = None
        self.pdf_view.load_document(None)
        self.undo_stack.clear()
        self._dirty = False

        self.settings.add_recent_project(str(path))
        self._refresh_recent_projects_menu()
        self._refresh_drawing_selector()
        self._refresh_review_table()
        self._update_window_title()

        if project.drawings:
            self._resolve_and_open_drawing(project.drawings[0])
        self.statusBar().showMessage(f"Project '{project.name}' loaded.", 5000)

    def _save_project(self) -> bool:
        if self.project is None:
            return False
        if not self.project.file_path:
            return self._save_project_as()
        try:
            db = ProjectDatabase(Path(self.project.file_path))
            db.connect()
            db.save_project(self.project)
            db.close()
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return False
        self._dirty = False
        self.undo_stack.setClean()
        self._update_window_title()
        self.statusBar().showMessage("Project saved.", 4000)
        return True

    def _save_project_as(self) -> bool:
        if self.project is None:
            return False
        default_name = _sanitize_filename(self.project.name)
        default_path = str(PROJECTS_DIR / default_name / f"{default_name}.bpdb")
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", default_path, "BalloonApp Project (*.bpdb)")
        if not path:
            return False
        if not path.lower().endswith(".bpdb"):
            path += ".bpdb"
        self.project.file_path = path
        saved = self._save_project()
        if saved:
            self.settings.add_recent_project(path)
            self._refresh_recent_projects_menu()
        return saved

    def _edit_project_properties(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "No Project", "Create or open a project first.")
            return
        dialog = ProjectPropertiesDialog(
            self.project.name, self.project.part_number, self.project.part_name,
            self.project.revision, self.project.customer, self.project.unit,
            self.project.notes, self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.values()
            self.project.name = values["name"]
            self.project.part_number = values["part_number"]
            self.project.part_name = values["part_name"]
            self.project.revision = values["revision"]
            self.project.customer = values["customer"]
            self.project.unit = values["unit"]
            self.project.notes = values["notes"]
            self.project.touch()
            self._mark_dirty()

    def _refresh_recent_projects_menu(self) -> None:
        self.recent_menu.clear()
        if not self.settings.recent_projects:
            action = self.recent_menu.addAction("(none)")
            action.setEnabled(False)
            return
        for path_str in self.settings.recent_projects:
            action = self.recent_menu.addAction(path_str)
            action.triggered.connect(lambda checked=False, p=path_str: self._open_recent(p))

    def _open_recent(self, path_str: str) -> None:
        if not self._confirm_discard_changes():
            return
        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "Not Found", f"Project file not found:\n{path}")
            return
        self._open_project_path(path)

    # ------------------------------------------------------------------
    # Drawing lifecycle
    # ------------------------------------------------------------------
    def _add_pdf_drawing(self) -> None:
        if self.project is None:
            if not self._new_project():
                return
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF Drawing", "", "PDF Files (*.pdf)")
        if not path:
            return
        try:
            pdf_doc = PdfDocument(path)
            pdf_doc.open()
            page_count = pdf_doc.page_count
        except PdfLoadError as exc:
            QMessageBox.critical(self, "Cannot Open PDF", str(exc))
            return

        drawing = Drawing(project_id=self.project.id, file_name=Path(path).name, original_path=str(path), page_count=page_count)
        self.project.drawings.append(drawing)
        self._mark_dirty()
        self._set_active_drawing(drawing, pdf_doc)
        self._refresh_drawing_selector()

    def _relink_drawing(self) -> None:
        if self.drawing is None:
            QMessageBox.information(self, "No Drawing", "No drawing is currently open.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Relink PDF Drawing", "", "PDF Files (*.pdf)")
        if not path:
            return
        self.drawing.last_known_good_path = path
        self._mark_dirty()
        self._resolve_and_open_drawing(self.drawing)

    def _resolve_and_open_drawing(self, drawing: Drawing) -> bool:
        path = resolve_source_path(drawing)
        if path is None:
            reply = QMessageBox.question(
                self, "Drawing Not Found",
                f"Cannot find the source PDF for '{drawing.file_name}'.\nLocate it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
            new_path, _ = QFileDialog.getOpenFileName(self, "Locate PDF Drawing", "", "PDF Files (*.pdf)")
            if not new_path:
                return False
            drawing.last_known_good_path = new_path
            self._mark_dirty()
            path = Path(new_path)
        try:
            pdf_doc = PdfDocument(path)
            pdf_doc.open()
        except PdfLoadError as exc:
            QMessageBox.critical(self, "Cannot Open PDF", str(exc))
            return False
        self._set_active_drawing(drawing, pdf_doc)
        return True

    def _set_active_drawing(self, drawing: Drawing, pdf_doc: PdfDocument) -> None:
        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
        self.drawing = drawing
        self.pdf_doc = pdf_doc
        self.pdf_view.load_document(pdf_doc)
        self.current_page = 0
        self._load_page(0)
        self._refresh_review_table()
        self._refresh_drawing_selector()

    def _refresh_drawing_selector(self) -> None:
        self.drawing_combo.blockSignals(True)
        self.drawing_combo.clear()
        if self.project:
            for d in self.project.drawings:
                self.drawing_combo.addItem(d.file_name or "(unnamed)", d.id)
            if self.drawing:
                idx = self.drawing_combo.findData(self.drawing.id)
                if idx >= 0:
                    self.drawing_combo.setCurrentIndex(idx)
        self.drawing_combo.blockSignals(False)
        self.drawing_combo.setVisible(bool(self.project and len(self.project.drawings) > 1))

    def _on_drawing_combo_changed(self, index: int) -> None:
        if self.project is None or index < 0:
            return
        drawing_id = self.drawing_combo.itemData(index)
        if self.drawing is not None and drawing_id == self.drawing.id:
            return
        drawing = self.project.get_drawing(drawing_id)
        if drawing:
            self._resolve_and_open_drawing(drawing)

    # ------------------------------------------------------------------
    # Page navigation
    # ------------------------------------------------------------------
    def _load_page(self, page_number: int) -> None:
        if self.drawing is None or self.pdf_doc is None:
            return
        page_number = max(0, min(page_number, self.drawing.page_count - 1))
        self.current_page = page_number
        balloons = self.project.balloons_for(self.drawing.id) if self.project else []
        self.pdf_view.set_page(self.current_page, balloons)
        self._update_page_controls()

    def _update_page_controls(self) -> None:
        has_drawing = self.drawing is not None
        self.page_spin.blockSignals(True)
        if has_drawing:
            self.page_spin.setMaximum(max(1, self.drawing.page_count))
            self.page_spin.setValue(self.current_page + 1)
            self.page_count_label.setText(f"/ {self.drawing.page_count}")
        else:
            self.page_spin.setMaximum(1)
            self.page_spin.setValue(1)
            self.page_count_label.setText("/ 0")
        self.page_spin.blockSignals(False)
        self.prev_page_action.setEnabled(has_drawing and self.current_page > 0)
        self.next_page_action.setEnabled(has_drawing and self.current_page < self.drawing.page_count - 1)

    # ------------------------------------------------------------------
    # Balloon helpers
    # ------------------------------------------------------------------
    def _find_balloon(self, balloon_id: str) -> Optional[Balloon]:
        if self.project is None:
            return None
        for b in self.project.balloons:
            if b.id == balloon_id:
                return b
        return None

    def _refresh_all(self) -> None:
        balloons = self.project.balloons_for(self.drawing.id) if (self.project and self.drawing) else []
        self.pdf_view.refresh_balloons(balloons)
        self._refresh_review_table()
        self._update_window_title()

    def _toggle_add_balloon_mode(self, checked: bool) -> None:
        self.pdf_view.add_balloon_mode = checked
        if checked:
            self.statusBar().showMessage("Add Balloon mode: click on the drawing to place a balloon.")
        else:
            self.statusBar().clearMessage()

    def _start_leader_pick(self) -> None:
        ids = self._selected_balloon_ids()
        if not ids:
            QMessageBox.information(self, "No Selection", "Select a balloon first.")
            return
        self.pdf_view.leader_mode_balloon_id = ids[0]
        self.statusBar().showMessage("Click on the drawing to set the leader line start point.")

    def _on_new_balloon_requested(self, pdf_x: float, pdf_y: float) -> None:
        if self.project is None or self.drawing is None:
            return
        balloon = Balloon(
            number=self.project.next_balloon_number(self.drawing.id),
            drawing_id=self.drawing.id,
            page_number=self.current_page,
            x=pdf_x, y=pdf_y,
            source=BalloonSource.MANUAL.value,
            status=ReviewStatus.ACCEPTED.value,
        )
        dialog = BalloonEditDialog(balloon, is_new=True, default_tolerances=self._current_default_tolerances(), parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply_to_balloon(balloon)
            cmd = AddBalloonsCommand(self.project, [balloon], self._refresh_all, text="Add Balloon")
            self.undo_stack.push(cmd)
            self._refresh_all()

    def _on_leader_point_picked(self, balloon_id: str, pdf_x: float, pdf_y: float) -> None:
        self.pdf_view.leader_mode_balloon_id = None
        balloon = self._find_balloon(balloon_id)
        if balloon is None:
            return
        before = {balloon_id: {"leader_x": balloon.leader_x, "leader_y": balloon.leader_y}}
        balloon.leader_x = pdf_x
        balloon.leader_y = pdf_y
        balloon.touch()
        after = {balloon_id: {"leader_x": pdf_x, "leader_y": pdf_y}}
        cmd = BalloonFieldChangeCommand(self.project, [balloon_id], before, after, self._refresh_all, text="Set Leader Point")
        self.undo_stack.push(cmd)
        self._refresh_all()
        self.statusBar().showMessage("Leader point set.", 3000)

    def _on_balloon_moved(self, balloon_id: str, pdf_x: float, pdf_y: float) -> None:
        balloon = self._find_balloon(balloon_id)
        if balloon is None or self.project is None:
            return
        if balloon_id not in self._drag_start_snapshots:
            self._drag_start_snapshots[balloon_id] = {"x": balloon.x, "y": balloon.y}
            self._drag_session_counter += 1
            self._drag_session_ids[balloon_id] = self._drag_session_counter
        balloon.x = pdf_x
        balloon.y = pdf_y
        balloon.touch()
        self._mark_dirty()
        # Push a mergeable command on every intermediate position update;
        # Qt's undo stack merges consecutive moves sharing the same session
        # id (see BalloonFieldChangeCommand.mergeWith), so a whole drag
        # gesture becomes a single undo step even though this fires on every
        # mouse-move tick. The session id is unique per drag *gesture*, so a
        # later, separate drag of the same balloon won't merge with this one.
        before = {balloon_id: self._drag_start_snapshots[balloon_id]}
        after = {balloon_id: {"x": pdf_x, "y": pdf_y}}
        cmd = BalloonFieldChangeCommand(
            self.project, [balloon_id], before, after, lambda: None, text="Move Balloon",
            merge_session_id=self._drag_session_ids[balloon_id],
        )
        self.undo_stack.push(cmd)

    def _on_balloon_drag_finished(self, balloon_id: str) -> None:
        # Clear the drag baseline/session so the *next* drag on this balloon
        # starts its own undo entry instead of merging into this finished one.
        self._drag_start_snapshots.pop(balloon_id, None)
        self._drag_session_ids.pop(balloon_id, None)

    def _on_leader_handle_moved(self, balloon_id: str, pdf_x: float, pdf_y: float) -> None:
        """Dragging a leader line's start point (see LeaderHandleItem). Mirrors
        _on_balloon_moved's mergeable-undo-per-gesture pattern; dragging a
        leader line that was only implicitly anchored to the balloon's bbox
        center makes it an explicit, independently-movable point from here on.
        """
        balloon = self._find_balloon(balloon_id)
        if balloon is None or self.project is None:
            return
        if balloon_id not in self._leader_drag_start_snapshots:
            self._leader_drag_start_snapshots[balloon_id] = {
                "leader_x": balloon.leader_x, "leader_y": balloon.leader_y,
            }
            self._drag_session_counter += 1
            self._leader_drag_session_ids[balloon_id] = self._drag_session_counter
        balloon.leader_x = pdf_x
        balloon.leader_y = pdf_y
        balloon.touch()
        self._mark_dirty()
        before = {balloon_id: self._leader_drag_start_snapshots[balloon_id]}
        after = {balloon_id: {"leader_x": pdf_x, "leader_y": pdf_y}}
        cmd = BalloonFieldChangeCommand(
            self.project, [balloon_id], before, after, lambda: None, text="Move Leader Point",
            merge_session_id=self._leader_drag_session_ids[balloon_id],
        )
        self.undo_stack.push(cmd)

    def _on_leader_handle_drag_finished(self, balloon_id: str) -> None:
        self._leader_drag_start_snapshots.pop(balloon_id, None)
        self._leader_drag_session_ids.pop(balloon_id, None)

    def _on_canvas_balloon_selected(self, balloon_id: str) -> None:
        self._select_table_row_for_balloon(balloon_id)
        # _select_table_row_for_balloon blocks the table's own signals (to
        # avoid re-triggering this same handler in a loop), so the usual
        # itemSelectionChanged -> select_balloon path never fires here --
        # do it directly so the leader-handle diamond still shows up.
        self.pdf_view.select_balloon(balloon_id)

    def _on_empty_space_clicked(self) -> None:
        self.review_table.clearSelection()
        self.pdf_view.select_balloon(None)

    def _select_table_row_for_balloon(self, balloon_id: str) -> None:
        for row in range(self.review_table.rowCount()):
            item = self.review_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == balloon_id:
                self.review_table.blockSignals(True)
                self.review_table.selectRow(row)
                self.review_table.blockSignals(False)
                break

    def _on_table_selection_changed(self) -> None:
        ids = self._selected_balloon_ids()
        if not ids:
            return
        balloon = self._find_balloon(ids[0])
        if balloon is None:
            return
        if balloon.page_number != self.current_page:
            self._load_page(balloon.page_number)
        self.pdf_view.select_balloon(balloon.id)

    def _selected_balloon_ids(self) -> list[str]:
        ids = []
        for index in self.review_table.selectionModel().selectedRows():
            item = self.review_table.item(index.row(), 0)
            if item:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    # ------------------------------------------------------------------
    # Review panel actions
    # ------------------------------------------------------------------
    def _filtered_sorted_balloons(self) -> list[Balloon]:
        if self.project is None or self.drawing is None:
            return []
        balloons = self.project.balloons_for(self.drawing.id)

        status_filter = self.filter_combo.currentText()
        if status_filter == "Pending":
            balloons = [b for b in balloons if b.status == ReviewStatus.PENDING.value]
        elif status_filter == "Accepted":
            balloons = [b for b in balloons if b.status == ReviewStatus.ACCEPTED.value]
        elif status_filter == "Edited":
            balloons = [b for b in balloons if b.status == ReviewStatus.EDITED.value]
        elif status_filter == "Rejected":
            balloons = [b for b in balloons if b.status == ReviewStatus.REJECTED.value]
        elif status_filter == "Manual":
            balloons = [b for b in balloons if b.source == BalloonSource.MANUAL.value]
        elif status_filter == "Auto":
            balloons = [b for b in balloons if b.source == BalloonSource.AUTO.value]

        search = self.search_edit.text().strip().lower()
        if search:
            def matches(b: Balloon) -> bool:
                haystacks = [str(b.number), (b.raw_text or "").lower(), (b.note or "").lower()]
                if b.nominal is not None:
                    haystacks.append(str(b.nominal))
                return any(search in h for h in haystacks)
            balloons = [b for b in balloons if matches(b)]

        sort_key = self.sort_combo.currentText()
        if sort_key == "Page":
            balloons.sort(key=lambda b: (b.page_number, b.number))
        elif sort_key == "Number":
            balloons.sort(key=lambda b: b.number)
        elif sort_key == "Confidence":
            balloons.sort(key=lambda b: b.confidence, reverse=True)
        elif sort_key == "Status":
            balloons.sort(key=lambda b: b.status)
        return balloons

    @staticmethod
    def _format_tolerance(b: Balloon) -> str:
        if b.tol_plus is not None and b.tol_minus is not None:
            if abs(b.tol_plus - b.tol_minus) < 1e-9:
                return f"±{b.tol_plus}"
            # tol_minus is defined via lower_limit = nominal - tol_minus, so
            # the deviation as it would actually be written on the drawing
            # is -tol_minus -- almost always positive (the common "+X/-Y"
            # case), but a same-sign stacked tolerance ("+.3 over +.1", see
            # ocr_parser._extract_leading_tolerance) stores tol_minus
            # negative to keep that formula correct, and must still display
            # with its own true sign ("+0.3/+0.1"), not a hardcoded "-".
            return f"{b.tol_plus:+g}/{-b.tol_minus:+g}"
        if b.lower_limit is not None and b.upper_limit is not None:
            return f"{b.lower_limit} to {b.upper_limit}"
        return ""

    def _refresh_review_table(self) -> None:
        balloons = self._filtered_sorted_balloons()
        table = self.review_table
        table.setRowCount(len(balloons))
        for row, b in enumerate(balloons):
            values = [
                str(b.number),
                str(b.page_number + 1),
                CharacteristicType.display_name(b.char_type, b.gdt_symbol),
                b.raw_text,
                "" if b.nominal is None else str(b.nominal),
                self._format_tolerance(b),
                b.inspection_method,
                b.status.capitalize(),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, b.id)
                table.setItem(row, col, item)
            method_combo = QComboBox(table)
            method_combo.addItems(COMMON_INSPECTION_METHODS)
            if method_combo.findText(b.inspection_method) < 0:
                method_combo.addItem(b.inspection_method)
            method_combo.setCurrentText(b.inspection_method)
            method_combo.setToolTip("Select an inspection method")
            method_combo.textActivated.connect(
                lambda method, balloon_id=b.id: self._set_inspection_method(balloon_id, method)
            )
            table.setCellWidget(row, 6, method_combo)
        table.resizeColumnsToContents()

    def _set_inspection_method(self, balloon_id: str, method: str) -> None:
        balloon = self._find_balloon(balloon_id)
        if self.project is None or balloon is None or balloon.inspection_method == method:
            return
        before = {"inspection_method": balloon.inspection_method, "modified_at": balloon.modified_at}
        balloon.inspection_method = method
        balloon.touch()
        after = {"inspection_method": balloon.inspection_method, "modified_at": balloon.modified_at}
        self.undo_stack.push(BalloonFieldChangeCommand(
            self.project, [balloon_id], {balloon_id: before}, {balloon_id: after},
            self._refresh_all, text="Change Inspection Method",
        ))

    def _set_status_for(self, ids: list[str], status: str, text: str) -> None:
        if not ids or self.project is None:
            return
        before, after = {}, {}
        for bid in ids:
            b = self._find_balloon(bid)
            if b is None:
                continue
            before[bid] = {"status": b.status}
            b.status = status
            b.touch()
            after[bid] = {"status": b.status}
        if after:
            cmd = BalloonFieldChangeCommand(self.project, list(after.keys()), before, after, self._refresh_all, text=text)
            self.undo_stack.push(cmd)
        self._refresh_all()

    def _accept_selected(self) -> None:
        self._set_status_for(self._selected_balloon_ids(), ReviewStatus.ACCEPTED.value, "Accept Balloon(s)")

    def _reject_selected(self) -> None:
        self._set_status_for(self._selected_balloon_ids(), ReviewStatus.REJECTED.value, "Reject Balloon(s)")

    def _accept_all_above_threshold(self) -> None:
        if self.project is None or self.drawing is None:
            return
        dialog = ConfidenceThresholdDialog(self.settings.confidence_threshold, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        threshold = dialog.threshold()
        balloons = self.project.balloons_for(self.drawing.id)
        ids = [b.id for b in balloons if b.status == ReviewStatus.PENDING.value and b.confidence >= threshold]
        if not ids:
            QMessageBox.information(self, "Nothing To Accept", "No pending proposals meet that confidence threshold.")
            return
        self._set_status_for(ids, ReviewStatus.ACCEPTED.value, "Accept All Above Threshold")

    def _reject_all_selected_or_pending(self) -> None:
        if self.project is None or self.drawing is None:
            return
        ids = self._selected_balloon_ids()
        if not ids:
            balloons = self.project.balloons_for(self.drawing.id)
            ids = [b.id for b in balloons if b.status == ReviewStatus.PENDING.value]
        if not ids:
            QMessageBox.information(self, "Nothing To Reject", "No selected or pending balloons to reject.")
            return
        self._set_status_for(ids, ReviewStatus.REJECTED.value, "Reject Balloon(s)")

    def _edit_selected_balloon(self) -> None:
        ids = self._selected_balloon_ids()
        if ids:
            self._edit_balloon(ids[0])

    def _edit_balloon(self, balloon_id: str) -> None:
        balloon = self._find_balloon(balloon_id)
        if balloon is None or self.project is None:
            return
        before = balloon.to_dict()
        dialog = BalloonEditDialog(balloon, is_new=False, default_tolerances=self._current_default_tolerances(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        dialog.apply_to_balloon(balloon)
        if balloon.source == BalloonSource.AUTO.value and balloon.status in (
            ReviewStatus.PENDING.value, ReviewStatus.ACCEPTED.value,
        ):
            if not balloon.is_unchanged_from_prediction():
                balloon.status = ReviewStatus.EDITED.value
        after = balloon.to_dict()
        cmd = BalloonFieldChangeCommand(self.project, [balloon.id], {balloon.id: before}, {balloon.id: after}, self._refresh_all, text="Edit Balloon")
        self.undo_stack.push(cmd)
        self._refresh_all()

    def _add_manual_balloon(self) -> None:
        if self.project is None or self.drawing is None:
            QMessageBox.information(self, "No Drawing", "Add a PDF drawing first.")
            return
        try:
            width, height = self.pdf_doc.page_size_pdf(self.current_page)
        except Exception:
            width, height = 612.0, 792.0
        balloon = Balloon(
            number=self.project.next_balloon_number(self.drawing.id),
            drawing_id=self.drawing.id,
            page_number=self.current_page,
            x=width / 2.0, y=height / 2.0,
            source=BalloonSource.MANUAL.value,
            status=ReviewStatus.ACCEPTED.value,
        )
        dialog = BalloonEditDialog(balloon, is_new=True, default_tolerances=self._current_default_tolerances(), parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply_to_balloon(balloon)
            cmd = AddBalloonsCommand(self.project, [balloon], self._refresh_all, text="Add Balloon")
            self.undo_stack.push(cmd)
            self._refresh_all()

    def _duplicate_selected_balloon(self) -> None:
        ids = self._selected_balloon_ids()
        if not ids or self.project is None or self.drawing is None:
            return
        original = self._find_balloon(ids[0])
        if original is None:
            return
        new_balloon = Balloon.from_dict(original.to_dict())
        new_balloon.id = new_id()
        new_balloon.number = self.project.next_balloon_number(self.drawing.id)
        new_balloon.x += 20.0
        new_balloon.y += 20.0
        new_balloon.source = BalloonSource.MANUAL.value
        new_balloon.status = ReviewStatus.ACCEPTED.value
        new_balloon.original_prediction = None
        cmd = AddBalloonsCommand(self.project, [new_balloon], self._refresh_all, text="Duplicate Balloon")
        self.undo_stack.push(cmd)
        self._refresh_all()

    def _split_selected_balloon(self) -> None:
        """Re-run auto-detection on the selected balloon's raw text and, if it
        now recognizes more than one characteristic, replace the balloon with
        one properly-typed balloon per characteristic (e.g. a hole's diameter
        and its depth, packed into one line of drawing text).

        Detection can only split what it can recognize; when the drawing's
        symbols/text don't resolve into more than one characteristic, this
        reports that plainly instead of guessing -- use Duplicate, then Edit
        each copy, for a fully manual split.
        """
        ids = self._selected_balloon_ids()
        if not ids or self.project is None or self.drawing is None:
            return
        original = self._find_balloon(ids[0])
        if original is None:
            return

        parsed_list = parse_characteristics(original.raw_text) if original.raw_text else []
        if len(parsed_list) < 2:
            QMessageBox.information(
                self, "Nothing To Split",
                "Automatic detection only recognizes one characteristic in this balloon's "
                "raw drawing text, so there's nothing to split automatically.\n\n"
                "Use Duplicate, then Edit each copy, to split it manually.",
            )
            return

        new_balloons: list[Balloon] = []
        for i, parsed in enumerate(parsed_list):
            b = Balloon.from_dict(original.to_dict())
            b.id = new_id()
            b.number = original.number if i == 0 else self.project.next_balloon_number(self.drawing.id)
            b.x = original.x + (i * 20.0)
            b.y = original.y + (i * 20.0)
            b.char_type = parsed.char_type
            b.nominal = parsed.nominal
            b.tol_plus = parsed.tol_plus
            b.tol_minus = parsed.tol_minus
            b.lower_limit = parsed.lower_limit
            b.upper_limit = parsed.upper_limit
            b.gdt_symbol = parsed.gdt_symbol
            b.gdt_tolerance = parsed.gdt_tolerance
            b.material_condition = parsed.material_condition
            b.datums = parsed.datums
            b.surface_finish = parsed.surface_finish
            b.thread_callout = parsed.thread_callout
            b.note = parsed.note or original.note
            b.original_prediction = None
            b.touch()
            new_balloons.append(b)

        cmd = SplitBalloonCommand(self.project, original, new_balloons, self._refresh_all)
        self.undo_stack.push(cmd)
        self._refresh_all()

    def _delete_selected_balloons(self) -> None:
        ids = self._selected_balloon_ids()
        if not ids or self.project is None:
            return
        balloons = [b for b in self.project.balloons if b.id in ids]
        if not balloons:
            return
        reply = QMessageBox.question(
            self, "Delete Balloon(s)", f"Delete {len(balloons)} balloon(s)? This can be undone with Ctrl+Z.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cmd = DeleteBalloonsCommand(self.project, balloons, self._refresh_all, text="Delete Balloon(s)")
        self.undo_stack.push(cmd)
        self._refresh_all()

    def _show_renumber_dialog(self) -> None:
        if self.project is None or self.drawing is None:
            QMessageBox.information(self, "No Drawing", "Add a PDF drawing first.")
            return
        dialog = RenumberDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_renumber(dialog.mode())

    def _apply_renumber(self, mode: str) -> None:
        balloons = self.project.balloons_for(self.drawing.id)
        if not balloons:
            return
        before = {b.id: {"number": b.number} for b in balloons}

        if mode == RenumberDialog.MODE_PAGE:
            target = [b for b in balloons if b.page_number == self.current_page]
            target.sort(key=lambda b: (round(b.y / 20.0), b.x))
        elif mode == RenumberDialog.MODE_DRAWING:
            target = sorted(balloons, key=lambda b: (b.page_number, round(b.y / 20.0), b.x))
        else:  # preserve
            target = sorted(balloons, key=lambda b: b.number)

        for i, b in enumerate(target, start=1):
            b.number = i
            b.touch()

        after = {b.id: {"number": b.number} for b in balloons}
        cmd = BalloonFieldChangeCommand(self.project, list(after.keys()), before, after, self._refresh_all, text="Renumber Balloons")
        self.undo_stack.push(cmd)
        self._refresh_all()

    def _on_review_rows_dropped(self, source_row: int, target_row: int) -> None:
        """Handle a drag-and-drop row reorder in the review table: renumber
        every balloon for this drawing to match the row's new position.

        Only safe when the table is showing every balloon for the drawing
        (Filter = "All") -- a filtered view is a subset, and renumbering it
        to 1..N would collide with the numbers of the balloons hidden by
        the filter.
        """
        if self.project is None or self.drawing is None:
            return
        if self.filter_combo.currentText() != "All":
            self.statusBar().showMessage('Set Filter to "All" to drag-reorder balloon numbers.', 6000)
            return

        displayed = self._filtered_sorted_balloons()
        if source_row >= len(displayed) or target_row >= len(displayed):
            return

        ordered_ids = [b.id for b in displayed]
        moved_id = ordered_ids.pop(source_row)
        ordered_ids.insert(target_row, moved_id)

        by_id = {b.id: b for b in displayed}
        before = {bid: {"number": by_id[bid].number} for bid in ordered_ids}
        for i, bid in enumerate(ordered_ids, start=1):
            by_id[bid].number = i
            by_id[bid].touch()
        after = {bid: {"number": by_id[bid].number} for bid in ordered_ids}

        cmd = BalloonFieldChangeCommand(self.project, ordered_ids, before, after, self._refresh_all, text="Reorder Balloons")
        self.undo_stack.push(cmd)
        self._refresh_all()

    def _focus_review_panel(self) -> None:
        self.search_edit.setFocus()

    # ------------------------------------------------------------------
    # Auto-ballooning
    # ------------------------------------------------------------------
    def _current_default_tolerances(self) -> DefaultTolerances:
        """The drawing's currently-configured default tolerance table (empty
        if none is set yet or there's no drawing) -- read-only, never
        prompts. Used to offer "Use Default Tolerance" in the balloon edit
        dialog regardless of whether auto-ballooning has run yet.
        """
        if self.drawing is None:
            return DefaultTolerances()
        return DefaultTolerances(
            by_decimal_places=dict(self.drawing.tol_by_decimal_places),
            angular=self.drawing.tol_angular,
        )

    def _ensure_default_tolerances(self) -> DefaultTolerances:
        """The first time a drawing is auto-ballooned, offer to set its
        general/default tolerance table (best-effort auto-detected from the
        current page's title block, reviewable/editable before use), then
        remember the choice on the Drawing so this isn't asked again.
        """
        if self.drawing is None:
            return DefaultTolerances()
        if self.drawing.tolerances_configured:
            return self._current_default_tolerances()

        page_text = ""
        if self.pdf_doc is not None:
            try:
                blocks = self.pdf_doc.extract_text_blocks(self.current_page)
                page_text = "\n".join(b.text for b in blocks)
            except Exception:
                logger.exception("Failed to extract page text for default-tolerance detection")
        detected = parse_default_tolerances(page_text)

        dialog = DefaultTolerancesDialog(
            detected.by_decimal_places, detected.angular,
            auto_detected=not detected.is_empty(), parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return DefaultTolerances()  # skip for this run only -- ask again next time

        self._apply_tolerance_dialog_values(dialog)
        return self._current_default_tolerances()

    def _apply_tolerance_dialog_values(self, dialog: DefaultTolerancesDialog) -> None:
        values = dialog.values()
        self.drawing.tol_by_decimal_places = values["tol_by_decimal_places"]
        self.drawing.tol_angular = values["tol_angular"]
        self.drawing.tolerances_configured = True
        self._mark_dirty()

    def _edit_default_tolerances(self) -> None:
        if self.project is None or self.drawing is None:
            QMessageBox.information(self, "No Drawing", "Open or add a PDF drawing first.")
            return
        dialog = DefaultTolerancesDialog(
            self.drawing.tol_by_decimal_places, self.drawing.tol_angular,
            auto_detected=self.drawing.tolerances_configured, parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_tolerance_dialog_values(dialog)

    def _auto_balloon_current_page(self) -> None:
        if self.project is None or self.drawing is None or self.pdf_doc is None:
            QMessageBox.information(self, "No Drawing", "Open or add a PDF drawing first.")
            return
        source_path = resolve_source_path(self.drawing)
        if source_path is None:
            QMessageBox.warning(self, "Drawing Not Found", "Cannot locate the source PDF. Relink the drawing first.")
            return
        default_tolerances = self._ensure_default_tolerances()
        existing = self.project.balloons_for(self.drawing.id)
        start_number = self.project.next_balloon_number(self.drawing.id)
        self._run_background(
            _auto_balloon_page_task, self._on_auto_balloon_page_done, "Auto-ballooning current page...",
            source_path, self.drawing.id, self.current_page, existing, start_number,
            self.settings.auto_balloon_dpi, self.settings.effective_tesseract_path(), default_tolerances,
        )

    def _on_auto_balloon_page_done(self, result: AutoBalloonResult) -> None:
        if result.balloons and self.project is not None:
            cmd = AddBalloonsCommand(self.project, result.balloons, self._refresh_all, text="Auto-Balloon Page")
            self.undo_stack.push(cmd)
        self._refresh_all()
        message = result.message or f"Auto-balloon complete: {len(result.balloons)} proposal(s)."
        self.statusBar().showMessage(message, 8000)
        if not result.ocr_available:
            QMessageBox.information(self, "OCR Unavailable", message)

    def _auto_balloon_entire_drawing(self) -> None:
        if self.project is None or self.drawing is None or self.pdf_doc is None:
            QMessageBox.information(self, "No Drawing", "Open or add a PDF drawing first.")
            return
        source_path = resolve_source_path(self.drawing)
        if source_path is None:
            QMessageBox.warning(self, "Drawing Not Found", "Cannot locate the source PDF. Relink the drawing first.")
            return
        reply = QMessageBox.question(
            self, "Auto-Balloon Entire Drawing",
            f"Scan all {self.drawing.page_count} page(s) for likely characteristics?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        default_tolerances = self._ensure_default_tolerances()
        existing = self.project.balloons_for(self.drawing.id)
        start_number = self.project.next_balloon_number(self.drawing.id)
        self._run_background(
            _auto_balloon_drawing_task, self._on_auto_balloon_drawing_done, "Auto-ballooning entire drawing...",
            source_path, self.drawing.id, self.drawing.page_count, existing, start_number,
            self.settings.auto_balloon_dpi, self.settings.effective_tesseract_path(), default_tolerances,
        )

    def _on_auto_balloon_drawing_done(self, results: list[AutoBalloonResult]) -> None:
        all_balloons = [b for r in results for b in r.balloons]
        if all_balloons and self.project is not None:
            cmd = AddBalloonsCommand(self.project, all_balloons, self._refresh_all, text="Auto-Balloon Entire Drawing")
            self.undo_stack.push(cmd)
        self._refresh_all()
        pages_without_ocr = sum(1 for r in results if not r.ocr_available)
        message = f"Auto-balloon complete: {len(all_balloons)} proposal(s) across {len(results)} page(s)."
        if pages_without_ocr:
            message += f" {pages_without_ocr} page(s) had no selectable text and OCR was unavailable."
        self.statusBar().showMessage(message, 10000)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _default_export_dir(self) -> str:
        if self.project and self.project.file_path:
            return str(Path(self.project.file_path).parent)
        return str(PROJECTS_DIR)

    def _export_excel(self) -> None:
        if self.project is None or self.drawing is None:
            QMessageBox.information(self, "No Drawing", "Open or add a PDF drawing first.")
            return
        dialog = ExportExcelOptionsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        include_pending = dialog.include_pending()
        default_name = f"{Path(self.drawing.file_name).stem}_inspection.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel Inspection Sheet", str(Path(self._default_export_dir()) / default_name), "Excel Workbook (*.xlsx)"
        )
        if not path:
            return
        balloons = self.project.balloons_for(self.drawing.id)
        self._run_background(
            export_excel, self._on_export_excel_done, f"Exporting Excel to {Path(path).name}...",
            self.project, self.drawing, balloons, path, include_pending,
        )

    def _on_export_excel_done(self, result_path: Path) -> None:
        self.statusBar().showMessage(f"Excel exported to {result_path}", 6000)
        QMessageBox.information(self, "Export Complete", f"Excel inspection sheet saved:\n{result_path}")

    def _export_ballooned_pdf(self) -> None:
        if self.project is None or self.drawing is None:
            QMessageBox.information(self, "No Drawing", "Open or add a PDF drawing first.")
            return
        dialog = ExportPdfOptionsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        include_pending, include_rejected = dialog.options()
        default_name = f"{Path(self.drawing.file_name).stem}_ballooned.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Ballooned PDF", str(Path(self._default_export_dir()) / default_name), "PDF Files (*.pdf)"
        )
        if not path:
            return
        balloons = self.project.balloons_for(self.drawing.id)
        self._run_background(
            export_ballooned_pdf, self._on_export_pdf_done, f"Exporting ballooned PDF to {Path(path).name}...",
            self.drawing, balloons, path, include_pending, include_rejected,
            balloon_size_percent=self.settings.balloon_size_percent,
        )

    def _on_export_pdf_done(self, result: PdfExportResult) -> None:
        message = f"Ballooned PDF exported to {result.output_path} ({result.balloon_count} balloon(s))."
        if result.used_raster_fallback:
            message += "\n\nNote: " + result.message
        self.statusBar().showMessage("Ballooned PDF exported.", 6000)
        QMessageBox.information(self, "Export Complete", message)

    def _show_teach_dialog(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "No Project", "Open or create a project first.")
            return
        stats = compute_teach_stats(self.project)
        dialog = TeachTrainingDialog(stats, on_export=self._start_training_export, parent=self)
        dialog.exec()

    def _start_training_export(self) -> None:
        if self.project is None:
            return
        self._run_background(export_training_dataset, self._on_training_export_done, "Exporting training dataset...", self.project)

    def _on_training_export_done(self, result: TrainingExportResult) -> None:
        message = (
            f"Training dataset exported to {result.output_root}\n\n"
            f"Images: {result.image_count}\n"
            f"Label files: {result.label_count} ({result.positive_label_count} positive boxes)\n"
            f"Crops: {result.crop_count}"
        )
        if result.skipped_drawings:
            message += f"\n\nSkipped (source PDF not found): {', '.join(result.skipped_drawings)}"
        self.statusBar().showMessage("Training dataset export complete.", 6000)
        QMessageBox.information(self, "Export Complete", message)

    # ------------------------------------------------------------------
    # Settings / About
    # ------------------------------------------------------------------
    def _show_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply_to_settings(self.settings)
            self.settings.save()
            self.statusBar().showMessage("Settings saved.", 3000)

    def _show_about_dialog(self) -> None:
        AboutDialog(self).exec()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self.pdf_view.shutdown()
        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
        self.settings.save()
        event.accept()


def main() -> int:
    from balloon_app.theme import apply_theme

    setup_logging()
    app = QApplication(sys.argv)
    apply_theme(app, AppSettings.load().theme)
    app.setApplicationName("BalloonApp")
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "resources" / "balloonapp.ico")))
    window = MainWindow()
    window.show()
    return app.exec()
