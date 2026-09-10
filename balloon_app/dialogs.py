"""All secondary dialogs used by the main window.

Kept in a single module (rather than one file per dialog) since each is
small; if this grows significantly, split by concern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from balloon_app import __version__
from balloon_app.config import AppSettings, DEFAULT_CONFIDENCE_THRESHOLD, resource_path
from balloon_app.data_model import Balloon, CharacteristicType, ReviewStatus
from balloon_app.ocr_parser import DefaultTolerances, compute_limits, decimal_places
from balloon_app.training_export import TeachStats

COMMON_INSPECTION_METHODS = [
    "", "Caliper", "Micrometer", "CMM", "Height Gauge", "Visual", "Pin Gauge",
    "Protractor", "GO/NO-GO", "Surface Roughness Tester",
]


class NewProjectDialog(QDialog):
    """Collects metadata for a brand-new project."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(420)

        self.name_edit = QLineEdit("Untitled Project")
        self.part_number_edit = QLineEdit()
        self.part_name_edit = QLineEdit()
        self.revision_edit = QLineEdit()
        self.customer_edit = QLineEdit()
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("Inches (in)", "in")
        self.unit_combo.addItem("Millimeters (mm)", "mm")
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setFixedHeight(70)

        unit_hint = QLabel(
            "Sets the unit of measure for every dimension on this drawing "
            "(shown as \"UoM\" on the exported FAIR). Choose it now, before ballooning."
        )
        unit_hint.setWordWrap(True)
        unit_hint.setStyleSheet("color: gray; font-size: 10px;")

        form = QFormLayout()
        form.addRow("Project Name:", self.name_edit)
        form.addRow("Part Number:", self.part_number_edit)
        form.addRow("Part Name:", self.part_name_edit)
        form.addRow("Revision:", self.revision_edit)
        form.addRow("Customer:", self.customer_edit)
        form.addRow("Unit of Measure:", self.unit_combo)
        form.addRow("", unit_hint)
        form.addRow("Notes:", self.notes_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Missing Name", "Please enter a project name.")
            return
        self.accept()

    def values(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "part_number": self.part_number_edit.text().strip(),
            "part_name": self.part_name_edit.text().strip(),
            "revision": self.revision_edit.text().strip(),
            "customer": self.customer_edit.text().strip(),
            "unit": self.unit_combo.currentData(),
            "notes": self.notes_edit.toPlainText().strip(),
        }


class ProjectPropertiesDialog(NewProjectDialog):
    """Same fields as :class:`NewProjectDialog`, pre-populated for editing."""

    def __init__(
        self,
        name: str,
        part_number: str,
        part_name: str,
        revision: str,
        customer: str,
        unit: str,
        notes: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Project Properties")
        self.name_edit.setText(name)
        self.part_number_edit.setText(part_number)
        self.part_name_edit.setText(part_name)
        self.revision_edit.setText(revision)
        self.customer_edit.setText(customer)
        idx = self.unit_combo.findData(unit)
        self.unit_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.notes_edit.setPlainText(notes)


def _optional_float_line_edit() -> QLineEdit:
    edit = QLineEdit()
    validator = QDoubleValidator()
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)
    edit.setValidator(validator)
    return edit


def _parse_optional_float(edit: QLineEdit) -> Optional[float]:
    text = edit.text().strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _set_optional_float(edit: QLineEdit, value: Optional[float]) -> None:
    edit.setText("" if value is None else _format_number(value))


def _format_number(value: float) -> str:
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return text if text else "0"


class _ToleranceRow(QWidget):
    """One "X.<n decimals> ±<tolerance>" row in :class:`DefaultTolerancesDialog`,
    with its own remove button."""

    removed = pyqtSignal(object)

    def __init__(self, decimal_places: int, tolerance: Optional[float], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._prefix = QLabel()
        layout.addWidget(self._prefix)
        self.places_spin = QSpinBox()
        self.places_spin.setRange(0, 6)
        self.places_spin.setValue(decimal_places)
        self.places_spin.setToolTip(
            "Number of decimal places this tolerance applies to (0 = whole number, e.g. \"30\")"
        )
        self.places_spin.valueChanged.connect(self._update_prefix)
        self._update_prefix(decimal_places)
        layout.addWidget(self.places_spin)
        layout.addWidget(QLabel("decimal(s) ±"))

        self.tol_edit = _optional_float_line_edit()
        _set_optional_float(self.tol_edit, tolerance)
        layout.addWidget(self.tol_edit, 1)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.setStyleSheet("QPushButton { padding: 0px; font-size: 16px; }")
        remove_btn.setAccessibleName("Remove tolerance row")
        remove_btn.setToolTip("Remove this row")
        remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(remove_btn)

    def _update_prefix(self, places: int) -> None:
        self._prefix.setText("X" if places == 0 else "X.")

    def decimal_places(self) -> int:
        return self.places_spin.value()

    def tolerance(self) -> Optional[float]:
        return _parse_optional_float(self.tol_edit)


class DefaultTolerancesDialog(QDialog):
    """Collects a drawing's general/default tolerance table -- the
    "TOLERANCES UNLESS OTHERWISE NOTED" convention, keyed by decimal-place
    count, plus a separate angular tolerance. Applied automatically, during
    auto-ballooning, to any detected dimension that has no explicit
    tolerance of its own.

    The decimal-place tiers are an open-ended, user-editable list (rather
    than a fixed X.X/X.XX/X.XXX set) since drawings vary in how many tiers
    they define.
    """

    def __init__(
        self,
        by_decimal_places: Optional[dict[int, float]] = None,
        angular: Optional[float] = None,
        auto_detected: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Default Tolerances")
        self.setMinimumWidth(440)

        intro_text = (
            "Detected from this drawing's title block -- review and adjust if needed."
            if auto_detected else
            "Not found on this drawing (or it has no title-block tolerance note). "
            "Enter values matching the drawing's convention, or leave any of them blank."
        )
        intro = QLabel(
            intro_text + "\n\nApplied automatically to any auto-detected dimension that doesn't "
            "carry its own explicit tolerance, based on its number of decimal places."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")

        self._rows: list[_ToleranceRow] = []
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(4)

        add_btn = QPushButton("+ Add Decimal Place")
        add_btn.clicked.connect(lambda: self._add_row())

        self.angular_edit = _optional_float_line_edit()
        _set_optional_float(self.angular_edit, angular)
        angular_form = QFormLayout()
        angular_form.addRow("Angles ±", self.angular_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(self.rows_layout)
        layout.addWidget(add_btn)
        layout.addLayout(angular_form)
        layout.addWidget(buttons)

        by_decimal_places = by_decimal_places or {}
        for places in sorted(by_decimal_places):
            self._add_row(places, by_decimal_places[places])
        if not by_decimal_places:
            for places in (0, 1, 2, 3):
                self._add_row(places, None)

    def _add_row(self, decimal_places: Optional[int] = None, tolerance: Optional[float] = None) -> None:
        used = {r.decimal_places() for r in self._rows}
        if decimal_places is None or decimal_places in used:
            decimal_places = next((p for p in range(7) if p not in used), decimal_places or 0)
        row = _ToleranceRow(decimal_places, tolerance)
        row.removed.connect(self._remove_row)
        self._rows.append(row)
        self.rows_layout.addWidget(row)

    def _remove_row(self, row: "_ToleranceRow") -> None:
        self._rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def values(self) -> dict:
        by_places: dict[int, float] = {}
        for row in self._rows:
            tol = row.tolerance()
            if tol is not None:
                by_places[row.decimal_places()] = tol
        return {
            "tol_by_decimal_places": by_places,
            "tol_angular": _parse_optional_float(self.angular_edit),
        }


class LearnedSymbolsDialog(QDialog):
    """Review/manage the marker-letter -> characteristic-type mappings
    BalloonIQ has learned from confirmed corrections (see
    AppSettings.learn_symbol).

    When a drawing's font substitutes a real dimensioning symbol (Ø, ▼, ⌵,
    ⌴, □) with an unrelated letter and that guess gets corrected, BalloonIQ
    can remember what the letter actually means -- global across every
    project, since the same CAD export tool/font produces the same
    mangling everywhere. This dialog is purely for transparency and
    cleanup: forgetting an entry here doesn't undo anything already
    ballooned, it just stops that marker from being auto-resolved that way
    on future drawings.
    """

    def __init__(self, learned_symbols: dict[str, str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Learned Symbols")
        self.setMinimumWidth(420)
        self._forgotten: set[str] = set()

        intro = QLabel(
            "Symbols BalloonIQ has learned from your corrections, applied automatically across "
            "every project from now on. Select one and click \"Forget Selected\" to stop applying it."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Marker", "Means"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for marker, char_type in sorted(learned_symbols.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(marker))
            self.table.setItem(row, 1, QTableWidgetItem(CharacteristicType.display_name(char_type)))

        if learned_symbols:
            empty_label = None
        else:
            empty_label = QLabel("Nothing learned yet -- correct a balloon's type when you spot a "
                                  "mangled symbol and BalloonIQ will offer to remember it.")
            empty_label.setWordWrap(True)
            empty_label.setStyleSheet("color: gray;")

        forget_btn = QPushButton("Forget Selected")
        forget_btn.clicked.connect(self._forget_selected)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        if empty_label is not None:
            layout.addWidget(empty_label)
        layout.addWidget(self.table)
        layout.addWidget(forget_btn)
        layout.addWidget(buttons)

    def _forget_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            marker_item = self.table.item(row, 0)
            if marker_item:
                self._forgotten.add(marker_item.text())
            self.table.removeRow(row)

    def forgotten_markers(self) -> set[str]:
        return self._forgotten


class BalloonEditDialog(QDialog):
    """Full edit form for one balloon's characteristic data.

    Used both for manual "Add Balloon" (with a fresh :class:`Balloon`) and
    for editing/reviewing an existing (possibly auto-proposed) balloon.
    """

    def __init__(
        self,
        balloon: Balloon,
        is_new: bool,
        default_tolerances: Optional[DefaultTolerances] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Add Balloon" if is_new else f"Edit Balloon #{balloon.number}")
        self.setMinimumWidth(480)
        self._balloon = balloon
        self._is_new = is_new
        self._default_tolerances = default_tolerances

        self.char_type_combo = QComboBox()
        for ct in CharacteristicType:
            self.char_type_combo.addItem(CharacteristicType.display_name(ct.value), ct.value)
        self._set_combo_by_data(self.char_type_combo, balloon.char_type)

        self.raw_text_edit = QLineEdit(balloon.raw_text)
        self.nominal_edit = _optional_float_line_edit()
        _set_optional_float(self.nominal_edit, balloon.nominal)
        self.tol_plus_edit = _optional_float_line_edit()
        _set_optional_float(self.tol_plus_edit, balloon.tol_plus)
        self.tol_minus_edit = _optional_float_line_edit()
        _set_optional_float(self.tol_minus_edit, balloon.tol_minus)
        self.lower_limit_edit = _optional_float_line_edit()
        _set_optional_float(self.lower_limit_edit, balloon.lower_limit)
        self.upper_limit_edit = _optional_float_line_edit()
        _set_optional_float(self.upper_limit_edit, balloon.upper_limit)

        calc_button = QPushButton("Calculate Limits from Nominal ± Tolerance")
        calc_button.clicked.connect(self._calculate_limits)

        default_tol_button = QPushButton("Use Default Tolerance")
        default_tol_button.setToolTip(
            "Fill Tolerance +/- from this drawing's default tolerance table (Tools -> Default "
            "Tolerances...), matched by the Nominal's decimal places (or by the angular default "
            "for an Angle characteristic)."
        )
        default_tol_button.clicked.connect(self._apply_default_tolerance)

        self.gdt_symbol_edit = QLineEdit(balloon.gdt_symbol or "")
        self.gdt_tolerance_edit = QLineEdit(balloon.gdt_tolerance or "")
        self.material_condition_edit = QLineEdit(balloon.material_condition or "")
        self.datums_edit = QLineEdit(balloon.datums or "")
        self.surface_finish_edit = QLineEdit(balloon.surface_finish or "")
        self.thread_callout_edit = QLineEdit(balloon.thread_callout or "")

        self.note_edit = QPlainTextEdit(balloon.note)
        self.note_edit.setFixedHeight(60)

        self.inspection_method_combo = QComboBox()
        self.inspection_method_combo.setEditable(True)
        self.inspection_method_combo.addItems(COMMON_INSPECTION_METHODS)
        self.inspection_method_combo.setCurrentText(balloon.inspection_method)

        self.critical_check = QCheckBox("Critical Characteristic")
        self.critical_check.setChecked(balloon.critical)

        self.status_combo = QComboBox()
        for status in ReviewStatus:
            self.status_combo.addItem(status.value.capitalize(), status.value)
        self._set_combo_by_data(self.status_combo, balloon.status)

        info_label = QLabel(
            f"Source: {balloon.source}   |   Confidence: {balloon.confidence:.2f}   |   "
            f"Model: {balloon.model_version or 'n/a'}"
        )
        info_label.setStyleSheet("color: gray;")

        form = QFormLayout()
        form.addRow("Characteristic Type:", self.char_type_combo)
        form.addRow("Raw Drawing Text:", self.raw_text_edit)
        form.addRow("Nominal:", self.nominal_edit)
        form.addRow("Tolerance +:", self.tol_plus_edit)
        form.addRow("Tolerance -:", self.tol_minus_edit)
        form.addRow("Lower Limit:", self.lower_limit_edit)
        form.addRow("Upper Limit:", self.upper_limit_edit)
        form.addRow("", calc_button)
        form.addRow("", default_tol_button)
        form.addRow("GD&T Symbol:", self.gdt_symbol_edit)
        form.addRow("GD&T Tolerance:", self.gdt_tolerance_edit)
        form.addRow("Material Condition:", self.material_condition_edit)
        form.addRow("Datums:", self.datums_edit)
        form.addRow("Surface Finish:", self.surface_finish_edit)
        form.addRow("Thread Callout:", self.thread_callout_edit)
        form.addRow("Note / Description:", self.note_edit)
        form.addRow("Inspection Method:", self.inspection_method_combo)
        form.addRow("", self.critical_check)
        form.addRow("Review Status:", self.status_combo)
        form.addRow(info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, data_value: str) -> None:
        idx = combo.findData(data_value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _calculate_limits(self) -> None:
        nominal = _parse_optional_float(self.nominal_edit)
        tol_plus = _parse_optional_float(self.tol_plus_edit)
        tol_minus = _parse_optional_float(self.tol_minus_edit)
        lower, upper = compute_limits(nominal, tol_plus, tol_minus)
        if lower is not None:
            _set_optional_float(self.lower_limit_edit, lower)
        if upper is not None:
            _set_optional_float(self.upper_limit_edit, upper)

    def _apply_default_tolerance(self) -> None:
        if self._default_tolerances is None or self._default_tolerances.is_empty():
            QMessageBox.information(
                self, "No Default Tolerance",
                "No default tolerance table is set for this drawing.\n\n"
                "Set one via Tools -> Default Tolerances...",
            )
            return

        char_type = self.char_type_combo.currentData()
        if char_type in (CharacteristicType.ANGLE.value, CharacteristicType.CHAMFER_ANGLE.value):
            tol = self._default_tolerances.angular
        else:
            tol = self._default_tolerances.for_decimal_places(decimal_places(self.nominal_edit.text().strip()))

        if tol is None:
            QMessageBox.information(
                self, "No Matching Default",
                "This drawing's default tolerance table has no entry for this value's "
                "decimal places (or, for an angle, no angular default).",
            )
            return

        _set_optional_float(self.tol_plus_edit, tol)
        _set_optional_float(self.tol_minus_edit, tol)
        self._calculate_limits()

    def apply_to_balloon(self, balloon: Balloon) -> None:
        """Write the dialog's fields back into ``balloon`` in place."""
        balloon.char_type = self.char_type_combo.currentData()
        balloon.raw_text = self.raw_text_edit.text().strip()
        balloon.nominal = _parse_optional_float(self.nominal_edit)
        balloon.tol_plus = _parse_optional_float(self.tol_plus_edit)
        balloon.tol_minus = _parse_optional_float(self.tol_minus_edit)
        balloon.lower_limit = _parse_optional_float(self.lower_limit_edit)
        balloon.upper_limit = _parse_optional_float(self.upper_limit_edit)
        balloon.gdt_symbol = self.gdt_symbol_edit.text().strip() or None
        balloon.gdt_tolerance = self.gdt_tolerance_edit.text().strip() or None
        balloon.material_condition = self.material_condition_edit.text().strip() or None
        balloon.datums = self.datums_edit.text().strip() or None
        balloon.surface_finish = self.surface_finish_edit.text().strip() or None
        balloon.thread_callout = self.thread_callout_edit.text().strip() or None
        balloon.note = self.note_edit.toPlainText().strip()
        balloon.inspection_method = self.inspection_method_combo.currentText().strip()
        balloon.critical = self.critical_check.isChecked()
        balloon.status = self.status_combo.currentData()
        balloon.touch()


class SettingsDialog(QDialog):
    """Editable application settings (Tesseract path, DPI, thresholds, YOLO model)."""

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)

        self.tesseract_edit = QLineEdit(settings.tesseract_path)
        tesseract_browse = QPushButton("Browse...")
        tesseract_browse.clicked.connect(self._browse_tesseract)
        tesseract_row = QHBoxLayout()
        tesseract_row.addWidget(self.tesseract_edit)
        tesseract_row.addWidget(tesseract_browse)
        tesseract_hint = QLabel(
            "Optional. Only needed if Tesseract is not on your system PATH.\n"
            "You can also set the TESSERACT_PATH environment variable instead."
        )
        tesseract_hint.setStyleSheet("color: gray; font-size: 10px;")
        tesseract_hint.setWordWrap(True)

        self.default_dpi_spin = QSpinBox()
        self.default_dpi_spin.setRange(72, 600)
        self.default_dpi_spin.setValue(settings.default_dpi)

        self.auto_dpi_spin = QSpinBox()
        self.auto_dpi_spin.setRange(72, 600)
        self.auto_dpi_spin.setValue(settings.auto_balloon_dpi)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(settings.confidence_threshold)

        self.yolo_path_edit = QLineEdit(settings.yolo_model_path)
        yolo_browse = QPushButton("Browse...")
        yolo_browse.clicked.connect(self._browse_yolo)
        yolo_row = QHBoxLayout()
        yolo_row.addWidget(self.yolo_path_edit)
        yolo_row.addWidget(yolo_browse)

        self.use_yolo_check = QCheckBox("Use YOLO model when available (falls back to rules/OCR otherwise)")
        self.use_yolo_check.setChecked(settings.use_yolo_if_available)

        form = QFormLayout()
        form.addRow("Tesseract Path:", tesseract_row)
        form.addRow("", tesseract_hint)
        form.addRow("Default Render DPI:", self.default_dpi_spin)
        form.addRow("Auto-Balloon DPI:", self.auto_dpi_spin)
        form.addRow("Default Confidence Threshold:", self.confidence_spin)
        form.addRow("YOLO Model (.pt) Path:", yolo_row)
        form.addRow("", self.use_yolo_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse_tesseract(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate tesseract.exe", "", "Executable (*.exe);;All Files (*)")
        if path:
            self.tesseract_edit.setText(path)

    def _browse_yolo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate YOLO model", str(Path.cwd()), "PyTorch Model (*.pt);;All Files (*)")
        if path:
            self.yolo_path_edit.setText(path)

    def apply_to_settings(self, settings: AppSettings) -> None:
        settings.tesseract_path = self.tesseract_edit.text().strip()
        settings.default_dpi = self.default_dpi_spin.value()
        settings.auto_balloon_dpi = self.auto_dpi_spin.value()
        settings.confidence_threshold = self.confidence_spin.value()
        settings.yolo_model_path = self.yolo_path_edit.text().strip()
        settings.use_yolo_if_available = self.use_yolo_check.isChecked()


class TeachTrainingDialog(QDialog):
    """Shows teach/training statistics and triggers a dataset export."""

    def __init__(self, stats: TeachStats, on_export: Callable[[], None], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Teach / Training Data")
        self.setMinimumWidth(420)
        self._on_export = on_export

        rows = [
            ("Total Balloons", stats.total_balloons),
            ("Auto Proposals", stats.auto_proposals),
            ("Accepted Auto Proposals", stats.accepted_auto),
            ("Edited Auto Proposals", stats.edited_auto),
            ("Rejected Auto Proposals", stats.rejected_auto),
            ("Manual Additions", stats.manual_additions),
            ("Labeled Pages Available for Export", stats.labeled_pages),
            ("Detector/Model Version(s) Used", ", ".join(stats.model_versions) or "none yet"),
            ("Learned Corrections", stats.learned_corrections),
            ("Repeated False Positives Suppressed", stats.suppressed_patterns),
        ]

        group = QGroupBox("Current Project Feedback Summary")
        form = QFormLayout()
        for label, value in rows:
            form.addRow(f"{label}:", QLabel(str(value)))
        group.setLayout(form)

        explanation = QLabel(
            "Exporting builds a YOLO-style dataset (images, .txt labels, class map, crops, "
            "and a manifest) from your reviewed balloons. Rejected proposals are recorded in "
            "the manifest for reference but are never exported as positive training labels.\n\n"
            "Reviewed corrections are also saved to local learning memory and reused on matching "
            "callouts. Two consistent rejections suppress that exact callout in future scans. "
            "Neural-model training remains a separate, explicitly controlled workflow."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: gray;")

        export_button = QPushButton("Export Training Dataset")
        export_button.clicked.connect(self._handle_export)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        button_row = QHBoxLayout()
        button_row.addWidget(export_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addWidget(explanation)
        layout.addLayout(button_row)

    def _handle_export(self) -> None:
        self._on_export()
        self.accept()


class ExportExcelOptionsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Export Excel Inspection Sheet")
        self.include_pending_check = QCheckBox("Include pending (unreviewed) auto-proposals")
        self.include_pending_check.setChecked(False)

        info = QLabel("By default, only accepted, edited, and manually added characteristics are exported.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.include_pending_check)
        layout.addWidget(buttons)

    def include_pending(self) -> bool:
        return self.include_pending_check.isChecked()


class ExportPdfOptionsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, stamp_size_percent: int = 100):
        super().__init__(parent)
        self.setWindowTitle("Export Ballooned PDF")

        self.include_pending_check = QCheckBox("Include pending proposals (shown in orange)")
        self.include_pending_check.setChecked(True)
        self.include_rejected_check = QCheckBox("Include rejected proposals (shown in muted red)")
        self.include_rejected_check.setChecked(False)

        info = QLabel("Accepted, edited, and manually added balloons are always included.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")

        self.stamp_size_spin = QSpinBox()
        self.stamp_size_spin.setRange(50, 200)
        self.stamp_size_spin.setSingleStep(10)
        self.stamp_size_spin.setSuffix("%")
        self.stamp_size_spin.setValue(stamp_size_percent)
        self.stamp_size_spin.setToolTip('Size of the "Ballooned Drawing" stamp in the top-left corner')
        stamp_size_row = QHBoxLayout()
        stamp_size_row.addWidget(QLabel('"Ballooned Drawing" stamp size:'))
        stamp_size_row.addWidget(self.stamp_size_spin)
        stamp_size_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.include_pending_check)
        layout.addWidget(self.include_rejected_check)
        layout.addLayout(stamp_size_row)
        layout.addWidget(buttons)

    def options(self) -> tuple[bool, bool]:
        return self.include_pending_check.isChecked(), self.include_rejected_check.isChecked()

    def stamp_size_percent(self) -> int:
        return self.stamp_size_spin.value()


class RenumberDialog(QDialog):
    """Choose a renumbering strategy for balloons."""

    MODE_PAGE = "page"
    MODE_DRAWING = "drawing"
    MODE_PRESERVE = "preserve"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Renumber Balloons")

        self.page_radio = QRadioButton("Current page only (left-to-right, top-to-bottom)")
        self.drawing_radio = QRadioButton("Entire drawing (page order, then left-to-right/top-to-bottom)")
        self.preserve_radio = QRadioButton("Preserve existing order (just make numbers sequential)")
        self.page_radio.setChecked(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.page_radio)
        layout.addWidget(self.drawing_radio)
        layout.addWidget(self.preserve_radio)
        layout.addWidget(buttons)

    def mode(self) -> str:
        if self.drawing_radio.isChecked():
            return self.MODE_DRAWING
        if self.preserve_radio.isChecked():
            return self.MODE_PRESERVE
        return self.MODE_PAGE


class ConfidenceThresholdDialog(QDialog):
    """Simple prompt for a confidence threshold, used by "Accept all above..."."""

    def __init__(self, initial: float = DEFAULT_CONFIDENCE_THRESHOLD, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Accept All Above Confidence Threshold")

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 1.0)
        self.spin.setSingleStep(0.05)
        self.spin.setValue(initial)

        form = QFormLayout()
        form.addRow("Confidence threshold:", self.spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def threshold(self) -> float:
        return self.spin.value()


class AboutDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("About BalloonIQ")
        self.setMinimumWidth(420)

        text = QLabel(
            f"<h2>BalloonIQ</h2>"
            f"<p>Version {__version__}</p>"
            "<p>An offline, local-first tool for ballooning mechanical-engineering PDF "
            "drawings and generating generic Excel inspection sheets.</p>"
            "<p><b>Important:</b> Automatic detection is an assistive feature, not a "
            "replacement for engineering review. OCR accuracy varies with PDF/scan quality, "
            "and complex GD&amp;T feature control frames may need manual correction. Always "
            "review proposed balloons before exporting.</p>"
            "<p>All project data stays on your computer by default.</p>"
        )
        text.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        logo = QLabel()
        logo.setPixmap(QPixmap(str(resource_path("balloonapp-logo.png"))).scaled(
            96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        ))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setAccessibleName("BalloonIQ logo")
        layout.addWidget(logo)
        layout.addWidget(text)
        layout.addWidget(buttons)
