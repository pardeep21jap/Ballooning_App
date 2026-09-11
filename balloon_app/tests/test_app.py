"""Regression tests for where a new project's .bpdb file gets created, and
for the review panel's action bar.

Bug 1: adding a PDF drawing when no project is open created the new
project's .bpdb file under PROJECTS_DIR (e.g. BalloonApp/projects/<name>/)
instead of in the same folder the user picked the PDF from.

Bug 2: the review panel's action bar had a one-click "Reject" button next
to Accept/Edit/Delete -- too easy to hit by mistake.

Bug 3: after removing that button, "Reject Selected/Pending" was still
reachable from the review panel's "More" menu -- the reject option was
requested to be removed completely from the review workflow, not just its
one-click button. (The balloon Edit dialog's general-purpose "Review
Status" field, which can set any status including Rejected, is out of
scope -- it's what keeps a project's already-rejected balloons normally
viewable/editable.)

Bug 4: a pending balloon's Status-column cell held a plain (non-clickable)
badge, and clicking into that column -- like clicking into the Method
column's combo box -- never reached the table's selection model, since a
cell widget swallows the mouse click before the view sees it. So a row
whose last click landed on the Method or Status column wasn't selected,
and the review panel's "Accept" button silently did nothing. The fix
makes the pending Status badge itself a clickable "accept this row"
button, so accepting no longer depends on row selection at all.

Bug 5: the review panel's summary legend (e.g. "2 accepted · 0 edited ·
0 rejected · 17 pending") still showed a "rejected" count even though the
reject workflow was removed completely per Bug 3 above -- the count was
always 0 and never actionable, so it was requested to be dropped from the
legend text entirely.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from balloon_app.app import MainWindow, _bpdb_path_beside_pdf
from balloon_app.data_model import Balloon, Drawing, Project, ReviewStatus
from balloon_app.dialogs import NewProjectDialog
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton, QWidget


def test_bpdb_path_beside_pdf_uses_pdf_folder(tmp_path: Path):
    pdf_path = tmp_path / "drawing.pdf"
    pdf_path.write_bytes(b"")

    db_path = _bpdb_path_beside_pdf(pdf_path, "My Project")

    assert db_path.parent == tmp_path
    assert db_path.name == "My_Project.bpdb"


def test_bpdb_path_beside_pdf_avoids_collision(tmp_path: Path):
    pdf_path = tmp_path / "drawing.pdf"
    (tmp_path / "My_Project.bpdb").write_bytes(b"")

    db_path = _bpdb_path_beside_pdf(pdf_path, "My Project")

    assert db_path == tmp_path / "My_Project_1.bpdb"


def test_new_project_beside_pdf_creates_bpdb_next_to_selected_pdf(tmp_path: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    drawings_dir = tmp_path / "customer_drawings"
    drawings_dir.mkdir()
    pdf_path = drawings_dir / "part.pdf"
    pdf_path.write_bytes(b"")

    monkeypatch.setattr(NewProjectDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(
        NewProjectDialog, "values",
        lambda self: {
            "name": "Part Project", "part_number": "", "part_name": "",
            "revision": "", "customer": "", "unit": "in", "notes": "",
        },
    )

    window = MainWindow()
    monkeypatch.setattr(window.settings, "save", lambda: None)
    try:
        created = window._new_project_beside_pdf(pdf_path)

        assert created is True
        assert window.project is not None
        assert Path(window.project.file_path).parent == drawings_dir
        assert (drawings_dir / "Part_Project.bpdb").exists()
    finally:
        window.close()
        app.processEvents()


def test_review_action_bar_has_no_reject_button():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        action_bar = window.findChild(QWidget, "reviewActions")
        buttons = action_bar.findChildren(QPushButton)
        labels = [b.text() for b in buttons]

        assert "Reject" not in labels
        assert "Accept" in labels
        assert "Edit" in labels
        assert "Delete" in labels
    finally:
        window.close()
        app.processEvents()


def test_review_actions_replace_more_menu_with_buttons(monkeypatch):
    app = QApplication.instance() or QApplication([])
    calls = []
    callbacks = [
        ("Accept All Above Threshold...", "_accept_all_above_threshold"),
        ("Add Manual", "_add_manual_balloon"),
        ("Duplicate", "_duplicate_selected_balloon"),
    ]
    for label, method in callbacks:
        monkeypatch.setattr(MainWindow, method, lambda self, checked=False, label=label: calls.append(label))
    window = MainWindow()
    try:
        action_bar = window.findChild(QWidget, "reviewActions")
        buttons = action_bar.findChildren(QPushButton)
        labels = [button.text() for button in buttons]
        assert labels == ["Accept", "Edit", "Delete", *[label for label, _ in callbacks]]
        assert all(button.menu() is None for button in buttons)
        for button in buttons[3:]:
            button.click()
        assert calls == [label for label, _ in callbacks]
    finally:
        window.close()
        app.processEvents()


class _FakeUndoStack:
    """Stands in for QUndoStack so the test can push/redo a real
    BalloonFieldChangeCommand without going through PyQt6's QUndoStack --
    pushing onto the real one and then letting Python garbage-collect it
    segfaults the interpreter on exit in this offscreen/headless test
    environment (reproducible with a bare QUndoCommand, unrelated to
    balloons or this fix), which is a pre-existing environment issue well
    outside the scope of the Status-badge fix under test here."""

    def push(self, cmd) -> None:
        cmd.redo()

    def isClean(self) -> bool:
        return False


def test_clicking_pending_status_badge_accepts_without_row_selection():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.undo_stack = _FakeUndoStack()
    window._confirm_discard_changes = lambda: True
    try:
        drawing = Drawing(file_name="part.pdf", page_count=1)
        balloon = Balloon(
            drawing_id=drawing.id, number=1, status=ReviewStatus.PENDING.value,
            inspection_method="Height Gauge",
        )
        project = Project(drawings=[drawing], balloons=[balloon])
        window.project = project
        window.drawing = drawing
        window._refresh_review_table()

        # Nothing selected -- clicking the toolbar Accept button would be a no-op.
        assert window.review_table.selectionModel().selectedRows() == []

        status_widget = window.review_table.cellWidget(0, 7)
        accept_button = status_widget.findChild(QPushButton)
        assert accept_button is not None
        assert accept_button.text() == "PENDING"

        accept_button.click()

        assert balloon.status == ReviewStatus.ACCEPTED.value
    finally:
        window.close()
        app.processEvents()


def test_accepting_without_an_inspection_method_is_blocked(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.undo_stack = _FakeUndoStack()
    window._confirm_discard_changes = lambda: True
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *args, **kwargs: warnings.append(args) or QMessageBox.StandardButton.Ok,
    )
    try:
        drawing = Drawing(file_name="part.pdf", page_count=1)
        balloon = Balloon(drawing_id=drawing.id, number=1, status=ReviewStatus.PENDING.value)
        assert balloon.inspection_method == ""
        project = Project(drawings=[drawing], balloons=[balloon])
        window.project = project
        window.drawing = drawing
        window._refresh_review_table()

        status_widget = window.review_table.cellWidget(0, 7)
        accept_button = status_widget.findChild(QPushButton)
        assert accept_button is not None

        accept_button.click()

        assert balloon.status == ReviewStatus.PENDING.value
        assert warnings, "expected a warning dialog when accepting without an inspection method"
    finally:
        window.close()
        app.processEvents()


def test_review_legend_has_no_rejected_count():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        # Before any project is open (the label's initial placeholder text).
        assert "rejected" not in window.review_legend.text().lower()

        drawing = Drawing(file_name="part.pdf", page_count=1)
        balloon = Balloon(drawing_id=drawing.id, number=1, status=ReviewStatus.ACCEPTED.value)
        project = Project(drawings=[drawing], balloons=[balloon])
        window.project = project
        window.drawing = drawing
        window._refresh_review_table()

        legend = window.review_legend.text()
        assert "rejected" not in legend.lower()
        assert "accepted" in legend.lower()
        assert "edited" in legend.lower()
        assert "pending" in legend.lower()
    finally:
        window.close()
        app.processEvents()


def test_app_has_no_renumber_action():
    from PyQt6.QtGui import QAction

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert not any("renumber" in action.text().lower() for action in window.findChildren(QAction))
    finally:
        window.close()
        app.processEvents()


def test_delete_balloons_renumbers_drawing_and_undo_restores_numbers():
    from balloon_app.app import DeleteBalloonsCommand

    app = QApplication.instance() or QApplication([])
    balloons = [Balloon(drawing_id="drawing", number=n, page_number=n % 2) for n in range(1, 6)]
    other = Balloon(drawing_id="other", number=9)
    project = Project(balloons=[balloons[4], other, *balloons[:4]])
    original = {b.id: b.number for b in project.balloons}
    refreshed = []
    command = DeleteBalloonsCommand(project, [balloons[1], balloons[3]], lambda: refreshed.append(True))
    command.redo()
    assert [b.number for b in (balloons[0], balloons[2], balloons[4])] == [1, 2, 3]
    assert other.number == 9
    assert {b.id for b in project.balloons} == {balloons[0].id, balloons[2].id, balloons[4].id, other.id}
    command.undo()
    assert {b.id: b.number for b in project.balloons} == original
    command.redo()
    assert [b.number for b in (balloons[0], balloons[2], balloons[4])] == [1, 2, 3]
    assert other.number == 9
    assert len(refreshed) == 3
    app.processEvents()


def test_app_has_no_split_balloon_action():
    from PyQt6.QtGui import QAction

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        labels = [action.text() for action in window.findChildren(QAction)]
        assert not any("split balloon" in label.lower() for label in labels)
        assert "Duplicate" in [button.text() for button in window.findChildren(QPushButton)]
        assert "Delete Balloon" in labels
    finally:
        window.close()
        app.processEvents()
