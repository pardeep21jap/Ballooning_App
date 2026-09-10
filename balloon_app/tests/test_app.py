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
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from balloon_app.app import MainWindow, _bpdb_path_beside_pdf
from balloon_app.dialogs import NewProjectDialog
from PyQt6.QtWidgets import QApplication, QDialog, QPushButton, QWidget


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


def test_review_more_menu_has_no_reject_option():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        action_bar = window.findChild(QWidget, "reviewActions")
        more_button = next(b for b in action_bar.findChildren(QPushButton) if b.text() == "More")
        menu_labels = [a.text() for a in more_button.menu().actions() if not a.isSeparator()]

        assert not any("reject" in label.lower() for label in menu_labels)
        # Unrelated More-menu actions must still be there.
        assert "Accept All Above Threshold..." in menu_labels
        assert "Add Manual" in menu_labels
    finally:
        window.close()
        app.processEvents()
