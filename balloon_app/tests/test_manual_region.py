"""Manual source-region selection stays independent of balloon placement."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QUndoStack
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from balloon_app.app import MainWindow
from balloon_app.data_model import Balloon
from balloon_app.pdf_view import PdfGraphicsView


@pytest.fixture
def canvas():
    qt_app = QApplication.instance() or QApplication([])
    view = PdfGraphicsView()
    view.resize(900, 700)
    view._on_rendered(view._request_counter, bytes([255]) * (800 * 600 * 3), 800, 600, 144)
    view.show()
    qt_app.processEvents()
    yield view
    view.shutdown()
    view.close()
    qt_app.processEvents()


def drag(view, start, end):
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(QPointF(*start)))
    QTest.mouseMove(view.viewport(), view.mapFromScene(QPointF(*end)))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(QPointF(*end)))


@pytest.mark.parametrize("rotation", [0, 90])
def test_region_converts_view_to_pdf(canvas, rotation):
    canvas.rotate(rotation)
    canvas.scale(.75, .75)
    received = []
    canvas.characteristicRegionPicked.connect(lambda bid, box: received.append((bid, box)))
    canvas.start_characteristic_region("manual")
    drag(canvas, (100, 100), (300, 200))
    assert received[0][0] == "manual"
    assert received[0][1] == pytest.approx((50, 50, 150, 100), abs=1)


@pytest.mark.parametrize("cancel", ["escape", "right_click", "unload"])
def test_cancel_region_emits_nothing(canvas, cancel):
    received = Mock()
    canvas.characteristicRegionPicked.connect(received)
    canvas.start_characteristic_region("manual")
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=canvas.mapFromScene(QPointF(100, 100)))
    if cancel == "escape":
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
    elif cancel == "right_click":
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.RightButton)
    else:
        canvas.load_document(None)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton)
    received.assert_not_called()


def test_zero_size_region_emits_nothing(canvas):
    received = Mock()
    canvas.characteristicRegionPicked.connect(received)
    canvas.start_characteristic_region("manual")
    drag(canvas, (100, 100), (100, 100))
    received.assert_not_called()


def test_region_clipped_to_page(canvas):
    received = []
    canvas.characteristicRegionPicked.connect(lambda bid, box: received.append(box))
    canvas.start_characteristic_region("manual")
    drag(canvas, (-10, -10), (100, 100))
    assert received == [(0, 0, 50, 50)]


@pytest.mark.parametrize("existing", [False, True])
def test_store_and_replace_region_undo(canvas, existing):
    balloon = Balloon(x=250, y=200, leader_x=10, leader_y=20)
    if existing:
        balloon.bbox_x0, balloon.bbox_y0, balloon.bbox_x1, balloon.bbox_y1 = (1, 2, 3, 4)
    before = balloon.to_dict()
    window = SimpleNamespace(
        project=SimpleNamespace(balloons=[balloon]), drawing=SimpleNamespace(id=balloon.drawing_id),
        current_page=0, pdf_doc=Mock(), undo_stack=QUndoStack(), _refresh_all=Mock(),
        _find_balloon=lambda bid: balloon,
    )
    window.pdf_doc.page_size_pdf.return_value = (400, 300)
    canvas.characteristicRegionPicked.connect(
        lambda bid, box: MainWindow._on_characteristic_region_picked(window, bid, box)
    )
    canvas.start_characteristic_region(balloon.id)
    drag(canvas, (100, 120), (300, 200))
    assert balloon.bbox() == (50, 60, 150, 100)
    assert (balloon.x, balloon.y, balloon.leader_x, balloon.leader_y) == (250, 200, 10, 20)
    window.undo_stack.undo()
    assert balloon.to_dict() == before
    window.undo_stack.redo()
    assert balloon.bbox() == (50, 60, 150, 100)


@pytest.mark.parametrize("box", [(1, 1, 1, 2), (2, 2, 1, 1), (float("nan"), 0, 10, 10), (500, 500, 600, 600)])
def test_invalid_region_not_stored(canvas, box):
    balloon = Balloon()
    window = SimpleNamespace(
        project=SimpleNamespace(balloons=[balloon]), drawing=SimpleNamespace(id=balloon.drawing_id),
        current_page=0, pdf_doc=Mock(), undo_stack=QUndoStack(), _refresh_all=Mock(),
        _find_balloon=lambda bid: balloon,
    )
    window.pdf_doc.page_size_pdf.return_value = (400, 300)
    MainWindow._on_characteristic_region_picked(window, balloon.id, box)
    assert not balloon.has_bbox()
    assert window.undo_stack.count() == 0


@pytest.mark.parametrize("source", ["manual", "auto"])
def test_action_only_starts_for_manual_balloon(monkeypatch, canvas, source):
    from balloon_app import app

    balloon = Balloon(source=source)
    window = SimpleNamespace(
        _selected_balloon_ids=lambda: [balloon.id], _find_balloon=lambda bid: balloon,
        current_page=0, pdf_view=Mock(),
    )
    monkeypatch.setattr(app.QMessageBox, "information", Mock())
    MainWindow._start_characteristic_region(window)
    if source == "manual":
        window.pdf_view.start_characteristic_region.assert_called_once_with(balloon.id)
    else:
        window.pdf_view.start_characteristic_region.assert_not_called()
