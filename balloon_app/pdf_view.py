"""PyQt6 QGraphicsView/Scene-based PDF viewer with an aligned balloon overlay.

Design notes
------------
Rather than scaling a fixed-resolution pixmap in and out (which blurs at
high zoom), zoom changes actually **re-render** the page at
``ACTUAL_SIZE_DPI * zoom_level`` in a background thread, so drawings stay
crisp at any zoom level. Because balloon items are positioned in that same
pixel space, they remain pixel-aligned with the drawing content after every
re-render -- panning and zooming never desynchronize the overlay from the
underlying page.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QMutex, QObject, QPointF, QRectF, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsObject, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from balloon_app.config import COLOR_SELECTED_OUTLINE, MAX_ZOOM, MIN_ZOOM, status_color
from balloon_app.data_model import Balloon
from balloon_app.pdf_engine import PdfDocument, pdf_to_pixel, pixel_to_pdf

logger = logging.getLogger("balloon_app.pdf_view")

ACTUAL_SIZE_DPI = 96.0
ZOOM_STEP = 1.25


class BalloonItem(QGraphicsObject):
    """A draggable, selectable circular balloon with its characteristic number."""

    moved = pyqtSignal(str, QPointF)
    doubleClicked = pyqtSignal(str)
    clickedItem = pyqtSignal(str)
    dragFinished = pyqtSignal(str)

    def __init__(self, balloon_id: str, number: int, source: str, status: str, radius: float):
        super().__init__()
        self.balloon_id = balloon_id
        self.number = number
        self.source = source
        self.status = status
        self.radius = radius
        self._emit_moves = True
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(10.0)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def boundingRect(self) -> QRectF:
        r = self.radius + 3
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: D102
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r, g, b, a = status_color(self.source, self.status)
        fill = QColor(r, g, b, a)
        if self.isSelected():
            pen = QPen(QColor(*COLOR_SELECTED_OUTLINE))
            pen.setWidth(3)
        else:
            pen = QPen(QColor(20, 20, 20))
            pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont()
        font.setBold(True)
        font.setPointSizeF(max(6.0, self.radius * 0.85))
        painter.setFont(font)
        painter.drawText(self.boundingRect(), int(Qt.AlignmentFlag.AlignCenter), str(self.number))

    def update_appearance(self, number: int, source: str, status: str) -> None:
        self.number = number
        self.source = source
        self.status = status
        self.update()

    def set_emit_moves(self, enabled: bool) -> None:
        self._emit_moves = enabled

    def itemChange(self, change, value):  # noqa: D102
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self._emit_moves:
            self.moved.emit(self.balloon_id, self.pos())
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: D102
        self.doubleClicked.emit(self.balloon_id)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: D102
        self.clickedItem.emit(self.balloon_id)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D102
        super().mouseReleaseEvent(event)
        self.dragFinished.emit(self.balloon_id)


class _RenderWorker(QObject):
    """Runs PyMuPDF page rendering on a dedicated background thread."""

    rendered = pyqtSignal(int, bytes, int, int, float)
    failed = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self._pdf_doc: Optional[PdfDocument] = None
        self._lock = QMutex()

    def set_document(self, pdf_doc: Optional[PdfDocument]) -> None:
        self._lock.lock()
        self._pdf_doc = pdf_doc
        self._lock.unlock()

    def render(self, request_id: int, page_number: int, dpi: float) -> None:
        self._lock.lock()
        doc = self._pdf_doc
        self._lock.unlock()
        if doc is None:
            self.failed.emit(request_id, "No document loaded")
            return
        try:
            data, width, height = doc.render_page_rgb(page_number, dpi)
            self.rendered.emit(request_id, data, width, height, dpi)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Page render failed")
            self.failed.emit(request_id, str(exc))


class PdfGraphicsView(QGraphicsView):
    """The main drawing canvas: renders PDF pages and overlays balloons."""

    balloonSelected = pyqtSignal(str)
    balloonMoved = pyqtSignal(str, float, float)
    balloonDragFinished = pyqtSignal(str)
    balloonDoubleClicked = pyqtSignal(str)
    newBalloonRequested = pyqtSignal(float, float)
    leaderPointPicked = pyqtSignal(str, float, float)
    statusMessage = pyqtSignal(str)
    renderStarted = pyqtSignal()
    renderFinished = pyqtSignal()
    _render_requested = pyqtSignal(int, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor(60, 60, 60)))

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._pdf_doc: Optional[PdfDocument] = None
        self._page_number = 0
        self._zoom_level = 1.0
        self._effective_dpi = ACTUAL_SIZE_DPI
        self._balloon_items: dict[str, BalloonItem] = {}
        self._leader_items: dict[str, QGraphicsLineItem] = {}
        self._current_balloons: list[Balloon] = []

        self.add_balloon_mode = False
        self.leader_mode_balloon_id: Optional[str] = None

        self._panning = False
        self._pan_start_pos = None

        self._request_counter = 0
        self._latest_request_id = -1

        self._thread = QThread(self)
        self._worker = _RenderWorker()
        self._worker.moveToThread(self._thread)
        self._worker.rendered.connect(self._on_rendered)
        self._worker.failed.connect(self._on_render_failed)
        self._render_requested.connect(self._worker.render)
        self._thread.start()

    def shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(2000)

    # ------------------------------------------------------------------
    # Document / page loading
    # ------------------------------------------------------------------
    def load_document(self, pdf_doc: Optional[PdfDocument]) -> None:
        self._pdf_doc = pdf_doc
        self._worker.set_document(pdf_doc)

    def set_page(self, page_number: int, balloons: list[Balloon], preserve_view: bool = False) -> None:
        self._page_number = page_number
        self._current_balloons = balloons
        self._request_render(preserve_view=preserve_view)

    def refresh_balloons(self, balloons: list[Balloon]) -> None:
        self._current_balloons = balloons
        self._sync_balloon_items()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom_level * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom_level / ZOOM_STEP)

    def actual_size(self) -> None:
        self.set_zoom(1.0)

    def fit_page(self) -> None:
        if self._pdf_doc is None:
            return
        try:
            page_w_pt, page_h_pt = self._pdf_doc.page_size_pdf(self._page_number)
        except Exception:
            return
        viewport_size = self.viewport().size()
        margin = 16
        avail_w = max(50, viewport_size.width() - margin)
        avail_h = max(50, viewport_size.height() - margin)
        # page size at ACTUAL_SIZE_DPI (i.e. zoom_level == 1.0)
        page_w_px = page_w_pt / 72.0 * ACTUAL_SIZE_DPI
        page_h_px = page_h_pt / 72.0 * ACTUAL_SIZE_DPI
        if page_w_px <= 0 or page_h_px <= 0:
            return
        zoom = min(avail_w / page_w_px, avail_h / page_h_px)
        self.set_zoom(zoom)

    def set_zoom(self, zoom_level: float) -> None:
        zoom_level = max(MIN_ZOOM, min(MAX_ZOOM, zoom_level))
        if abs(zoom_level - self._zoom_level) < 1e-6:
            return
        self._zoom_level = zoom_level
        self._request_render(preserve_view=True)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: D102
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            elif delta < 0:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _request_render(self, preserve_view: bool) -> None:
        if self._pdf_doc is None:
            return
        self._effective_dpi = ACTUAL_SIZE_DPI * self._zoom_level

        self._pending_center_pdf: Optional[tuple[float, float]] = None
        if preserve_view and self._pixmap_item is not None:
            center_scene = self.mapToScene(self.viewport().rect().center())
            self._pending_center_pdf = pixel_to_pdf(center_scene.x(), center_scene.y(), self._last_render_dpi())

        self._request_counter += 1
        self.renderStarted.emit()
        self._render_requested.emit(self._request_counter, self._page_number, self._effective_dpi)

    def _last_render_dpi(self) -> float:
        return getattr(self, "_current_render_dpi", self._effective_dpi)

    def _on_rendered(self, request_id: int, data: bytes, width: int, height: int, dpi: float) -> None:
        if request_id < self._request_counter:
            return  # stale result from a superseded zoom/page change
        self._latest_request_id = request_id
        self._current_render_dpi = dpi

        image = QImage(data, width, height, width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())

        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setZValue(0.0)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)

        self._scene.setSceneRect(0, 0, width, height)
        self._sync_balloon_items()

        if getattr(self, "_pending_center_pdf", None) is not None:
            px, py = pdf_to_pixel(*self._pending_center_pdf, dpi)
            self.centerOn(px, py)
            self._pending_center_pdf = None
        else:
            self.centerOn(width / 2, height / 2)

        self.renderFinished.emit()
        self.statusMessage.emit(f"Page rendered at {int(dpi)} DPI ({int(self._zoom_level * 100)}%).")

    def _on_render_failed(self, request_id: int, message: str) -> None:
        if request_id < self._request_counter:
            return
        self.statusMessage.emit(f"Failed to render page: {message}")
        self.renderFinished.emit()

    # ------------------------------------------------------------------
    # Balloon overlay sync
    # ------------------------------------------------------------------
    def _sync_balloon_items(self) -> None:
        from balloon_app.config import BALLOON_RADIUS_PDF_POINTS

        dpi = self._last_render_dpi()
        radius_px = BALLOON_RADIUS_PDF_POINTS * dpi / 72.0

        page_balloons = [b for b in self._current_balloons if b.page_number == self._page_number]
        current_ids = {b.id for b in page_balloons}

        for stale_id in list(self._balloon_items):
            if stale_id not in current_ids:
                self._scene.removeItem(self._balloon_items.pop(stale_id))
                if stale_id in self._leader_items:
                    self._scene.removeItem(self._leader_items.pop(stale_id))

        for balloon in page_balloons:
            px, py = pdf_to_pixel(balloon.x, balloon.y, dpi)
            item = self._balloon_items.get(balloon.id)
            if item is None:
                item = BalloonItem(balloon.id, balloon.number, balloon.source, balloon.status, radius_px)
                item.moved.connect(self._on_item_moved)
                item.doubleClicked.connect(self.balloonDoubleClicked.emit)
                item.clickedItem.connect(self.balloonSelected.emit)
                item.dragFinished.connect(self.balloonDragFinished.emit)
                self._scene.addItem(item)
                self._balloon_items[balloon.id] = item
            item.set_emit_moves(False)
            item.radius = radius_px
            item.update_appearance(balloon.number, balloon.source, balloon.status)
            item.setPos(px, py)
            item.set_emit_moves(True)

            leader_source = None
            if balloon.leader_x is not None and balloon.leader_y is not None:
                leader_source = (balloon.leader_x, balloon.leader_y)
            elif balloon.has_bbox():
                bx0, by0, bx1, by1 = balloon.bbox()  # type: ignore[misc]
                leader_source = ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)

            line_item = self._leader_items.get(balloon.id)
            if leader_source is not None:
                lx, ly = pdf_to_pixel(leader_source[0], leader_source[1], dpi)
                if line_item is None:
                    line_item = QGraphicsLineItem()
                    line_item.setPen(QPen(QColor(0, 0, 0, 180), 1.2))
                    line_item.setZValue(5.0)
                    self._scene.addItem(line_item)
                    self._leader_items[balloon.id] = line_item
                line_item.setLine(lx, ly, px, py)
            elif line_item is not None:
                self._scene.removeItem(line_item)
                del self._leader_items[balloon.id]

    def _on_item_moved(self, balloon_id: str, scene_pos: QPointF) -> None:
        # NOTE: deliberately does not trigger a full _sync_balloon_items() resync
        # here -- the dragged item already reflects its own position via Qt's
        # built-in item-move handling, and resyncing from (still-stale) model
        # data mid-drag would fight the drag and cause visible jitter. Only the
        # leader line's end point needs to track the item live; the model
        # update (and any full resync) happens once the caller persists the
        # new position.
        dpi = self._last_render_dpi()
        pdf_x, pdf_y = pixel_to_pdf(scene_pos.x(), scene_pos.y(), dpi)
        self.balloonMoved.emit(balloon_id, pdf_x, pdf_y)
        line_item = self._leader_items.get(balloon_id)
        if line_item is not None:
            line = line_item.line()
            line_item.setLine(line.x1(), line.y1(), scene_pos.x(), scene_pos.y())

    def select_balloon(self, balloon_id: Optional[str]) -> None:
        for bid, item in self._balloon_items.items():
            item.setSelected(bid == balloon_id)

    # ------------------------------------------------------------------
    # Mouse interaction: manual panning + add-balloon / leader picking
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: D102
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.itemAt(event.pos())

            if self.add_balloon_mode:
                pdf_x, pdf_y = pixel_to_pdf(scene_pos.x(), scene_pos.y(), self._last_render_dpi())
                self.newBalloonRequested.emit(pdf_x, pdf_y)
                return

            if self.leader_mode_balloon_id is not None:
                pdf_x, pdf_y = pixel_to_pdf(scene_pos.x(), scene_pos.y(), self._last_render_dpi())
                self.leaderPointPicked.emit(self.leader_mode_balloon_id, pdf_x, pdf_y)
                return

            if isinstance(item, BalloonItem):
                super().mousePressEvent(event)
                return

            self._panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: D102
        if self._panning and self._pan_start_pos is not None:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D102
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._pan_start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(event)
