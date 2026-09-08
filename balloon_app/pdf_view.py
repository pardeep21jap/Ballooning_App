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
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QEvent, QMutex, QObject, QPointF, QRectF, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsObject, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from balloon_app.config import (
    COLOR_SELECTED_OUTLINE,
    MAX_ZOOM,
    MIN_ZOOM,
    STAMP_CORNER_RADIUS_PERCENT,
    STAMP_FONT_SIZE_PDF_POINTS,
    STAMP_MARGIN_PDF_POINTS,
    STAMP_TEXT,
    status_color,
)
from balloon_app.data_model import Balloon, ReviewStatus
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
        font.setPointSizeF(max(1.0, self.radius * 0.85))
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


class LeaderHandleItem(QGraphicsObject):
    """A small draggable square marking a leader line's start point (the end
    that touches the drawing feature, as opposed to the end at the balloon).

    Distinct from :class:`BalloonItem` mainly in shape (a diamond, so it
    reads as "not the balloon" at a glance) and in not carrying a number.
    """

    moved = pyqtSignal(str, QPointF)
    dragFinished = pyqtSignal(str)

    def __init__(self, balloon_id: str, half_size: float):
        super().__init__()
        self.balloon_id = balloon_id
        self.half_size = half_size
        self._emit_moves = True
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(9.0)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Drag to move the leader line's start point.")

    def boundingRect(self) -> QRectF:
        r = self.half_size + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: D102
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(20, 20, 20))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        s = self.half_size
        painter.drawPolygon([QPointF(0, -s), QPointF(s, 0), QPointF(0, s), QPointF(-s, 0)])

    def set_emit_moves(self, enabled: bool) -> None:
        self._emit_moves = enabled

    def itemChange(self, change, value):  # noqa: D102
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self._emit_moves:
            self.moved.emit(self.balloon_id, self.pos())
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D102
        super().mouseReleaseEvent(event)
        self.dragFinished.emit(self.balloon_id)


class StampItem(QGraphicsObject):
    """Static "Ballooned Drawing" badge shown in the page's top-left corner.

    Mirrors the look of the stamp drawn onto exported PDFs (pdf_export.py's
    ``_stamp_page``) so the on-screen preview matches the export -- a
    transparent-background box with a red border and bold red text. It is
    not interactive (no selection/drag), just an overlay.
    """

    def __init__(self, text: str, width: float, height: float, font_size: float):
        super().__init__()
        self.text = text
        self.width = width
        self.height = height
        self.font_size = font_size
        self.setZValue(20.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: D102
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(191, 0, 0)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        outer_radius = self.height * STAMP_CORNER_RADIUS_PERCENT
        outer_pen = QPen(color)
        outer_pen.setWidthF(max(1.0, self.height * 0.09))
        painter.setPen(outer_pen)
        painter.drawRoundedRect(self.boundingRect(), outer_radius, outer_radius)

        font = QFont()
        font.setBold(True)
        font.setPointSizeF(max(1.0, self.font_size))
        painter.setPen(QPen(color))
        painter.setFont(font)
        painter.drawText(self.boundingRect(), int(Qt.AlignmentFlag.AlignCenter), self.text)

    def set_geometry(self, width: float, height: float, font_size: float) -> None:
        self.prepareGeometryChange()
        self.width = width
        self.height = height
        self.font_size = font_size
        self.update()


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
    leaderHandleMoved = pyqtSignal(str, float, float)
    leaderHandleDragFinished = pyqtSignal(str)
    emptySpaceClicked = pyqtSignal()
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
        self._update_canvas_background()

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._pdf_doc: Optional[PdfDocument] = None
        self._page_number = 0
        self._welcome_logo = QPixmap(str(Path(__file__).parent / "resources" / "balloonapp-logo.png"))
        self._zoom_level = 1.0
        self._view_rotation = 0
        self.balloon_size_percent = 100
        self.stamp_size_percent = 100
        self._effective_dpi = ACTUAL_SIZE_DPI
        self._balloon_items: dict[str, BalloonItem] = {}
        self._leader_items: dict[str, QGraphicsLineItem] = {}
        self._leader_handle_items: dict[str, LeaderHandleItem] = {}
        self._stamp_item: Optional[StampItem] = None
        self._selected_balloon_id: Optional[str] = None
        self._current_balloons: list[Balloon] = []

        self.add_balloon_mode = False
        self.leader_mode_balloon_id: Optional[str] = None

        self._panning = False
        self._pan_start_pos = None
        self._click_start_pos = None

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

    def _update_canvas_background(self) -> None:
        light_theme = self.palette().window().color().lightness() > 128
        self.setBackgroundBrush(QBrush(QColor("#e9edf2" if light_theme else "#3c3c3c")))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._update_canvas_background()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._pdf_doc is not None or self._pixmap_item is not None or self._welcome_logo.isNull():
            return
        # Paint in viewport coordinates, outside the drawing scene, so the
        # welcome logo stays centered and never affects zoom or export.
        area = self.viewport().rect()
        side = min(420, int(min(area.width(), area.height()) * 0.6))
        if side <= 0:
            return
        size = self._welcome_logo.size().scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF((area.width()-size.width()) / 2, (area.height()-size.height()) / 2,
                        size.width(), size.height())
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(0.4)
        painter.drawPixmap(target, self._welcome_logo, QRectF(self._welcome_logo.rect()))
        painter.end()

    # ------------------------------------------------------------------
    # Document / page loading
    # ------------------------------------------------------------------
    def load_document(self, pdf_doc: Optional[PdfDocument]) -> None:
        # Invalidate queued results before clearing the scene. A render from
        # the previous document must not put its page back after closing.
        self._request_counter += 1
        self._pdf_doc = pdf_doc
        self._worker.set_document(pdf_doc)
        self._scene.clear()
        self._pixmap_item = None
        self._balloon_items.clear()
        self._leader_items.clear()
        self._leader_handle_items.clear()
        self._stamp_item = None
        self._current_balloons = []
        self._selected_balloon_id = None
        self._page_number = 0
        self._pending_center_pdf = None
        self._latest_request_id = -1
        self.add_balloon_mode = False
        self.leader_mode_balloon_id = None
        self._panning = False
        self._pan_start_pos = None
        self._click_start_pos = None
        self.unsetCursor()
        self._view_rotation = 0
        self.setTransform(QTransform())
        self._scene.setSceneRect(0, 0, 0, 0)
        self.viewport().update()
        self.renderFinished.emit()

    def set_page(self, page_number: int, balloons: list[Balloon], preserve_view: bool = False) -> None:
        self._page_number = page_number
        self._current_balloons = balloons
        self._request_render(preserve_view=preserve_view)

    def refresh_balloons(self, balloons: list[Balloon]) -> None:
        self._current_balloons = balloons
        self._sync_balloon_items()
        self._sync_stamp_item()

    def set_balloon_size(self, percent: int) -> None:
        self.balloon_size_percent = max(50, min(200, percent))
        self._sync_balloon_items()

    def set_stamp_size(self, percent: int) -> None:
        self.stamp_size_percent = max(50, min(200, percent))
        self._sync_stamp_item()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    # ------------------------------------------------------------------
    # View rotation
    # ------------------------------------------------------------------
    @property
    def view_rotation(self) -> int:
        return self._view_rotation

    def rotate_view_cw(self) -> None:
        self._set_view_rotation(self._view_rotation + 90)

    def rotate_view_ccw(self) -> None:
        self._set_view_rotation(self._view_rotation - 90)

    def _set_view_rotation(self, degrees: int) -> None:
        # Rebuilt from scratch each time (rather than composing successive
        # QGraphicsView.rotate() calls) so repeated rotation never
        # accumulates floating-point drift away from an exact multiple of
        # 90 degrees.
        self._view_rotation = degrees % 360
        transform = QTransform()
        transform.rotate(self._view_rotation)
        self.setTransform(transform)

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
        if self._view_rotation % 180 == 90:
            # Sideways view: the page's on-screen bounding box has its
            # width/height swapped relative to the unrotated page.
            page_w_px, page_h_px = page_h_px, page_w_px
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
        self._sync_stamp_item()

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
        radius_px = BALLOON_RADIUS_PDF_POINTS * self.balloon_size_percent / 100 * dpi / 72.0

        page_balloons = [b for b in self._current_balloons if b.page_number == self._page_number]
        current_ids = {b.id for b in page_balloons}

        for stale_id in list(self._balloon_items):
            if stale_id not in current_ids:
                self._scene.removeItem(self._balloon_items.pop(stale_id))
                if stale_id in self._leader_items:
                    self._scene.removeItem(self._leader_items.pop(stale_id))
                if stale_id in self._leader_handle_items:
                    self._scene.removeItem(self._leader_handle_items.pop(stale_id))

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
            item.prepareGeometryChange()
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

                handle = self._leader_handle_items.get(balloon.id)
                if handle is None:
                    handle = LeaderHandleItem(balloon.id, half_size=radius_px * 0.4)
                    handle.setVisible(balloon.id == self._selected_balloon_id)
                    handle.moved.connect(self._on_leader_handle_item_moved)
                    handle.dragFinished.connect(self.leaderHandleDragFinished.emit)
                    self._scene.addItem(handle)
                    self._leader_handle_items[balloon.id] = handle
                handle.set_emit_moves(False)
                handle.prepareGeometryChange()
                handle.half_size = radius_px * 0.4
                handle.setPos(lx, ly)
                handle.set_emit_moves(True)
            else:
                if line_item is not None:
                    self._scene.removeItem(line_item)
                    del self._leader_items[balloon.id]
                if balloon.id in self._leader_handle_items:
                    self._scene.removeItem(self._leader_handle_items.pop(balloon.id))

    def _sync_stamp_item(self) -> None:
        """Position the "Ballooned Drawing" badge in the page's top-left
        corner, sized in the same PDF-points-scaled-by-DPI way as balloons
        (see _sync_balloon_items) so it matches the exported PDF's stamp.

        The badge certifies the drawing as fully reviewed, so it's only
        shown once every balloon on the drawing (all pages, not just the
        one currently displayed -- see ``_current_balloons``) is Accepted;
        a drawing with no balloons at all isn't "ballooned" either.
        """
        if self._pixmap_item is None:
            return

        all_accepted = bool(self._current_balloons) and all(
            b.status in (ReviewStatus.ACCEPTED.value, ReviewStatus.EDITED.value)
            for b in self._current_balloons
        )
        if not all_accepted:
            if self._stamp_item is not None:
                self._scene.removeItem(self._stamp_item)
                self._stamp_item = None
            return

        dpi = self._last_render_dpi()
        px_per_pt = dpi / 72.0
        scale = self.stamp_size_percent / 100
        margin = STAMP_MARGIN_PDF_POINTS * scale * px_per_pt
        font_size = STAMP_FONT_SIZE_PDF_POINTS * scale * px_per_pt
        page_width = self._pixmap_item.pixmap().width()
        # Snug the pill to the actual text width plus a little breathing
        # room, matching the export's stamp (see pdf_export.py's
        # _stamp_page) instead of a fixed width that leaves a wide dead gap
        # on either side of STAMP_TEXT.
        preview_font = QFont()
        preview_font.setBold(True)
        preview_font.setPointSizeF(max(1.0, font_size))
        text_width = QFontMetricsF(preview_font).horizontalAdvance(STAMP_TEXT)
        horizontal_padding = font_size * 1.0
        width = min(text_width + 2 * horizontal_padding, page_width - 2 * margin)
        height = font_size * 2.0

        if self._stamp_item is None:
            self._stamp_item = StampItem(STAMP_TEXT, width, height, font_size)
            self._scene.addItem(self._stamp_item)
        else:
            self._stamp_item.set_geometry(width, height, font_size)
        self._stamp_item.setPos(margin, margin)

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

    def _on_leader_handle_item_moved(self, balloon_id: str, scene_pos: QPointF) -> None:
        # Mirrors _on_item_moved above, but for the leader line's *start*
        # point: only the line's live endpoint is updated here, not the
        # model -- the caller persists the new position once dragging ends.
        dpi = self._last_render_dpi()
        pdf_x, pdf_y = pixel_to_pdf(scene_pos.x(), scene_pos.y(), dpi)
        self.leaderHandleMoved.emit(balloon_id, pdf_x, pdf_y)
        line_item = self._leader_items.get(balloon_id)
        if line_item is not None:
            line = line_item.line()
            line_item.setLine(scene_pos.x(), scene_pos.y(), line.x2(), line.y2())

    def select_balloon(self, balloon_id: Optional[str]) -> None:
        self._selected_balloon_id = balloon_id
        for bid, item in self._balloon_items.items():
            item.setSelected(bid == balloon_id)
        for bid, handle in self._leader_handle_items.items():
            handle.setVisible(bid == balloon_id)

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

            if isinstance(item, (BalloonItem, LeaderHandleItem)):
                super().mousePressEvent(event)
                return

            self._panning = True
            self._pan_start_pos = event.pos()
            self._click_start_pos = event.pos()  # fixed, to distinguish a click from a pan-drag
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
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # A press-release on empty space with negligible movement is a
            # click, not a pan-drag -- deselect whatever balloon was selected
            # (and its leader handle) rather than leaving it selected forever.
            if self._click_start_pos is not None and (event.pos() - self._click_start_pos).manhattanLength() <= 4:
                self.emptySpaceClicked.emit()
            self._pan_start_pos = None
            self._click_start_pos = None
            return
        super().mouseReleaseEvent(event)
