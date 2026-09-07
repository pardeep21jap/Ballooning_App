"""Canvas unload must clear visible ink and reject late render callbacks."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from balloon_app.pdf_view import PdfGraphicsView, StampItem


def test_unload_clears_drawing_and_ignores_pending_render():
    app = QApplication.instance() or QApplication([])
    view = PdfGraphicsView()
    try:
        old_request = view._request_counter
        pixels = bytes([255] * 12)
        view._on_rendered(old_request, pixels, 2, 2, 96)
        assert view.scene().items()
        view.add_balloon_mode = True
        view.leader_mode_balloon_id = "old-balloon"
        view._selected_balloon_id = "old-balloon"
        view.load_document(None)
        assert view.scene().items() == []
        assert view._pixmap_item is None
        assert view._current_balloons == []
        assert view._selected_balloon_id is None
        assert not view.add_balloon_mode
        assert view.leader_mode_balloon_id is None
        messages = []
        view.statusMessage.connect(messages.append)
        view._on_rendered(old_request, pixels, 2, 2, 96)
        view._on_render_failed(old_request, "Closed document")
        assert view.scene().items() == []
        assert messages == []
        # A later document's render still works after the old scene clears.
        view._on_rendered(view._request_counter, pixels, 2, 2, 96)
        assert view._pixmap_item is not None
        assert view.scene().items()
    finally:
        view.shutdown()
        view.close()
        app.processEvents()


def test_stamp_item_appears_top_left_and_scales_with_stamp_size():
    app = QApplication.instance() or QApplication([])
    view = PdfGraphicsView()
    try:
        # A realistically large page -- the stamp's width is snugged to its
        # text plus a little padding (see _sync_stamp_item), and clamped
        # against the page width as a safety cap for pathologically small
        # pages; a tiny 200x200 stub page would hit that cap at 200% stamp
        # size well before the scaling behavior under test ever mattered.
        page_size = 2000
        pixels = bytes([255] * (page_size * page_size * 3))
        view._on_rendered(view._request_counter, pixels, page_size, page_size, 96)

        assert isinstance(view._stamp_item, StampItem)
        assert view._stamp_item in view.scene().items()
        pos = view._stamp_item.pos()
        assert 0 < pos.x() < 20
        assert 0 < pos.y() < 20

        base_width = view._stamp_item.width
        base_font_size = view._stamp_item.font_size

        view.set_stamp_size(200)
        assert view.stamp_size_percent == 200
        assert view._stamp_item.font_size == pytest.approx(base_font_size * 2)
        assert view._stamp_item.width > base_width

        # A fresh render (e.g. after switching pages) must keep the badge,
        # not silently drop it -- same expectation as balloon items.
        view._on_rendered(view._request_counter, pixels, page_size, page_size, 96)
        assert view._stamp_item is not None
        assert view._stamp_item in view.scene().items()

        # load_document(None) tears the whole scene down; nothing should
        # dangle a reference to the destroyed C++ item afterwards.
        view.load_document(None)
        assert view._stamp_item is None
    finally:
        view.shutdown()
        view.close()
        app.processEvents()
