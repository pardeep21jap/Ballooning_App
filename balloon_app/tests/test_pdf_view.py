"""Canvas unload must clear visible ink and reject late render callbacks."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from balloon_app.pdf_view import PdfGraphicsView


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
