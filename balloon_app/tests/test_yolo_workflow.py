"""Regression coverage for settings, worker reuse, and detector fallback."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pymupdf
import pytest

from balloon_app import app, auto_balloon
from balloon_app.config import RULES_OCR_MODEL_VERSION, AppSettings
from balloon_app.pdf_engine import PdfDocument


@pytest.mark.parametrize("whole_drawing", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_actions_forward_yolo_settings(monkeypatch, whole_drawing, enabled):
    window = SimpleNamespace(
        project=Mock(), drawing=SimpleNamespace(id="drawing", page_count=2),
        pdf_doc=object(), current_page=0,
        settings=AppSettings(use_yolo_if_available=enabled, yolo_model_path="custom.pt"),
        _ensure_default_tolerances=lambda: None, _run_background=Mock(),
        _on_auto_balloon_page_done=Mock(), _on_auto_balloon_drawing_done=Mock(),
    )
    monkeypatch.setattr(app, "resolve_source_path", lambda drawing: "drawing.pdf")
    monkeypatch.setattr(app.QMessageBox, "question", lambda *args: app.QMessageBox.StandardButton.Yes)
    action = app.MainWindow._auto_balloon_entire_drawing if whole_drawing else app.MainWindow._auto_balloon_current_page
    action(window)
    args = window._run_background.call_args.args
    assert args[0] is (app._auto_balloon_drawing_task if whole_drawing else app._auto_balloon_page_task)
    assert args[-2:] == (enabled, "custom.pt")


@pytest.mark.parametrize("whole_drawing", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_workers_create_and_reuse_detector(monkeypatch, whole_drawing, enabled):
    factory = Mock(return_value=SimpleNamespace(available=True))
    monkeypatch.setattr(app, "YoloDetector", factory)
    monkeypatch.setattr(app, "PdfDocument", Mock())
    pipeline = Mock(return_value=SimpleNamespace(balloons=[]))
    monkeypatch.setattr(app, "auto_balloon_page", pipeline)
    worker = app._auto_balloon_drawing_task if whole_drawing else app._auto_balloon_page_task
    worker("drawing.pdf", "drawing", 2 if whole_drawing else 0, [], 1, 200, None,
           use_yolo_if_available=enabled, yolo_model_path="custom.pt")
    assert pipeline.call_count == (2 if whole_drawing else 1)
    if enabled:
        factory.assert_called_once_with("custom.pt")
    else:
        factory.assert_not_called()
    for call in pipeline.call_args_list:
        assert call.kwargs.get("detector") is (factory.return_value if enabled else None)


@pytest.mark.parametrize("failure", ["missing_path", "directory", "empty", "missing_package", "load_error"])
def test_unavailable_yolo_falls_back(monkeypatch, tmp_path, failure):
    model_path = tmp_path / "model.pt"
    if failure == "directory":
        model_path.mkdir()
    elif failure == "empty":
        model_path = ""
    elif failure != "missing_path":
        model_path.touch()
    if failure == "missing_package":
        monkeypatch.setitem(sys.modules, "ultralytics", None)
    elif failure == "load_error":
        monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Mock(side_effect=RuntimeError("bad model"))))
    detector = auto_balloon.YoloDetector(model_path)
    assert not detector.available
    fallback = Mock(available=True)
    fallback.detect.return_value = [auto_balloon.Detection((40, 40, 90, 60), "linear_dimension", .8, "12.50")]
    monkeypatch.setattr(auto_balloon, "RulesOcrDetector", Mock(return_value=fallback))
    result = _run_page(tmp_path, detector)
    fallback.detect.assert_called_once()
    assert result.ocr_available and result.used_ocr
    assert result.balloons[0].nominal == 12.5
    assert result.balloons[0].model_version == RULES_OCR_MODEL_VERSION


def _run_page(tmp_path, detector, native=False):
    path = tmp_path / "drawing.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        if native:
            page.insert_text((100, 150), "12.50", fontsize=12)
        doc.save(path)
    doc = PdfDocument(path)
    doc.open()
    try:
        return auto_balloon.auto_balloon_page(doc, "drawing", 0, [], 1, dpi=144, detector=detector)
    finally:
        doc.close()


def test_native_text_still_takes_priority(tmp_path):
    detector = Mock(available=True)
    result = _run_page(tmp_path, detector, native=True)
    detector.detect.assert_not_called()
    assert result.balloons[0].nominal == 12.5
    assert result.balloons[0].model_version == RULES_OCR_MODEL_VERSION
    assert not result.used_ocr


@pytest.mark.parametrize("has_boxes", [False, True])
def test_yolo_results_are_not_identified_as_ocr(tmp_path, has_boxes):
    detector = auto_balloon.YoloDetector.__new__(auto_balloon.YoloDetector)
    detector.available = True
    detector.detect = Mock(return_value=[auto_balloon.Detection((40, 40, 90, 60), "diameter", .8)] if has_boxes else [])
    result = _run_page(tmp_path, detector)
    assert not result.used_ocr
    if has_boxes:
        assert result.balloons[0].char_type == "diameter"
        assert result.balloons[0].model_version == "yolo"
    else:
        assert "YOLO" in result.message
