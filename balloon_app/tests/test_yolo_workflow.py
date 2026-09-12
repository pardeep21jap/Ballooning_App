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


@pytest.mark.parametrize("enabled", [False, True])
def test_native_text_unchanged_without_available_yolo(tmp_path, enabled):
    detector = auto_balloon.YoloDetector(tmp_path / "missing.pt") if enabled else None
    result = _run_page(tmp_path, detector, native=True)
    assert len(result.balloons) == 1
    assert result.balloons[0].nominal == 12.5
    assert result.balloons[0].model_version == RULES_OCR_MODEL_VERSION
    assert not result.used_ocr


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("has_boxes", [False, True])
def test_yolo_supplements_existing_detections(monkeypatch, tmp_path, native, has_boxes):
    fallback = Mock(available=True)
    fallback.detect.return_value = [auto_balloon.Detection((40, 40, 90, 60), "linear_dimension", .8, "12.50")]
    factory = Mock(return_value=fallback)
    monkeypatch.setattr(auto_balloon, "RulesOcrDetector", factory)
    detector = auto_balloon.YoloDetector.__new__(auto_balloon.YoloDetector)
    detector.available = True
    detector.detect = Mock(return_value=[auto_balloon.Detection((100, 40, 150, 60), "diameter", .8)] if has_boxes else [])
    result = _run_page(tmp_path, detector, native=native)
    detector.detect.assert_called_once()
    if native:
        factory.assert_not_called()
    else:
        fallback.detect.assert_called_once()
    assert result.used_ocr is (not native)
    assert len(result.balloons) == (2 if has_boxes else 1)
    rules = [b for b in result.balloons if b.model_version == RULES_OCR_MODEL_VERSION]
    assert len(rules) == 1
    assert rules[0].nominal == 12.5
    assert [b.number for b in result.balloons] == list(range(1, len(result.balloons) + 1))
    if has_boxes:
        yolo = [b for b in result.balloons if b.model_version == "yolo"]
        assert len(yolo) == 1
        assert yolo[0].char_type == "diameter"
    assert not result.message


@pytest.mark.parametrize("rules_box,yolo_box,enabled,expected", [
    ((40, 40, 100, 60), (40, 40, 100, 60), True, 1),
    ((40, 40, 100, 60), (60, 40, 120, 60), True, 1),  # IoU exactly 0.5
    ((40, 40, 100, 60), (61, 40, 121, 60), True, 2),  # below 0.5
    ((40, 40, 100, 60), (95, 40, 155, 60), True, 2),  # nearby, slight overlap
    ((40, 40, 100, 60), (102, 40, 162, 60), True, 2),  # nearby, separate
    (None, (40, 40, 100, 60), True, 1),
    ((40, 40, 100, 60), None, True, 1),
    ((40, 40, 100, 60), (40, 40, 100, 60), False, 1),
])
def test_yolo_overlap_prefers_rules(monkeypatch, tmp_path, rules_box, yolo_box, enabled, expected,
                                  yolo_type="linear_dimension"):
    fallback = Mock(available=True)
    fallback.detect.return_value = (
        [auto_balloon.Detection(rules_box, "linear_dimension", .8, "12.50 \u00b10.10")]
        if rules_box else []
    )
    monkeypatch.setattr(auto_balloon, "RulesOcrDetector", Mock(return_value=fallback))
    detector = auto_balloon.YoloDetector.__new__(auto_balloon.YoloDetector)
    detector.available = True
    detector.detect = Mock(return_value=(
        [auto_balloon.Detection(yolo_box, yolo_type, .95)] if yolo_box else []
    ))
    result = _run_page(tmp_path, detector if enabled else None)
    assert len(result.balloons) == expected
    rules = [b for b in result.balloons if b.model_version == RULES_OCR_MODEL_VERSION]
    if rules_box:
        assert len(rules) == 1
        assert rules[0].raw_text == "12.50 \u00b10.10"
        assert rules[0].nominal == 12.5
        assert rules[0].tol_plus == .1
        assert rules[0].char_type == "linear_dimension"
    else:
        assert result.balloons[0].model_version == "yolo"
    if not enabled:
        detector.detect.assert_not_called()


@pytest.mark.parametrize("rules_box,yolo_box,yolo_type,expected", [
    ((40, 40, 100, 60), (20, 20, 140, 80), "linear_dimension", 1),
    ((20, 20, 140, 80), (40, 40, 100, 60), "linear_dimension", 1),
    ((40, 40, 100, 60), (52, 20, 172, 80), "linear_dimension", 1),  # containment 0.80
    ((40, 40, 100, 60), (53, 20, 173, 80), "linear_dimension", 2),  # below 0.80
    ((40, 40, 100, 60), (95, 20, 215, 80), "linear_dimension", 2),
    ((40, 40, 100, 60), (20, 20, 140, 80), "diameter", 2),
    ((20, 20, 140, 80), (40, 40, 100, 60), "diameter", 2),
    ((40, 40, 100, 60), (45, 40, 105, 60), "diameter", 2),  # IoU 0.846
    ((40, 40, 100, 60), (40, 40, 94, 60), "diameter", 1),  # IoU 0.90
    ((40, 40, 100, 60), (20, 20, 140, 80), "other", 1),
    ((40, 40, 100, 60), (20, 20, 140, 80), "", 1),
])
def test_yolo_containment_and_type_compatibility(monkeypatch, tmp_path, rules_box, yolo_box,
                                               yolo_type, expected):
    test_yolo_overlap_prefers_rules(monkeypatch, tmp_path, rules_box, yolo_box, True, expected,
                                  yolo_type=yolo_type)
