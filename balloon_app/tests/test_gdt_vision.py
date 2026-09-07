"""Independent PDF drawings exercise symbol vision and balloon metadata."""
import cv2
import numpy as np
import pymupdf as fitz
import pytest

from balloon_app.auto_balloon import Detection, auto_balloon_page
from balloon_app.gdt_vision import SYMBOLS, classify_symbol, find_gdt_frames
from balloon_app.pdf_engine import PdfDocument


def draw_callout(page, name, datums=False):
    page.draw_rect((40, 70, 185, 130), width=0.8)
    page.draw_line((100, 70), (100, 130), width=0.8)
    page.draw_line((40, 100), (15, 100), width=0.8)
    page.draw_line((15, 90), (15, 170), width=0.8)
    page.insert_text((108, 110), "0.01", fontsize=22)
    if datums:
        page.draw_rect((185, 70, 220, 130), width=0.8)
        page.insert_text((195, 110), "A", fontsize=22)

    def line(a, b):
        page.draw_line(a, b, width=1)

    if name == "Straightness":
        line((49, 100), (91, 100))
    elif name == "Flatness":
        page.draw_polyline([(48, 113), (59, 87), (92, 87), (81, 113)], closePath=True, width=1)
    elif name in ("Circularity", "Concentricity", "Position", "Cylindricity"):
        page.draw_circle((70, 100), 14, width=1)
        if name == "Concentricity":
            page.draw_circle((70, 100), 9, width=1)
        if name == "Position":
            line((48, 100), (92, 100))
            line((70, 78), (70, 122))
        if name == "Cylindricity":
            line((49, 115), (61, 85))
            line((79, 115), (91, 85))
    elif name == "Angularity":
        line((49, 115), (91, 115))
        line((49, 115), (87, 86))
    elif name == "Perpendicularity":
        line((49, 117), (91, 117))
        line((70, 117), (70, 82))
    elif name == "Parallelism":
        line((52, 117), (65, 83))
        line((73, 117), (86, 83))
    elif name == "Symmetry":
        line((49, 100), (91, 100))
        line((59, 88), (81, 88))
        line((59, 112), (81, 112))
    elif name in ("Profile of a Line", "Profile of a Surface"):
        page.draw_bezier((49, 112), (49, 84), (91, 84), (91, 112), width=1)
        if name == "Profile of a Surface":
            line((49, 112), (91, 112))
    elif name in ("Circular Runout", "Total Runout"):
        starts = [63] if name == "Circular Runout" else [53, 74]
        for x in starts:
            line((x, 117), (x+13, 83))
            page.draw_polyline([(x+13, 83), (x+6, 92), (x+15, 94)], closePath=True, fill=(0, 0, 0), width=0.5)
        if len(starts) == 2:
            line((53, 117), (74, 117))


@pytest.mark.parametrize("name", SYMBOLS)
@pytest.mark.parametrize("dpi", [100, 200])
def test_all_symbols_from_vector_pdf(tmp_path, name, dpi):
    path = tmp_path / "frame.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    draw_callout(page, name, datums=True)
    doc.save(path)
    doc.close()
    pdf = PdfDocument(path)
    result = auto_balloon_page(pdf, "drawing", 0, [], 1, dpi=dpi)
    pdf.close()
    assert len(result.balloons) == 1
    balloon = result.balloons[0]
    assert balloon.char_type == "gdt_frame"
    assert balloon.gdt_symbol == name
    assert balloon.gdt_tolerance == "0.01"
    assert balloon.datums == "A"
    # A feature control frame's stated value is the maximum allowed
    # variation, zero being the best case: nominal is the box value, upper
    # limit equals it, lower limit is 0.0.
    assert balloon.nominal == pytest.approx(0.01)
    assert balloon.tol_plus == pytest.approx(0.0)
    assert balloon.tol_minus == pytest.approx(0.01)
    assert balloon.lower_limit == pytest.approx(0.0)
    assert balloon.upper_limit == pytest.approx(0.01)
    assert balloon.status == "pending"


@pytest.mark.parametrize("name", SYMBOLS)
def test_scanned_symbol_with_blur_and_noise(name):
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    draw_callout(page, name)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.6)
    noise = np.random.default_rng(10).normal(0, 3, gray.shape)
    scan = np.clip(blurred.astype(float) + noise, 0, 255).astype(np.uint8)
    frames = find_gdt_frames(scan)
    doc.close()
    assert len(frames) == 1
    assert frames[0].symbol == name


@pytest.mark.parametrize("kind", ["blank", "rectangle", "letter", "triangle"])
def test_unknown_shapes_are_not_named(kind):
    ink = np.zeros((80, 80), np.uint8)
    if kind == "rectangle":
        cv2.rectangle(ink, (15, 20), (65, 60), 255, 2)
    elif kind == "letter":
        cv2.putText(ink, "A", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.7, 255, 2)
    elif kind == "triangle":
        cv2.polylines(ink, [np.array([(15, 60), (40, 20), (65, 60)])], True, 255, 2)
    assert classify_symbol(ink) is None


def test_unframed_circle_is_not_a_callout():
    image = np.full((400, 400), 255, np.uint8)
    cv2.circle(image, (100, 100), 20, 0, 2)
    cv2.putText(image, "0.01", (140, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    assert find_gdt_frames(image) == []


@pytest.mark.parametrize("name", SYMBOLS)
@pytest.mark.parametrize("mixed_page", [False, True])
def test_scanned_pdf_uses_cell_ocr_without_duplicate_balloons(tmp_path, monkeypatch, name, mixed_page):
    source = fitz.open()
    page = source.new_page(width=400, height=400)
    draw_callout(page, name, datums=True)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    scan = fitz.open()
    page = scan.new_page(width=400, height=400)
    page.insert_image(page.rect, pixmap=pix)
    if mixed_page:
        page.insert_text((250, 250), "25.50", fontsize=12)
    path = tmp_path / "scan.pdf"
    scan.save(path)
    scan.close()
    source.close()

    class FakeOcr:
        available = True
        error = None

        def __init__(self, *args):
            self.reads = iter(["0.01", "A"])

        def detect(self, image):
            # Simulate full-page OCR already finding the value. The frame
            # result must replace it, not create a second linear balloon.
            return [Detection(bbox=(216, 180, 300, 222), label="linear_dimension",
                              confidence=0.7, raw_text="0.01")]

        def read_cell(self, image):
            assert image.size > 0
            return next(self.reads)

    monkeypatch.setattr("balloon_app.auto_balloon.RulesOcrDetector", FakeOcr)
    pdf = PdfDocument(path)
    result = auto_balloon_page(pdf, "drawing", 0, [], 1, dpi=144)
    pdf.close()
    assert len(result.balloons) == (2 if mixed_page else 1)
    balloon = next(b for b in result.balloons if b.char_type == "gdt_frame")
    assert balloon.gdt_symbol == name
    assert balloon.gdt_tolerance == "0.01"
    assert balloon.datums == "A"
    assert result.used_ocr


def test_scanned_frame_does_not_invent_missing_tolerance(tmp_path, monkeypatch):
    source = fitz.open()
    page = source.new_page(width=400, height=400)
    draw_callout(page, "Flatness")
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_image(page.rect, pixmap=pix)
    path = tmp_path / "missing_ocr.pdf"
    doc.save(path)
    doc.close()
    source.close()

    class MissingOcr:
        available = False
        error = "Tesseract unavailable"

        def __init__(self, *args):
            pass

    monkeypatch.setattr("balloon_app.auto_balloon.RulesOcrDetector", MissingOcr)
    pdf = PdfDocument(path)
    result = auto_balloon_page(pdf, "drawing", 0, [], 1, dpi=144)
    pdf.close()
    assert result.balloons == []
    assert not result.ocr_available
    assert "Tesseract" in result.message
