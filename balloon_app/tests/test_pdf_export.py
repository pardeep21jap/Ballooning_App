"""Tests for balloon placement in exported ballooned PDFs."""

from __future__ import annotations

import numpy as np
import pytest

from balloon_app.data_model import Balloon, Drawing
from balloon_app.pdf_export import export_ballooned_pdf

fitz = pytest.importorskip("pymupdf")


@pytest.mark.parametrize("percent, diameter", [(50, 9), (100, 18), (200, 36)])
def test_export_uses_selected_balloon_size(tmp_path, percent, diameter):
    source = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.save(source)
    doc.close()
    drawing = Drawing(file_name="source.pdf", original_path=str(source), page_count=1)
    balloon = Balloon(number=1, drawing_id=drawing.id, x=100, y=100)
    output = tmp_path / "output.pdf"
    export_ballooned_pdf(drawing, [balloon], output, balloon_size_percent=percent)
    with fitz.open(output) as exported:
        circle = exported[0].get_drawings()[0]["rect"]
        assert circle.width == pytest.approx(diameter)
        assert circle.height == pytest.approx(diameter)
        assert "1" in exported[0].get_text()


def test_balloon_lands_on_target_on_rotated_page(tmp_path):
    """Regression test: on a page with /Rotate 90, a balloon placed at a
    page.rect-space point (the app's stored convention -- see
    pdf_engine.py) must render at that same visual location in the output
    PDF. Before the pdf_export.py fix, draw_circle/insert_textbox were fed
    the stored coordinates directly, but PyMuPDF's drawing APIs expect
    raw/mediabox-space coordinates, so every balloon on a rotated page was
    drawn far from its intended target.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=400)
    page.insert_text((20, 40), "0.500", fontsize=12)
    page.set_rotation(90)
    doc.save(str(source_path))
    doc.close()

    # Same point a correctly-fixed extract_text_blocks() would report for
    # that text, in the app's page.rect (rotated display) convention.
    target_x, target_y = 44.0, 24.0

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    balloon = Balloon(number=1, drawing_id=drawing.id, page_number=0, x=target_x, y=target_y)

    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [balloon], output_path)

    out_doc = fitz.open(str(output_path))
    out_page = out_doc[0]
    dpi = 150.0
    zoom = dpi / 72.0
    pix = out_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    out_doc.close()

    # A default Balloon (source="manual", status="accepted") is filled with
    # COLOR_MANUAL, a distinct blue -- isolate it from the page's own black
    # "0.500" text and white background rather than matching any non-white
    # pixel (which the source text itself would also satisfy).
    blue_mask = (img[:, :, 2] > 150) & (img[:, :, 0] < 150)
    ys, xs = np.where(blue_mask)
    assert len(xs) > 0, "balloon circle was not drawn at all"
    drawn_x, drawn_y = xs.mean(), ys.mean()

    expected_x, expected_y = target_x * zoom, target_y * zoom
    assert abs(drawn_x - expected_x) < 20, f"balloon x={drawn_x} far from expected {expected_x}"
    assert abs(drawn_y - expected_y) < 20, f"balloon y={drawn_y} far from expected {expected_y}"


def test_balloon_number_is_rendered_on_unrotated_page(tmp_path):
    """Regression test: the balloon number must be drawn even on an
    ordinary, unrotated page.

    The text box used to be ``radius`` wide but only ``radius * 0.75``
    tall -- too short for insert_textbox to fit even a single digit at the
    configured font size, so it silently drew nothing on *every* export,
    independent of page rotation. The box must be the full circle-diameter
    square instead.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=400)
    doc.save(str(source_path))
    doc.close()

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    balloon = Balloon(number=12, drawing_id=drawing.id, page_number=0, x=100.0, y=100.0)

    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [balloon], output_path)

    out_doc = fitz.open(str(output_path))
    spans = [
        span
        for block in out_doc[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]
    out_doc.close()
    assert any(s["text"].strip() == "12" for s in spans), (
        f"balloon number '12' was not found as rendered text (found: {[s['text'] for s in spans]})"
    )


def test_balloon_number_text_is_rendered_on_rotated_page(tmp_path):
    """Regression test: the balloon's number label must actually be drawn,
    at the balloon's location and right-side up, on a rotated page.

    Beyond the text-box sizing bug above, insert_textbox's rect must be
    derotated the same way draw_circle/draw_line's points are (to land in
    the right place), and also needs `rotate=page.rotation` so the glyphs
    themselves are rotated to read upright once PyMuPDF applies the page's
    own rotation for display.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=400)
    page.set_rotation(90)
    doc.save(str(source_path))
    doc.close()

    target_x, target_y = 100.0, 100.0
    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    balloon = Balloon(number=7, drawing_id=drawing.id, page_number=0, x=target_x, y=target_y)

    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [balloon], output_path)

    out_doc = fitz.open(str(output_path))
    out_page = out_doc[0]
    spans = [
        span
        for block in out_page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]

    matches = [s for s in spans if s["text"].strip() == "7"]
    assert matches, f"balloon number '7' was not found as rendered text (found: {[s['text'] for s in spans]})"

    # It should also sit at the balloon's location, not somewhere stray.
    label_rect = fitz.Rect(matches[0]["bbox"]) * out_page.rotation_matrix
    label_rect.normalize()
    out_doc.close()
    label_center = fitz.Point((label_rect.x0 + label_rect.x1) / 2, (label_rect.y0 + label_rect.y1) / 2)
    assert label_center.distance_to(fitz.Point(target_x, target_y)) < 15
