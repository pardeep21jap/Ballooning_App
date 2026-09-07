"""Tests for balloon placement in exported ballooned PDFs."""

from __future__ import annotations

import numpy as np
import pytest

from balloon_app.data_model import Balloon, Drawing
from balloon_app.pdf_export import STAMP_TEXT, export_ballooned_pdf

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


@pytest.mark.parametrize("percent, expected_text_width", [(50, 28.75), (100, 57.50), (200, 114.99)])
def test_export_uses_selected_stamp_size(tmp_path, percent, expected_text_width):
    source = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=400, height=400)
    doc.save(source)
    doc.close()
    drawing = Drawing(file_name="source.pdf", original_path=str(source), page_count=1)
    output = tmp_path / "output.pdf"
    export_ballooned_pdf(drawing, [], output, stamp_size_percent=percent)
    with fitz.open(output) as exported:
        spans = [
            span
            for block in exported[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip() == STAMP_TEXT
        ]
        assert spans, "stamp text not found"
        bbox = spans[0]["bbox"]
        assert (bbox[2] - bbox[0]) == pytest.approx(expected_text_width, abs=1.0)


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


@pytest.mark.parametrize("number", [9, 14])
def test_balloon_number_is_vertically_centered_in_circle(tmp_path, number):
    """Regression test: the balloon's number must sit in the vertical
    center of its circle, not float above it.

    insert_textbox lays a line out top-down from the box's top edge,
    reserving the font's full ascender+descender height -- room a bare
    digit (no descender) never uses -- so a box simply spanning the
    circle's diameter left the digit's actual ink sitting well above
    center, with all the leftover space stuck below it.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.save(str(source_path))
    doc.close()

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    center_x, center_y = 100.0, 100.0
    balloon = Balloon(number=number, drawing_id=drawing.id, page_number=0, x=center_x, y=center_y)

    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [balloon], output_path)

    out_doc = fitz.open(str(output_path))
    out_page = out_doc[0]
    zoom = 4.0
    pix = out_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    out_doc.close()

    radius = 9.0
    x0, y0 = int((center_x - radius) * zoom), int((center_y - radius) * zoom)
    x1, y1 = int((center_x + radius) * zoom), int((center_y + radius) * zoom)
    crop = img[y0:y1, x0:x1]
    h, w, _ = crop.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # Exclude the crop's corners (outside the circle, plain white page
    # background) so only white pixels actually inside the balloon --
    # i.e. the digit itself -- count.
    inside_circle = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (radius * zoom * 0.85)
    white_mask = (crop[:, :, 0] > 200) & (crop[:, :, 1] > 200) & (crop[:, :, 2] > 200) & inside_circle
    ys, _xs = np.where(white_mask)
    assert len(ys) > 0, "balloon number was not found inside the circle"
    vertical_offset_pt = (ys.mean() - h / 2) / zoom
    assert abs(vertical_offset_pt) < 1.0, f"number is {vertical_offset_pt:+.2f}pt off circle center vertically"


def test_leader_line_stops_at_balloon_edge_not_center(tmp_path):
    """Regression test: the leader line must terminate at the balloon
    circle's circumference, not its center. Drawing all the way to the
    center reads -- through the circle's fill_opacity, which isn't fully
    opaque -- as a line pointing into the middle of the balloon rather than
    stopping cleanly at its edge.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.save(str(source_path))
    doc.close()

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    center_x, center_y = 150.0, 150.0
    leader_x, leader_y = 50.0, 50.0
    balloon = Balloon(
        number=1, drawing_id=drawing.id, page_number=0,
        x=center_x, y=center_y, leader_x=leader_x, leader_y=leader_y,
    )

    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [balloon], output_path)

    out_doc = fitz.open(str(output_path))
    lines = [
        item for d in out_doc[0].get_drawings() for item in d["items"] if item[0] == "l"
    ]
    out_doc.close()
    assert lines, "leader line was not drawn"
    _, p1, p2 = lines[0]
    center = fitz.Point(center_x, center_y)
    # The endpoint nearer the balloon must sit on the circle's radius, not
    # coincide with its center.
    balloon_end = p2 if p1.distance_to(center) > p2.distance_to(center) else p1
    assert balloon_end.distance_to(center) == pytest.approx(9.0, abs=0.5)


def test_stamp_text_is_vertically_centered_in_its_border(tmp_path):
    """Regression test: the "BALLOONED DRAWING" stamp text must sit in the
    vertical center of its pill-shaped border, not float above it -- the
    same top-down insert_textbox layout bug fixed for balloon numbers
    (see test_balloon_number_is_vertically_centered_in_circle) also
    affected the stamp, which shares the same box for its border and text.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=600, height=400)
    doc.save(str(source_path))
    doc.close()

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [], output_path)

    out_doc = fitz.open(str(output_path))
    out_page = out_doc[0]
    borders = [d["rect"] for d in out_page.get_drawings() if d["type"] == "s"]
    assert borders, "stamp border was not drawn"
    border = borders[0]
    border_center_y = (border.y0 + border.y1) / 2.0

    zoom = 8.0
    pix = out_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    out_doc.close()

    # Inset from the border rect so only the text's own ink counts, not the
    # border stroke itself (both are drawn in the same dark-red color).
    inset = 2.0
    x0, y0 = int((border.x0 + inset) * zoom), int((border.y0 + inset) * zoom)
    x1, y1 = int((border.x1 - inset) * zoom), int((border.y1 - inset) * zoom)
    crop = img[y0:y1, x0:x1]
    red_mask = (crop[:, :, 0] > 140) & (crop[:, :, 1] < 90) & (crop[:, :, 2] < 90)
    ys, _xs = np.where(red_mask)
    assert len(ys) > 0, "stamp text ink was not found inside the border"
    text_center_y = border.y0 + inset + ys.mean() / zoom
    assert abs(text_center_y - border_center_y) < 1.0, (
        f"stamp text is {text_center_y - border_center_y:+.2f}pt off the border's vertical center"
    )


def test_stamp_is_rendered_in_top_left_corner(tmp_path):
    """Every exported page must be stamped 'Ballooned Drawing' near the
    visual top-left corner, even on a page with no balloons on it.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=400)
    doc.save(str(source_path))
    doc.close()

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [], output_path)

    out_doc = fitz.open(str(output_path))
    spans = [
        span
        for block in out_doc[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]
    out_doc.close()

    matches = [s for s in spans if s["text"].strip() == STAMP_TEXT]
    assert matches, f"stamp text not found (found: {[s['text'] for s in spans]})"
    bbox = matches[0]["bbox"]
    assert bbox[0] < 100 and bbox[1] < 100, f"stamp not near top-left corner: {bbox}"


def test_stamp_is_upright_in_top_left_corner_on_rotated_page(tmp_path):
    """The stamp must stay in the page's visual top-left corner and read
    upright even when the source page has a /Rotate value, mirroring the
    derotation handling used for balloons.
    """
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=400)
    page.set_rotation(90)
    doc.save(str(source_path))
    doc.close()

    drawing = Drawing(file_name="source.pdf", original_path=str(source_path), page_count=1)
    output_path = tmp_path / "out.pdf"
    export_ballooned_pdf(drawing, [], output_path)

    out_doc = fitz.open(str(output_path))
    out_page = out_doc[0]
    spans = [
        span
        for block in out_page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]

    matches = [s for s in spans if s["text"].strip() == STAMP_TEXT]
    assert matches, f"stamp text not found (found: {[s['text'] for s in spans]})"
    # Map the drawn (mediabox-space) bbox back into display space, the same
    # way test_balloon_number_text_is_rendered_on_rotated_page does, to
    # check it landed in the visual top-left corner of the rotated page.
    label_rect = fitz.Rect(matches[0]["bbox"]) * out_page.rotation_matrix
    label_rect.normalize()
    out_doc.close()
    assert label_rect.x0 < 100 and label_rect.y0 < 100, (
        f"stamp not near visual top-left corner: {label_rect}"
    )
