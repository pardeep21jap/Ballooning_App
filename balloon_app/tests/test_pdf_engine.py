"""Tests for pure coordinate-conversion helpers in pdf_engine."""

from __future__ import annotations

import math

from balloon_app.pdf_engine import (
    dpi_to_zoom,
    pdf_to_pixel,
    pixel_to_pdf,
    rect_pdf_to_pixel,
    rect_pixel_to_pdf,
)


def test_dpi_to_zoom():
    assert math.isclose(dpi_to_zoom(72.0), 1.0)
    assert math.isclose(dpi_to_zoom(144.0), 2.0)
    assert math.isclose(dpi_to_zoom(36.0), 0.5)


def test_pdf_to_pixel_and_back_roundtrip():
    dpi = 150.0
    original = (100.25, 200.75)
    px = pdf_to_pixel(*original, dpi)
    back = pixel_to_pdf(*px, dpi)
    assert math.isclose(back[0], original[0], abs_tol=1e-9)
    assert math.isclose(back[1], original[1], abs_tol=1e-9)


def test_pdf_to_pixel_scaling():
    # At 72 DPI (1:1), pixel coords should equal PDF point coords.
    px, py = pdf_to_pixel(50.0, 60.0, 72.0)
    assert math.isclose(px, 50.0)
    assert math.isclose(py, 60.0)

    # At 144 DPI (2x), pixel coords should double.
    px2, py2 = pdf_to_pixel(50.0, 60.0, 144.0)
    assert math.isclose(px2, 100.0)
    assert math.isclose(py2, 120.0)


def test_rect_roundtrip():
    dpi = 200.0
    rect = (10.0, 20.0, 110.0, 220.0)
    px_rect = rect_pdf_to_pixel(rect, dpi)
    back = rect_pixel_to_pdf(px_rect, dpi)
    for a, b in zip(rect, back):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_pixel_to_pdf_zero_dpi_safe():
    # Defensive: must not raise a ZeroDivisionError.
    assert pixel_to_pdf(10.0, 10.0, 0.0) == (0.0, 0.0)
