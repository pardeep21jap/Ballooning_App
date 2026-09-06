"""Local GD&T symbol recognition on rendered PDFs and scans.

Only inspect enclosed cells followed by a matching value cell. Compare the
isolated ink with geometric templates, rejecting weak or ambiguous matches.
No font mapping, network service, or downloaded model is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


SYMBOLS = {
    "Straightness": "⏤", "Flatness": "⏥", "Circularity": "○",
    "Cylindricity": "⌭", "Angularity": "∠", "Perpendicularity": "⟂",
    "Parallelism": "∥", "Position": "⌖", "Concentricity": "◎",
    "Symmetry": "⌯", "Profile of a Line": "⌒",
    "Profile of a Surface": "⌓", "Total Runout": "⌇", "Circular Runout": "↗",
}


@dataclass(frozen=True)
class GdtFrame:
    symbol: str
    confidence: float
    # Pixel coordinates, excluding the symbol compartment.
    cells: tuple[tuple[int, int, int, int], ...]

    @property
    def value_bbox(self) -> tuple[int, int, int, int]:
        return self.cells[0]


def _normalize(ink: np.ndarray) -> np.ndarray | None:
    points = cv2.findNonZero(ink)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if w < 3 and h < 3:
        return None
    scale = 48 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(ink[y:y+h, x:x+w], (nw, nh), interpolation=cv2.INTER_AREA)
    result = np.zeros((64, 64), np.uint8)
    result[(64-nh)//2:(64-nh)//2+nh, (64-nw)//2:(64-nw)//2+nw] = (resized > 90).astype(np.uint8)
    return result


def _template(name: str, thickness: int, variant: int) -> np.ndarray:
    canvas = np.zeros((100, 100), np.uint8)

    def line(a, b):
        cv2.line(canvas, a, b, 255, thickness, cv2.LINE_AA)

    def circle(radius):
        cv2.circle(canvas, (50, 50), radius, 255, thickness, cv2.LINE_AA)

    def arrow(x):
        line((x, 78), (x+25, 22))
        if variant == 0:
            cv2.fillConvexPoly(canvas, np.array([(x+25, 22), (x+12, 36), (x+28, 40)]), 255)
        else:
            line((x+25, 22), (x+12, 36))
            line((x+25, 22), (x+28, 40))

    if name == "Straightness":
        line((18, 50), (82, 50))
    elif name == "Flatness":
        corners = [(18, 72), (38, 28), (82, 28), (62, 72)] if variant == 0 else [(12, 72), (30, 28), (88, 28), (70, 72)]
        cv2.polylines(canvas, [np.array(corners)], True, 255, thickness, cv2.LINE_AA)
    elif name in ("Circularity", "Position", "Concentricity", "Cylindricity"):
        circle((23 if variant == 0 else 26) if name == "Position" else 28)
        if name == "Position":
            line((10, 50), (90, 50))
            line((50, 10), (50, 90))
        elif name == "Concentricity":
            circle(18 if variant == 0 else 21)
        elif name == "Cylindricity":
            line((12, 78), (35, 22))
            line((65, 78), (88, 22))
    elif name == "Angularity":
        line((18, 72), (82, 72))
        line((18, 72), (75, 28 if variant == 0 else 18))
    elif name == "Perpendicularity":
        line((18, 78), (82, 78))
        line((50, 78), (50, 22))
    elif name == "Parallelism":
        slant = 25 if variant == 0 else 21
        line((22, 78), (22+slant, 22))
        line((53, 78), (53+slant, 22))
    elif name == "Symmetry":
        line((18, 50), (82, 50))
        line((32, 30), (68, 30))
        line((32, 70), (68, 70))
    elif name in ("Profile of a Line", "Profile of a Surface"):
        cv2.ellipse(canvas, (50, 72), (32, 32 if variant == 0 else 24), 0, 180, 360, 255, thickness, cv2.LINE_AA)
        if name == "Profile of a Surface":
            line((18, 72), (82, 72))
    elif name == "Circular Runout":
        arrow(36)
    elif name == "Total Runout":
        arrow(20)
        arrow(53)
        line((20, 78), (53, 78))
    return canvas


@lru_cache(maxsize=1)
def _templates():
    result = []
    for name in SYMBOLS:
        for thickness in (2, 4, 6):
            for variant in (0, 1):
                base = _template(name, thickness, variant)
                for stretch in (0.8, 1.0, 1.2):
                    mask = _normalize(cv2.resize(base, (round(100*stretch), 100)))
                    distance = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
                    result.append((name, mask, distance))
    return result


def classify_symbol(ink: np.ndarray) -> tuple[str, float] | None:
    """Return a symbol name and conservative match score, or abstain.

    Input is a cropped cell interior with nonzero foreground ink.
    """
    mask = _normalize(ink)
    if mask is None:
        return None
    distance = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    for i, contour in enumerate(contours):
        depth, parent = 0, hierarchy[0][i][3]
        while parent >= 0:
            depth += 1
            parent = hierarchy[0][parent][3]
        if depth % 2 and cv2.contourArea(contour) > 8:
            holes += 1
    expected_holes = {"Circularity": 1, "Concentricity": 2, "Position": 4,
                      "Flatness": 1, "Profile of a Surface": 1}
    scores: dict[str, float] = {}
    for name, template, template_distance in _templates():
        if name in expected_holes and holes != expected_holes[name]:
            continue
        if name == "Circularity":
            outer = max(contours, key=cv2.contourArea)
            if len(cv2.approxPolyDP(outer, 0.025 * cv2.arcLength(outer, True), True)) < 6:
                continue
        error = (float(template_distance[mask > 0].mean())
                 + float(distance[template > 0].mean())) / 2
        scores[name] = min(scores.get(name, float("inf")), error)
    ranked = sorted(scores.items(), key=lambda item: item[1])
    (name, error), (_, runner_up) = ranked[:2]
    if error > 1.25 or runner_up - error < 0.35:
        return None
    return name, min(0.9, max(0.55, 0.9 - error * 0.15))


def find_gdt_frames(image: np.ndarray) -> list[GdtFrame]:
    """Locate adjacent rectangular compartments and read the first symbol.

    Frame borders supply context that separates symbols from part geometry.
    Small scan gaps are closed before finding cell interiors. Coordinates
    use the rendered page, so PDF vector and raster sources share this path.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 14 or w < 10 or h > gray.shape[0] * 0.25 or w > 12*h:
            continue
        if cv2.contourArea(contour) < 0.88 * (w-1) * (h-1):
            continue
        # Exclude the outer border of a multi-cell frame: its interior
        # contains a full-height divider, whereas each cell is mostly clear.
        pad = max(2, round(h * 0.06))
        interior = ink[y+pad:y+h-pad, x+pad:x+w-pad]
        if interior.size == 0 or np.any(np.mean(interior > 0, axis=0) > 0.95):
            continue
        cells.append((x, y, x+w, y+h))
    cells.sort()
    frames = []
    consumed = set()
    for cell in cells:
        x0, y0, x1, y1 = cell
        h = y1-y0
        if cell in consumed or not 0.45*h <= x1-x0 <= 2*h:
            continue

        def next_cell(current):
            matches = [c for c in cells if -2 <= c[0]-current[2] <= max(5, h*0.12)
                       and c[0] > current[0]
                       and abs(c[1]-y0) <= max(3, h*0.06)
                       and abs(c[3]-y1) <= max(3, h*0.06)]
            return min(matches, key=lambda c: c[0]) if matches else None

        value = next_cell(cell)
        if value is None:
            continue
        pad = max(2, round(h * 0.06))
        match = classify_symbol(ink[y0+pad:y1-pad, x0+pad:x1-pad])
        if match is None:
            continue
        following = [value]
        while len(following) < 8:
            subsequent = next_cell(following[-1])
            if subsequent is None:
                break
            following.append(subsequent)
        consumed.update(following)
        frames.append(GdtFrame(match[0], match[1], tuple(following)))
    return frames
