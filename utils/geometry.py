from __future__ import annotations

from math import hypot
from typing import Sequence

Point = Sequence[float]
Line = tuple[Point, Point]


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def distance(p1: Point, p2: Point) -> float:
    return hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def line_side(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = float(point[0]), float(point[1])
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def point_line_distance(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = float(point[0]), float(point[1])
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = hypot(x2 - x1, y2 - y1)
    return numerator / denominator if denominator else 0.0
