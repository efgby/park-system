from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.geometry import bbox_center


@dataclass(frozen=True)
class DirectionResult:
    track_id: int
    direction: str
    zone: str | None
    confidence: float
    reason: str


class DirectionClassifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.parking = config.get("parking", {})
        self.lines = config.get("lines", {})
        self.trajectory_history = int(config.get("trajectory_history", 30))

    def classify(self, track_id: int, points: list[tuple[float, float]]) -> DirectionResult:
        valid_points = points[-self.trajectory_history :]
        if len(valid_points) < 3:
            return DirectionResult(track_id, "insufficient", None, 0.0, "trajectory_too_short")

        zone = self._final_zone(valid_points)
        if zone is None:
            return DirectionResult(track_id, "unknown", None, 0.0, "no_zone_match")

        zone_name = str(self.parking.get(zone, {}).get("name", zone))
        return DirectionResult(track_id, zone_name.upper(), zone, 1.0, "final_point_in_zone")

    def _final_zone(self, points: list[tuple[float, float]]) -> str | None:
        final_point = points[-1]
        candidates: list[tuple[str, float]] = []
        for zone, zone_data in self.lines.items():
            p1 = tuple(zone_data["p1"])
            p2 = tuple(zone_data["p2"])
            if self._point_in_zone(final_point, p1, p2):
                candidates.append((zone, self._zone_score(final_point, p1, p2)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    def _point_in_zone(self, point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        x, y = point
        left = min(p1[0], p2[0])
        right = max(p1[0], p2[0])
        top = min(p1[1], p2[1])
        bottom = max(p1[1], p2[1])
        return left <= x <= right and top <= y <= bottom

    def _zone_score(self, point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
        center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        return -abs(point[0] - center[0]) - abs(point[1] - center[1])