from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque


class TrajectoryStore:
    def __init__(self, max_length: int) -> None:
        self.max_length = int(max_length)
        self.track_history: dict[int, Deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=self.max_length))

    def update(self, track_id: int, bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.track_history[track_id].append(point)
        return point

    def update_point(self, track_id: int, point: tuple[float, float]) -> None:
        self.track_history[track_id].append(point)

    def get(self, track_id: int) -> list[tuple[float, float]]:
        return list(self.track_history.get(track_id, []))

    def prune(self, active_track_ids: set[int]) -> None:
        stale_ids = [track_id for track_id in self.track_history if track_id not in active_track_ids]
        for track_id in stale_ids:
            del self.track_history[track_id]

    def count_points(self, track_id: int) -> int:
        return len(self.track_history.get(track_id, []))