from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: list[float]
    confidence: float
    class_id: int
    class_name: str


class BaseTracker(Protocol):
    def update(self, detections: list[dict]) -> list[TrackedVehicle]:
        ...

    def get_stats(self) -> dict[str, int]:
        ...