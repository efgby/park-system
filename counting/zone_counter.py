from __future__ import annotations

from typing import Any

from utils.geometry import bbox_center


class ZoneCounter:
    def __init__(self, parking_config: dict, line_config: dict, counting_config: dict) -> None:
        self.parking = {
            zone: {
                "name": zone_data["name"],
                "capacity": int(zone_data["capacity"]),
                "occupied": 0,
            }
            for zone, zone_data in parking_config.items()
        }
        self.lines = {
            zone: (tuple(zone_data["p1"]), tuple(zone_data["p2"]))
            for zone, zone_data in line_config.items()
        }
        self.counting_config = counting_config
        self.track_history: dict[int, list[tuple[float, float]]] = {}
        self.track_state: dict[int, dict[str, Any]] = {}

    def update(self, tracks: list[dict[str, Any]], frame_index: int) -> dict[str, Any]:
        for track in tracks:
            track_id = track["track_id"]
            center = tuple(track.get("center") or bbox_center(track["bbox"]))
            self.track_history.setdefault(track_id, []).append(center)
            self.track_state.setdefault(track_id, {"last_frame": frame_index})
            self.track_state[track_id]["last_frame"] = frame_index
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return {
            "parking": self.parking,
            "lines": self.lines,
            "track_history": self.track_history,
            "track_state": self.track_state,
        }
