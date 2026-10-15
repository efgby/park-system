from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class Visualizer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def _track_value(self, track: Any, key: str, default: Any = None) -> Any:
        if isinstance(track, dict):
            return track.get(key, default)
        return getattr(track, key, default)

    def draw(self, frame, detections: list[dict[str, Any]], tracks: list[dict[str, Any]], parking_state: dict[str, Any], trajectories: dict[int, list[tuple[float, float]]] | None = None, frame_index: int | None = None, debug_mode: bool = False):
        annotated = frame.copy()
        trajectories = trajectories or {}
        rendered_bbox_count = 0
        rendered_label_count = 0
        rendered_trajectory_count = 0
        rendered_ids: set[int] = set()

        for track in tracks:
            x1, y1, x2, y2 = map(int, self._track_value(track, "bbox"))
            track_id = self._track_value(track, "track_id")
            score_value = self._track_value(track, "score", self._track_value(track, "confidence", 0.0))
            score = float(score_value) if score_value is not None else 0.0
            class_name = self._track_value(track, "class_name")
            if track_id in rendered_ids:
                raise RuntimeError(f"Duplicate rendering ID {track_id}")
            rendered_ids.add(track_id)
            label_prefix = f"Vehicle {track_id}"
            if debug_mode:
                label_prefix = f"Vehicle {track_id} [raw={self._track_value(track, 'raw_track_id')}]"
            print(f"[DRAW BBOX] frame={frame_index} business_id={track_id} raw_track_id={self._track_value(track, 'raw_track_id')} source=current_business_track bbox={[x1, y1, x2, y2]}")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rendered_bbox_count += 1
            print(f"[DRAW LABEL] frame={frame_index} business_id={track_id} raw_track_id={self._track_value(track, 'raw_track_id')} source=current_business_track bbox={[x1, y1, x2, y2]}")
            cv2.putText(
                annotated,
                f'{label_prefix} | {class_name} | {score:.2f}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            rendered_label_count += 1

            points = trajectories.get(track_id, [])
            points = points[-30:]
            if points:
                if not isinstance(points[0], tuple):
                    raise TypeError(f"trajectory history must store tuples, got {type(points[0])}")
                if len(points) >= 2:
                    polyline = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
                    cv2.polylines(annotated, [polyline], False, (0, 255, 255), 2)
                    rendered_trajectory_count += 1
                elif len(points) == 1:
                    cv2.circle(annotated, (int(points[0][0]), int(points[0][1])), 3, (0, 255, 255), -1)
                    rendered_trajectory_count += 1

        if rendered_bbox_count != len(tracks):
            raise AssertionError(
                f"Rendering mismatch: output_tracks={len(tracks)}, rendered_bbox={rendered_bbox_count}"
            )

        if parking_state:
            for zone, line in parking_state.get("lines", {}).items():
                p1, p2 = line
                cv2.line(annotated, tuple(map(int, p1)), tuple(map(int, p2)), (255, 0, 0), 2)
                cv2.putText(
                    annotated,
                    zone,
                    tuple(map(int, p1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

            y_offset = 30
            cv2.putText(annotated, "Parking Status", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y_offset += 35
            for zone, zone_data in parking_state.get("parking", {}).items():
                available = max(0, zone_data["capacity"] - zone_data["occupied"])
                text = f'{zone} {zone_data["name"]}: {zone_data["occupied"]} occupied, {available} available'
                cv2.putText(annotated, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 28

        return annotated, {
            "rendered_bbox": rendered_bbox_count,
            "rendered_labels": rendered_label_count,
            "rendered_trajectories": rendered_trajectory_count,
        }
