from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrackLifecycleState(str, Enum):
    ACTIVE = "ACTIVE"
    EXITING = "EXITING"
    FINISHED = "FINISHED"


@dataclass
class BusinessVehicle:
    business_vehicle_id: int
    raw_track_ids: set[int] = field(default_factory=set)
    state: TrackLifecycleState = TrackLifecycleState.ACTIVE
    created_frame: int = 0
    last_seen_frame: int = 0
    last_bbox: list[float] = field(default_factory=list)
    last_bottom_center: tuple[float, float] | None = None
    last_velocity: tuple[float, float] | None = None
    exit_frame: int | None = None
    exit_confirm_counter: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)
    suppressed_exit_fragments: int = 0
    exit_region: str | None = None
    exit_reason: str | None = None
    merged_raw_track_ids: list[int] = field(default_factory=list)


class TrackLifecycleManager:
    def __init__(self, tracking_config: dict[str, Any], frame_size: tuple[int, int]) -> None:
        self.right_exit_margin = int(tracking_config.get("right_exit_margin", 80))
        self.bottom_exit_margin = int(tracking_config.get("bottom_exit_margin", 60))
        self.exit_confirm_frames = int(tracking_config.get("exit_confirm_frames", 15))
        self.exit_fragment_max_gap = int(tracking_config.get("exit_fragment_max_gap", 3))
        self.exit_fragment_max_distance = float(tracking_config.get("exit_fragment_max_distance", 80.0))
        self.frame_width, self.frame_height = frame_size
        self.right_exit_x = self.frame_width - self.right_exit_margin
        self.bottom_exit_y = self.frame_height - self.bottom_exit_margin

        self.business_vehicles: dict[int, BusinessVehicle] = {}
        self.raw_to_business: dict[int, int] = {}
        self.next_business_vehicle_id = 1
        self.total_suppressed_exit_fragments = 0
        self.total_fragment_merges = 0
        self.first_fragmentation_info: dict[str, Any] | None = None

    def update(self, raw_tracks: list[dict[str, Any]], frame_id: int) -> list[dict[str, Any]]:
        business_tracks: list[dict[str, Any]] = []
        observed_business_ids: set[int] = set()
        emitted_business_ids: set[int] = set()

        for raw_track in raw_tracks:
            raw_track_id = int(raw_track["track_id"])
            bbox = list(map(float, raw_track["bbox"]))
            score = float(raw_track["score"])
            class_id = int(raw_track["class_id"])
            class_name = str(raw_track["class_name"])
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width / 2.0
            center_y = y1 + height / 2.0
            bottom_center = (center_x, y2)
            distance_to_right_border = self.frame_width - x2
            distance_to_bottom_border = self.frame_height - y2
            exit_region = self._exit_region_for_bbox(bbox)

            business_vehicle = self._resolve_business_vehicle(raw_track_id, bbox, bottom_center, frame_id)
            observed_business_ids.add(business_vehicle.business_vehicle_id)

            if raw_track_id not in business_vehicle.raw_track_ids:
                business_vehicle.raw_track_ids.add(raw_track_id)
                business_vehicle.merged_raw_track_ids.append(raw_track_id)
                if len(business_vehicle.raw_track_ids) > 1:
                    self.total_fragment_merges += 1
                    print(
                        f"[BUSINESS ID MERGE] Frame: {frame_id} Business Vehicle: {business_vehicle.business_vehicle_id} "
                        f"Old raw ID: {self._previous_raw_id(business_vehicle)} New raw ID: {raw_track_id} "
                        f"IoU: {self._best_iou_with_last_bbox(business_vehicle, bbox):.2f} "
                        f"Distance: {self._distance_to_last_bottom_center(business_vehicle, bottom_center):.1f} "
                        f"Time gap: {frame_id - business_vehicle.last_seen_frame if business_vehicle.last_seen_frame else 0} "
                        f"Direction similarity: {self._direction_similarity(business_vehicle, bottom_center):.2f} "
                        f"Exit region: {exit_region} Reason: single-lane exit fragment"
                    )

            prev_bbox = business_vehicle.last_bbox
            prev_bottom_center = business_vehicle.last_bottom_center
            prev_frame = business_vehicle.last_seen_frame
            if prev_bottom_center is not None:
                business_vehicle.last_velocity = (
                    bottom_center[0] - prev_bottom_center[0],
                    bottom_center[1] - prev_bottom_center[1],
                )

            business_vehicle.last_seen_frame = frame_id
            business_vehicle.last_bbox = bbox
            business_vehicle.last_bottom_center = bottom_center
            business_vehicle.history.append(bottom_center)
            if len(business_vehicle.history) > 30:
                business_vehicle.history = business_vehicle.history[-30:]

            if business_vehicle.state == TrackLifecycleState.ACTIVE and self._should_enter_exit(business_vehicle, bbox):
                business_vehicle.state = TrackLifecycleState.EXITING
                business_vehicle.exit_frame = frame_id
                business_vehicle.exit_confirm_counter = 0
                business_vehicle.exit_region = exit_region
                business_vehicle.exit_reason = "internal_to_exit_zone_motion"
                print(
                    f"[EXIT STATE] frame={frame_id} business_id={business_vehicle.business_vehicle_id} raw_id={raw_track_id} "
                    f"state=ACTIVE->EXITING bbox={self._fmt_bbox(bbox)} bottom_center={self._fmt_point(bottom_center)} "
                    f"distance_to_right_border={distance_to_right_border:.1f} distance_to_bottom_border={distance_to_bottom_border:.1f} "
                    f"created_frame={business_vehicle.created_frame} prev_frame={prev_frame} prev_bbox={self._fmt_bbox(prev_bbox)}"
                )

            if business_vehicle.state == TrackLifecycleState.EXITING:
                business_vehicle.exit_confirm_counter += 1
                if business_vehicle.exit_confirm_counter >= self.exit_confirm_frames:
                    business_vehicle.state = TrackLifecycleState.FINISHED
                    print(
                        f"[EXIT FINISHED] frame={frame_id} business_id={business_vehicle.business_vehicle_id} "
                        f"raw_ids={sorted(business_vehicle.raw_track_ids)} exit_frame={business_vehicle.exit_frame} "
                        f"created_frame={business_vehicle.created_frame}"
                    )

            if business_vehicle.state != TrackLifecycleState.FINISHED and business_vehicle.business_vehicle_id not in emitted_business_ids:
                emitted_business_ids.add(business_vehicle.business_vehicle_id)
                business_tracks.append(
                    {
                        "track_id": business_vehicle.business_vehicle_id,
                        "raw_track_id": raw_track_id,
                        "raw_track_ids": sorted(business_vehicle.raw_track_ids),
                        "bbox": bbox,
                        "score": score,
                        "class_id": class_id,
                        "class_name": class_name,
                        "state": business_vehicle.state.value,
                        "created_frame": business_vehicle.created_frame,
                        "last_seen_frame": business_vehicle.last_seen_frame,
                        "last_bbox": business_vehicle.last_bbox,
                        "last_bottom_center": business_vehicle.last_bottom_center,
                        "last_velocity": business_vehicle.last_velocity,
                        "exit_frame": business_vehicle.exit_frame,
                        "distance_to_right_border": distance_to_right_border,
                        "distance_to_bottom_border": distance_to_bottom_border,
                        "exit_region": exit_region,
                    }
                )

            if self.first_fragmentation_info is None and len(business_vehicle.raw_track_ids) > 1:
                self.first_fragmentation_info = {
                    "frame_id": frame_id,
                    "business_vehicle_id": business_vehicle.business_vehicle_id,
                    "raw_track_ids": sorted(business_vehicle.raw_track_ids),
                    "raw_track_id": raw_track_id,
                    "bbox": bbox,
                    "distance_to_right_border": distance_to_right_border,
                    "distance_to_bottom_border": distance_to_bottom_border,
                    "state": business_vehicle.state.value,
                    "created_frame": business_vehicle.created_frame,
                    "exit_region": exit_region,
                }

        self._suppress_exit_fragments(raw_tracks, frame_id)
        self._prune_finished(observed_business_ids)
        return business_tracks

    def _resolve_business_vehicle(
        self,
        raw_track_id: int,
        bbox: list[float],
        bottom_center: tuple[float, float],
        frame_id: int,
    ) -> BusinessVehicle:
        existing_business_id = self.raw_to_business.get(raw_track_id)
        if existing_business_id is not None:
            return self.business_vehicles[existing_business_id]

        candidate = self._find_merge_candidate(bbox, bottom_center, frame_id)
        if candidate is not None:
            self.raw_to_business[raw_track_id] = candidate.business_vehicle_id
            return candidate

        business_vehicle = BusinessVehicle(
            business_vehicle_id=self.next_business_vehicle_id,
            raw_track_ids={raw_track_id},
            state=TrackLifecycleState.ACTIVE,
            created_frame=frame_id,
            last_seen_frame=frame_id,
            last_bbox=bbox,
            last_bottom_center=bottom_center,
            history=[bottom_center],
        )
        self.business_vehicles[self.next_business_vehicle_id] = business_vehicle
        self.raw_to_business[raw_track_id] = self.next_business_vehicle_id
        self.next_business_vehicle_id += 1
        return business_vehicle

    def _find_merge_candidate(
        self,
        bbox: list[float],
        bottom_center: tuple[float, float],
        frame_id: int,
    ) -> BusinessVehicle | None:
        candidates = [
            vehicle
            for vehicle in self.business_vehicles.values()
            if vehicle.state == TrackLifecycleState.EXITING
            and vehicle.last_bottom_center is not None
            and frame_id - vehicle.last_seen_frame <= self.exit_fragment_max_gap
            and self._is_exit_region(bbox)
        ]
        if not candidates:
            return None

        best_vehicle: BusinessVehicle | None = None
        best_score = -1.0
        for vehicle in candidates:
            distance = self._distance(vehicle.last_bottom_center, bottom_center) if vehicle.last_bottom_center else float("inf")
            iou = self._iou(vehicle.last_bbox, bbox) if vehicle.last_bbox else 0.0
            direction_similarity = self._direction_similarity(vehicle, bottom_center)
            if distance > self.exit_fragment_max_distance:
                continue
            if direction_similarity < 0.0:
                continue
            score = iou * 2.0 + max(0.0, 1.0 - distance / self.exit_fragment_max_distance) + direction_similarity
            if score > best_score:
                best_score = score
                best_vehicle = vehicle

        return best_vehicle

    def _suppress_exit_fragments(self, raw_tracks: list[dict[str, Any]], frame_id: int) -> None:
        exiting_vehicles = [vehicle for vehicle in self.business_vehicles.values() if vehicle.state == TrackLifecycleState.EXITING]
        for raw_track in raw_tracks:
            raw_track_id = int(raw_track["track_id"])
            business_id = self.raw_to_business.get(raw_track_id)
            if business_id is None:
                continue
            vehicle = self.business_vehicles[business_id]
            if vehicle.state != TrackLifecycleState.EXITING:
                continue
            bbox = list(map(float, raw_track["bbox"]))
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            for exiting in exiting_vehicles:
                if exiting.business_vehicle_id != vehicle.business_vehicle_id:
                    continue
                if exiting.last_bottom_center is None:
                    continue
                distance = self._distance(exiting.last_bottom_center, (center_x, center_y))
                time_gap = frame_id - exiting.last_seen_frame
                if self._is_exit_region(bbox) and distance <= self.exit_fragment_max_distance and time_gap <= self.exit_fragment_max_gap:
                    exiting.suppressed_exit_fragments += 1
                    self.total_suppressed_exit_fragments += 1
                    print(
                        f"[SUPPRESS EXIT FRAGMENT] frame={frame_id} business_id={exiting.business_vehicle_id} raw_id={raw_track_id} "
                        f"bbox={self._fmt_bbox(bbox)} distance_to_right_border={self.frame_width - x2:.1f} "
                        f"distance_to_bottom_border={self.frame_height - y2:.1f} time_gap={time_gap} spatial_distance={distance:.1f}"
                    )

    def _prune_finished(self, observed_business_ids: set[int]) -> None:
        finished_ids = [
            business_id
            for business_id, vehicle in self.business_vehicles.items()
            if vehicle.state == TrackLifecycleState.FINISHED and business_id not in observed_business_ids
        ]
        for business_id in finished_ids:
            del self.business_vehicles[business_id]

    def _should_enter_exit(self, vehicle: BusinessVehicle, bbox: list[float]) -> bool:
        if len(vehicle.history) < 3:
            return False
        if vehicle.last_velocity is None:
            return False

        vx, vy = vehicle.last_velocity
        moving_outward = vx > 0 or vy > 0
        in_exit_margin = self._is_exit_region(bbox)
        had_internal_history = any(
            point[0] < self.right_exit_x and point[1] < self.bottom_exit_y for point in vehicle.history[:-1]
        )
        return moving_outward and in_exit_margin and had_internal_history

    def _exit_region_for_bbox(self, bbox: list[float]) -> str:
        x1, y1, x2, y2 = bbox
        right = x2 >= self.right_exit_x
        bottom = y2 >= self.bottom_exit_y
        if right and bottom:
            return "RIGHT_BOTTOM"
        if right:
            return "RIGHT"
        if bottom:
            return "BOTTOM"
        return "INTERIOR"

    def _is_exit_region(self, bbox: list[float]) -> bool:
        return self._exit_region_for_bbox(bbox) != "INTERIOR"

    def get_summary(self) -> dict[str, Any]:
        active = sum(1 for vehicle in self.business_vehicles.values() if vehicle.state == TrackLifecycleState.ACTIVE)
        exiting = sum(1 for vehicle in self.business_vehicles.values() if vehicle.state == TrackLifecycleState.EXITING)
        finished = sum(1 for vehicle in self.business_vehicles.values() if vehicle.state == TrackLifecycleState.FINISHED)
        return {
            "active": active,
            "exiting": exiting,
            "finished": finished,
            "suppressed_exit_fragments": self.total_suppressed_exit_fragments,
            "fragment_merges": self.total_fragment_merges,
        }

    def get_first_fragmentation_info(self) -> dict[str, Any] | None:
        return self.first_fragmentation_info

    def get_business_vehicle(self, business_vehicle_id: int) -> BusinessVehicle | None:
        return self.business_vehicles.get(business_vehicle_id)

    def get_active_business_ids(self) -> set[int]:
        return {
            business_id
            for business_id, vehicle in self.business_vehicles.items()
            if vehicle.state != TrackLifecycleState.FINISHED
        }

    def _previous_raw_id(self, vehicle: BusinessVehicle) -> int | None:
        if len(vehicle.raw_track_ids) < 2:
            return None
        return sorted(vehicle.raw_track_ids)[-2]

    def _best_iou_with_last_bbox(self, vehicle: BusinessVehicle, bbox: list[float]) -> float:
        return self._iou(vehicle.last_bbox, bbox) if vehicle.last_bbox else 0.0

    def _distance_to_last_bottom_center(self, vehicle: BusinessVehicle, bottom_center: tuple[float, float]) -> float:
        if vehicle.last_bottom_center is None:
            return 0.0
        return self._distance(vehicle.last_bottom_center, bottom_center)

    def _direction_similarity(self, vehicle: BusinessVehicle, bottom_center: tuple[float, float]) -> float:
        if vehicle.last_bottom_center is None or vehicle.last_velocity is None:
            return 0.0
        vx, vy = vehicle.last_velocity
        dx = bottom_center[0] - vehicle.last_bottom_center[0]
        dy = bottom_center[1] - vehicle.last_bottom_center[1]
        dot = vx * dx + vy * dy
        norm_a = (vx * vx + vy * vy) ** 0.5
        norm_b = (dx * dx + dy * dy) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _iou(box_a: list[float], box_b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _fmt_bbox(bbox: list[float]) -> list[int]:
        return [int(value) for value in bbox]

    @staticmethod
    def _fmt_point(point: tuple[float, float]) -> list[int]:
        return [int(point[0]), int(point[1])]