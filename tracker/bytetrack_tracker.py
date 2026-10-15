from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


@dataclass
class TrackState:
    track_id: int
    bbox: list[float]
    score: float
    class_id: int
    class_name: str
    is_activated: bool = False
    state: str = "new"
    missed_frames: int = 0
    frame_id: int = 0
    start_frame: int = 0


class ByteTrackTracker:
    def __init__(self, tracking_config: dict) -> None:
        self.tracking_config = tracking_config
        self.track_thresh = float(tracking_config.get("track_thresh", tracking_config.get("track_threshold", 0.5)))
        self.high_thresh = float(tracking_config.get("high_thresh", self.track_thresh))
        self.low_thresh = float(tracking_config.get("low_thresh", 0.1))
        self.new_track_thresh = float(tracking_config.get("new_track_thresh", self.high_thresh))
        self.match_thresh = float(tracking_config.get("match_thresh", tracking_config.get("match_threshold", 0.8)))
        self.track_buffer = int(tracking_config.get("track_buffer", 60))
        self.next_track_id = 1
        self.active_tracks: dict[int, TrackState] = {}
        self.lost_tracks: dict[int, TrackState] = {}
        self.removed_tracks: dict[int, TrackState] = {}
        self.frame_id = 0
        self.debug_target_box = [418.0, 166.0, 442.0, 186.0]
        self.debug_target_last_track_id: int | None = None
        self.debug_target_last_bbox: list[float] | None = None

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.frame_id += 1
        candidates = [d for d in detections if float(d.get("score", 0.0)) >= self.low_thresh]
        high_candidates = [d for d in candidates if float(d.get("score", 0.0)) >= self.high_thresh]
        low_candidates = [d for d in candidates if self.low_thresh <= float(d.get("score", 0.0)) < self.high_thresh]

        matches, unmatched_tracks, unmatched_detections = self._match_tracks(high_candidates, list(self.active_tracks.keys()))
        matched_track_ids = {track_id for track_id, _ in matches}

        updated_tracks: list[TrackState] = []
        for track_id, detection_index in matches:
            detection = high_candidates[detection_index]
            track = self.active_tracks[track_id]
            track.bbox = list(map(float, detection["bbox"]))
            track.score = float(detection["score"])
            track.class_id = int(detection["class_id"])
            track.class_name = str(detection["class_name"])
            track.missed_frames = 0
            track.is_activated = True
            track.state = "tracked"
            track.frame_id = self.frame_id
            updated_tracks.append(track)

        for track_id in unmatched_tracks:
            track = self.active_tracks[track_id]
            track.missed_frames += 1
            track.state = "lost"
            self.lost_tracks[track_id] = track
            if track.missed_frames <= self.track_buffer:
                updated_tracks.append(track)
            else:
                track.state = "removed"
                self.removed_tracks[track_id] = track
                del self.active_tracks[track_id]
                self.lost_tracks.pop(track_id, None)

        lost_track_ids = list(self.lost_tracks.keys())
        if low_candidates and lost_track_ids:
            low_matches, low_unmatched_tracks, low_unmatched_detections = self._match_tracks(low_candidates, lost_track_ids)
            for track_id, detection_index in low_matches:
                detection = low_candidates[detection_index]
                track = self.lost_tracks[track_id]
                track.bbox = list(map(float, detection["bbox"]))
                track.score = float(detection["score"])
                track.class_id = int(detection["class_id"])
                track.class_name = str(detection["class_name"])
                track.missed_frames = 0
                track.is_activated = True
                track.state = "tracked"
                track.frame_id = self.frame_id
                self.active_tracks[track_id] = track
                self.lost_tracks.pop(track_id, None)
                updated_tracks.append(track)
            unmatched_detections = [index for index in unmatched_detections if index < len(high_candidates)]
            for detection_index in low_unmatched_detections:
                detection = low_candidates[detection_index]
                if float(detection.get("score", 0.0)) < self.new_track_thresh:
                    continue
                self._init_track(detection, updated_tracks)
        else:
            for detection_index in unmatched_detections:
                detection = high_candidates[detection_index]
                if float(detection.get("score", 0.0)) < self.new_track_thresh:
                    continue
                self._init_track(detection, updated_tracks)

        self._prune_tracks()
        self._debug_target_check(candidates, updated_tracks)
        current_tracks = [
            {
                "track_id": track.track_id,
                "bbox": track.bbox,
                "score": track.score,
                "class_id": track.class_id,
                "class_name": track.class_name,
            }
            for track in updated_tracks
            if track.state != "removed"
        ]
        seen_track_ids: set[int] = set()
        seen_bboxes: set[tuple[int, int, int, int]] = set()
        for track in current_tracks:
            track_id = int(track["track_id"])
            bbox = tuple(map(int, track["bbox"]))
            if track_id in seen_track_ids:
                raise RuntimeError(f"Duplicate active track_id detected: {track_id}")
            seen_track_ids.add(track_id)
            if bbox in seen_bboxes:
                print(f"[DUPLICATE ACTIVE TRACK] frame={self.frame_id} track_id={track_id} bbox={list(bbox)}")
            seen_bboxes.add(bbox)
        if len(current_tracks) != len(seen_track_ids):
            raise RuntimeError(
                f"Tracker output contains duplicate track IDs: output={len(current_tracks)} unique={len(seen_track_ids)}"
            )
        return current_tracks

    def _init_track(self, detection: dict[str, Any], updated_tracks: list[TrackState]) -> None:
        track = TrackState(
            track_id=self.next_track_id,
            bbox=list(map(float, detection["bbox"])),
            score=float(detection["score"]),
            class_id=int(detection["class_id"]),
            class_name=str(detection["class_name"]),
            is_activated=True,
            state="tracked",
            frame_id=self.frame_id,
            start_frame=self.frame_id,
        )
        self.active_tracks[self.next_track_id] = track
        self.next_track_id += 1
        updated_tracks.append(track)

    def _match_tracks(self, detections: list[dict[str, Any]], track_ids: list[int]):
        if not track_ids or not detections:
            return [], track_ids[:], list(range(len(detections)))

        track_boxes = [self.active_tracks[track_id].bbox if track_id in self.active_tracks else self.lost_tracks[track_id].bbox for track_id in track_ids]
        det_boxes = [detection["bbox"] for detection in detections]

        cost_matrix = [[1.0 - self._iou(track_box, det_box) for det_box in det_boxes] for track_box in track_boxes]
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(detections)))

        while True:
            best_cost = 1.0
            best_pair: tuple[int, int] | None = None
            for track_index, row in enumerate(cost_matrix):
                for detection_index, cost in enumerate(row):
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (track_index, detection_index)
            if best_pair is None:
                break

            track_index, detection_index = best_pair
            iou = 1.0 - best_cost
            if iou < self.match_thresh:
                break

            track_id = track_ids[track_index]
            matches.append((track_id, detection_index))
            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(detection_index)
            cost_matrix[track_index] = [math.inf] * len(cost_matrix[track_index])
            for row in cost_matrix:
                row[detection_index] = math.inf

        return matches, list(unmatched_tracks), list(unmatched_detections)

    def _prune_tracks(self) -> None:
        stale_ids = [track_id for track_id, track in self.lost_tracks.items() if track.missed_frames > self.track_buffer]
        for track_id in stale_ids:
            track = self.lost_tracks.pop(track_id)
            track.state = "removed"
            self.removed_tracks[track_id] = track

    def _debug_target_check(self, detections: list[dict[str, Any]], tracks: list[TrackState]) -> None:
        best_detection = None
        best_iou = 0.0
        for detection in detections:
            iou = self._iou(self.debug_target_box, list(map(float, detection["bbox"])))
            if iou > best_iou:
                best_iou = iou
                best_detection = detection

        if best_detection is None or best_iou < 0.5:
            return

        candidate_track = None
        candidate_iou = 0.0
        for track in tracks:
            iou = self._iou(list(map(float, track.bbox)), list(map(float, best_detection["bbox"])))
            if iou > candidate_iou:
                candidate_iou = iou
                candidate_track = track

        if candidate_track is None:
            return

        previous_id = self.debug_target_last_track_id
        previous_bbox = self.debug_target_last_bbox
        if previous_id is not None and previous_id != candidate_track.track_id and previous_bbox is not None:
            previous_iou = self._iou(previous_bbox, list(map(float, candidate_track.bbox)))
            if previous_iou >= 0.5:
                print(
                    f"[TRACK WARNING] Possible ID switch | Frame: {self.frame_id} | Previous ID: {previous_id} | New ID: {candidate_track.track_id} | IoU: {previous_iou:.2f}"
                )

        self.debug_target_last_track_id = candidate_track.track_id
        self.debug_target_last_bbox = list(map(float, candidate_track.bbox))

    def get_stats(self) -> dict[str, int]:
        return {
            "tracked": sum(1 for track in self.active_tracks.values() if track.state == "tracked"),
            "lost": len(self.lost_tracks),
            "removed": len(self.removed_tracks),
            "output": sum(1 for track in self.active_tracks.values() if track.state == "tracked"),
        }

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
