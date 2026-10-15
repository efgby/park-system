from __future__ import annotations

from dataclasses import dataclass
from importlib import util
import inspect
import importlib
from pathlib import Path
from typing import Any

import math
import sys
import numpy as np

from tracker.base_tracker import TrackedVehicle


@dataclass
class _OCSortTrack:
    track_id: int
    bbox: list[float]
    score: float | None
    class_id: int | None
    class_name: str | None
    missed_frames: int = 0


class OCSortTracker:
    def __init__(self, tracking_config: dict[str, Any]) -> None:
        try:
            _OCSortImpl = self._load_backend()
        except Exception as exc:  # pragma: no cover - fail fast if the real OC-SORT backend is unavailable
            raise RuntimeError("OC-SORT backend import failed") from exc

        self.tracking_config = tracking_config
        self.track_thresh = float(tracking_config.get("track_thresh", tracking_config.get("track_threshold", 0.5)))
        self.match_thresh = float(tracking_config.get("match_thresh", tracking_config.get("match_threshold", 0.8)))
        self.track_buffer = int(tracking_config.get("track_buffer", 30))
        self._ocsort = _OCSortImpl(
            det_thresh=self.track_thresh,
            iou_threshold=self.match_thresh,
            max_age=self.track_buffer,
        )
        self.next_track_id = 1
        self.active_tracks: dict[int, _OCSortTrack] = {}
        self.lost_tracks: dict[int, _OCSortTrack] = {}
        self.removed_tracks: dict[int, _OCSortTrack] = {}
        self.frame_id = 0
        self.last_tracker_time_ms = 0.0
        self.last_duplicate_active_events = 0
        self.last_suspected_fragmentations = 0
        self._last_track_bbox: dict[int, list[float]] = {}
        backend_module = importlib.import_module(self._ocsort.__class__.__module__)
        self.backend_module = self._ocsort.__class__.__module__
        self.backend_class = self._ocsort.__class__.__name__
        self.backend_source_path = str(Path(inspect.getfile(backend_module)).resolve())
        self._debug_frames = {0, 1, 2, 160, 180, 188, 200}

    def _load_backend(self):
        project_root = Path(__file__).resolve().parents[1]
        backend_path = project_root / "third_party" / "OC_SORT" / "trackers" / "ocsort_tracker" / "ocsort.py"
        if not backend_path.exists():
            raise FileNotFoundError(f"OC-SORT backend not found: {backend_path}")

        backend_dir = str(backend_path.parent)
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        spec = util.spec_from_file_location("third_party.OC_SORT.trackers.ocsort_tracker.ocsort", backend_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load OC-SORT backend from {backend_path}")

        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "OCSort"):
            raise ImportError(f"OC-SORT backend module missing OCSort class: {backend_path}")
        return module.OCSort

    def update(self, detections: list[dict]) -> list[TrackedVehicle]:
        import time

        start_time = time.perf_counter()
        self.frame_id += 1
        candidates = [d for d in detections if float(d.get("score", 0.0)) >= self.track_thresh]
        ocsort_input = np.asarray(
            [
                [
                    float(detection["bbox"][0]),
                    float(detection["bbox"][1]),
                    float(detection["bbox"][2]),
                    float(detection["bbox"][3]),
                    float(detection["score"]),
                ]
                for detection in candidates
            ],
            dtype=np.float32,
        )
        if ocsort_input.size == 0:
            ocsort_input = np.empty((0, 5), dtype=np.float32)
        img_info = (600, 800)
        img_size = (600, 800)
        if self.frame_id - 1 in self._debug_frames:
            print(
                "[OCSORT INPUT]\n"
                f"frame={self.frame_id - 1}\n"
                f"detections={ocsort_input.tolist()}\n"
                f"img_info={img_info}\n"
                f"img_size={img_size}"
            )
        raw_tracks = self._ocsort.update(ocsort_input, img_info, img_size)
        if self.frame_id - 1 in self._debug_frames:
            print(
                "[OCSORT RAW]\n"
                f"frame={self.frame_id - 1}\n"
                f"type={type(raw_tracks)}\n"
                f"shape={getattr(raw_tracks, 'shape', None)}\n"
                f"dtype={getattr(raw_tracks, 'dtype', None)}\n"
                f"raw={raw_tracks.tolist() if hasattr(raw_tracks, 'tolist') else raw_tracks}"
            )
        current_tracks: list[TrackedVehicle] = []
        for raw_track in raw_tracks:
            if len(raw_track) < 5:
                raise RuntimeError(f"Unexpected OC-SORT output shape: {raw_track}")
            x1, y1, x2, y2, track_id = raw_track[:5]
            score = None
            class_id = None
            class_name = None
            parsed_bbox = [float(x1), float(y1), float(x2), float(y2)]
            if parsed_bbox[2] <= parsed_bbox[0] or parsed_bbox[3] <= parsed_bbox[1] or (parsed_bbox[2] - parsed_bbox[0]) * (parsed_bbox[3] - parsed_bbox[1]) <= 0:
                print(
                    "[OCSORT PARSE ERROR]\n"
                    f"frame={self.frame_id - 1}\n"
                    f"raw={raw_track.tolist() if hasattr(raw_track, 'tolist') else raw_track}\n"
                    f"parsed_bbox={parsed_bbox}"
                )
            if self.frame_id - 1 in self._debug_frames:
                print(
                    "[OCSORT PARSE]\n"
                    f"frame={self.frame_id - 1}\n"
                    f"RAW={raw_track.tolist() if hasattr(raw_track, 'tolist') else raw_track}\n"
                    f"PARSED: track_id={int(track_id)} bbox={parsed_bbox} confidence={score} class_id={class_id} class_name={class_name}"
                )
            current_tracks.append(
                TrackedVehicle(
                    track_id=int(track_id),
                    bbox=parsed_bbox,
                    confidence=score,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        duplicate_active_events = 0
        suspected_fragmentations = 0
        seen_ids: set[int] = set()
        seen_bboxes: set[tuple[int, int, int, int]] = set()
        for track in current_tracks:
            bbox = tuple(map(int, track.bbox))
            if track.track_id in seen_ids:
                raise RuntimeError(f"Duplicate active track_id detected: {track.track_id}")
            seen_ids.add(track.track_id)
            if bbox in seen_bboxes:
                duplicate_active_events += 1
                print(
                    f"[DUPLICATE ACTIVE TRACK] tracker=ocsort frame={self.frame_id} id_a={track.track_id} id_b={track.track_id} bbox_a={list(bbox)} bbox_b={list(bbox)} iou=1.00"
                )
            seen_bboxes.add(bbox)

        self.last_tracker_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_duplicate_active_events = duplicate_active_events
        self.last_suspected_fragmentations = suspected_fragmentations
        self._last_track_bbox = {track.track_id: list(track.bbox) for track in current_tracks}
        return current_tracks

    def get_stats(self) -> dict[str, int]:
        return {
            "tracked": len(self.active_tracks),
            "lost": len(self.lost_tracks),
            "removed": len(self.removed_tracks),
            "output": len(self.active_tracks),
            "duplicate_active_events": self.last_duplicate_active_events,
            "suspected_fragmentations": self.last_suspected_fragmentations,
        }

