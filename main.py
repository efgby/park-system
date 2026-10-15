from pathlib import Path
import csv
import statistics
import time
import hashlib

import cv2
import yaml

from detector.rtdetr_detector import RTDETRDetector
from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.ocsort_tracker import OCSortTracker
from direction import DirectionClassifier
from trajectory import TrajectoryStore
from video.video_source import VideoSource
from visualization.visualizer import Visualizer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _track_value(track, key: str):
    if hasattr(track, key):
        return getattr(track, key)
    if isinstance(track, dict) and key in track:
        return track[key]
    if key == "confidence" and isinstance(track, dict) and "score" in track:
        return track["score"]
    if key == "score" and isinstance(track, dict) and "confidence" in track:
        return track["confidence"]
    raise KeyError(key)


def _bbox_iou(box_a, box_b) -> float:
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


def _track_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _select_right_bottom_target(track_history: dict[int, list[dict[str, object]]]) -> list[int]:
    best_track_id = None
    best_score = None
    for track_id, rows in track_history.items():
        if len(rows) < 3:
            continue
        first_row = rows[0]
        last_row = rows[-1]
        first_center = _track_center([first_row["x1"], first_row["y1"], first_row["x2"], first_row["y2"]])
        last_center = _track_center([last_row["x1"], last_row["y1"], last_row["x2"], last_row["y2"]])
        dx = last_center[0] - first_center[0]
        dy = last_center[1] - first_center[1]
        if dx <= 0 or dy <= 0:
            continue
        first_area = max(0.0, (first_row["x2"] - first_row["x1"])) * max(0.0, (first_row["y2"] - first_row["y1"]))
        last_area = max(0.0, (last_row["x2"] - last_row["x1"])) * max(0.0, (last_row["y2"] - last_row["y1"]))
        if last_area < first_area:
            continue
        border_score = last_center[0] + last_center[1]
        score = (dx + dy, border_score)
        if best_score is None or score > best_score:
            best_score = score
            best_track_id = track_id
    if best_track_id is None:
        return []
    return [row["track_id"] for row in track_history[best_track_id]]


def _track_fingerprint_rows(rows: list[dict[str, object]]) -> list[tuple[int, int, int, int, int, int]]:
    return [
        (
            int(row["frame_id"]),
            int(row["track_id"]),
            int(round(float(row["x1"]))),
            int(round(float(row["y1"]))),
            int(round(float(row["x2"]))),
            int(round(float(row["y2"]))),
        )
        for row in rows
    ]


def _summarize_tracks(rows: list[dict[str, object]]) -> dict[str, object]:
    by_track: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_track.setdefault(int(row["track_id"]), []).append(row)

    lifespans = {track_id: len(track_rows) for track_id, track_rows in by_track.items()}
    summary = {
        "rows": len(rows),
        "unique_raw_ids": len(by_track),
        "tracks_le_5": sum(1 for lifespan in lifespans.values() if lifespan <= 5),
        "tracks_le_10": sum(1 for lifespan in lifespans.values() if lifespan <= 10),
        "tracks_le_30": sum(1 for lifespan in lifespans.values() if lifespan <= 30),
        "track_lifespans": [
            {
                "track_id": track_id,
                "first_frame": track_rows[0]["frame_id"],
                "last_frame": track_rows[-1]["frame_id"],
                "lifespan": len(track_rows),
                "observation_count": len(track_rows),
            }
            for track_id, track_rows in sorted(by_track.items())
        ],
    }
    return summary


def _find_right_bottom_vehicle(rows: list[dict[str, object]]) -> dict[str, object]:
    by_track: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_track.setdefault(int(row["track_id"]), []).append(row)

    best_track_id = None
    best_score = None
    for track_id, track_rows in by_track.items():
        if len(track_rows) < 3:
            continue
        first_row = track_rows[0]
        last_row = track_rows[-1]
        first_center = _track_center([first_row["x1"], first_row["y1"], first_row["x2"], first_row["y2"]])
        last_center = _track_center([last_row["x1"], last_row["y1"], last_row["x2"], last_row["y2"]])
        dx = last_center[0] - first_center[0]
        dy = last_center[1] - first_center[1]
        if dx <= 0 or dy <= 0:
            continue
        first_area = max(0.0, (first_row["x2"] - first_row["x1"])) * max(0.0, (first_row["y2"] - first_row["y1"]))
        last_area = max(0.0, (last_row["x2"] - last_row["x1"])) * max(0.0, (last_row["y2"] - last_row["y1"]))
        if last_area < first_area:
            continue
        border_score = last_center[0] + last_center[1]
        score = (dx + dy, border_score)
        if best_score is None or score > best_score:
            best_score = score
            best_track_id = track_id

    if best_track_id is None:
        return {"track_id": None, "raw_id_sequence": []}

    track_rows = by_track[best_track_id]
    return {
        "track_id": best_track_id,
        "raw_id_sequence": [row["track_id"] for row in track_rows],
    }


def _print_summary(label: str, summary: dict[str, object], tracker_ms: list[float], fragments: list[dict[str, object]], duplicates: list[dict[str, object]], right_bottom: dict[str, object]) -> None:
    average_ms = statistics.mean(tracker_ms) if tracker_ms else 0.0
    p95_ms = statistics.quantiles(tracker_ms, n=20)[18] if len(tracker_ms) >= 20 else (max(tracker_ms) if tracker_ms else 0.0)
    print(label)
    print(f"Rows = {summary['rows']}")
    print(f"Unique raw IDs = {summary['unique_raw_ids']}")
    print(f"Fragments = {len(fragments)}")
    print(f"Duplicate active tracks = {len(duplicates)}")
    print(f"Tracks <=5 frames = {summary['tracks_le_5']}")
    print(f"Tracks <=10 frames = {summary['tracks_le_10']}")
    print(f"Tracks <=30 frames = {summary['tracks_le_30']}")
    print("Right-bottom vehicle:")
    print(f"Raw ID sequence = {right_bottom['raw_id_sequence']}")
    print(f"Average tracker ms = {average_ms:.3f}")
    print(f"P95 tracker ms = {p95_ms:.3f}")


def _trajectory_status(points: list[tuple[float, float]]) -> str:
    return "insufficient" if len(points) < 3 else "valid"


def _direction_label(direction: str | None) -> str:
    if not direction:
        return "UNKNOWN"
    normalized = str(direction).upper()
    if normalized in {"LEFT", "FRONT", "RIGHT"}:
        return normalized
    return "UNKNOWN"


def _resolve_tracker_type(tracking_config: dict[str, object]) -> str:
    tracker_type = str(tracking_config.get("tracker_type", "bytetrack")).lower()
    if tracker_type in {"byte", "bytetrack"}:
        return "bytetrack"
    if tracker_type in {"oc", "ocsort"}:
        return "ocsort"
    return tracker_type


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "config.yaml")

    video_source = VideoSource(config["video"], project_root)
    detector = RTDETRDetector(config["model"], config["classes"], project_root)
    tracking_config = config.get("tracking", {})
    tracker_type = _resolve_tracker_type(tracking_config)
    tracker = OCSortTracker(tracking_config) if tracker_type == "ocsort" else ByteTrackTracker(tracking_config)
    visualizer = Visualizer(config)
    trajectory_store = TrajectoryStore(config.get("trajectory_history", 30))
    direction_classifier = DirectionClassifier(config)
    enable_business_lifecycle = bool(tracking_config.get("enable_business_lifecycle", False))

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_name = tracker_type
    output_path = output_dir / f"result_{tracker_name}_fixed.mp4"
    csv_path = output_dir / f"tracks_{tracker_name}_fixed.csv"
    if csv_path.exists():
        csv_path.unlink()
    start_frame = int(config.get("video", {}).get("start_frame", 0))
    end_frame = int(config.get("video", {}).get("end_frame", 211))
    if start_frame > 0:
        video_source.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = None
    frame_index = start_frame
    csv_rows: list[dict[str, object]] = []
    frame_tracks: dict[int, list[dict[str, object]]] = {}
    detection_count = 0
    tracker_output_count = 0
    update_times_ms: list[float] = []
    suspected_fragmentations: list[dict[str, object]] = []
    duplicate_active_tracks: list[dict[str, object]] = []
    target_history: dict[int, list[dict[str, object]]] = {}
    direction_results: dict[int, dict[str, object]] = {}
    counted_track_ids: set[int] = set()
    direction_counts = {"LEFT": 0, "FRONT": 0, "RIGHT": 0}
    if enable_business_lifecycle:
        raise RuntimeError("Business lifecycle is disabled for the tracker A/B test")

    print(
        "[TRACKER INIT]\n"
        f"requested={tracker_type}\n"
        f"instance_class={tracker.__class__.__name__}\n"
        f"module={tracker.__class__.__module__}\n"
        f"csv={csv_path.name}"
    )
    print(
        "[CSV OUTPUT]\n"
        f"tracker={tracker_name}\n"
        f"absolute_path={csv_path.resolve()}"
    )

    try:
        while True:
            if end_frame is not None and frame_index > int(end_frame):
                break
            success, frame = video_source.read()
            if not success:
                break

            if writer is None:
                height, width = frame.shape[:2]
                fps = video_source.fps or 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            detections = detector.detect(frame)
            detection_count += len(detections)
            if frame_index < start_frame + 5:
                print(
                    "[TRACKER CALL]\n"
                    f"frame={frame_index}\n"
                    f"tracker_type={tracker_type}\n"
                    f"tracker_class={tracker.__class__.__name__}\n"
                    f"detections={len(detections)}"
                )
            update_start = time.perf_counter()
            tracks = tracker.update(detections)
            update_times_ms.append((time.perf_counter() - update_start) * 1000.0)
            stats = tracker.get_stats()
            tracker_output_count += len(tracks)
            if 160 <= frame_index <= 211:
                print(
                    "[WRAPPER STATE]\n"
                    f"frame={frame_index}\n"
                    f"current detections={len(detections)}\n"
                    f"current matched tracks={len(tracks)}\n"
                    f"tracked_stracks={stats.get('tracked')}\n"
                    f"lost_stracks={stats.get('lost')}\n"
                    f"removed_stracks={stats.get('removed')}\n"
                    f"wrapper returned tracks={len(tracks)}"
                )
            if frame_index < start_frame + 5:
                output_ids = [track.track_id if hasattr(track, "track_id") else track["track_id"] for track in tracks]
                print(
                    f"outputs={len(tracks)}\n"
                    f"output_ids={output_ids}"
                )
            active_track_ids = {track.track_id if hasattr(track, "track_id") else track["track_id"] for track in tracks}
            current_frame_tracks: list[dict[str, object]] = []
            for track in tracks:
                bbox = track.bbox if hasattr(track, "bbox") else track["bbox"]
                track_id = track.track_id if hasattr(track, "track_id") else track["track_id"]
                confidence_value = _track_value(track, "confidence")
                confidence = float(confidence_value) if confidence_value is not None else None
                class_name_value = _track_value(track, "class_name") if hasattr(track, "class_name") or (isinstance(track, dict) and "class_name" in track) else None
                class_name = str(class_name_value) if class_name_value is not None else None
                trajectory_store.update(track_id, bbox)
                trajectory_points = trajectory_store.get(track_id)
                direction_result = direction_classifier.classify(track_id, trajectory_points)
                direction_results[track_id] = {
                    "direction": direction_result.direction,
                    "zone": direction_result.zone,
                    "confidence": direction_result.confidence,
                    "reason": direction_result.reason,
                }
                direction_label = _direction_label(direction_result.direction)
                if (
                    _trajectory_status(trajectory_points) == "valid"
                    and direction_label in direction_counts
                    and track_id not in counted_track_ids
                ):
                    direction_counts[direction_label] += 1
                    counted_track_ids.add(track_id)
                row = {
                    "frame_id": frame_index,
                    "track_id": track_id,
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                    "confidence": confidence,
                    "class_name": class_name,
                    "direction": direction_results[track_id]["direction"],
                    "direction_zone": direction_results[track_id]["zone"],
                    "direction_confidence": direction_results[track_id]["confidence"],
                    "direction_reason": direction_results[track_id]["reason"],
                }
                csv_rows.append(row)
                current_frame_tracks.append(row)
                target_history.setdefault(track_id, []).append(row)
            trajectory_store.prune(active_track_ids)
            frame_tracks[frame_index] = current_frame_tracks

            if frame_index > start_frame:
                previous_tracks = frame_tracks.get(frame_index - 1, [])
                for previous_track in previous_tracks:
                    for current_track in current_frame_tracks:
                        if previous_track["track_id"] == current_track["track_id"]:
                            continue
                        iou = _bbox_iou(
                            [previous_track["x1"], previous_track["y1"], previous_track["x2"], previous_track["y2"]],
                            [current_track["x1"], current_track["y1"], current_track["x2"], current_track["y2"]],
                        )
                        if iou >= 0.85:
                            suspected_fragmentations.append(
                                {
                                    "tracker": tracker_name,
                                    "frame": frame_index,
                                    "old_id": previous_track["track_id"],
                                    "new_id": current_track["track_id"],
                                    "iou": iou,
                                    "old_bbox": [previous_track["x1"], previous_track["y1"], previous_track["x2"], previous_track["y2"]],
                                    "new_bbox": [current_track["x1"], current_track["y1"], current_track["x2"], current_track["y2"]],
                                }
                            )

            seen_boxes: list[dict[str, object]] = []
            for current_track in current_frame_tracks:
                current_bbox = [current_track["x1"], current_track["y1"], current_track["x2"], current_track["y2"]]
                for seen_track in seen_boxes:
                    iou = _bbox_iou(current_bbox, [seen_track["x1"], seen_track["y1"], seen_track["x2"], seen_track["y2"]])
                    if iou >= 0.85 and seen_track["track_id"] != current_track["track_id"]:
                        duplicate_active_tracks.append(
                            {
                                "frame": frame_index,
                                "id_a": seen_track["track_id"],
                                "id_b": current_track["track_id"],
                                "iou": iou,
                                "bbox_a": [seen_track["x1"], seen_track["y1"], seen_track["x2"], seen_track["y2"]],
                                "bbox_b": current_bbox,
                            }
                        )
                seen_boxes.append(current_track)

            if frame_index % 30 == 0:
                print(f"Frame: {frame_index}")
                print(f"Detections: {len(detections)}")
                print(f"Tracked: {stats['tracked']}")
                print(f"Lost: {stats['lost']}")
                print(f"Removed: {stats['removed']}")
                print(f"Output tracks: {stats['output']}")
                for track in tracks[:5]:
                    confidence_value = track.confidence if hasattr(track, "confidence") else track["score"]
                    confidence_text = f"{confidence_value:.2f}" if confidence_value is not None else "None"
                    print(
                        f"Track ID: {track.track_id if hasattr(track, 'track_id') else track['track_id']} | bbox: {track.bbox if hasattr(track, 'bbox') else track['bbox']} | confidence: {confidence_text} | class: {track.class_name if hasattr(track, 'class_name') else track['class_name']}"
                    )

            annotated_frame, render_stats = visualizer.draw(
                frame,
                detections,
                tracks,
                {},
                {track_id: trajectory_store.get(track_id) for track_id in active_track_ids},
                frame_index=frame_index,
                debug_mode=False,
            )

            hud_lines = [
                "Parking Monitoring",
                f"LEFT:   {direction_counts['LEFT']}",
                f"FRONT:  {direction_counts['FRONT']}",
                f"RIGHT:  {direction_counts['RIGHT']}",
                f"TOTAL:  {len(counted_track_ids)}",
                f"Frame: {frame_index} / {end_frame}",
            ]
            hud_y = 28
            for hud_line in hud_lines:
                cv2.putText(annotated_frame, hud_line, (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                hud_y += 26

            if frame_index % 30 == 0:
                print(f"Rendered bbox: {render_stats['rendered_bbox']}")
                print(f"Rendered labels: {render_stats['rendered_labels']}")
                print(f"Rendered trajectories: {render_stats['rendered_trajectories']}")

            writer.write(annotated_frame)
            cv2.imshow("Parking Monitoring", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            frame_index += 1
    finally:
        video_source.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer_csv = csv.DictWriter(csv_file, fieldnames=["frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_name", "direction", "direction_zone", "direction_confidence", "direction_reason"])
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)

        fingerprint_rows = _track_fingerprint_rows(csv_rows[:10])
        print(f"FINGERPRINT={fingerprint_rows}")
        print(f"LEFT = {direction_counts['LEFT']}")
        print(f"FRONT = {direction_counts['FRONT']}")
        print(f"RIGHT = {direction_counts['RIGHT']}")
        print(f"TOTAL = {len(counted_track_ids)}")
        print(f"counted track IDs:")
        print(f"LEFT: {sorted(track_id for track_id, info in direction_results.items() if _direction_label(info.get('direction')) == 'LEFT' and track_id in counted_track_ids)}")
        print(f"FRONT: {sorted(track_id for track_id, info in direction_results.items() if _direction_label(info.get('direction')) == 'FRONT' and track_id in counted_track_ids)}")
        print(f"RIGHT: {sorted(track_id for track_id, info in direction_results.items() if _direction_label(info.get('direction')) == 'RIGHT' and track_id in counted_track_ids)}")

        summary = _summarize_tracks(csv_rows)
        right_bottom = _find_right_bottom_vehicle(csv_rows)

        print(f"CSV saved: {csv_path}")
        print(f"Video saved: {output_path}")
        print(
            "[CSV FINAL]\n"
            f"tracker={tracker_name}\n"
            f"absolute_path={csv_path.resolve()}\n"
            f"rows={len(csv_rows)}\n"
            f"unique_ids={summary['unique_raw_ids']}\n"
            f"file_size={csv_path.stat().st_size}\n"
            f"sha256={hashlib.sha256(csv_path.read_bytes()).hexdigest()}"
        )
        print(f"tracker={tracker_name}")
        print(f"frame_range={start_frame}-{end_frame}")
        print(f"total_detections={detection_count}")
        print(f"total_tracker_outputs={tracker_output_count}")
        print(f"unique_raw_ids={summary['unique_raw_ids']}")
        print(f"suspected_fragmentations={len(suspected_fragmentations)}")
        print(f"duplicate_active_track_events={len(duplicate_active_tracks)}")
        print(f"average_tracker_time_ms={statistics.mean(update_times_ms) if update_times_ms else 0.0:.3f}")
        print(f"p95_tracker_time_ms={statistics.quantiles(update_times_ms, n=20)[18] if len(update_times_ms) >= 20 else (max(update_times_ms) if update_times_ms else 0.0):.3f}")
        print("TRACK_LIFESPANS:")
        for item in summary["track_lifespans"]:
            print(item)
        print("SUSPECTED_FRAGMENTATIONS:")
        for item in suspected_fragmentations:
            print(item)
        print("DUPLICATE_ACTIVE_TRACKS:")
        for item in duplicate_active_tracks:
            print(item)

        if tracker_name == "ocsort":
            byte_csv = output_dir / "bytetrack_tracks.csv"
            if byte_csv.exists():
                try:
                    with byte_csv.open("r", encoding="utf-8", newline="") as byte_file, csv_path.open("r", encoding="utf-8", newline="") as ocsort_file:
                        byte_rows = list(csv.DictReader(byte_file))
                        ocsort_rows = list(csv.DictReader(ocsort_file))
                    if _track_fingerprint_rows(byte_rows[:10]) == _track_fingerprint_rows(ocsort_rows[:10]):
                        print("[AB TEST INVALID]")
                        print("ByteTrack and OC-SORT outputs are exactly identical.")
                        return
                except Exception:
                    pass

        _print_summary(
            "BYTETrack" if tracker_name == "bytetrack" else "OC-SORT",
            summary,
            update_times_ms,
            suspected_fragmentations,
            duplicate_active_tracks,
            right_bottom,
        )

        print("Tracker run complete.")


if __name__ == "__main__":
    main()
