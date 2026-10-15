from __future__ import annotations

from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, video_config: dict, project_root: Path) -> None:
        self.video_type = video_config.get("type", "file")
        self.source = video_config.get("source")
        self.project_root = project_root
        self.capture = self._open_capture()
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture.isOpened() else 0.0

    def _resolve_source(self) -> str:
        if self.video_type == "file":
            source_path = Path(self.source)
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            return str(source_path)
        return str(self.source)

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._resolve_source())

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
