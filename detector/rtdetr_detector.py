from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np


try:
    import torch
except Exception:  # pragma: no cover - torch may be absent during bootstrap
    torch = None


class RTDETRDetector:
    def __init__(self, model_config: dict, class_config: dict, project_root: Path) -> None:
        self.model_config = model_config
        self.class_config = class_config
        self.project_root = project_root
        self.confidence_threshold = float(model_config.get("confidence_threshold", 0.4))
        self.enabled_classes = {name for name, enabled in class_config.items() if enabled}
        self.device = "cpu"
        self.model = None
        self.postprocessor = None
        self.class_names = {3: "car", 6: "bus", 8: "truck", 4: "motorcycle"}
        self._debug_printed = False
        self.input_size = 640
        self._load_model()

    def _ensure_rtdetr_path(self) -> None:
        rtdetr_root = self.project_root / "rtdetr" / "rtdetr_pytorch"
        if rtdetr_root.exists():
            path_str = str(rtdetr_root)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _load_model(self) -> None:
        if torch is None:
            print("[WARN] torch is not installed.")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._ensure_rtdetr_path()
            from src.core import YAMLConfig
        except Exception as exc:
            print(f"[WARN] Failed to import official RT-DETR: {exc}")
            return

        # 读取 config.yaml 中指定的官方 RT-DETR YAML
        config_path = self.model_config.get("config")
        weights_path = self.model_config.get("weights")

        if not config_path:
            print("[WARN] RT-DETR config path is missing.")
            return

        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = self.project_root / config_file

        weights_file = Path(weights_path)
        if not weights_file.is_absolute():
            weights_file = self.project_root / weights_file

        if not config_file.exists():
            print(f"[WARN] RT-DETR config not found: {config_file}")
            return

        if not weights_file.exists():
            print(f"[WARN] RT-DETR weights not found: {weights_file}")
            return

        try:
            cfg = YAMLConfig(str(config_file))

            checkpoint = torch.load(
                weights_file,
                map_location="cpu"
            )

            state = self._extract_state_dict(checkpoint)

            cfg.model.load_state_dict(state)

            self.model = cfg.model.to(self.device)
            self.model.eval()

            self.postprocessor = cfg.postprocessor

            print(
                f"[INFO] RT-DETR loaded successfully "
                f"on {self.device}"
            )

        except Exception as exc:
            print(f"[ERROR] Failed to initialize RT-DETR: {exc}")
            self.model = None

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            if "ema" in checkpoint:
                ema = checkpoint["ema"]
                if isinstance(ema, dict) and "module" in ema:
                    return ema["module"]
                if isinstance(ema, dict):
                    return ema
            if "model" in checkpoint:
                model_state = checkpoint["model"]
                if isinstance(model_state, dict):
                    return model_state
        if isinstance(checkpoint, dict):
            return checkpoint
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    def detect(self, frame) -> list[dict[str, Any]]:
        if self.model is None:
            return []

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results, meta = self._predict(image)
        detections: list[dict[str, Any]] = []

        if not results:
            return detections

        boxes, scores, class_ids = self._normalize_predictions(results)
        boxes = self._restore_boxes(boxes, meta)

        if not self._debug_printed:
            self._debug_printed = True
            print(f"[DEBUG] Original frame size: width={frame.shape[1]}, height={frame.shape[0]}")
            print(f"[DEBUG] Model input size: width={self.input_size}, height={self.input_size}")
            print(f"[DEBUG] Raw detections: {len(boxes)}")
            for index, (bbox, score, class_id) in enumerate(zip(boxes[:5], scores[:5], class_ids[:5])):
                print(
                    f"[DEBUG] Postprocessed[{index}]: bbox={bbox.tolist()}, "
                    f"score={float(score):.4f}, class_id={int(class_id)}, "
                    f"class_name={self.class_names.get(int(class_id), str(int(class_id)))}"
                )

        for bbox, score, class_id in zip(boxes, scores, class_ids):
            class_name = self.class_names.get(int(class_id), str(int(class_id)))
            if class_name not in self.enabled_classes:
                continue
            detections.append(
                {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "score": float(score),
                    "class_id": int(class_id),
                    "class_name": class_name,
                }
            )

        return detections

    def _predict(self, image: np.ndarray):
        tensor, meta = self._preprocess(image)
        with torch.no_grad():
            outputs = self.model(tensor)

        if self.postprocessor is None:
            raise RuntimeError("RT-DETR postprocessor is not available.")

        target_sizes = torch.tensor([[meta["padded_height"], meta["padded_width"]]], device=tensor.device)
        return self.postprocessor(outputs, target_sizes), meta

    def _preprocess(self, image: np.ndarray):
        original_height, original_width = image.shape[:2]
        scale = min(self.input_size / original_width, self.input_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=resized.dtype)
        padded[:resized_height, :resized_width] = resized

        tensor = torch.from_numpy(padded).to(torch.float32) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        meta = {
            "scale": scale,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "padded_width": self.input_size,
            "padded_height": self.input_size,
            "original_width": original_width,
            "original_height": original_height,
        }
        return tensor.to(self.device), meta

    def _restore_boxes(self, boxes: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        restored = boxes.copy()
        restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, meta["resized_width"]) / meta["scale"]
        restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, meta["resized_height"]) / meta["scale"]
        restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, meta["original_width"])
        restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, meta["original_height"])
        return restored

    def _normalize_predictions(self, results: Any):
        result = results[0]
        if isinstance(result, dict):
            boxes = result.get("boxes")
            scores = result.get("scores")
            labels = result.get("labels")
        else:
            boxes = getattr(result, "boxes", None)
            scores = getattr(result, "scores", None)
            labels = getattr(result, "labels", None)
            if boxes is not None and hasattr(boxes, "xyxy"):
                scores = boxes.conf
                labels = boxes.cls
                boxes = boxes.xyxy

        if boxes is None or scores is None or labels is None:
            return [], [], []

        boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
        scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
        labels = labels.cpu().numpy().astype(int) if hasattr(labels, "cpu") else np.asarray(labels).astype(int)
        keep = scores >= self.confidence_threshold
        return boxes[keep], scores[keep], labels[keep]
