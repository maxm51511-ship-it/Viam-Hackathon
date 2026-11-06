"""YOLO instance-segmentation inference (Ultralytics).

Ultralytics is an OPTIONAL dependency (``requirements-ml.txt``): the rest of
the pipeline runs on heuristics without it. A pretrained COCO model does NOT
know spaghetti failures — weights must be fine-tuned on our dataset
(``datasets/print_defects``) before YOLO output is trusted.

Class names come from the model; they are mapped onto ``MaterialClass`` so the
dataset label vocabulary and the classifier stay in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from printbot.types import Detection2D, MaterialClass

log = logging.getLogger(__name__)

_NAME_MAP = {c.value: c for c in MaterialClass}


class YoloSegmenter:
    def __init__(self, cfg: dict[str, Any]):
        self.weights = Path(cfg["weights"])
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("confidence", 0.35))
        self.iou = float(cfg.get("iou", 0.5))
        self.device = cfg.get("device")
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.weights}. Train with tools/train_yolo.py "
                "or set yolo.weights in config/settings.json")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics not installed: pip install -r requirements-ml.txt") from e
        self._model = YOLO(str(self.weights))
        unknown = [n for n in self._model.names.values() if n not in _NAME_MAP]
        if unknown:
            log.warning("model classes not in MaterialClass vocabulary (ignored): %s", unknown)

    def predict(self, bgr: np.ndarray) -> list[Detection2D]:
        if self._model is None:
            self.load()
        res = self._model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                  device=self.device, verbose=False, retina_masks=True)[0]
        return results_to_detections(res, bgr.shape[:2])


def results_to_detections(res, hw: tuple[int, int]) -> list[Detection2D]:
    """Convert one Ultralytics ``Results`` object into ``Detection2D``s."""
    out: list[Detection2D] = []
    if res.boxes is None:
        return out
    names = res.names
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    masks: Optional[np.ndarray] = None
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy() > 0.5
    for i, (b, c, k) in enumerate(zip(boxes, confs, clss)):
        label = _NAME_MAP.get(names[k])
        if label is None:
            continue
        m = None
        if masks is not None and masks[i].shape == hw:
            m = masks[i]
        out.append(Detection2D(label=label, confidence=float(c),
                               bbox_xyxy=tuple(int(v) for v in b), mask=m, source="yolo"))
    return out
