"""Finds vehicles in camera frames with a pretrained YOLO model."""

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).parent.parent / "weights" / "yolo11n.pt"
VEHICLE_NAMES = {"vehicle", "car", "bus", "truck"}  # Our fine-tuned class, then the COCO ones.
MIN_CONFIDENCE = 0.25
IMAGE_SIZE = 640


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # Left, top, right, bottom, in pixels.
    confidence: float
    label: str


class VehicleDetector:
    def __init__(self, weights: Path = WEIGHTS, min_confidence: float = MIN_CONFIDENCE, image_size: int = IMAGE_SIZE):
        self.model = YOLO(str(weights))
        self.min_confidence = min_confidence
        self.image_size = image_size
        self.classes = [number for number, name in self.model.names.items() if name in VEHICLE_NAMES]
        if not self.classes:
            raise ValueError(f"{weights} has no vehicle classes, only {self.model.names}")

    def detect(self, frames: list) -> list[list[Detection]]:
        """One list of detections per BGR frame."""
        results = self.model.predict(
            frames,
            conf=self.min_confidence,
            imgsz=self.image_size,
            classes=self.classes,
            verbose=False,
        )
        return [
            [
                Detection(tuple(box.xyxy[0].tolist()), float(box.conf), self.model.names[int(box.cls)])
                for box in result.boxes
            ]
            for result in results
        ]
