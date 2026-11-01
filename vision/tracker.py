"""Follows detected vehicles from frame to frame, one ByteTrack tracker per camera."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from simulator.camera_geometry import HEIGHT, WIDTH
from vision.detector import Detection

MIN_DETECTION_CONFIDENCE = 0.1  # The detector must pass weak boxes on: they keep tracks alive.
LOST_TRACK_S = 1.0  # How long an unseen vehicle keeps its ID.
NOMINAL_FRAME_RATE = 30  # ByteTrack measures its lost-track limit against this rate.


@dataclass(frozen=True)
class Track:
    id: int
    box: tuple[float, float, float, float]  # Left, top, right, bottom, in pixels.
    confidence: float


class CameraTracker:
    def __init__(self, fps: float):
        settings = SimpleNamespace(
            track_high_thresh=0.25,  # Stronger detections are matched first.
            track_low_thresh=MIN_DETECTION_CONFIDENCE,  # Weaker ones only continue existing tracks.
            new_track_thresh=0.25,
            track_buffer=round(LOST_TRACK_S * NOMINAL_FRAME_RATE),
            match_thresh=0.8,
            fuse_score=True,
        )
        self.tracker = BYTETracker(settings, frame_rate=fps)
        # ByteTrack scales the limit by frame rate and truncates, which loses a frame at low rates.
        self.tracker.max_time_lost = max(round(LOST_TRACK_S * fps), 1)

    def update(self, detections: list[Detection]) -> list[Track]:
        """Call once per frame, in order, even when nothing was detected."""
        rows = [(*d.box, d.confidence, 0) for d in detections]
        boxes = Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (HEIGHT, WIDTH))
        return [
            Track(int(row[4]), tuple(float(value) for value in row[:4]), float(row[5]))
            for row in self.tracker.update(boxes)
        ]
