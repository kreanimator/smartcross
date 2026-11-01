"""Lets the controller run on what the cameras see.

The simulation is only used to draw the scene. Its vehicles reach the observation solely
as pixels, passing through the detector, the trackers and the analyzers.

A missing or blank frame says nothing about the road, so it never reaches the detector. Brief
dropouts keep the last reading. A longer one marks the approach unavailable, because reporting
no vehicles would look like an empty road. A camera frozen on a real picture is not noticed.
"""

import numpy as np

from controller.observation import ApproachObservation, TrafficObservation
from controller.traffic_lights import Approach
from simulator.camera_geometry import camera_specs
from simulator.cameras import render
from simulator.simulation import STEPS_PER_OBSERVATION
from vision.tracker import CameraTracker
from vision.traffic_analyzer import ApproachAnalyzer

STEPS_PER_FRAME = 12  # 5 frames per simulated second: 10 tracked no better and costs double.
FPS = 5
BLANK_CONTRAST = 5  # Spread of pixel values. Real frames measured 55 to 60; a dead feed is about 0.
BAD_FRAMES_BEFORE_UNAVAILABLE = FPS  # One second.
GOOD_FRAMES_BEFORE_AVAILABLE = FPS  # Also long enough for tracks to have a speed again.


def frame_is_usable(frame: np.ndarray | None) -> bool:
    return frame is not None and frame[::8, ::8].std() >= BLANK_CONTRAST


class CameraHealth:
    """Whether a camera can be trusted, slow to change so one bad frame doesn't flip it."""

    def __init__(self):
        self.available = True
        self.opposite_frames = 0

    def update(self, usable: bool) -> bool:
        if usable == self.available:
            self.opposite_frames = 0
            return self.available
        self.opposite_frames += 1
        limit = GOOD_FRAMES_BEFORE_AVAILABLE if usable else BAD_FRAMES_BEFORE_UNAVAILABLE
        if self.opposite_frames >= limit:
            self.available = usable
            self.opposite_frames = 0
        return self.available


class CameraSensor:
    is_usable = staticmethod(frame_is_usable)

    def __init__(self, app, world, detector, frame_fault=None):
        """detector must keep weak boxes, as vision.tracker.MIN_DETECTION_CONFIDENCE asks.

        frame_fault(approach, frame, now_s) can replace a frame, to test failing cameras.
        """
        self.app, self.world, self.detector = app, world, detector
        self.frame_fault = frame_fault or (lambda _approach, frame, _now_s: frame)
        specs = camera_specs()
        self.trackers = {approach: CameraTracker(FPS) for approach in Approach}
        self.analyzers = {approach: ApproachAnalyzer(specs[approach.value], approach) for approach in Approach}
        self.health = {approach: CameraHealth() for approach in Approach}
        self.seen = {approach: ApproachObservation() for approach in Approach}

    def read(self, simulation) -> TrafficObservation | None:
        if simulation.steps % STEPS_PER_FRAME == 0:
            self._look(simulation)
        if simulation.steps % STEPS_PER_OBSERVATION == 0:
            return TrafficObservation(simulation.traffic.now_s, "cv", dict(self.seen))
        return None

    def _look(self, simulation) -> None:
        now_s = simulation.traffic.now_s
        self.world.show(simulation)
        render(self.app)
        frames = {
            approach: self.frame_fault(approach, self.world.cameras[approach.value].frame(), now_s)
            for approach in Approach
        }
        usable = [approach for approach, frame in frames.items() if self.is_usable(frame)]
        detections = dict(zip(usable, self.detector.detect([frames[a] for a in usable]))) if usable else {}
        for approach in Approach:
            # Trackers and analyzers run on every frame, so tracks age out during an outage.
            tracks = self.trackers[approach].update(detections.get(approach, []))
            seen = self.analyzers[approach].update(now_s, tracks)
            available = self.health[approach].update(approach in detections)
            if not available:
                self.seen[approach] = ApproachObservation(available=False)
            elif approach in detections:
                self.seen[approach] = seen
