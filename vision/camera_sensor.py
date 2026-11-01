"""Lets the controller run on what the cameras see.

The simulation is only used to draw the scene. Its vehicles reach the observation solely
as pixels, passing through the detector, the trackers and the analyzers.
"""

from controller.observation import ApproachObservation, TrafficObservation
from controller.traffic_lights import Approach
from simulator.camera_geometry import camera_specs
from simulator.cameras import render
from simulator.simulation import STEPS_PER_OBSERVATION
from vision.tracker import CameraTracker
from vision.traffic_analyzer import ApproachAnalyzer

STEPS_PER_FRAME = 12  # 5 frames per simulated second: 10 tracked no better and costs double.
FPS = 5


class CameraSensor:
    def __init__(self, app, world, detector):
        """detector must keep weak boxes, as vision.tracker.MIN_DETECTION_CONFIDENCE asks."""
        self.app, self.world, self.detector = app, world, detector
        specs = camera_specs()
        self.trackers = {approach: CameraTracker(FPS) for approach in Approach}
        self.analyzers = {approach: ApproachAnalyzer(specs[approach.value], approach) for approach in Approach}
        self.seen = {approach: ApproachObservation() for approach in Approach}

    def read(self, simulation) -> TrafficObservation | None:
        if simulation.steps % STEPS_PER_FRAME == 0:
            self._look(simulation)
        if simulation.steps % STEPS_PER_OBSERVATION == 0:
            return TrafficObservation(simulation.traffic.now_s, "cv", dict(self.seen))
        return None

    def _look(self, simulation) -> None:
        self.world.show(simulation)
        render(self.app)
        frames = [self.world.cameras[approach.value].frame() for approach in Approach]
        for approach, detections in zip(Approach, self.detector.detect(frames)):
            tracks = self.trackers[approach].update(detections)
            self.seen[approach] = self.analyzers[approach].update(simulation.traffic.now_s, tracks)
