"""Turns vehicle tracks into the traffic observation the controller reads.

A track's lowest box edge is the front of the car, because the cameras face oncoming traffic.
Measuring that point on the road plane gives its distance to the stop line, and how fast
that distance shrinks gives its speed. Only the camera's calibration is used, never truth.
"""

from collections import deque
from dataclasses import dataclass, field

from controller.observation import ApproachObservation, TrafficObservation
from controller.traffic_lights import Approach
from simulator.camera_geometry import CameraSpec, in_polygon
from simulator.perfect_sensor import NEAR_LINE_M
from simulator.roads import distance_to_line
from simulator.traffic import STOPPED_SPEED
from vision.tracker import Track

SPEED_WINDOW_S = 1.0  # Longer is steadier but slower to notice a stop.
MIN_SPEED_WINDOW_S = 0.5  # Until a track has this much history its speed is unknown.
FORGET_AFTER_S = 2.0  # History of a track unseen this long is dropped.


@dataclass
class _Followed:
    history: deque = field(default_factory=deque)  # (time, distance to the stop line)
    last_seen_s: float = 0.0
    stopped_s: float = 0.0


class ApproachAnalyzer:
    def __init__(self, spec: CameraSpec, approach: Approach):
        self.spec = spec
        self.approach = approach
        self.roi = spec.roi(approach)
        self.followed: dict[int, _Followed] = {}
        self.last_update_s: float | None = None

    def update(self, now_s: float, tracks: list[Track]) -> ApproachObservation:
        """Call once per frame, in time order."""
        dt = 0.0 if self.last_update_s is None else now_s - self.last_update_s
        self.last_update_s = now_s

        speeds = []
        waits = []
        near_line = queued = 0
        for track in tracks:
            front = ((track.box[0] + track.box[2]) / 2, track.box[3])
            ground = self.spec.ground_point(front)
            if ground is None or not in_polygon(front, self.roi):
                continue
            distance = distance_to_line(self.approach, *ground)
            followed = self.followed.setdefault(track.id, _Followed())
            followed.last_seen_s = now_s
            followed.history.append((now_s, distance))
            while now_s - followed.history[0][0] > SPEED_WINDOW_S:
                followed.history.popleft()

            speed = self._speed(followed)
            if speed is not None:
                speeds.append(speed)
            if speed is not None and speed < STOPPED_SPEED:
                queued += 1
                followed.stopped_s += dt
            else:
                near_line += distance <= NEAR_LINE_M
            waits.append(followed.stopped_s)

        self.followed = {id: f for id, f in self.followed.items() if now_s - f.last_seen_s <= FORGET_AFTER_S}
        return ApproachObservation(
            queued=queued,
            approaching=len(waits) - queued,
            near_line=near_line,
            oldest_wait_s=max(waits, default=0.0),
            average_speed_mps=sum(speeds) / len(speeds) if speeds else None,
        )

    @staticmethod
    def _speed(followed: _Followed) -> float | None:
        start_s, start = followed.history[0]
        end_s, end = followed.history[-1]
        if end_s - start_s < MIN_SPEED_WINDOW_S:
            return None
        return max((start - end) / (end_s - start_s), 0.0)


class TrafficAnalyzer:
    """One analyzer per approach, each reading the camera that faces it."""

    def __init__(self, specs: dict[str, CameraSpec]):
        self.analyzers = {approach: ApproachAnalyzer(specs[approach.value], approach) for approach in Approach}

    def update(self, now_s: float, tracks: dict[Approach, list[Track]]) -> TrafficObservation:
        approaches = {approach: analyzer.update(now_s, tracks[approach]) for approach, analyzer in self.analyzers.items()}
        return TrafficObservation(now_s, "cv", approaches)
