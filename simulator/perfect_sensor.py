"""Reads the simulator's exact state: only for tests and for scoring vision against the truth.

Nothing that runs the intersection may import this. Live control sees traffic through the
cameras alone, and tests/test_vision_only.py checks that it stays that way.
"""

from controller.observation import NEAR_LINE_M, ApproachObservation, TrafficObservation
from simulator.simulation import STEPS_PER_OBSERVATION
from simulator.traffic import STOPPED_SPEED, Lane, Traffic


def observe_lane(lane: Lane) -> ApproachObservation:
    before_line = [v for v in lane.vehicles if v.distance >= 0]
    if not before_line:
        return ApproachObservation()
    moving = [v for v in before_line if v.speed >= STOPPED_SPEED]
    queued = len(before_line) - len(moving)
    return ApproachObservation(
        queued=queued,
        approaching=len(moving),
        near_line=sum(v.distance <= NEAR_LINE_M for v in moving),
        oldest_wait_s=max(v.stopped_s for v in before_line),
        average_speed_mps=sum(v.speed for v in before_line) / len(before_line),
    )


def observe(traffic: Traffic) -> TrafficObservation:
    approaches = {approach: observe_lane(lane) for approach, lane in traffic.lanes.items()}
    return TrafficObservation(traffic.now_s, "perfect_sensor", approaches)


class PerfectSensor:
    def read(self, simulation):
        return observe(simulation.traffic) if simulation.steps % STEPS_PER_OBSERVATION == 0 else None
