"""Demand-based strategy: give green to whichever direction needs it more.

Each direction has a pressure: its queued vehicles, partly its moving ones, and how
long its longest-waiting vehicle has waited. Green switches when the other side's
pressure is clearly higher, or when green has lasted max_green_s and the other side
has anyone at all. That limit is what stops a busy road starving a quiet one.
"""

from dataclasses import dataclass

from controller.observation import TrafficObservation
from controller.traffic_lights import Approach, LightSnapshot, Phase, Timings

APPROACHES = {"NS": (Approach.NORTH, Approach.SOUTH), "EW": (Approach.EAST, Approach.WEST)}
GREEN_DIRECTION = {Phase.NS_GREEN: "NS", Phase.EW_GREEN: "EW"}
OTHER = {"NS": "EW", "EW": "NS"}

APPROACHING_WEIGHT = 0.5  # A moving vehicle counts as half of a queued one.
WAIT_S_PER_VEHICLE = 10  # Ten seconds of waiting counts as one queued vehicle.
SWITCH_MARGIN = 2  # Switching costs lost time, so the other side must clearly need it.


@dataclass(frozen=True)
class Decision:
    """A logged switch: when, to which direction, why, and the pressures behind it."""

    time_s: float
    direction: str
    reason: str
    pressure_ns: float
    pressure_ew: float


def pressure(observation: TrafficObservation, direction: str) -> float:
    seen = [observation.approaches[approach] for approach in APPROACHES[direction]]
    vehicles = sum(a.queued + APPROACHING_WEIGHT * a.approaching for a in seen)
    return vehicles + max(a.oldest_wait_s for a in seen) / WAIT_S_PER_VEHICLE


class Adaptive:
    def __init__(self, max_green_s: float = 45, timings: Timings = Timings()):
        self.max_green_s = max_green_s
        self.min_green_s = timings.min_green_s
        self.target = "NS"
        self.decisions: list[Decision] = []

    def decide(self, observation: TrafficObservation, lights: LightSnapshot) -> str:
        green = GREEN_DIRECTION.get(lights.phase)
        if green is None or lights.seconds_in_phase < self.min_green_s:
            return self.target  # Nothing can change yet, so there is nothing to decide.

        mine, theirs = pressure(observation, green), pressure(observation, OTHER[green])
        if theirs > 0 and lights.seconds_in_phase >= self.max_green_s:
            self._switch_to(OTHER[green], "max green", observation)
        elif theirs > mine + SWITCH_MARGIN:
            self._switch_to(OTHER[green], "more demand", observation)
        return self.target

    def _switch_to(self, direction: str, reason: str, observation: TrafficObservation) -> None:
        self.target = direction
        self.decisions.append(
            Decision(
                observation.timestamp_sim_s,
                direction,
                reason,
                pressure(observation, "NS"),
                pressure(observation, "EW"),
            )
        )
