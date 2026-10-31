"""Demand-based strategy: keep green while it is being used, unless the other side can't wait.

Green is held while vehicles are still clearing the stop line. Once it has lasted long
enough to be worth the time a switch loses, three checks can end that hold early, so the
other side doesn't pay for a long queue here: its longest wait, its queue length, and
its total vehicles. Once this side has cleared, green moves to the other side as soon as
it has more pressure. Green also ends at max_green_s if anyone at all is waiting.
"""

from dataclasses import dataclass

from controller.observation import TrafficObservation
from controller.traffic_lights import Approach, LightSnapshot, Phase, Timings

APPROACHES = {"NS": (Approach.NORTH, Approach.SOUTH), "EW": (Approach.EAST, Approach.WEST)}
GREEN_DIRECTION = {Phase.NS_GREEN: "NS", Phase.EW_GREEN: "EW"}
OTHER = {"NS": "EW", "EW": "NS"}

APPROACHING_WEIGHT = 0.5  # A moving vehicle counts as half of a queued one.
WAIT_S_PER_VEHICLE = 10  # Ten seconds of waiting counts as one queued vehicle.
MIN_GREEN_BEFORE_INTERRUPT_S = 15  # Switching loses 5 s, so a busy green must earn it first.
MAX_WAIT_S = 40  # The other side's longest wait that a clearing queue may cause.
MAX_QUEUE = 10  # The other side's longest queue; a lane holds about 13 vehicles.
OUTGROWN_MARGIN = 6  # How many more vehicles the other side needs to interrupt clearing.


@dataclass(frozen=True)
class Decision:
    """A logged switch: when, to which direction, why, and the pressures behind it."""

    time_s: float
    direction: str
    reason: str
    pressure_ns: float
    pressure_ew: float


def seen_from(observation: TrafficObservation, direction: str):
    return [observation.approaches[approach] for approach in APPROACHES[direction]]


def pressure(observation: TrafficObservation, direction: str) -> float:
    seen = seen_from(observation, direction)
    vehicles = sum(a.queued + APPROACHING_WEIGHT * a.approaching for a in seen)
    return vehicles + max(a.oldest_wait_s for a in seen) / WAIT_S_PER_VEHICLE


def vehicles_left(observation: TrafficObservation, direction: str) -> int:
    return sum(a.queued + a.approaching for a in seen_from(observation, direction))


def is_clearing(observation: TrafficObservation, direction: str) -> bool:
    return any(a.queued or a.near_line for a in seen_from(observation, direction))


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

        reason = self._reason_to_switch(observation, green, lights.seconds_in_phase)
        if reason:
            self.target = OTHER[green]
            self.decisions.append(
                Decision(
                    observation.timestamp_sim_s,
                    self.target,
                    reason,
                    pressure(observation, "NS"),
                    pressure(observation, "EW"),
                )
            )
        return self.target

    def _reason_to_switch(self, observation: TrafficObservation, green: str, seconds_green: float) -> str | None:
        other = OTHER[green]
        mine, theirs = pressure(observation, green), pressure(observation, other)
        if theirs == 0:
            return None
        if seconds_green >= self.max_green_s:
            return "max green"
        if not is_clearing(observation, green):
            return "queue cleared" if theirs > mine else None
        if seconds_green < MIN_GREEN_BEFORE_INTERRUPT_S:
            return None

        waiting = seen_from(observation, other)
        if max(a.oldest_wait_s for a in waiting) >= MAX_WAIT_S:
            return "long wait"
        if max(a.queued for a in waiting) >= MAX_QUEUE:
            return "long queue"
        if vehicles_left(observation, other) > vehicles_left(observation, green) + OUTGROWN_MARGIN:
            return "outgrown"
        return None
