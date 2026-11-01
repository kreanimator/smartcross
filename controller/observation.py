"""What the controller knows about traffic. Sensors fill this in; it can come from a
perfect sensor or from cameras, and the controller cannot tell which."""

from dataclasses import dataclass

from controller.traffic_lights import Approach


@dataclass(frozen=True)
class ApproachObservation:
    queued: int = 0
    approaching: int = 0
    near_line: int = 0  # How many of the approaching vehicles are close to the stop line.
    oldest_wait_s: float = 0.0
    average_speed_mps: float | None = None  # None when no vehicles are seen.
    available: bool = True  # False means this approach cannot be seen, not that it is empty.

    def __post_init__(self):
        if min(self.queued, self.approaching, self.near_line, self.oldest_wait_s) < 0:
            raise ValueError(f"invalid observation: {self}")


@dataclass(frozen=True)
class TrafficObservation:
    timestamp_sim_s: float
    source: str
    approaches: dict[Approach, ApproachObservation]
