"""What the controller knows about traffic. Sensors fill this in; it can come from a
perfect sensor or from cameras, and the controller cannot tell which."""

from dataclasses import dataclass

from controller.traffic_lights import Approach


@dataclass(frozen=True)
class ApproachObservation:
    queued: int = 0
    approaching: int = 0
    oldest_wait_s: float = 0.0
    average_speed_mps: float | None = None  # None when no vehicles are seen.

    def __post_init__(self):
        if self.queued < 0 or self.approaching < 0 or self.oldest_wait_s < 0:
            raise ValueError(f"invalid observation: {self}")


@dataclass(frozen=True)
class TrafficObservation:
    timestamp_sim_s: float
    source: str
    approaches: dict[Approach, ApproachObservation]
