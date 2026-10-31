"""Safety state machine: the only code that decides what the lamps show.

Strategies call request_phase() to say which direction they want. The machine
gets there through yellow and all-red, and only once the timings allow it.
"""

import math
from dataclasses import dataclass
from enum import StrEnum


class Approach(StrEnum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class Lamp(StrEnum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class Phase(StrEnum):
    ALL_RED = "ALL_RED"
    NS_GREEN = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    EW_GREEN = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"


def _lamps(north_south: Lamp, east_west: Lamp) -> dict[Approach, Lamp]:
    return {
        Approach.NORTH: north_south,
        Approach.SOUTH: north_south,
        Approach.EAST: east_west,
        Approach.WEST: east_west,
    }


# No phase here lights both directions, so conflicting greens cannot be shown.
LAMPS = {
    Phase.ALL_RED: _lamps(Lamp.RED, Lamp.RED),
    Phase.NS_GREEN: _lamps(Lamp.GREEN, Lamp.RED),
    Phase.NS_YELLOW: _lamps(Lamp.YELLOW, Lamp.RED),
    Phase.EW_GREEN: _lamps(Lamp.RED, Lamp.GREEN),
    Phase.EW_YELLOW: _lamps(Lamp.RED, Lamp.YELLOW),
}

GREEN_FOR = {"NS": Phase.NS_GREEN, "EW": Phase.EW_GREEN}
YELLOW_AFTER = {Phase.NS_GREEN: Phase.NS_YELLOW, Phase.EW_GREEN: Phase.EW_YELLOW}


@dataclass(frozen=True)
class Timings:
    """Simulation parameters in simulated seconds, not real-world guidance."""

    min_green_s: float = 10
    yellow_s: float = 3
    all_red_s: float = 2

    def __post_init__(self):
        if self.min_green_s < 0 or self.yellow_s <= 0 or self.all_red_s <= 0:
            raise ValueError(f"invalid timings: {self}")


@dataclass(frozen=True)
class LightSnapshot:
    phase: Phase
    lamps: dict[Approach, Lamp]
    seconds_in_phase: float


class TrafficLights:
    def __init__(self, timings: Timings = Timings()):
        self.timings = timings
        self.reset()

    def reset(self) -> None:
        """Go straight to all-red and forget any request (a new simulation run)."""
        self._phase = Phase.ALL_RED
        self._seconds_in_phase = 0.0
        self._requested = None

    def request_phase(self, direction: str) -> None:
        """Ask for 'NS' or 'EW' to be green. The latest request replaces earlier ones."""
        if direction not in GREEN_FOR:
            raise ValueError(f"unknown direction: {direction!r}")
        self._requested = direction

    def update(self, dt: float) -> None:
        """Advance by dt simulated seconds. A dt of 0 changes nothing, which is how pause works.

        At most one transition happens per update and leftover time is dropped,
        so even a huge dt shows every phase and only ever makes a phase longer.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError(f"invalid dt: {dt}")
        self._seconds_in_phase += dt
        next_phase = self._next_phase()
        if next_phase is not self._phase:
            self._phase = next_phase
            self._seconds_in_phase = 0.0

    def get_states(self) -> LightSnapshot:
        return LightSnapshot(self._phase, dict(LAMPS[self._phase]), self._seconds_in_phase)

    def _next_phase(self) -> Phase:
        phase, elapsed, timings = self._phase, self._seconds_in_phase, self.timings

        if phase is Phase.ALL_RED:
            if self._requested and elapsed >= timings.all_red_s:
                return GREEN_FOR[self._requested]
        elif phase in YELLOW_AFTER:
            if phase is not GREEN_FOR[self._requested] and elapsed >= timings.min_green_s:
                return YELLOW_AFTER[phase]
        elif elapsed >= timings.yellow_s:
            return Phase.ALL_RED
        return phase
