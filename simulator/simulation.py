"""One simulation run: traffic and lights advancing together on a fixed time step."""

from controller.controller import TrafficController
from controller.fixed_timer import FixedTimer
from simulator.traffic import Traffic

TIME_STEP_S = 1 / 60
MAX_CATCH_UP_S = 0.25  # After a stall the simulation falls behind instead of jumping ahead.


class Simulation:
    def __init__(self, seed: int = 42):
        self.traffic = Traffic(seed)
        self.controller = TrafficController(FixedTimer())
        self.lights = self.controller.update(0)

    def step(self) -> None:
        """Vehicles react to the lamps as shown, then the lights advance for the next step."""
        self.traffic.update(TIME_STEP_S, self.lights.lamps)
        self.lights = self.controller.update(TIME_STEP_S)


class Playback:
    """Turns real time into simulation steps.

    Pausing and speed only change when steps run, never what they compute, so a run
    gives the same results however it is watched.
    """

    def __init__(self):
        self.paused = False
        self.speed = 1
        self.unsimulated_s = 0.0

    def steps_due(self, real_dt: float) -> int:
        if self.paused:
            return 0
        self.unsimulated_s = min(self.unsimulated_s + real_dt * self.speed, MAX_CATCH_UP_S * self.speed)
        steps = int(self.unsimulated_s / TIME_STEP_S)
        self.unsimulated_s -= steps * TIME_STEP_S
        return steps
