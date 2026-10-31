"""One simulation run: traffic and lights advancing together on a fixed time step."""

from controller.controller import TrafficController
from controller.fixed_timer import FixedTimer
from simulator.traffic import Traffic

TIME_STEP_S = 1 / 60


class Simulation:
    def __init__(self, seed: int = 42):
        self.traffic = Traffic(seed)
        self.controller = TrafficController(FixedTimer())
        self.lights = self.controller.update(0)

    def step(self) -> None:
        """Vehicles react to the lamps as shown, then the lights advance for the next step."""
        self.traffic.update(TIME_STEP_S, self.lights.lamps)
        self.lights = self.controller.update(TIME_STEP_S)
