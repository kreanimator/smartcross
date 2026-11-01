"""One simulation run: traffic and lights advancing together on a fixed time step."""

from controller.adaptive import Adaptive
from controller.controller import TrafficController
from controller.fixed_timer import FixedTimer
from simulator.traffic import ARRIVALS_PER_MINUTE, Traffic

TIME_STEP_S = 1 / 60
STEPS_PER_OBSERVATION = 60  # The controller gets a new observation once per simulated second.
MAX_CATCH_UP_S = 0.25  # After a stall the simulation falls behind instead of jumping ahead.
STRATEGIES = {"fixed": FixedTimer, "adaptive": Adaptive}


class Simulation:
    def __init__(self, seed: int = 42, strategy="fixed", arrivals_per_minute=ARRIVALS_PER_MINUTE, *, sensor):
        """strategy is a name from STRATEGIES, or a strategy object.

        sensor.read(simulation) runs after every step and returns an observation when one is due.
        It has no default, so traffic can never reach the controller without someone choosing how.
        """
        self.traffic = Traffic(seed, arrivals_per_minute)
        self.controller = TrafficController(STRATEGIES[strategy]() if isinstance(strategy, str) else strategy)
        self.sensor = sensor
        self.steps = 0
        self.lights = self.controller.lights.get_states()
        self.lights = self.controller.update(0, self.sensor.read(self))

    def step(self) -> None:
        """Vehicles react to the lamps as shown, then the lights advance for the next step."""
        self.traffic.update(TIME_STEP_S, self.lights.lamps)
        self.steps += 1
        self.lights = self.controller.update(TIME_STEP_S, self.sensor.read(self))


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
