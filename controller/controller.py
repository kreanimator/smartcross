"""Connects a strategy to the lights. The strategy only ever requests; the lights decide.

The state machine holds a green for as long as nobody asks otherwise, so if observations stop
altogether a fixed-time plan takes over until they return.
"""

from controller.fixed_timer import FixedTimer
from controller.observation import TrafficObservation
from controller.traffic_lights import LightSnapshot, TrafficLights

NO_OBSERVATION_LIMIT_S = 5.0


class TrafficController:
    def __init__(self, strategy, lights: TrafficLights | None = None):
        self.strategy = strategy
        self.lights = lights or TrafficLights()
        self.fallback = FixedTimer()
        self.now_s = 0.0
        self.last_observation_s = 0.0

    def update(self, dt: float, observation: TrafficObservation | None = None) -> LightSnapshot:
        """Advance the lights. The strategy is only consulted when there is a new observation."""
        self.now_s += dt
        if observation:
            self.last_observation_s = self.now_s
            self.lights.request_phase(self.strategy.decide(observation, self.lights.get_states()))
        elif self.now_s - self.last_observation_s > NO_OBSERVATION_LIMIT_S:
            nothing = TrafficObservation(self.now_s, "none", {})
            self.lights.request_phase(self.fallback.decide(nothing, self.lights.get_states()))
        self.lights.update(dt)
        return self.lights.get_states()
