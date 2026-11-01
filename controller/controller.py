"""Connects a strategy to the lights. The strategy only ever requests; the lights decide.

A fixed-time plan takes over in two cases where following the strategy would starve a direction:
when observations stop altogether, until they return, and when a green has hit the lights' limit
while still being requested, until the strategy asks for something else.
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
        self.last_request = None
        self.distrusted_request = None
        self.forced_ends_seen = 0

    def update(self, dt: float, observation: TrafficObservation | None = None) -> LightSnapshot:
        """Advance the lights. The strategy is only consulted when there is a new observation."""
        self.now_s += dt
        if observation:
            self.last_observation_s = self.now_s
            self.lights.request_phase(self._choose(observation))
        elif self.now_s - self.last_observation_s > NO_OBSERVATION_LIMIT_S:
            nothing = TrafficObservation(self.now_s, "none", {})
            self.lights.request_phase(self.fallback.decide(nothing, self.lights.get_states()))
        self.lights.update(dt)
        if self.lights.forced_ends != self.forced_ends_seen:
            self.forced_ends_seen = self.lights.forced_ends
            self.distrusted_request = self.last_request
        return self.lights.get_states()

    def _choose(self, observation: TrafficObservation) -> str:
        lights = self.lights.get_states()
        self.last_request = self.strategy.decide(observation, lights)
        if self.last_request != self.distrusted_request:
            self.distrusted_request = None
            return self.last_request
        return self.fallback.decide(observation, lights)
