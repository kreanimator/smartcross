"""Connects a strategy to the lights. The strategy only ever requests; the lights decide."""

from controller.observation import TrafficObservation
from controller.traffic_lights import LightSnapshot, TrafficLights


class TrafficController:
    def __init__(self, strategy, lights: TrafficLights | None = None):
        self.strategy = strategy
        self.lights = lights or TrafficLights()

    def update(self, dt: float, observation: TrafficObservation | None = None) -> LightSnapshot:
        """Advance the lights. The strategy is only consulted when there is a new observation."""
        if observation:
            self.lights.request_phase(self.strategy.decide(observation, self.lights.get_states()))
        self.lights.update(dt)
        return self.lights.get_states()
