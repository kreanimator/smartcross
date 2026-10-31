"""Connects a strategy to the lights. The strategy only ever requests; the lights decide."""

from controller.traffic_lights import LightSnapshot, TrafficLights


class TrafficController:
    def __init__(self, strategy, lights: TrafficLights | None = None):
        self.strategy = strategy
        self.lights = lights or TrafficLights()
        self.now_s = 0.0

    def update(self, dt: float) -> LightSnapshot:
        self.now_s += dt
        self.lights.request_phase(self.strategy.decide(self.now_s))
        self.lights.update(dt)
        return self.lights.get_states()
