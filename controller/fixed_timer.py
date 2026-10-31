"""Fixed-time strategy: ask for each direction in turn, whatever the traffic is doing."""

from controller.observation import TrafficObservation
from controller.traffic_lights import LightSnapshot, Timings


class FixedTimer:
    def __init__(self, green_ns_s: float = 20, green_ew_s: float = 20, timings: Timings = Timings()):
        clearance_s = timings.yellow_s + timings.all_red_s
        self.ns_window_s = green_ns_s + clearance_s
        self.cycle_s = self.ns_window_s + green_ew_s + clearance_s

    def decide(self, observation: TrafficObservation, lights: LightSnapshot) -> str:
        return "NS" if observation.timestamp_sim_s % self.cycle_s < self.ns_window_s else "EW"
