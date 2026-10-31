"""Vehicles moving along the four lanes. Plain logic with no rendering, so it is easy to test.

A vehicle's distance is from its front bumper to its stop line, and goes negative past the line.
"""

import math
import random
from dataclasses import dataclass

from controller.traffic_lights import Approach, Lamp
from simulator.roads import ROAD_LENGTH, STOP_LINE

MODELS = ["sedan", "suv", "hatchback-sports", "van", "taxi", "sedan-sports", "suv-luxury", "delivery", "truck"]
VEHICLE_LENGTH = 5  # Metres; the longest model is 4.9.
MIN_GAP = 2  # Bumper to bumper when queued.
STOP_MARGIN = 0.5  # How far before the stop line a vehicle halts.
MAX_SPEED = 13  # Metres per second, about 47 km/h.
ACCELERATION = 2.5
BRAKING = 3
SPAWN_DISTANCE = ROAD_LENGTH / 2 - STOP_LINE
EXIT_DISTANCE = -(ROAD_LENGTH / 2 + STOP_LINE)

ARRIVALS_PER_MINUTE = {Approach.NORTH: 8, Approach.SOUTH: 8, Approach.EAST: 12, Approach.WEST: 12}


@dataclass
class Vehicle:
    id: str
    approach: Approach
    model: str
    distance: float
    speed: float = MAX_SPEED
    committed: bool = False  # Too close to stop when the lamp turned yellow, so it keeps going.

    def stopping_distance(self) -> float:
        return self.speed**2 / (2 * BRAKING) + STOP_MARGIN


class Lane:
    """One approach's vehicles, front first. Arrivals come from the lane's own seeded random
    stream, so the schedule is the same whatever the lights do."""

    def __init__(self, approach: Approach, seed: int, arrivals_per_minute: float):
        self.approach = approach
        self.rng = random.Random(f"{seed}:{approach.value}")
        self.mean_gap_s = 60 / arrivals_per_minute if arrivals_per_minute else math.inf
        self.next_arrival_s = self._time_between_arrivals()
        self.vehicles: list[Vehicle] = []
        self.spawned = 0
        self.lamp = Lamp.RED

    def update(self, dt: float, now_s: float, lamp: Lamp) -> None:
        if lamp is Lamp.YELLOW and self.lamp is not Lamp.YELLOW:
            for vehicle in self.vehicles:
                vehicle.committed = 0 <= vehicle.distance < vehicle.stopping_distance()
        self.lamp = lamp

        leader = None
        for vehicle in self.vehicles:
            self._drive(vehicle, leader, dt)
            leader = vehicle
        self.vehicles = [v for v in self.vehicles if v.distance > EXIT_DISTANCE]
        self._spawn_arrivals(now_s)

    def _drive(self, vehicle: Vehicle, leader: Vehicle | None, dt: float) -> None:
        """Go as fast as possible while still able to stop within the free road ahead."""
        room = math.inf
        if leader:
            room = vehicle.distance - leader.distance - VEHICLE_LENGTH - MIN_GAP
        if self._must_stop_at_line(vehicle):
            room = min(room, vehicle.distance - STOP_MARGIN)
        room = max(room, 0)

        safe_speed = math.sqrt(2 * BRAKING * room)
        vehicle.speed = min(vehicle.speed + ACCELERATION * dt, MAX_SPEED, safe_speed, room / dt)
        vehicle.distance -= vehicle.speed * dt

    def _must_stop_at_line(self, vehicle: Vehicle) -> bool:
        if vehicle.distance < 0:
            return False
        return self.lamp is Lamp.RED or (self.lamp is Lamp.YELLOW and not vehicle.committed)

    def _spawn_arrivals(self, now_s: float) -> None:
        """Add one due arrival. If the lane is backed up to the spawn point, it waits its turn."""
        if self.next_arrival_s > now_s:
            return
        if self.vehicles and SPAWN_DISTANCE - self.vehicles[-1].distance < VEHICLE_LENGTH + MIN_GAP:
            return
        self.spawned += 1
        name = f"{self.approach.value}-{self.spawned}"
        self.vehicles.append(Vehicle(name, self.approach, self.rng.choice(MODELS), SPAWN_DISTANCE))
        self.next_arrival_s += self._time_between_arrivals()

    def _time_between_arrivals(self) -> float:
        return self.rng.expovariate(1 / self.mean_gap_s) if self.mean_gap_s != math.inf else math.inf


class Traffic:
    def __init__(self, seed: int = 42, arrivals_per_minute: dict[Approach, float] = ARRIVALS_PER_MINUTE):
        self.now_s = 0.0
        self.lanes = {
            approach: Lane(approach, seed, arrivals_per_minute[approach]) for approach in Approach
        }

    @property
    def vehicles(self) -> list[Vehicle]:
        return [vehicle for lane in self.lanes.values() for vehicle in lane.vehicles]

    def update(self, dt: float, lamps: dict[Approach, Lamp]) -> None:
        if dt <= 0:
            raise ValueError(f"invalid dt: {dt}")
        self.now_s += dt
        for approach, lane in self.lanes.items():
            lane.update(dt, self.now_s, lamps[approach])
