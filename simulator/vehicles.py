"""Draws the vehicles of the traffic simulation. All movement is decided in traffic.py."""

from ursina import destroy

from simulator.assets import model_entity
from simulator.roads import heading_degrees, lane_point
from simulator.traffic import VEHICLE_LENGTH, Vehicle

SCALE = 1.5  # The kit's cars are toy-sized; this makes a sedan about 3.8 m long.
MODEL_FACING = 180  # The kit's models point along -z.


class VehicleViews:
    def __init__(self):
        self.entities = {}

    def clear(self) -> None:
        for entity in self.entities.values():
            destroy(entity)
        self.entities.clear()

    def show(self, vehicles: list[Vehicle]) -> None:
        ids = {vehicle.id for vehicle in vehicles}
        for id in [id for id in self.entities if id not in ids]:
            destroy(self.entities.pop(id))

        for vehicle in vehicles:
            if vehicle.id not in self.entities:
                rotation_y = heading_degrees(vehicle.approach) + MODEL_FACING
                self.entities[vehicle.id] = model_entity("cars", vehicle.model, rotation_y=rotation_y, scale=SCALE)
            # Models are centred, and distance is measured to the front bumper.
            x, z = lane_point(vehicle.approach, vehicle.distance + VEHICLE_LENGTH / 2)
            self.entities[vehicle.id].position = (x, 0, z)
