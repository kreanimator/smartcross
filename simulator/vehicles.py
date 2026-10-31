"""Vehicle models placed in the scene. Movement comes with the traffic simulation."""

from ursina import Entity

from controller.traffic_lights import Approach
from simulator.assets import model_entity
from simulator.roads import heading_degrees, lane_point

MODELS = ["sedan", "suv", "hatchback-sports", "van", "taxi", "sedan-sports", "suv-luxury", "delivery", "truck"]
SCALE = 1.5  # The kit's cars are toy-sized; this makes a sedan about 3.8 m long.
VEHICLE_SPACING = 7
MODEL_FACING = 180  # The kit's models point along -z.


def place_vehicle(model: str, approach: Approach, distance: float) -> Entity:
    x, z = lane_point(approach, distance)
    return model_entity("cars", model, position=(x, 0, z), rotation_y=heading_degrees(approach) + MODEL_FACING, scale=SCALE)
