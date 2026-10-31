"""Draws the roads described in roads.py."""

from ursina import Entity, color

from controller.traffic_lights import Approach
from simulator.roads import LANE_WIDTH, OUTWARD, ROAD_LENGTH, ROAD_WIDTH, STOP_LINE, lane_point

ASPHALT = color.rgb32(62, 62, 66)
GRASS = color.rgb32(92, 128, 72)
PAVEMENT = color.rgb32(150, 150, 145)


def build_roads() -> None:
    Entity(model="plane", scale=ROAD_LENGTH * 2, color=GRASS)
    _strip((0, 0), ROAD_LENGTH, ROAD_WIDTH, ASPHALT)
    _strip((0, 0), ROAD_WIDTH, ROAD_LENGTH, ASPHALT)
    for side in (-1, 1):
        offset = side * (ROAD_WIDTH / 2 + 1)
        _strip((0, offset), ROAD_LENGTH, 2, PAVEMENT, height=0.02)
        _strip((offset, 0), 2, ROAD_LENGTH, PAVEMENT, height=0.02)
    _strip((0, 0), STOP_LINE * 2, STOP_LINE * 2, ASPHALT, height=0.04)

    for approach, (out_x, out_z) in OUTWARD.items():
        _stop_line(approach)
        for along in range(int(STOP_LINE) + 4, ROAD_LENGTH // 2, 8):
            x, z = out_x * along, out_z * along
            size_x, size_z = (0.2, 3) if out_x == 0 else (3, 0.2)
            _strip((x, z), size_x, size_z, color.white, height=0.06)


def _stop_line(approach: Approach) -> None:
    x, z = lane_point(approach, 0)
    size_x, size_z = (LANE_WIDTH, 0.5) if OUTWARD[approach][0] == 0 else (0.5, LANE_WIDTH)
    _strip((x, z), size_x, size_z, color.white, height=0.06)


def _strip(center, size_x, size_z, strip_color, height=0.01) -> Entity:
    """A flat rectangle on the ground; height keeps overlapping layers from flickering."""
    return Entity(
        model="cube",
        position=(center[0], height / 2, center[1]),
        scale=(size_x, height, size_z),
        color=strip_color,
    )
