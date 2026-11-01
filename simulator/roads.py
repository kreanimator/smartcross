"""Road geometry. World units are metres: x is east, z is north, y is up."""

from controller.traffic_lights import Approach

LANE_WIDTH = 3.5
ROAD_WIDTH = 2 * LANE_WIDTH
ROAD_LENGTH = 200
STOP_LINE = ROAD_WIDTH / 2 + 2  # Distance of each stop line from the centre.

# Unit vector (x, z) from the centre toward where each approach's traffic comes from.
OUTWARD = {
    Approach.NORTH: (0, 1),
    Approach.SOUTH: (0, -1),
    Approach.EAST: (1, 0),
    Approach.WEST: (-1, 0),
}


def lane_point(approach: Approach, distance: float) -> tuple[float, float]:
    """(x, z) in the approach's incoming lane, distance metres before its stop line.

    Traffic drives on the right, and a driver heading (hx, hz) has their right at (hz, -hx).
    """
    out_x, out_z = OUTWARD[approach]
    right_x, right_z = -out_z, out_x
    along = STOP_LINE + distance
    return (out_x * along + right_x * LANE_WIDTH / 2, out_z * along + right_z * LANE_WIDTH / 2)


def distance_to_line(approach: Approach, x: float, z: float) -> float:
    """How far before its stop line a ground point is on this approach; negative past the line."""
    out_x, out_z = OUTWARD[approach]
    return out_x * x + out_z * z - STOP_LINE


def heading_degrees(approach: Approach) -> float:
    """Entity rotation_y of a vehicle driving toward the intersection."""
    return {
        Approach.NORTH: 180,
        Approach.SOUTH: 0,
        Approach.EAST: 270,
        Approach.WEST: 90,
    }[approach]
