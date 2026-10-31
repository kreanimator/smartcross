from controller.traffic_lights import Approach
from simulator.roads import LANE_WIDTH, STOP_LINE, heading_degrees, lane_point


def test_each_approach_drives_on_the_right():
    # A driver heading south has west on their right, and so on around the compass.
    assert lane_point(Approach.NORTH, 0) == (-LANE_WIDTH / 2, STOP_LINE)
    assert lane_point(Approach.SOUTH, 0) == (LANE_WIDTH / 2, -STOP_LINE)
    assert lane_point(Approach.EAST, 0) == (STOP_LINE, LANE_WIDTH / 2)
    assert lane_point(Approach.WEST, 0) == (-STOP_LINE, -LANE_WIDTH / 2)


def test_distance_moves_away_from_the_intersection():
    assert lane_point(Approach.NORTH, 10) == (-LANE_WIDTH / 2, STOP_LINE + 10)
    assert lane_point(Approach.WEST, 10) == (-STOP_LINE - 10, -LANE_WIDTH / 2)


def test_headings_point_at_the_intersection():
    # In Ursina, rotation_y 0 faces +z (north) and 90 faces +x (east).
    assert heading_degrees(Approach.SOUTH) == 0
    assert heading_degrees(Approach.WEST) == 90
    assert heading_degrees(Approach.NORTH) == 180
    assert heading_degrees(Approach.EAST) == 270
