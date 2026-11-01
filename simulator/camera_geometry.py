"""Where each camera stands and what it sees, as plain maths with no renderer.

Pixels are measured from the top-left of the frame. World points are (x, y, z) in metres.
"""

import math
from dataclasses import dataclass

import numpy as np

from controller.traffic_lights import Approach
from simulator.roads import LANE_WIDTH, OUTWARD, lane_point

WIDTH, HEIGHT = 640, 480
FOV = 50  # Horizontal, in degrees.
NEAR_M = 1.0  # Points closer to the camera than this cannot be projected.
OVERHEAD = "overhead"
MAST_HEIGHT = 12
MAST_DISTANCE = 16  # How far past the centre of the intersection each camera stands.
VIEW_DISTANCE = 22  # The road point it aims at, measured back from the stop line.
OVERHEAD_HEIGHT = 65

# How far back from the stop line each camera watches its lane. At 60 m a car is only about
# 19 pixels wide on a roadside camera. The overhead frame ends just past 15 m.
ROADSIDE_ROI_M = 60
OVERHEAD_ROI_M = 15


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    approaches: tuple[Approach, ...]  # The lanes this camera calibrates.
    roi_distance_m: float

    def project(self, point) -> tuple[float, float] | None:
        """The pixel showing a world point, or None if it is not in front of the camera.

        The result can fall outside the frame. The world is y-up and left-handed.
        """
        forward = _unit(np.subtract(self.target, self.position))
        right = _unit(np.cross((0, 1, 0), forward))
        up = np.cross(forward, right)
        offset = np.subtract(point, self.position)
        depth = offset @ forward
        if depth < NEAR_M:
            return None
        half_width = math.tan(math.radians(FOV) / 2)
        half_height = half_width * HEIGHT / WIDTH
        return (
            float((1 + offset @ right / (depth * half_width)) / 2 * WIDTH),
            float((1 - offset @ up / (depth * half_height)) / 2 * HEIGHT),
        )

    def project_box(self, low, high) -> tuple[float, float, float, float] | None:
        """The (left, top, right, bottom) pixels around a world box, cut to the frame.

        None if the box is out of frame or reaches behind the camera.
        """
        corners = [self.project((x, y, z)) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
        if None in corners:
            return None
        xs, ys = zip(*corners)
        left, top, right, bottom = max(min(xs), 0), max(min(ys), 0), min(max(xs), WIDTH), min(max(ys), HEIGHT)
        if left >= right or top >= bottom:
            return None
        return (left, top, right, bottom)

    def stop_line(self, approach: Approach) -> list[tuple[float, float]]:
        """The two ends of an approach's stop line, in pixels."""
        return [self.project(point) for point in _lane_edges(approach, 0)]

    def roi(self, approach: Approach) -> list[tuple[float, float]]:
        """The watched stretch of an approach's lane on the ground, as a pixel polygon."""
        near_a, near_b = _lane_edges(approach, 0)
        far_a, far_b = _lane_edges(approach, self.roi_distance_m)
        return [self.project(point) for point in (near_a, near_b, far_b, far_a)]

    def calibration(self) -> dict:
        """Everything vision code may know about this camera, in a JSON-ready form."""
        return {
            "name": self.name,
            "position": self.position,
            "target": self.target,
            "width": WIDTH,
            "height": HEIGHT,
            "horizontal_fov_degrees": FOV,
            "roi_distance_m": self.roi_distance_m,
            "approaches": {
                approach.value: {"stop_line": self.stop_line(approach), "roi": self.roi(approach)}
                for approach in self.approaches
            },
        }


def camera_specs() -> dict[str, CameraSpec]:
    """One camera per approach, across the intersection and facing oncoming traffic."""
    specs = {
        # A tiny offset, because looking exactly straight down leaves "up" undefined.
        OVERHEAD: CameraSpec(OVERHEAD, (0, OVERHEAD_HEIGHT, -0.01), (0, 0, 0), tuple(Approach), OVERHEAD_ROI_M)
    }
    for approach, (out_x, out_z) in OUTWARD.items():
        position = (-out_x * MAST_DISTANCE, MAST_HEIGHT, -out_z * MAST_DISTANCE)
        target_x, target_z = lane_point(approach, VIEW_DISTANCE)
        specs[approach.value] = CameraSpec(
            approach.value, position, (target_x, 0, target_z), (approach,), ROADSIDE_ROI_M
        )
    return specs


def in_polygon(point, polygon) -> bool:
    """Whether a pixel is inside a polygon, counting how many edges a ray from it crosses."""
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, [*polygon[1:], polygon[0]]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _unit(vector) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _lane_edges(approach: Approach, distance: float) -> list[tuple[float, float, float]]:
    """The two sides of an approach's lane, distance metres before its stop line, on the ground."""
    x, z = lane_point(approach, distance)
    out_x, out_z = OUTWARD[approach]
    across_x, across_z = -out_z * LANE_WIDTH / 2, out_x * LANE_WIDTH / 2
    return [(x - across_x, 0, z - across_z), (x + across_x, 0, z + across_z)]
