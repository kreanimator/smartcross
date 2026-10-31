"""Virtual cameras: each renders off-screen into an array the vision code can read."""

import numpy as np
from panda3d.core import Camera, PerspectiveLens, Texture
from ursina import scene

from controller.traffic_lights import Approach
from simulator.roads import OUTWARD, lane_point

WIDTH, HEIGHT = 640, 480
FOV = 50
SKY = (0.55, 0.72, 0.9, 1)
OVERHEAD = "overhead"
MAST_HEIGHT = 12
MAST_DISTANCE = 16  # How far past the centre of the intersection each camera stands.
VIEW_DISTANCE = 22  # The road point it aims at, measured back from the stop line.


class FrameCamera:
    def __init__(self, app, name: str, position, target):
        self.texture = Texture(name)
        self.buffer = app.win.make_texture_buffer(name, WIDTH, HEIGHT, self.texture, to_ram=True)
        self.buffer.set_clear_color(SKY)

        lens = PerspectiveLens()
        lens.set_fov(FOV)
        lens.set_aspect_ratio(WIDTH / HEIGHT)
        self.node = scene.attach_new_node(Camera(name, lens))
        self.node.set_pos(*position)
        self.node.look_at(*target)
        self.buffer.make_display_region().set_camera(self.node)

    def frame(self) -> np.ndarray:
        """The last rendered frame as a top-down BGR array, the layout OpenCV expects."""
        pixels = np.frombuffer(self.texture.get_ram_image_as("BGR"), np.uint8)
        return pixels.reshape(HEIGHT, WIDTH, 3)[::-1].copy()


def build_cameras(app) -> dict[str, FrameCamera]:
    """One camera per approach, across the intersection and facing oncoming traffic."""
    cameras = {OVERHEAD: FrameCamera(app, OVERHEAD, (0, 65, -0.01), (0, 0, 0))}
    for approach, (out_x, out_z) in OUTWARD.items():
        position = (-out_x * MAST_DISTANCE, MAST_HEIGHT, -out_z * MAST_DISTANCE)
        target_x, target_z = lane_point(approach, VIEW_DISTANCE)
        cameras[approach.value] = FrameCamera(app, approach.value, position, (target_x, 0, target_z))
    return cameras
