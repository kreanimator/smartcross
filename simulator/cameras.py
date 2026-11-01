"""Virtual cameras: each renders off-screen into an array the vision code can read."""

import numpy as np
from panda3d.core import Camera, PerspectiveLens, Texture
from ursina import scene

from simulator.camera_geometry import FOV, HEIGHT, NEAR_M, WIDTH, CameraSpec, camera_specs

SKY = (0.55, 0.72, 0.9, 1)


class FrameCamera:
    def __init__(self, app, spec: CameraSpec):
        self.spec = spec
        self.texture = Texture(spec.name)
        self.buffer = app.win.make_texture_buffer(spec.name, WIDTH, HEIGHT, self.texture, to_ram=True)
        self.buffer.set_clear_color(SKY)

        self.lens = PerspectiveLens()
        self.lens.set_fov(FOV)
        self.lens.set_aspect_ratio(WIDTH / HEIGHT)
        self.lens.set_near(NEAR_M)
        self.node = scene.attach_new_node(Camera(spec.name, self.lens))
        self.node.set_pos(*spec.position)
        self.node.look_at(*spec.target)
        self.buffer.make_display_region().set_camera(self.node)

    def frame(self) -> np.ndarray:
        """The last rendered frame as a top-down BGR array, the layout OpenCV expects."""
        pixels = np.frombuffer(self.texture.get_ram_image_as("BGR"), np.uint8)
        return pixels.reshape(HEIGHT, WIDTH, 3)[::-1].copy()


def build_cameras(app) -> dict[str, FrameCamera]:
    return {name: FrameCamera(app, spec) for name, spec in camera_specs().items()}
