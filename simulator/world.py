"""The intersection scene.

    python -m simulator.world                      # open a window; drag to orbit, scroll to zoom
    python -m simulator.world --frames DIR         # save one frame per camera, no window
    python -m simulator.world --frames DIR --egl   # same, on a machine with no X server
"""

import argparse
import random
from pathlib import Path

from panda3d.core import loadPrcFileData
from PIL import Image
from ursina import EditorCamera, Entity, Ursina, camera

from controller.traffic_lights import Approach, LightSnapshot, TrafficLights
from simulator.assets import extract_models, model_entity
from simulator.cameras import build_cameras
from simulator.lighting import add_sunlight
from simulator.roads import build_roads
from simulator.signals import SignalHead
from simulator.vehicles import MODELS, VEHICLE_SPACING, place_vehicle

BUILDING_SCALE = 10
# Model, x, z, rotation_y. Set back from the roads so they never block a camera.
BUILDINGS = [
    ("building-a", 32, 32, 180),
    ("building-f", 58, 30, 180),
    ("building-c", -30, 32, 90),
    ("building-k", -32, 60, 90),
    ("building-d", 30, -32, 270),
    ("building-m", 60, -34, 0),
    ("building-h", -34, -30, 0),
    ("building-n", -62, -32, 90),
]
SHADOW_WARMUP_FRAMES = 2  # The shadow map is not ready until the second frame.
SAMPLE_QUEUES = {Approach.NORTH: 6, Approach.SOUTH: 3, Approach.EAST: 5, Approach.WEST: 2}


class World:
    def __init__(self, app):
        Entity.default_shader = None  # Everything inherits the shader from add_sunlight.
        extract_models()
        build_roads()
        for model, x, z, rotation_y in BUILDINGS:
            model_entity("city", model, position=(x, 0, z), rotation_y=rotation_y, scale=BUILDING_SCALE)
        self.signals = {approach: SignalHead(approach) for approach in Approach}
        add_sunlight()
        self.cameras = build_cameras(app)

    def show_lights(self, snapshot: LightSnapshot) -> None:
        for approach, lamp in snapshot.lamps.items():
            self.signals[approach].show(lamp)


def place_sample_traffic(seed: int = 42) -> None:
    rng = random.Random(seed)
    for approach, count in SAMPLE_QUEUES.items():
        for i in range(count):
            place_vehicle(rng.choice(MODELS), approach, 3 + i * VEHICLE_SPACING)


def sample_lights() -> LightSnapshot:
    lights = TrafficLights()
    lights.request_phase("EW")
    lights.update(lights.timings.all_red_s)
    return lights.get_states()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, help="save one PNG per camera here and exit")
    parser.add_argument("--egl", action="store_true", help="render without an X server")
    args = parser.parse_args()

    if args.egl:
        loadPrcFileData("", "load-display p3headlessgl")
    window_type = "offscreen" if args.frames else "onscreen"
    app = Ursina(window_type=window_type, size=(64, 64) if args.frames else (1280, 720), development_mode=False)

    world = World(app)
    place_sample_traffic()
    world.show_lights(sample_lights())

    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
        for _ in range(SHADOW_WARMUP_FRAMES):
            app.graphicsEngine.render_frame()
        for name, frame_camera in world.cameras.items():
            Image.fromarray(frame_camera.frame()[:, :, ::-1]).save(args.frames / f"{name}.png")
        return

    camera.clip_plane_far = 1000
    camera.z = -90
    EditorCamera(rotation=(35, 30, 0))
    app.run()


if __name__ == "__main__":
    main()
