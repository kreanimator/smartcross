"""The intersection scene.

    python -m simulator.world                      # open the interactive window
    python -m simulator.world --frames DIR         # simulate, save one frame per camera, no window
    python -m simulator.world --frames DIR --egl   # same, on a machine with no X server
"""

import argparse
from pathlib import Path

from panda3d.core import loadPrcFileData
from PIL import Image
from ursina import Entity, Ursina

from controller.traffic_lights import Approach
from simulator.assets import extract_models, model_entity
from simulator.cameras import build_cameras
from simulator.lighting import add_sunlight
from simulator.road_view import build_roads
from simulator.signals import SignalHead
from simulator.simulation import STRATEGIES, TIME_STEP_S, Simulation
from simulator.vehicles import VehicleViews
from simulator.viewer import Viewer

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
        self.vehicles = VehicleViews()

    def show(self, simulation: Simulation) -> None:
        for approach, lamp in simulation.lights.lamps.items():
            self.signals[approach].show(lamp)
        self.vehicles.show(simulation.traffic.vehicles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, help="save one PNG per camera here and exit")
    parser.add_argument("--seconds", type=float, default=60, help="simulated time before saving frames")
    parser.add_argument("--seed", type=int, default=42, help="seed for the arrival schedule")
    parser.add_argument("--strategy", choices=STRATEGIES, default="fixed", help="how the lights decide")
    parser.add_argument("--egl", action="store_true", help="render without an X server")
    args = parser.parse_args()

    if args.egl:
        loadPrcFileData("", "load-display p3headlessgl")
    window_type = "offscreen" if args.frames else "onscreen"
    app = Ursina(window_type=window_type, size=(64, 64) if args.frames else (1280, 720), development_mode=False)

    world = World(app)

    if args.frames:
        simulation = Simulation(args.seed, args.strategy)
        for _ in range(int(args.seconds / TIME_STEP_S)):
            simulation.step()
        world.show(simulation)
        args.frames.mkdir(parents=True, exist_ok=True)
        for _ in range(SHADOW_WARMUP_FRAMES):
            app.graphicsEngine.render_frame()
        for name, frame_camera in world.cameras.items():
            Image.fromarray(frame_camera.frame()[:, :, ::-1]).save(args.frames / f"{name}.png")
        return

    Viewer(world, args.seed, args.strategy)
    app.run()


if __name__ == "__main__":
    main()
