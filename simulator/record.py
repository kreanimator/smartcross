"""Runs the simulation and saves timestamped frames from all five cameras.

    python -m simulator.record --out DIR                      # 60 s at 5 frames per second
    python -m simulator.record --out DIR --start 60 --seconds 0  # a single frame at 60 s
    python -m simulator.record --out DIR --overlay            # also save marked-up debug sheets
    python -m simulator.record --out DIR --arrivals 14 14     # north-south, east-west cars per minute
    python -m simulator.record --out DIR --egl                # on a machine with no X server

The folder holds:
    run.json             settings and each camera's calibration
    frames.jsonl         one line per sampled moment: time, lights and ground truth
    <camera>/NNNNNN.png  the clean frames, numbered by their line in frames.jsonl
    overlay/NNNNNN.png   with --overlay, all cameras marked up on one sheet

Ground truth in frames.jsonl is for evaluation and debugging only, never for the controller.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from panda3d.core import loadPrcFileData
from PIL import Image
from ursina import Ursina

from controller.traffic_lights import Approach
from simulator.overlay import annotate, mosaic
from simulator.perfect_sensor import observe_lane
from simulator.simulation import STRATEGIES, TIME_STEP_S, Simulation
from simulator.traffic import ARRIVALS_PER_MINUTE
from simulator.world import World

SHADOW_WARMUP_FRAMES = 2  # The shadow map is not ready until the second frame.


def steps_per_sample(fps: float) -> int:
    """Samples must land exactly on simulation steps, or their timestamps would be wrong."""
    steps = 1 / (fps * TIME_STEP_S) if fps > 0 else 0
    if steps < 1 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"fps must divide {1 / TIME_STEP_S:.0f} exactly, got {fps}")
    return round(steps)


def manifest_entry(index: int, simulation: Simulation, world: World) -> dict:
    lights = simulation.lights
    return {
        "frame": index,
        "timestamp_sim_s": round(simulation.traffic.now_s, 6),
        "phase": lights.phase.value,
        "lamps": {approach.value: lamp.value for approach, lamp in lights.lamps.items()},
        "vehicles": [_vehicle_truth(vehicle, world) for vehicle in simulation.traffic.vehicles],
        "truth": {
            approach.value: asdict(observe_lane(lane)) for approach, lane in simulation.traffic.lanes.items()
        },
    }


def _vehicle_truth(vehicle, world: World) -> dict:
    low, high = world.vehicles.bounds(vehicle.id)
    boxes = {name: camera.spec.project_box(low, high) for name, camera in world.cameras.items()}
    return {
        "id": vehicle.id,
        "approach": vehicle.approach.value,
        "model": vehicle.model,
        "distance_m": round(vehicle.distance, 3),
        "speed_mps": round(vehicle.speed, 3),
        "boxes": {name: [round(value, 1) for value in box] for name, box in boxes.items() if box},
    }


def record(
    app, world: World, out: Path, *, seed=42, strategy="fixed", start=0.0, seconds=60.0, fps=5, overlay=False,
    arrivals=ARRIVALS_PER_MINUTE,
):
    """Saves the run to out and returns its manifest entries."""
    every = steps_per_sample(fps)
    calibrations = {name: camera.spec.calibration() for name, camera in world.cameras.items()}
    out.mkdir(parents=True, exist_ok=True)
    for name in world.cameras:
        (out / name).mkdir(exist_ok=True)
    if overlay:
        (out / "overlay").mkdir(exist_ok=True)
    (out / "run.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "strategy": strategy,
                "start_s": start,
                "seconds": seconds,
                "fps": fps,
                "arrivals_per_minute": {approach.value: rate for approach, rate in arrivals.items()},
                "cameras": calibrations,
            },
            indent=2,
        )
    )

    simulation = Simulation(seed, strategy, arrivals)
    for _ in range(round(start / TIME_STEP_S)):
        simulation.step()
    for _ in range(SHADOW_WARMUP_FRAMES):
        app.graphicsEngine.render_frame()

    samples = []
    with open(out / "frames.jsonl", "w") as manifest:
        for index in range(round(seconds * fps) + 1):
            if index:
                for _ in range(every):
                    simulation.step()
            world.show(simulation)
            app.graphicsEngine.render_frame()

            sample = manifest_entry(index, simulation, world)
            samples.append(sample)
            images = {
                name: Image.fromarray(camera.frame()[:, :, ::-1]) for name, camera in world.cameras.items()
            }
            for name, image in images.items():
                image.save(out / name / f"{index:06d}.png")
            if overlay:
                marked = [annotate(image, calibrations[name], sample) for name, image in images.items()]
                mosaic(marked).save(out / "overlay" / f"{index:06d}.png")
            # Last, and flushed, so a run that is cut short never lists a frame it did not save.
            manifest.write(json.dumps(sample) + "\n")
            manifest.flush()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder to save the run in")
    parser.add_argument("--seconds", type=float, default=60, help="simulated time to record")
    parser.add_argument("--start", type=float, default=0, help="simulated time before the first frame")
    parser.add_argument("--fps", type=float, default=5, help="frames per simulated second; must divide 60")
    parser.add_argument("--seed", type=int, default=42, help="seed for the arrival schedule")
    parser.add_argument("--strategy", choices=STRATEGIES, default="fixed", help="how the lights decide")
    parser.add_argument("--arrivals", type=float, nargs=2, metavar=("NS", "EW"), help="cars per minute on each lane")
    parser.add_argument("--overlay", action="store_true", help="also save marked-up debug sheets")
    parser.add_argument("--egl", action="store_true", help="render without an X server")
    args = parser.parse_args()
    try:
        steps_per_sample(args.fps)
    except ValueError as error:
        parser.error(str(error))

    if args.egl:
        loadPrcFileData("", "load-display p3headlessgl")
    app = Ursina(window_type="offscreen", size=(64, 64), development_mode=False)
    arrivals = ARRIVALS_PER_MINUTE
    if args.arrivals:
        north_south, east_west = args.arrivals
        arrivals = {
            Approach.NORTH: north_south, Approach.SOUTH: north_south,
            Approach.EAST: east_west, Approach.WEST: east_west,
        }
    samples = record(
        app, World(app), args.out, seed=args.seed, strategy=args.strategy,
        start=args.start, seconds=args.seconds, fps=args.fps, overlay=args.overlay, arrivals=arrivals,
    )
    print(f"saved {len(samples)} frames per camera to {args.out}")


if __name__ == "__main__":
    main()
