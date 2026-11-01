"""Shows what dead cameras do to control, with and without noticing they are dead.

    python -m evaluation.camera_faults --weights FILE.pt --dead north
    python -m evaluation.camera_faults --weights FILE.pt --dead east west --seeds 3

The named cameras send black frames from --start until --stop. "Ignored" treats those frames as
real pictures, which is how an unnoticed failure behaves. Delay figures come from ground truth.
"""

import argparse
from pathlib import Path

import numpy as np

from controller.adaptive import Adaptive
from controller.traffic_lights import Approach
from evaluation.compare import SCENARIOS
from evaluation.metrics import summarize
from simulator.simulation import TIME_STEP_S, Simulation

COLUMNS = {
    "time lost": "mean_time_lost_s",
    "stopped": "mean_delay_s",
    "p95 stopped": "p95_delay_s",
    "worst wait": "worst_wait_s",
    "per hour": "throughput_per_hour",
    "unfinished": "unfinished",
}


def run(app, world, detector, mode: str, args, seed: int):
    from vision.camera_sensor import CameraSensor

    class IgnoresOutages(CameraSensor):
        is_usable = staticmethod(lambda frame: True)

    def fault(approach, frame, now_s):
        dead = mode != "healthy" and approach.value in args.dead and args.start <= now_s < args.stop
        return np.zeros_like(frame) if dead else frame

    world.vehicles.clear()
    sensor_type = IgnoresOutages if mode == "ignored" else CameraSensor
    strategy = Adaptive()
    simulation = Simulation(seed, strategy, SCENARIOS[args.scenario], sensor=sensor_type(app, world, detector, fault))
    longest_green_s = 0.0
    for _ in range(int(args.seconds / TIME_STEP_S)):
        simulation.step()
        if args.start <= simulation.traffic.now_s < args.stop:
            longest_green_s = max(longest_green_s, simulation.lights.seconds_in_phase)
    return summarize(simulation.traffic), longest_green_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dead", nargs="+", choices=[a.value for a in Approach], required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="default")
    parser.add_argument("--start", type=float, default=100)
    parser.add_argument("--stop", type=float, default=500)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=400)
    args = parser.parse_args()

    from ursina import Ursina

    from simulator.world import World
    from vision.detector import VehicleDetector
    from vision.tracker import MIN_DETECTION_CONFIDENCE

    app = Ursina(window_type="offscreen", size=(64, 64), development_mode=False)
    world = World(app)
    detector = VehicleDetector(args.weights, MIN_DETECTION_CONFIDENCE)

    seeds = range(args.first_seed, args.first_seed + args.seeds)
    print(f"{' + '.join(args.dead)} dead from {args.start:g} to {args.stop:g} s of {args.seconds:g}, {args.scenario}")
    print(f"{'':12s}" + "".join(f"{name:>13s}" for name in (*COLUMNS, "longest phase")), flush=True)
    for mode in ("healthy", "ignored", "handled"):
        runs = [run(app, world, detector, mode, args, seed) for seed in seeds]
        values = [np.mean([getattr(r[0], field) for r in runs]) for field in COLUMNS.values()]
        values.append(max(r[1] for r in runs))
        print(f"{mode:12s}" + "".join(f"{value:13.1f}" for value in values), flush=True)


if __name__ == "__main__":
    main()
