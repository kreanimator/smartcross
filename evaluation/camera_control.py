"""Runs adaptive control on camera input and compares it with the perfect sensor.

    python -m evaluation.camera_control --weights FILE.pt
    python -m evaluation.camera_control --weights FILE.pt --scenarios overloaded --seeds 3
    python -m evaluation.camera_control --weights FILE.pt --output FILE.json

Every strategy on a seed gets identical arrivals. All delay figures come from ground truth,
and truth is compared with what the cameras saw only for this report, never for control.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from controller.adaptive import Adaptive
from evaluation.compare import DURATION_S, SCENARIOS, STRATEGY_NAMES, run
from evaluation.metrics import Summary, summarize
from evaluation.perception import Errors
from simulator.perfect_sensor import observe
from simulator.simulation import STEPS_PER_OBSERVATION, TIME_STEP_S, Simulation

CAMERAS = "adaptive cameras"
BASELINE_WORKERS = 4
COLUMNS = {
    "time lost": "mean_time_lost_s",
    "stopped": "mean_delay_s",
    "p95 stopped": "p95_delay_s",
    "worst wait": "worst_wait_s",
    "per hour": "throughput_per_hour",
    "unfinished": "unfinished",
}


def run_on_cameras(app, world, detector, scenario: str, seed: int) -> tuple[Summary, Errors, Counter]:
    from vision.camera_sensor import CameraSensor

    world.vehicles.clear()
    strategy = Adaptive()
    simulation = Simulation(seed, strategy, SCENARIOS[scenario], sensor=CameraSensor(app, world, detector))
    errors = Errors()
    for _ in range(int(DURATION_S / TIME_STEP_S)):
        simulation.step()
        if simulation.steps % STEPS_PER_OBSERVATION == 0:
            truth = observe(simulation.traffic).approaches
            for approach, seen in simulation.sensor.seen.items():
                if not seen.available:
                    continue
                errors.add("queued", seen.queued, truth[approach].queued)
                errors.add("oldest_wait_s", seen.oldest_wait_s, truth[approach].oldest_wait_s)
    return summarize(simulation.traffic), errors, Counter(d.reason for d in strategy.decisions)


def print_scenario(scenario: str, results: dict, seeds: range) -> None:
    print(f"{'':18s}" + "".join(f"{name:>12s}" for name in COLUMNS))
    for strategy in (*STRATEGY_NAMES, CAMERAS):
        runs = [results[strategy, seed] for seed in seeds]
        values = [np.mean([getattr(r, field) for r in runs]) for field in COLUMNS.values()]
        print(f"{strategy:18s}" + "".join(f"{value:12.1f}" for value in values))
    for baseline in STRATEGY_NAMES:
        wins = sum(results[CAMERAS, s].mean_time_lost_s < results[baseline, s].mean_time_lost_s for s in seeds)
        print(f"cameras lost less time than {baseline} on {wins} of {len(seeds)} seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=300, help="the default seeds played no part in tuning")
    parser.add_argument("--output", type=Path, help="save every run to this JSON file")
    args = parser.parse_args()

    seeds = range(args.first_seed, args.first_seed + args.seeds)
    # The baselines go first: forking worker processes is unsafe once graphics and CUDA are running.
    jobs = [(scenario, strategy, seed) for scenario in args.scenarios for strategy in STRATEGY_NAMES for seed in seeds]
    with ProcessPoolExecutor(max_workers=BASELINE_WORKERS) as pool:
        baselines = dict(zip(jobs, pool.map(run, jobs)))

    from ursina import Ursina

    from simulator.world import World
    from vision.detector import VehicleDetector
    from vision.tracker import MIN_DETECTION_CONFIDENCE

    app = Ursina(window_type="offscreen", size=(64, 64), development_mode=False)
    world = World(app)
    detector = VehicleDetector(args.weights, MIN_DETECTION_CONFIDENCE)

    saved = []
    for scenario in args.scenarios:
        results = {(strategy, seed): baselines[scenario, strategy, seed] for strategy in STRATEGY_NAMES for seed in seeds}
        print(f"\n{scenario}, {DURATION_S} simulated seconds, mean over seeds {seeds.start}-{seeds.stop - 1}", flush=True)
        errors, reasons = Errors(), Counter()
        for seed in seeds:
            summary, run_errors, run_reasons = run_on_cameras(app, world, detector, scenario, seed)
            results[CAMERAS, seed] = summary
            reasons += run_reasons
            for key, values in run_errors.values.items():
                errors.values[key] += values
        print_scenario(scenario, results, seeds)
        print(f"queued: error, bias, p95, worst {errors.row('queued')}")
        print(f"wait:   error, bias, p95, worst {errors.row('oldest_wait_s')}")
        print(f"camera switches by reason: {dict(reasons)}", flush=True)
        saved += [
            {"scenario": scenario, "strategy": strategy, "seed": seed, **asdict(summary)}
            for (strategy, seed), summary in results.items()
        ]
    if args.output:
        args.output.write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
