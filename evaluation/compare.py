"""Runs fixed and adaptive control on identical arrivals and compares them.

    python -m evaluation.compare                     # print the comparison
    python -m evaluation.compare --output FILE.json  # also save every run and its settings
    python -m evaluation.compare --first-seed 200    # use other seeds
    python -m evaluation.compare --decisions         # print one adaptive run's logged switches
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from controller.fixed_timer import FixedTimer
from controller.traffic_lights import Approach
from evaluation.metrics import Summary, summarize
from simulator.simulation import TIME_STEP_S, Simulation

DURATION_S = 1800
SEED_COUNT = 10


def rates(north_south: float, east_west: float) -> dict[Approach, float]:
    return {
        Approach.NORTH: north_south,
        Approach.SOUTH: north_south,
        Approach.EAST: east_west,
        Approach.WEST: east_west,
    }


# Arrivals per minute on each lane.
SCENARIOS = {
    "default": rates(8, 12),
    "balanced": rates(8, 8),
    "directional rush": rates(4, 16),
    "low traffic": rates(2, 2),
}

# The best (north-south, east-west) green times found by trying plans on each scenario.
# A real engineer would set these from a traffic count, so adaptive must beat these too.
MATCHED_GREENS = {
    "default": (12, 18),
    "balanced": (10, 10),
    "directional rush": (10, 35),
    "low traffic": (10, 10),
}
STRATEGY_NAMES = ("fixed 20/20", "fixed matched", "adaptive")


def make_strategy(name: str, scenario: str):
    if name == "fixed matched":
        return FixedTimer(*MATCHED_GREENS[scenario])
    return "fixed" if name == "fixed 20/20" else "adaptive"


def run(job: tuple[str, str, int]) -> Summary:
    scenario, strategy, seed = job
    simulation = Simulation(seed, make_strategy(strategy, scenario), SCENARIOS[scenario])
    for _ in range(int(DURATION_S / TIME_STEP_S)):
        simulation.step()
    return summarize(simulation.traffic)


def print_comparison(results: dict[tuple[str, str, int], Summary], seeds: range) -> None:
    columns = {
        "time lost": "mean_time_lost_s",
        "stopped": "mean_delay_s",
        "stopped NS": "mean_delay_ns_s",
        "stopped EW": "mean_delay_ew_s",
        "p95 stopped": "p95_delay_s",
        "worst wait": "worst_wait_s",
        "per hour": "throughput_per_hour",
        "unfinished": "unfinished",
    }
    print(f"Mean over seeds {seeds.start}-{seeds.stop - 1}, {DURATION_S} simulated seconds each. Times in seconds per vehicle.\n")
    print(f"{'scenario':18s}{'strategy':15s}" + "".join(f"{name:>12s}" for name in columns))
    for scenario in SCENARIOS:
        for strategy in STRATEGY_NAMES:
            runs = [results[scenario, strategy, seed] for seed in seeds]
            values = [np.mean([getattr(r, field) for r in runs]) for field in columns.values()]
            print(f"{scenario:18s}{strategy:15s}" + "".join(f"{v:12.1f}" for v in values))

        for baseline in STRATEGY_NAMES[:2]:
            wins = sum(
                results[scenario, "adaptive", seed].mean_time_lost_s < results[scenario, baseline, seed].mean_time_lost_s
                for seed in seeds
            )
            print(f"{'':33s}adaptive lost less time than {baseline} on {wins} of {len(seeds)} seeds")
        print()


def print_decisions(seed: int) -> None:
    simulation = Simulation(seed, "adaptive", SCENARIOS["default"])
    for _ in range(int(300 / TIME_STEP_S)):
        simulation.step()
    for d in simulation.controller.strategy.decisions:
        print(f"{d.time_s:6.0f} s  to {d.direction}  {d.reason:12s} pressure NS {d.pressure_ns:5.1f}  EW {d.pressure_ew:5.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="save every run and its settings to this JSON file")
    parser.add_argument("--first-seed", type=int, default=0, help="first of the seeds to run")
    parser.add_argument("--decisions", action="store_true", help="print one adaptive run's switches and exit")
    args = parser.parse_args()

    if args.decisions:
        print_decisions(args.first_seed)
        return

    seeds = range(args.first_seed, args.first_seed + SEED_COUNT)
    jobs = [(scenario, strategy, seed) for scenario in SCENARIOS for strategy in STRATEGY_NAMES for seed in seeds]
    with ProcessPoolExecutor() as pool:
        results = dict(zip(jobs, pool.map(run, jobs)))
    print_comparison(results, seeds)

    if args.output:
        with open(args.output, "w") as file:
            json.dump(
                {
                    "duration_s": DURATION_S,
                    "time_step_s": TIME_STEP_S,
                    "scenarios": SCENARIOS,
                    "matched_greens": MATCHED_GREENS,
                    "runs": [{"job": job, **asdict(summary)} for job, summary in results.items()],
                },
                file,
                indent=2,
            )


if __name__ == "__main__":
    main()
