"""Compares camera-derived traffic observations with the simulator's exact ones.

    python -m evaluation.perception RUN_DIR --weights FILE.pt

Errors are the estimate minus the truth, so a negative bias means the cameras see too little.
The truth covers the whole lane, while the cameras only watch their ROI.
"""

import argparse
import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from controller.traffic_lights import Approach, Lamp
from evaluation.detection import BATCH_FRAMES
from simulator.camera_geometry import camera_specs

WARM_UP_S = 30  # Cars already waiting at the first frame make early waits unknowable.
FIELDS = ("queued", "approaching", "near_line", "oldest_wait_s", "average_speed_mps")
QUEUE_BANDS = [(0, 0), (1, 3), (4, 6), (7, 9), (10, 1000)]
CROSSING_TOLERANCE_S = 0.5  # Under the gap between cars leaving a queue, so matches stay unambiguous.


class Errors:
    def __init__(self):
        self.values = defaultdict(list)

    def add(self, key, estimate, truth) -> None:
        if estimate is not None and truth is not None:
            self.values[key].append(estimate - truth)

    def row(self, key) -> str:
        values = self.values[key]
        if not values:
            return f"{'-':>7}{'-':>7}{'-':>7}{'-':>6}"
        sizes = sorted(map(abs, values))
        mean_abs = sum(sizes) / len(sizes)
        p95 = sizes[min(int(len(sizes) * 0.95), len(sizes) - 1)]
        return f"{mean_abs:7.2f}{sum(values) / len(values):+7.2f}{p95:7.1f}{max(values, key=abs):+6.0f}"


def queue_band(queued: int) -> tuple[int, int]:
    return next(band for band in QUEUE_BANDS if band[0] <= queued <= band[1])


def compare(errors: Errors, by_queue: Errors, estimate: dict, truth: dict) -> None:
    for name in FIELDS:
        errors.add(name, estimate[name], truth[name])
    by_queue.add(queue_band(truth["queued"]), estimate["queued"], truth["queued"])


def true_crossings(samples: list[dict], approach: str) -> list[float]:
    """When each vehicle's front passed the stop line, estimated between the recorded frames."""
    times = []
    last = {}
    for sample in samples:
        now_s = sample["timestamp_sim_s"]
        for vehicle in sample["vehicles"]:
            if vehicle["approach"] != approach:
                continue
            distance = vehicle["distance_m"]
            if vehicle["id"] in last:
                before_s, before = last[vehicle["id"]]
                if before >= 0 > distance:
                    times.append(before_s + (now_s - before_s) * before / (before - distance))
            last[vehicle["id"]] = (now_s, distance)
    return sorted(times)


def match_crossings(seen: list[float], true: list[float], tolerance_s: float) -> tuple[list[float], int, int]:
    """Pairs each crossing with at most one true one, in time order.

    Returns the time errors of the pairs, then how many true crossings were missed and how many
    seen ones match nothing.
    """
    seen, true = sorted(seen), sorted(true)
    errors = []
    i = j = missed = false = 0
    while i < len(seen) and j < len(true):
        error = seen[i] - true[j]
        if abs(error) <= tolerance_s:
            errors.append(error)
            i += 1
            j += 1
        elif error < 0:
            false += 1
            i += 1
        else:
            missed += 1
            j += 1
    return errors, missed + len(true) - j, false + len(seen) - i


def lamp_at(samples: list[dict], approach: str, time_s: float) -> str:
    """The lamp shown at the sampled frame just after a moment."""
    times = [sample["timestamp_sim_s"] for sample in samples]
    return samples[min(bisect_left(times, time_s), len(samples) - 1)]["lamps"][approach]


def crossing_report(name: str, samples: list[dict], seen: list[float], first_s: float, last_s: float) -> dict:
    """Scores one approach. A crossing right at either end of the window can count as missed or false."""
    within = lambda times: [t for t in times if first_s <= t <= last_s]
    seen, true = within(seen), within(true_crossings(samples, name))
    errors, missed, false = match_crossings(seen, true, CROSSING_TOLERANCE_S)
    return {
        "true": len(true),
        "seen": len(seen),
        "errors": errors,
        "missed": missed,
        "false": false,
        "seen_on_red": sum(lamp_at(samples, name, t) == Lamp.RED for t in seen),
        "true_on_red": sum(lamp_at(samples, name, t) == Lamp.RED for t in true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a folder saved by simulator.record")
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    import cv2

    from vision.detector import VehicleDetector
    from vision.tracker import MIN_DETECTION_CONFIDENCE, CameraTracker
    from vision.traffic_analyzer import ApproachAnalyzer

    detector = VehicleDetector(args.weights, MIN_DETECTION_CONFIDENCE)
    samples = [json.loads(line) for line in (args.run / "frames.jsonl").read_text().splitlines()]
    run = json.loads((args.run / "run.json").read_text())
    specs = camera_specs()
    first_s = samples[0]["timestamp_sim_s"]

    errors, by_queue = Errors(), Errors()
    crossings = {}
    for approach in Approach:
        name = approach.value
        tracker, analyzer = CameraTracker(run["fps"]), ApproachAnalyzer(specs[name], approach)
        for start in range(0, len(samples), BATCH_FRAMES):
            batch = samples[start : start + BATCH_FRAMES]
            frames = [cv2.imread(str(args.run / name / f"{sample['frame']:06d}.png")) for sample in batch]
            for sample, detections in zip(batch, detector.detect(frames)):
                observation = analyzer.update(sample["timestamp_sim_s"], tracker.update(detections))
                if sample["timestamp_sim_s"] - first_s >= WARM_UP_S:
                    compare(errors, by_queue, asdict(observation), sample["truth"][name])
        seen = [crossing.time_s for crossing in analyzer.crossings]
        crossings[name] = crossing_report(name, samples, seen, first_s + WARM_UP_S, samples[-1]["timestamp_sim_s"])

    print(f"{'measure':<20}{'error':>7}{'bias':>7}{'p95':>7}{'worst':>6}")
    for name in FIELDS:
        print(f"{name:<20}{errors.row(name)}")
    print("queued, by true queue length")
    for band in QUEUE_BANDS:
        print(f"{f'{band[0]}-{band[1]} cars':<20}{by_queue.row(band)}   {len(by_queue.values[band])} frames")

    print(f"stop-line crossings, within {CROSSING_TOLERANCE_S} s")
    print(f"{'camera':<8}{'true':>6}{'seen':>6}{'missed':>8}{'false':>7}{'bias s':>8}{'p95 s':>7}{'red seen':>10}{'red true':>10}")
    for name, report in crossings.items():
        sizes = sorted(map(abs, report["errors"]))
        p95 = sizes[min(int(len(sizes) * 0.95), len(sizes) - 1)] if sizes else 0.0
        bias = sum(report["errors"]) / len(sizes) if sizes else 0.0
        print(
            f"{name:<8}{report['true']:>6}{report['seen']:>6}{report['missed']:>8}{report['false']:>7}"
            f"{bias:>+8.2f}{p95:>7.2f}{report['seen_on_red']:>10}{report['true_on_red']:>10}"
        )


if __name__ == "__main__":
    main()
