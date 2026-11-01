"""Compares camera-derived traffic observations with the simulator's exact ones.

    python -m evaluation.perception RUN_DIR --weights FILE.pt

Errors are the estimate minus the truth, so a negative bias means the cameras see too little.
The truth covers the whole lane, while the cameras only watch their ROI.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from controller.traffic_lights import Approach
from evaluation.detection import BATCH_FRAMES
from simulator.camera_geometry import camera_specs

WARM_UP_S = 30  # Cars already waiting at the first frame make early waits unknowable.
FIELDS = ("queued", "approaching", "near_line", "oldest_wait_s", "average_speed_mps")
QUEUE_BANDS = [(0, 0), (1, 3), (4, 6), (7, 9), (10, 1000)]


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

    print(f"{'measure':<20}{'error':>7}{'bias':>7}{'p95':>7}{'worst':>6}")
    for name in FIELDS:
        print(f"{name:<20}{errors.row(name)}")
    print("queued, by true queue length")
    for band in QUEUE_BANDS:
        print(f"{f'{band[0]}-{band[1]} cars':<20}{by_queue.row(band)}   {len(by_queue.values[band])} frames")


if __name__ == "__main__":
    main()
