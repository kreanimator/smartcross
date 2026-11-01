"""Scores a detector against the ground truth of a recorded run.

    python -m evaluation.detection RUN_DIR
    python -m evaluation.detection RUN_DIR --weights FILE.pt    # score other weights
    python -m evaluation.detection RUN_DIR --output FILE.json   # also save every detection
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulator.camera_geometry import OVERHEAD

MATCH_IOU = 0.5
MOSTLY_VISIBLE = 0.5  # Share of a vehicle's box that nothing nearer covers.
SIZE_BANDS = [(0, 20), (20, 40), (40, 80), (80, 10_000)]  # Box width in pixels.
BATCH_FRAMES = 32


def iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared)


def match(detections: list, truths: list) -> tuple[list[int | None], int]:
    """Pairs detections with truth boxes, most confident first, each truth at most once.

    detections are (box, confidence). Returns each truth's matched detection index or None,
    and how many detections matched nothing.
    """
    matched: list[int | None] = [None] * len(truths)
    false_positives = 0
    for index in sorted(range(len(detections)), key=lambda i: -detections[i][1]):
        box = detections[index][0]
        free = [(iou(box, truth), t) for t, truth in enumerate(truths) if matched[t] is None]
        best_iou, best = max(free, default=(0.0, None))
        if best is not None and best_iou >= MATCH_IOU:
            matched[best] = index
        else:
            false_positives += 1
    return matched, false_positives


def visible_fractions(boxes: list) -> list[float]:
    """The share of each box not covered by nearer boxes.

    On a flat road, the lower a box's bottom edge is in the frame, the nearer the vehicle.
    Boxes are looser than vehicles, so this is an estimate that errs towards hidden.
    """
    fractions = []
    for box in boxes:
        left, top, right, bottom = (round(value) for value in box)
        area = np.zeros((max(bottom - top, 1), max(right - left, 1)), dtype=bool)
        for other in boxes:
            if other[3] > box[3]:
                area[
                    max(round(other[1]) - top, 0) : max(round(other[3]) - top, 0),
                    max(round(other[0]) - left, 0) : max(round(other[2]) - left, 0),
                ] = True
        fractions.append(1 - float(area.mean()))
    return fractions


def size_band(box) -> tuple[int, int]:
    width = box[2] - box[0]
    return next(band for band in SIZE_BANDS if band[0] <= width < band[1])


@dataclass
class Score:
    found: Counter = field(default_factory=Counter)
    total: Counter = field(default_factory=Counter)
    false_positives: int = 0
    detections: int = 0

    def add(self, truths: list, detections: list) -> None:
        matched, false_positives = match(detections, truths)
        self.false_positives += false_positives
        self.detections += len(detections)
        for box, visible, hit in zip(truths, visible_fractions(truths), matched):
            keys = ["all", "visible" if visible >= MOSTLY_VISIBLE else "hidden"]
            if visible >= MOSTLY_VISIBLE:
                keys.append(size_band(box))
            for key in keys:
                self.total[key] += 1
                self.found[key] += hit is not None

    def recall(self, key) -> float | None:
        return self.found[key] / self.total[key] if self.total[key] else None

    def precision(self) -> float | None:
        return 1 - self.false_positives / self.detections if self.detections else None


def _percent(value: float | None) -> str:
    return "   -" if value is None else f"{value:4.0%}"


def print_scores(scores: dict[str, Score]) -> None:
    bands = [f"{low}-{high}px" if high < 10_000 else f"{low}px+" for low, high in SIZE_BANDS]
    print("recall by box width is for mostly visible vehicles only")
    print(f"{'camera':<9}{'precision':>10}{'recall':>8}{'visible':>9}{'hidden':>8}" + "".join(f"{band:>10}" for band in bands))
    for name, score in scores.items():
        print(
            f"{name:<9}{_percent(score.precision()):>10}{_percent(score.recall('all')):>8}"
            f"{_percent(score.recall('visible')):>9}{_percent(score.recall('hidden')):>8}"
            + "".join(f"{_percent(score.recall(band)):>10}" for band in SIZE_BANDS)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a folder saved by simulator.record")
    parser.add_argument("--output", type=Path, help="save every detection to this JSON file")
    parser.add_argument("--weights", type=Path, help="model to score, instead of the pretrained one")
    parser.add_argument("--min-confidence", type=float, help="ignore weaker detections")
    args = parser.parse_args()

    import cv2

    from vision.detector import MIN_CONFIDENCE, WEIGHTS, VehicleDetector

    detector = VehicleDetector(args.weights or WEIGHTS, args.min_confidence or MIN_CONFIDENCE)
    samples = [json.loads(line) for line in (args.run / "frames.jsonl").read_text().splitlines()]
    cameras = [name for name in json.loads((args.run / "run.json").read_text())["cameras"] if name != OVERHEAD]

    scores = {name: Score() for name in cameras}
    saved = []
    for name in cameras:
        for start in range(0, len(samples), BATCH_FRAMES):
            batch = samples[start : start + BATCH_FRAMES]
            frames = [cv2.imread(str(args.run / name / f"{sample['frame']:06d}.png")) for sample in batch]
            for sample, detections in zip(batch, detector.detect(frames)):
                truths = [v["boxes"][name] for v in sample["vehicles"] if name in v["boxes"]]
                scores[name].add(truths, [(d.box, d.confidence) for d in detections])
                saved.append(
                    {
                        "camera": name,
                        "frame": sample["frame"],
                        "detections": [{"box": d.box, "confidence": d.confidence, "label": d.label} for d in detections],
                    }
                )
    print_scores(scores)
    if args.output:
        args.output.write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
