"""Turns recorded runs into a labelled YOLO training set, using their ground truth.

    python -m vision.dataset --out DIR --train RUN [RUN ...] --val RUN [RUN ...]

Runs in --train and --val must use different seeds, or validation only measures memory.
Only the roadside cameras are used: the overhead view is for people, never a vision input.
"""

import argparse
import json
import shutil
from pathlib import Path

from evaluation.detection import visible_fractions
from simulator.camera_geometry import HEIGHT, OVERHEAD, WIDTH

MIN_VISIBLE = 0.3  # A mostly hidden vehicle would teach the model to box the one in front.
MIN_BOX_PIXELS = 6


def labels_for(boxes: list) -> list[str]:
    """YOLO label lines (class, centre x, centre y, width, height, all 0-1) for one frame."""
    lines = []
    for box, visible in zip(boxes, visible_fractions(boxes)):
        left, top, right, bottom = box
        if visible >= MIN_VISIBLE and min(right - left, bottom - top) >= MIN_BOX_PIXELS:
            lines.append(
                f"0 {(left + right) / 2 / WIDTH:.6f} {(top + bottom) / 2 / HEIGHT:.6f} "
                f"{(right - left) / WIDTH:.6f} {(bottom - top) / HEIGHT:.6f}"
            )
    return lines


def add_run(run: Path, out: Path, split: str) -> int:
    images, labels = out / "images" / split, out / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    cameras = [name for name in json.loads((run / "run.json").read_text())["cameras"] if name != OVERHEAD]
    added = 0
    for line in (run / "frames.jsonl").read_text().splitlines():
        sample = json.loads(line)
        for name in cameras:
            stem = f"{run.name}_{name}_{sample['frame']:06d}"
            shutil.copy(run / name / f"{sample['frame']:06d}.png", images / f"{stem}.png")
            boxes = [vehicle["boxes"][name] for vehicle in sample["vehicles"] if name in vehicle["boxes"]]
            (labels / f"{stem}.txt").write_text("".join(f"{label}\n" for label in labels_for(boxes)))
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    seeds = {split: {json.loads((run / "run.json").read_text())["seed"] for run in runs}
             for split, runs in (("train", args.train), ("val", args.val))}
    if seeds["train"] & seeds["val"]:
        parser.error(f"seeds used for both training and validation: {seeds['train'] & seeds['val']}")

    train = sum(add_run(run, args.out, "train") for run in args.train)
    val = sum(add_run(run, args.out, "val") for run in args.val)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n"
    )
    print(f"saved {train} training and {val} validation images to {args.out}")


if __name__ == "__main__":
    main()
