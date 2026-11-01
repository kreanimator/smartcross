"""Fine-tunes the pretrained detector on a set made by vision.dataset.

    python -m vision.train DATASET_DIR --name NAME   # best weights: runs/NAME/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from vision.detector import IMAGE_SIZE, WEIGHTS

RUNS = Path(__file__).parent.parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    YOLO(str(WEIGHTS)).train(
        data=str(args.dataset.resolve() / "data.yaml"),
        project=str(RUNS),
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=IMAGE_SIZE,
        seed=args.seed,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
