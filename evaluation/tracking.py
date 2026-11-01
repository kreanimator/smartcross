"""Scores vehicle tracks against the ground truth of a recorded run.

    python -m evaluation.tracking RUN_DIR --weights FILE.pt
    python -m evaluation.tracking RUN_DIR --weights FILE.pt --every 2   # use every 2nd frame
    python -m evaluation.tracking RUN_DIR --weights FILE.pt --output FILE.json

Two rows per camera: "all" is everything in frame, and "lane" is the stretch the controller
relies on, the camera's own lane from its stop line to the end of its ROI.

Vehicles too hidden or small to label are never counted as missed, but a track on one
is not a false alarm either.
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.detection import BATCH_FRAMES, MATCH_IOU, iou, visible_fractions
from simulator.camera_geometry import OVERHEAD, in_polygon

MOSTLY_TRACKED = 0.8  # Share of its visible frames a vehicle must be followed in.
MIN_TRACK_FRAMES_S = 0.5  # Shorter tracks are not counted as a vehicle.
MIN_VISIBLE = 0.3
MIN_BOX_PIXELS = 6


def is_expected(box, visible: float) -> bool:
    """Whether a vehicle could fairly be expected to be seen in this frame."""
    return visible >= MIN_VISIBLE and min(box[2] - box[0], box[3] - box[1]) >= MIN_BOX_PIXELS


@dataclass
class TrackingScore:
    """CLEAR MOT counts, plus how many tracks each vehicle was split into."""

    frames_expected: int = 0
    missed: int = 0
    false_alarms: int = 0
    id_switches: int = 0
    last_track: dict = field(default_factory=dict)  # Vehicle ID to the track last following it.
    last_scored: dict = field(default_factory=dict)  # The same, as of when the vehicle was last watched.
    expected_frames: Counter = field(default_factory=Counter)
    followed_frames: Counter = field(default_factory=Counter)
    tracks_of: dict = field(default_factory=lambda: defaultdict(set))
    track_frames: Counter = field(default_factory=Counter)  # All frames each track existed in.

    def add(self, truths: dict, tracks: dict, watched_vehicles=None, watched_tracks=None) -> None:
        """truths and tracks map IDs to boxes, for one frame.

        Everything is used for matching, so a track on some other vehicle is not a false alarm.
        Only the watched vehicles and tracks are scored; by default, all of them.
        """
        watched_vehicles = truths.keys() if watched_vehicles is None else watched_vehicles
        watched_tracks = tracks.keys() if watched_tracks is None else watched_tracks
        pairs = self._assign(truths, tracks)
        for track in watched_tracks:
            self.track_frames[track] += 1
        for vehicle, track in pairs.items():
            if vehicle in watched_vehicles:
                if self.last_scored.get(vehicle, track) != track:
                    self.id_switches += 1
                self.last_scored[vehicle] = track
                self.tracks_of[vehicle].add(track)
            self.last_track[vehicle] = track
        self.false_alarms += len(set(watched_tracks) - set(pairs.values()))

        ids = list(truths)
        for vehicle, visible in zip(ids, visible_fractions([truths[v] for v in ids])):
            if vehicle in watched_vehicles and is_expected(truths[vehicle], visible):
                self.frames_expected += 1
                self.expected_frames[vehicle] += 1
                if vehicle in pairs:
                    self.followed_frames[vehicle] += 1
                else:
                    self.missed += 1

    def _assign(self, truths: dict, tracks: dict) -> dict:
        """Vehicle to track. A pair from the last frame that still overlaps is kept first."""
        pairs = {}
        for vehicle, track in self.last_track.items():
            if vehicle in truths and track in tracks and iou(truths[vehicle], tracks[track]) >= MATCH_IOU:
                pairs[vehicle] = track
        taken = set(pairs.values())
        candidates = sorted(
            (
                (iou(box, track_box), vehicle, track)
                for vehicle, box in truths.items()
                if vehicle not in pairs
                for track, track_box in tracks.items()
                if track not in taken
            ),
            key=lambda item: -item[0],
        )
        for overlap, vehicle, track in candidates:
            if overlap >= MATCH_IOU and vehicle not in pairs and track not in taken:
                pairs[vehicle] = track
                taken.add(track)
        return pairs

    def mota(self) -> float | None:
        """1 is perfect. Misses, false alarms and ID switches all count against it."""
        if not self.frames_expected:
            return None
        return 1 - (self.missed + self.false_alarms + self.id_switches) / self.frames_expected

    def vehicles(self) -> int:
        return len(self.expected_frames)

    def mostly_tracked(self) -> float | None:
        if not self.expected_frames:
            return None
        followed = sum(
            self.followed_frames[vehicle] >= MOSTLY_TRACKED * frames for vehicle, frames in self.expected_frames.items()
        )
        return followed / len(self.expected_frames)

    def extra_tracks(self) -> int:
        """Times a vehicle was given another track: how often it would be counted twice."""
        return sum(len(tracks) - 1 for vehicle, tracks in self.tracks_of.items() if tracks)

    def counted(self, min_frames: float) -> int:
        """How many vehicles a counter would report: every track lasting long enough."""
        return sum(frames >= min_frames for frames in self.track_frames.values())


def lane_view(sample: dict, name: str, calibration: dict, tracks: list) -> tuple[set, set]:
    """The vehicles in a camera's own watched lane, and the tracks standing on its ROI."""
    vehicles = {
        v["id"]
        for v in sample["vehicles"]
        if name in v["boxes"] and v["approach"] == name and 0 <= v["distance_m"] <= calibration["roi_distance_m"]
    }
    roi = calibration["approaches"][name]["roi"]
    on_roi = {t.id for t in tracks if in_polygon(((t.box[0] + t.box[2]) / 2, t.box[3]), roi)}
    return vehicles, on_roi


def print_scores(scores: dict[str, TrackingScore], fps: float) -> None:
    print(f"{'camera':<12}{'MOTA':>7}{'missed':>8}{'false':>7}{'switches':>10}{'vehicles':>10}{'followed':>10}{'extra':>7}{'counted':>9}")
    for name, score in scores.items():
        mota = "-" if score.mota() is None else f"{score.mota():.1%}"
        followed = "-" if score.mostly_tracked() is None else f"{score.mostly_tracked():.0%}"
        print(
            f"{name:<12}{mota:>7}{score.missed:>8}{score.false_alarms:>7}{score.id_switches:>10}"
            f"{score.vehicles():>10}{followed:>10}{score.extra_tracks():>7}{score.counted(MIN_TRACK_FRAMES_S * fps):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a folder saved by simulator.record")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1, help="use only every Nth frame")
    parser.add_argument("--output", type=Path, help="save every track to this JSON file")
    args = parser.parse_args()

    import cv2

    from vision.detector import VehicleDetector
    from vision.tracker import MIN_DETECTION_CONFIDENCE, CameraTracker

    detector = VehicleDetector(args.weights, MIN_DETECTION_CONFIDENCE)
    samples = [json.loads(line) for line in (args.run / "frames.jsonl").read_text().splitlines()][:: args.every]
    run = json.loads((args.run / "run.json").read_text())
    fps = run["fps"] / args.every

    scores, saved = {}, []
    for name in [name for name in run["cameras"] if name != OVERHEAD]:
        tracker, score, lane_score = CameraTracker(fps), TrackingScore(), TrackingScore()
        for start in range(0, len(samples), BATCH_FRAMES):
            batch = samples[start : start + BATCH_FRAMES]
            frames = [cv2.imread(str(args.run / name / f"{sample['frame']:06d}.png")) for sample in batch]
            for sample, detections in zip(batch, detector.detect(frames)):
                tracks = tracker.update(detections)
                truths = {v["id"]: v["boxes"][name] for v in sample["vehicles"] if name in v["boxes"]}
                boxes = {track.id: track.box for track in tracks}
                score.add(truths, boxes)
                lane_score.add(truths, boxes, *lane_view(sample, name, run["cameras"][name], tracks))
                saved.append(
                    {"camera": name, "frame": sample["frame"], "tracks": {t.id: t.box for t in tracks}}
                )
        scores[f"{name} all"] = score
        scores[f"{name} lane"] = lane_score
    print(f"{fps:g} frames per second")
    print_scores(scores, fps)
    if args.output:
        args.output.write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
