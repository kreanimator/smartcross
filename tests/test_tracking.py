import pytest

pytest.importorskip("ultralytics")

from evaluation.tracking import TrackingScore, is_expected  # noqa: E402
from simulator.camera_geometry import in_polygon  # noqa: E402
from vision.detector import Detection  # noqa: E402
from vision.tracker import CameraTracker  # noqa: E402

FPS = 10


def car(x, y=100, confidence=0.9):
    return Detection((x, y, x + 40, y + 30), confidence, "vehicle")


def run(tracker, frames):
    return [tracker.update(detections) for detections in frames]


def test_moving_car_keeps_one_id():
    history = run(CameraTracker(FPS), [[car(100 + i * 5)] for i in range(20)])
    assert len({track.id for tracks in history for track in tracks}) == 1
    assert all(len(tracks) == 1 for tracks in history)


def test_track_box_follows_the_car():
    history = run(CameraTracker(FPS), [[car(100 + i * 5)] for i in range(20)])
    assert history[-1][0].box[0] == pytest.approx(195, abs=3)


def test_two_cars_get_different_ids_that_last():
    frames = [[car(100 + i * 2), car(400 - i * 2)] for i in range(20)]
    history = run(CameraTracker(FPS), frames)
    assert len(history[-1]) == 2
    assert len({track.id for tracks in history for track in tracks}) == 2


def test_car_missing_briefly_keeps_its_id():
    frames = [[car(100 + i * 2)] if not 10 <= i < 15 else [] for i in range(25)]
    history = run(CameraTracker(FPS), frames)
    assert len({track.id for tracks in history for track in tracks}) == 1


def test_car_missing_too_long_gets_a_new_id():
    frames = [[car(100)] if not 10 <= i < 30 else [] for i in range(40)]
    history = run(CameraTracker(FPS), frames)
    assert len({track.id for tracks in history for track in tracks}) == 2


def test_weak_detections_continue_a_track_but_do_not_start_one():
    strong_then_weak = [[car(100 + i * 2)] for i in range(10)] + [[car(120 + i * 2, confidence=0.15)] for i in range(10)]
    assert len(run(CameraTracker(FPS), strong_then_weak)[-1]) == 1

    only_weak = [[car(100 + i * 2, confidence=0.15)] for i in range(10)]
    assert not any(run(CameraTracker(FPS), only_weak))


def test_cameras_do_not_share_tracks():
    north, east = CameraTracker(FPS), CameraTracker(FPS)
    for i in range(10):
        north.update([car(100 + i)])
        tracks = east.update([])
    assert tracks == []


BOX = (0, 0, 50, 50)
ELSEWHERE = (300, 300, 350, 350)


def test_perfect_tracking_scores_one():
    score = TrackingScore()
    for _ in range(5):
        score.add({"a": BOX, "b": ELSEWHERE}, {1: BOX, 2: ELSEWHERE})
    assert score.mota() == 1
    assert score.extra_tracks() == 0
    assert score.mostly_tracked() == 1


def test_changing_track_is_an_id_switch_and_an_extra_track():
    score = TrackingScore()
    score.add({"a": BOX}, {1: BOX})
    score.add({"a": BOX}, {2: BOX})
    assert score.id_switches == 1
    assert score.extra_tracks() == 1
    assert score.mota() == pytest.approx(0.5)


def test_gap_then_same_track_is_a_miss_not_a_switch():
    score = TrackingScore()
    score.add({"a": BOX}, {1: BOX})
    score.add({"a": BOX}, {})
    score.add({"a": BOX}, {1: BOX})
    assert (score.missed, score.id_switches) == (1, 0)


def test_track_on_nothing_is_a_false_alarm():
    score = TrackingScore()
    score.add({"a": BOX}, {1: BOX, 2: ELSEWHERE})
    assert score.false_alarms == 1


def test_existing_pair_survives_a_better_overlapping_track():
    score = TrackingScore()
    score.add({"a": BOX}, {1: BOX})
    score.add({"a": BOX}, {1: (2, 2, 52, 52), 2: BOX})
    assert score.id_switches == 0
    assert score.false_alarms == 1


def test_hidden_vehicle_is_not_expected_or_missed():
    far, near = (100, 100, 150, 150), (100, 100, 150, 160)  # The nearer box covers the far one.
    score = TrackingScore()
    score.add({"far": far, "near": near}, {1: near})
    assert score.frames_expected == 1
    assert score.missed == 0
    assert score.false_alarms == 0


def test_tiny_box_is_not_expected():
    assert not is_expected((0, 0, 4, 40), 1.0)
    assert is_expected((0, 0, 8, 40), 1.0)


def test_counted_ignores_short_tracks():
    score = TrackingScore()
    for frame in range(5):
        score.add({"a": BOX}, {1: BOX, 2: ELSEWHERE} if frame == 0 else {1: BOX})
    assert score.counted(min_frames=3) == 1


def test_unwatched_vehicle_is_not_scored_but_its_track_is_no_false_alarm():
    score = TrackingScore()
    score.add({"a": BOX, "b": ELSEWHERE}, {2: ELSEWHERE}, watched_vehicles={"a"}, watched_tracks={2})
    assert score.frames_expected == 1
    assert score.missed == 1
    assert score.false_alarms == 0


def test_watched_track_on_nothing_is_a_false_alarm():
    score = TrackingScore()
    score.add({"a": BOX}, {1: ELSEWHERE}, watched_vehicles={"a"}, watched_tracks={1})
    assert score.false_alarms == 1


def test_unwatched_track_on_nothing_is_ignored():
    score = TrackingScore()
    score.add({"a": BOX}, {1: BOX, 2: ELSEWHERE}, watched_vehicles={"a"}, watched_tracks={1})
    assert score.false_alarms == 0
    assert score.mota() == 1


def test_switch_while_unwatched_does_not_count():
    score = TrackingScore()
    score.add({"a": BOX}, {1: BOX}, watched_vehicles=set(), watched_tracks=set())
    score.add({"a": BOX}, {2: BOX}, watched_vehicles={"a"}, watched_tracks={2})
    assert score.id_switches == 0
    assert score.extra_tracks() == 0


def test_in_polygon_with_slanted_edges():
    polygon = [(0, 0), (10, 0), (20, 10), (-10, 10)]
    assert in_polygon((5, 5), polygon)
    assert in_polygon((-4, 8), polygon)
    assert not in_polygon((-4, 2), polygon)
    assert not in_polygon((5, 11), polygon)
