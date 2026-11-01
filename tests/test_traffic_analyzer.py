import pytest

pytest.importorskip("ultralytics")

from controller.traffic_lights import Approach  # noqa: E402
from evaluation.perception import Errors, crossing_report, match_crossings, queue_band, true_crossings  # noqa: E402
from simulator.camera_geometry import camera_specs  # noqa: E402
from simulator.roads import distance_to_line, lane_point  # noqa: E402
from vision.tracker import Track  # noqa: E402
from vision.traffic_analyzer import ApproachAnalyzer  # noqa: E402

SPECS = camera_specs()
FPS = 5


def track_at(approach, distance, id=1):
    """A track whose front edge sits on the lane, distance metres before the stop line."""
    x, y = SPECS[approach.value].project((*_xz(approach, distance),))
    return Track(id, (x - 15, y - 30, x + 15, y), 0.9)


def _xz(approach, distance):
    x, z = lane_point(approach, distance)
    return (x, 0, z)


def drive(approach, distances_by_id, analyzer=None):
    analyzer = analyzer or ApproachAnalyzer(SPECS[approach.value], approach)
    frames = len(next(iter(distances_by_id.values())))
    result = None
    for frame in range(frames):
        tracks = [track_at(approach, distances[frame], id) for id, distances in distances_by_id.items()]
        result = analyzer.update(frame / FPS, tracks)
    return result


@pytest.mark.parametrize("approach", list(Approach), ids=lambda a: a.value)
def test_ground_point_undoes_projection(approach):
    spec = SPECS[approach.value]
    for distance in (0, 15, 40, 80):
        point = _xz(approach, distance)
        x, z = spec.ground_point(spec.project(point))
        assert (x, z) == pytest.approx((point[0], point[2]), abs=1e-6)
        assert distance_to_line(approach, x, z) == pytest.approx(distance, abs=1e-6)


def test_sky_has_no_ground_point():
    assert SPECS["north"].ground_point((320, 0)) is None


def test_distance_is_negative_past_the_line():
    x, z = lane_point(Approach.EAST, -4)
    assert distance_to_line(Approach.EAST, x, z) == pytest.approx(-4)


def test_stopped_cars_are_queued_and_their_wait_grows():
    seen = drive(Approach.NORTH, {1: [10] * 30, 2: [17] * 30})
    assert (seen.queued, seen.approaching) == (2, 0)
    assert seen.oldest_wait_s == pytest.approx(30 / FPS - 0.2 - 0.6, abs=0.3)
    assert seen.average_speed_mps == pytest.approx(0, abs=0.01)


def test_moving_car_is_approaching_at_its_speed():
    seen = drive(Approach.EAST, {1: [55 - i * 10 / FPS for i in range(20)]})
    assert (seen.queued, seen.approaching) == (0, 1)
    assert seen.average_speed_mps == pytest.approx(10, abs=0.1)
    assert seen.oldest_wait_s == 0


def test_near_line_counts_only_close_moving_cars():
    close = [25 - i * 8 / FPS for i in range(10)]
    far = [58 - i * 8 / FPS for i in range(10)]
    stopped = [12] * 10
    seen = drive(Approach.SOUTH, {1: close, 2: far, 3: stopped})
    assert seen.near_line == 1
    assert (seen.queued, seen.approaching) == (1, 2)


def test_new_track_counts_as_approaching_with_unknown_speed():
    seen = drive(Approach.NORTH, {1: [20]})
    assert (seen.queued, seen.approaching) == (0, 1)
    assert seen.average_speed_mps is None


def test_cars_past_the_line_or_beyond_the_roi_are_not_counted():
    seen = drive(Approach.WEST, {1: [-3] * 10, 2: [93] * 10})
    assert (seen.queued, seen.approaching) == (0, 0)
    assert seen.average_speed_mps is None


def test_nothing_seen_is_an_empty_observation():
    analyzer = ApproachAnalyzer(SPECS["north"], Approach.NORTH)
    seen = analyzer.update(0, [])
    assert (seen.queued, seen.approaching, seen.near_line, seen.oldest_wait_s) == (0, 0, 0, 0)


def test_wait_stops_growing_once_a_car_drives_off_and_clears():
    analyzer = ApproachAnalyzer(SPECS["north"], Approach.NORTH)
    drive(Approach.NORTH, {1: [10] * 30}, analyzer)
    seen = analyzer.update(30 / FPS, [])
    assert seen.oldest_wait_s == 0


def test_forgotten_track_starts_over():
    analyzer = ApproachAnalyzer(SPECS["north"], Approach.NORTH)
    drive(Approach.NORTH, {1: [10] * 30}, analyzer)
    analyzer.update(20, [])
    assert analyzer.followed == {}


def test_error_row_reports_absolute_mean_bias_and_worst():
    errors = Errors()
    errors.add("queued", 3, 5)
    errors.add("queued", 5, 5)
    errors.add("queued", 6, 5)
    errors.add("queued", None, 5)
    assert errors.values["queued"] == [-2, 0, 1]
    assert errors.row("queued").split() == ["1.00", "-0.33", "2.0", "-2"]


def test_queue_bands_cover_every_length():
    assert queue_band(0) == (0, 0)
    assert queue_band(5) == (4, 6)
    assert queue_band(40) == (10, 1000)


def crossings_for(approach, distances):
    analyzer = ApproachAnalyzer(SPECS[approach.value], approach)
    drive(approach, {1: distances}, analyzer)
    return analyzer.crossings


@pytest.mark.parametrize("approach", list(Approach), ids=lambda a: a.value)
def test_a_car_driving_over_the_line_crosses_once(approach):
    crossings = crossings_for(approach, [8, 5.4, 2.8, 0.2, -2.4, -4.0])
    assert len(crossings) == 1
    assert crossings[0].track_id == 1
    assert crossings[0].time_s == pytest.approx(3 / FPS + 0.2 / 2.6 / FPS, abs=0.01)
    assert crossings[0].speed_mps == pytest.approx(13, abs=0.5)


def test_a_car_halted_short_of_the_line_does_not_cross():
    assert crossings_for(Approach.NORTH, [3, 1.5, 0.5, 0.5, 0.4, 0.6, 0.5]) == []


def test_a_slight_overshoot_is_not_a_crossing():
    assert crossings_for(Approach.NORTH, [1, 0.5, -0.5, -0.6, -0.5]) == []


def test_a_track_never_followed_in_the_lane_does_not_cross():
    assert crossings_for(Approach.NORTH, [-2, -3, -4]) == []


def test_true_crossings_are_read_from_the_recorded_distances():
    samples = [
        {"timestamp_sim_s": t, "vehicles": [{"id": "n1", "approach": "north", "distance_m": d}]}
        for t, d in ((0.0, 3.0), (1.0, 1.0), (2.0, -1.0), (3.0, -3.0))
    ]
    assert true_crossings(samples, "north") == [1.5]
    assert true_crossings(samples, "east") == []


def test_matching_counts_misses_and_false_crossings():
    errors, missed, false = match_crossings([1.1, 5.0, 9.0], [1.0, 3.0, 9.2], 0.5)
    assert errors == pytest.approx([0.1, -0.2])
    assert (missed, false) == (1, 1)


def test_one_true_crossing_cannot_match_twice():
    errors, missed, false = match_crossings([1.0, 1.1], [1.0], 0.5)
    assert (len(errors), missed, false) == (1, 0, 1)


def test_crossings_on_red_are_counted_from_the_recorded_lamps():
    samples = [
        {
            "timestamp_sim_s": t,
            "lamps": {"north": lamp},
            "vehicles": [{"id": "n1", "approach": "north", "distance_m": d}],
        }
        for t, d, lamp in ((0.0, 1.0, "GREEN"), (1.0, -1.0, "RED"), (2.0, -3.0, "RED"))
    ]
    report = crossing_report("north", samples, [0.5, 1.5], 0.0, 2.0)
    assert report["true_on_red"] == 1
    assert report["seen_on_red"] == 2
