import pytest
from PIL import Image

from controller.traffic_lights import Approach
from simulator.camera_geometry import HEIGHT, OVERHEAD, WIDTH, camera_specs
from simulator.overlay import annotate, mosaic
from simulator.roads import lane_point

SPECS = camera_specs()
ROADSIDE = [SPECS[approach.value] for approach in Approach]


def in_frame(pixel):
    return 0 <= pixel[0] <= WIDTH and 0 <= pixel[1] <= HEIGHT


def test_there_are_five_cameras():
    assert set(SPECS) == {OVERHEAD, "north", "south", "east", "west"}


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS)
def test_camera_aim_is_the_centre_of_the_frame(spec):
    assert spec.project(spec.target) == pytest.approx((WIDTH / 2, HEIGHT / 2))


def test_points_behind_the_camera_do_not_project():
    north = SPECS["north"]  # It stands south of the intersection and looks north.
    assert north.project((0, 0, -100)) is None


def test_camera_image_is_not_mirrored_or_upside_down():
    north = SPECS["north"]  # Looking north, east is to the right and up is up.
    x, y = north.project((0, 0, 30))
    assert north.project((5, 0, 30))[0] > x
    assert north.project((0, 5, 30))[1] < y


def test_overhead_shows_north_at_the_top_and_east_on_the_right():
    overhead = SPECS[OVERHEAD]
    assert overhead.project((0, 0, 10))[1] < HEIGHT / 2
    assert overhead.project((10, 0, 0))[0] > WIDTH / 2


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS)
def test_every_calibrated_line_and_roi_is_inside_the_frame(spec):
    for approach in spec.approaches:
        assert all(in_frame(pixel) for pixel in spec.stop_line(approach))
        assert all(in_frame(pixel) for pixel in spec.roi(approach))


@pytest.mark.parametrize("spec", ROADSIDE, ids=[spec.name for spec in ROADSIDE])
def test_farther_along_the_lane_is_higher_in_a_roadside_frame(spec):
    approach = spec.approaches[0]
    near_x, near_z = lane_point(approach, 0)
    far_x, far_z = lane_point(approach, spec.roi_distance_m)
    assert spec.project((far_x, 0, far_z))[1] < spec.project((near_x, 0, near_z))[1]


def test_overhead_calibrates_every_approach_and_roadside_only_its_own():
    assert set(SPECS[OVERHEAD].calibration()["approaches"]) == {approach.value for approach in Approach}
    for spec in ROADSIDE:
        assert set(spec.calibration()["approaches"]) == {spec.name}


def test_box_is_cut_to_the_frame():
    north = SPECS["north"]
    left, top, right, bottom = north.project_box((-100, 0, 20), (100, 2, 22))
    assert (left, right) == (0, WIDTH)
    assert 0 <= top < bottom <= HEIGHT


def test_box_out_of_frame_or_behind_the_camera_has_no_pixels():
    north = SPECS["north"]
    assert north.project_box((500, 0, 20), (502, 2, 22)) is None
    assert north.project_box((0, 0, -30), (2, 2, -28)) is None
    assert north.project_box((0, 0, -30), (2, 2, 28)) is None


def test_overlay_marks_a_copy_and_leaves_the_clean_frame_alone():
    clean = Image.new("RGB", (WIDTH, HEIGHT))
    sample = {
        "timestamp_sim_s": 1.2,
        "phase": "NS_GREEN",
        "vehicles": [
            {"id": "north-1", "boxes": {"north": [100, 100, 200, 200]}},
            {"id": "east-1", "boxes": {}},
        ],
    }
    marked = annotate(clean, SPECS["north"].calibration(), sample)
    assert clean.getbbox() is None
    assert marked.getpixel((150, 100)) != (0, 0, 0)


def test_mosaic_keeps_every_frame_at_full_size():
    frames = [Image.new("RGB", (WIDTH, HEIGHT), (index * 40 + 40, 0, 0)) for index in range(5)]
    sheet = mosaic(frames)
    assert sheet.size == (WIDTH * 3, HEIGHT * 2)
    assert sheet.getpixel((WIDTH + 5, HEIGHT + 5))[0] == frames[4].getpixel((0, 0))[0]
