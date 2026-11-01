"""Checks the calibration against what the renderer really draws. Skipped with no graphics."""

import json

import numpy as np
import pytest
from PIL import Image

from controller.traffic_lights import Approach
from simulator.camera_geometry import HEIGHT, OVERHEAD, WIDTH
from simulator.record import record, steps_per_sample
from simulator.simulation import Simulation
from simulator.traffic import Vehicle

MIN_BOX_OVERLAP = 0.7


@pytest.fixture(scope="module")
def scene_parts():
    ursina = pytest.importorskip("ursina")
    try:
        app = ursina.Ursina(window_type="offscreen", size=(64, 64), development_mode=False)
    except Exception as error:
        pytest.skip(f"no graphics: {error}")
    from simulator.world import World

    world = World(app)
    for _ in range(2):
        app.graphicsEngine.render_frame()
    return app, world


def render_with(app, world, vehicles):
    world.vehicles.show(vehicles)
    app.graphicsEngine.render_frame()


def changed_box(before, after):
    """The pixels around everything that differs between two frames, as (left, top, right, bottom)."""
    rows, columns = np.nonzero(np.abs(before.astype(int) - after.astype(int)).sum(axis=2) > 30)
    return (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1) if len(rows) else None


def overlap(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return width * height / (area(a) + area(b) - width * height)


@pytest.mark.parametrize("name", [OVERHEAD, "north", "south", "east", "west"])
def test_projection_matches_the_renderer(scene_parts, name):
    from panda3d.core import Point2, Point3

    _, world = scene_parts
    camera = world.cameras[name]
    x, y, z = camera.spec.target
    for point in [(x, y, z), (x + 3, y, z + 2), (x - 4, y + 2, z - 3), (x + 5, y + 1, z + 4)]:
        projected = Point2()
        assert camera.lens.project(camera.node.get_relative_point(camera.node.get_top(), Point3(*point)), projected)
        expected = ((projected.x + 1) / 2 * WIDTH, (1 - projected.y) / 2 * HEIGHT)
        assert camera.spec.project(point) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("approach", list(Approach), ids=lambda approach: approach.value)
def test_drawn_stop_line_lies_on_its_calibrated_pixels(scene_parts, approach):
    app, world = scene_parts
    render_with(app, world, [])
    frame = world.cameras[approach.value].frame()
    (x0, y0), (x1, y1) = world.cameras[approach.value].spec.stop_line(approach)
    white = 0
    samples = np.linspace(0.1, 0.9, 20)
    for t in samples:
        pixel = frame[round(y0 + (y1 - y0) * t), round(x0 + (x1 - x0) * t)]
        white += pixel.min() > 150
    assert white >= 0.8 * len(samples)


@pytest.mark.parametrize("approach", list(Approach), ids=lambda approach: approach.value)
def test_a_frame_shows_the_vehicle_where_it_is_right_now(scene_parts, approach):
    """No frame of lag: one render after a move, the car is exactly where the box says."""
    app, world = scene_parts
    camera = world.cameras[approach.value]
    render_with(app, world, [])
    empty = camera.frame()

    for distance in (40, 12):  # Move it a long way, so a stale frame could not pass.
        car = Vehicle("test-car", approach, "van", distance)
        render_with(app, world, [car])
        seen = changed_box(empty, camera.frame())
        expected = camera.spec.project_box(*world.vehicles.bounds(car.id))
        assert seen is not None
        assert overlap(seen, expected) >= MIN_BOX_OVERLAP
    world.vehicles.clear()


def test_record_saves_aligned_frames_and_truth(scene_parts, tmp_path):
    app, world = scene_parts
    samples = record(app, world, tmp_path, seed=3, start=30, seconds=1, fps=5, overlay=True)
    world.vehicles.clear()

    assert [sample["frame"] for sample in samples] == list(range(6))
    assert [sample["timestamp_sim_s"] for sample in samples] == pytest.approx([30 + i * 0.2 for i in range(6)])

    lines = (tmp_path / "frames.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == samples

    run = json.loads((tmp_path / "run.json").read_text())
    assert run["seed"] == 3 and set(run["cameras"]) == set(world.cameras)

    for name in world.cameras:
        with Image.open(tmp_path / name / "000005.png") as image:
            assert image.size == (WIDTH, HEIGHT)
    with Image.open(tmp_path / "overlay" / "000005.png") as sheet:
        assert sheet.size == (WIDTH * 3, HEIGHT * 2)


def test_recorded_truth_matches_a_run_nobody_watched(scene_parts, tmp_path):
    app, world = scene_parts
    samples = record(app, world, tmp_path, seed=8, seconds=10, fps=1)
    world.vehicles.clear()

    unwatched = Simulation(8)
    for _ in range(600):
        unwatched.step()
    expected = {v.id: round(v.distance, 3) for v in unwatched.traffic.vehicles}
    assert {v["id"]: v["distance_m"] for v in samples[-1]["vehicles"]} == expected


@pytest.mark.parametrize("fps, steps", [(60, 1), (30, 2), (5, 12), (1, 60), (0.5, 120)])
def test_sampling_lands_on_simulation_steps(fps, steps):
    assert steps_per_sample(fps) == steps


@pytest.mark.parametrize("fps", [7, 0, -5, 120])
def test_sampling_rejects_rates_between_steps(fps):
    with pytest.raises(ValueError):
        steps_per_sample(fps)
