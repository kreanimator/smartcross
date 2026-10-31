import math
import random

import pytest

from controller.traffic_lights import Approach, Lamp, Phase, Timings, TrafficLights

TIMINGS = Timings(min_green_s=10, yellow_s=3, all_red_s=2)

LEGAL_TRANSITIONS = {
    (Phase.ALL_RED, Phase.NS_GREEN),
    (Phase.ALL_RED, Phase.EW_GREEN),
    (Phase.NS_GREEN, Phase.NS_YELLOW),
    (Phase.NS_YELLOW, Phase.ALL_RED),
    (Phase.EW_GREEN, Phase.EW_YELLOW),
    (Phase.EW_YELLOW, Phase.ALL_RED),
}

MIN_DURATION = {
    Phase.ALL_RED: TIMINGS.all_red_s,
    Phase.NS_GREEN: TIMINGS.min_green_s,
    Phase.EW_GREEN: TIMINGS.min_green_s,
    Phase.NS_YELLOW: TIMINGS.yellow_s,
    Phase.EW_YELLOW: TIMINGS.yellow_s,
}


def phase_of(lights):
    return lights.get_states().phase


def lights_in(direction):
    """Lights that have just turned green for direction."""
    lights = TrafficLights(TIMINGS)
    lights.request_phase(direction)
    lights.update(TIMINGS.all_red_s)
    assert phase_of(lights) is Phase[f"{direction}_GREEN"]
    return lights


def test_starts_all_red():
    snapshot = TrafficLights(TIMINGS).get_states()
    assert snapshot.phase is Phase.ALL_RED
    assert set(snapshot.lamps.values()) == {Lamp.RED}
    assert snapshot.seconds_in_phase == 0


def test_stays_all_red_without_a_request():
    lights = TrafficLights(TIMINGS)
    lights.update(1000)
    assert phase_of(lights) is Phase.ALL_RED


def test_first_green_waits_for_all_red_clearance():
    lights = TrafficLights(TIMINGS)
    lights.request_phase("NS")
    lights.update(TIMINGS.all_red_s - 0.1)
    assert phase_of(lights) is Phase.ALL_RED
    lights.update(0.1)
    assert phase_of(lights) is Phase.NS_GREEN


@pytest.mark.parametrize(
    "start, target, sequence",
    [
        ("NS", "EW", [Phase.NS_YELLOW, Phase.ALL_RED, Phase.EW_GREEN]),
        ("EW", "NS", [Phase.EW_YELLOW, Phase.ALL_RED, Phase.NS_GREEN]),
    ],
)
def test_change_goes_through_yellow_and_all_red(start, target, sequence):
    lights = lights_in(start)
    lights.request_phase(target)
    seen = []
    for _ in range(3):
        lights.update(1000)
        seen.append(phase_of(lights))
    assert seen == sequence


def test_request_cannot_cut_minimum_green_short():
    lights = lights_in("NS")
    lights.request_phase("EW")
    lights.update(TIMINGS.min_green_s - 0.1)
    assert phase_of(lights) is Phase.NS_GREEN
    lights.update(0.1)
    assert phase_of(lights) is Phase.NS_YELLOW


def test_green_holds_while_its_direction_is_requested():
    lights = lights_in("NS")
    lights.request_phase("NS")
    lights.update(1000)
    assert phase_of(lights) is Phase.NS_GREEN


def test_latest_request_wins():
    lights = lights_in("NS")
    lights.request_phase("EW")
    lights.request_phase("NS")
    lights.update(1000)
    assert phase_of(lights) is Phase.NS_GREEN


def test_request_during_yellow_still_clears_the_intersection():
    lights = lights_in("NS")
    lights.request_phase("EW")
    lights.update(TIMINGS.min_green_s)
    lights.request_phase("NS")
    lights.update(1000)
    assert phase_of(lights) is Phase.ALL_RED
    lights.update(1000)
    assert phase_of(lights) is Phase.NS_GREEN


@pytest.mark.parametrize("direction", ["", "ns", "NORTH", None, 1])
def test_invalid_request_is_rejected_and_ignored(direction):
    lights = lights_in("NS")
    with pytest.raises(ValueError):
        lights.request_phase(direction)
    lights.update(1000)
    assert phase_of(lights) is Phase.NS_GREEN


@pytest.mark.parametrize("dt", [-0.1, math.nan, math.inf])
def test_invalid_dt_is_rejected_and_changes_nothing(dt):
    lights = lights_in("NS")
    before = lights.get_states()
    with pytest.raises(ValueError):
        lights.update(dt)
    assert lights.get_states() == before


def test_zero_dt_pauses():
    lights = lights_in("NS")
    lights.request_phase("EW")
    lights.update(5)
    before = lights.get_states()
    lights.update(0)
    assert lights.get_states() == before


def test_reset_returns_to_safe_start_and_forgets_requests():
    lights = lights_in("NS")
    lights.request_phase("EW")
    lights.reset()
    assert lights.get_states() == TrafficLights(TIMINGS).get_states()
    lights.update(1000)
    assert phase_of(lights) is Phase.ALL_RED


def test_snapshot_cannot_change_the_lamps():
    lights = TrafficLights(TIMINGS)
    lights.get_states().lamps[Approach.NORTH] = Lamp.GREEN
    assert lights.get_states().lamps[Approach.NORTH] is Lamp.RED


@pytest.mark.parametrize("kwargs", [{"min_green_s": -1}, {"yellow_s": 0}, {"all_red_s": 0}])
def test_unsafe_timings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Timings(**kwargs)


@pytest.mark.parametrize("seed", range(20))
def test_safety_holds_under_random_requests_and_dt(seed):
    rng = random.Random(seed)
    lights = TrafficLights(TIMINGS)
    previous = lights.get_states()

    for _ in range(5000):
        if rng.random() < 0.2:
            lights.request_phase(rng.choice(["NS", "EW"]))
        dt = rng.choice([0, 0.05, 0.5, 5, 1000])
        lights.update(dt)
        current = lights.get_states()

        ns_green = Lamp.GREEN in (current.lamps[Approach.NORTH], current.lamps[Approach.SOUTH])
        ew_green = Lamp.GREEN in (current.lamps[Approach.EAST], current.lamps[Approach.WEST])
        assert not (ns_green and ew_green)

        if current.phase is not previous.phase:
            assert (previous.phase, current.phase) in LEGAL_TRANSITIONS
            assert previous.seconds_in_phase + dt >= MIN_DURATION[previous.phase]
        previous = current
