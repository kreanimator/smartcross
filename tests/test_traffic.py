import pytest

from controller.traffic_lights import Approach, Lamp
from simulator.roads import STOP_LINE
from simulator.simulation import TIME_STEP_S, Simulation
from simulator.traffic import (
    EXIT_DISTANCE,
    MAX_SPEED,
    MIN_GAP,
    SPAWN_DISTANCE,
    STOP_MARGIN,
    VEHICLE_LENGTH,
    Lane,
    Traffic,
    Vehicle,
)

DT = TIME_STEP_S
NORTH_ONLY = {Approach.NORTH: 30, Approach.SOUTH: 0, Approach.EAST: 0, Approach.WEST: 0}


def all_lamps(lamp):
    return {approach: lamp for approach in Approach}


def seconds(duration):
    return int(duration / DT)


def lane_with(*distances, speed=0.0):
    lane = Lane(Approach.NORTH, seed=1, arrivals_per_minute=0)
    lane.vehicles = [
        Vehicle(f"car-{i}", Approach.NORTH, "sedan", distance, speed) for i, distance in enumerate(distances)
    ]
    return lane


def drive(lane, lamp, duration):
    for _ in range(seconds(duration)):
        lane.update(DT, 0, lamp)


def test_vehicle_stops_before_a_red_light():
    lane = lane_with(SPAWN_DISTANCE, speed=MAX_SPEED)
    drive(lane, Lamp.RED, 60)
    vehicle = lane.vehicles[0]
    assert vehicle.speed == pytest.approx(0, abs=0.05)
    assert 0 <= vehicle.distance < STOP_MARGIN + 0.1


def test_vehicle_cannot_cross_when_red_appears_right_in_front_of_it():
    lane = lane_with(1.0, speed=MAX_SPEED)
    drive(lane, Lamp.RED, 5)
    assert lane.vehicles[0].distance >= 0


def test_queue_keeps_its_gaps_and_leaves_on_green():
    lane = lane_with(20, 40, 60, 80, speed=MAX_SPEED)
    drive(lane, Lamp.RED, 60)
    for leader, follower in zip(lane.vehicles, lane.vehicles[1:]):
        gap = follower.distance - leader.distance - VEHICLE_LENGTH
        assert gap == pytest.approx(MIN_GAP, abs=0.2)
    drive(lane, Lamp.GREEN, 60)
    assert lane.vehicles == []


def test_vehicles_leave_the_map_past_the_exit():
    lane = lane_with(0, speed=MAX_SPEED)
    drive(lane, Lamp.GREEN, -EXIT_DISTANCE / MAX_SPEED + 1)
    assert lane.vehicles == []


def test_vehicle_too_close_to_stop_runs_the_yellow():
    lane = lane_with(10, speed=MAX_SPEED)
    drive(lane, Lamp.YELLOW, 2)
    assert lane.vehicles[0].distance < 0


def test_vehicle_with_room_stops_for_yellow_without_changing_its_mind():
    lane = lane_with(50, speed=MAX_SPEED)
    drive(lane, Lamp.YELLOW, 20)
    assert 0 <= lane.vehicles[0].distance < STOP_MARGIN + 0.1


def test_vehicles_past_the_line_ignore_the_lights():
    lane = lane_with(-1, speed=MAX_SPEED)
    drive(lane, Lamp.RED, 1)
    assert lane.vehicles[0].distance < -10


def arrivals(lamp, duration=600):
    """The (id, model) of every vehicle entering the north lane, in order."""
    traffic = Traffic(seed=3, arrivals_per_minute=NORTH_ONLY)
    entered = {}
    for _ in range(seconds(duration)):
        traffic.update(DT, all_lamps(lamp))
        for vehicle in traffic.vehicles:
            entered[vehicle.id] = vehicle.model
    return list(entered.items())


def test_backed_up_lane_delays_arrivals_without_changing_them():
    blocked, free = arrivals(Lamp.RED), arrivals(Lamp.GREEN)
    assert 0 < len(blocked) < len(free)
    assert blocked == free[: len(blocked)]


def test_invalid_dt_is_rejected():
    with pytest.raises(ValueError):
        Traffic().update(0, all_lamps(Lamp.RED))


def state_of(simulation):
    return [(v.id, v.model, v.distance, v.speed) for v in simulation.traffic.vehicles]


def test_same_seed_gives_the_same_run_and_another_seed_does_not():
    first, second, other = Simulation(seed=7), Simulation(seed=7), Simulation(seed=8)
    for _ in range(seconds(120)):
        first.step()
        second.step()
        other.step()
    assert state_of(first) == state_of(second)
    assert state_of(first) != state_of(other)


@pytest.mark.parametrize("seed", range(5))
def test_safety_over_a_long_fixed_timer_run(seed):
    simulation = Simulation(seed)
    before = {}
    crossed = 0

    for _ in range(seconds(1200)):
        lamps = simulation.lights.lamps
        simulation.step()
        now = {}
        for lane in simulation.traffic.lanes.values():
            for leader, follower in zip(lane.vehicles, lane.vehicles[1:]):
                assert follower.distance - leader.distance >= VEHICLE_LENGTH  # No overlap.
            for vehicle in lane.vehicles:
                now[vehicle.id] = vehicle.distance
                if before.get(vehicle.id, -1) >= 0 > vehicle.distance:
                    crossed += 1
                    assert lamps[vehicle.approach] is not Lamp.RED  # No red-light crossing.
                assert vehicle.distance <= before.get(vehicle.id, SPAWN_DISTANCE)  # No reversing.
        before = now
        assert_no_conflict_in_junction(simulation)

    assert crossed > 300  # Traffic really did flow.


def in_junction(vehicle):
    """True when any part of the vehicle is inside the box between the stop lines."""
    return vehicle.distance < 0 and vehicle.distance + VEHICLE_LENGTH > -2 * STOP_LINE


def assert_no_conflict_in_junction(simulation):
    inside = {v.approach for v in simulation.traffic.vehicles if in_junction(v)}
    north_south = inside & {Approach.NORTH, Approach.SOUTH}
    east_west = inside & {Approach.EAST, Approach.WEST}
    assert not (north_south and east_west)
