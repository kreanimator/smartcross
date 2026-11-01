import subprocess
import sys

import pytest

from controller.adaptive import (
    MAX_QUEUE,
    MAX_WAIT_S,
    MIN_GREEN_BEFORE_INTERRUPT_S,
    OUTGROWN_MARGIN,
    SATURATED_VEHICLES,
    Adaptive,
    pressure,
)
from controller.observation import ApproachObservation, TrafficObservation
from controller.traffic_lights import LAMPS, Approach, LightSnapshot, Phase
from evaluation.compare import rates
from evaluation.metrics import summarize
from simulator.perfect_sensor import observe
from simulator.simulation import TIME_STEP_S, Simulation
from simulator.traffic import STOPPED_SPEED, Traffic, Vehicle


def observation(north=0, east=0, wait=0.0):
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(queued=north, oldest_wait_s=wait if north else 0.0)
    seen[Approach.EAST] = ApproachObservation(queued=east, oldest_wait_s=wait if east else 0.0)
    return TrafficObservation(100.0, "test", seen)


def green_ns(seconds):
    return LightSnapshot(Phase.NS_GREEN, {}, seconds)


def on_ns():
    return Adaptive(max_green_s=45)


def test_pressure_counts_queues_moving_vehicles_and_waiting():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(queued=3, approaching=2, oldest_wait_s=20)
    seen[Approach.SOUTH] = ApproachObservation(queued=1, oldest_wait_s=5)
    assert pressure(TrafficObservation(0, "test", seen), "NS") == 3 + 1 + 2 * 0.5 + 20 / 10


def test_holds_green_while_its_queue_clears():
    strategy = on_ns()
    assert strategy.decide(observation(north=2, east=5), green_ns(15)) == "NS"
    assert strategy.decisions == []


def test_holds_green_while_vehicles_are_about_to_reach_the_line():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(approaching=1, near_line=1)
    seen[Approach.EAST] = ApproachObservation(queued=3)
    strategy = on_ns()
    assert strategy.decide(TrafficObservation(100.0, "test", seen), green_ns(15)) == "NS"


def test_switches_once_cleared_if_the_other_side_has_more():
    strategy = on_ns()
    assert strategy.decide(observation(east=1), green_ns(15)) == "EW"
    assert strategy.decisions[-1].reason == "queue cleared"


def test_far_away_vehicles_do_not_count_as_clearing():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(approaching=1)
    seen[Approach.EAST] = ApproachObservation(queued=1)
    strategy = on_ns()
    assert strategy.decide(TrafficObservation(100.0, "test", seen), green_ns(15)) == "EW"
    assert strategy.decisions[-1].reason == "queue cleared"


def test_long_wait_interrupts_a_clearing_queue():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(queued=SATURATED_VEHICLES - 1)
    seen[Approach.EAST] = ApproachObservation(queued=1, oldest_wait_s=MAX_WAIT_S)
    strategy = on_ns()
    assert strategy.decide(TrafficObservation(100.0, "test", seen), green_ns(15)) == "EW"
    assert strategy.decisions[-1].reason == "long wait"


def test_nothing_interrupts_clearing_before_the_green_has_earned_it():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(queued=1)
    seen[Approach.EAST] = ApproachObservation(queued=MAX_QUEUE + 5, oldest_wait_s=MAX_WAIT_S + 5)
    strategy = on_ns()
    too_soon = green_ns(MIN_GREEN_BEFORE_INTERRUPT_S - 1)
    assert strategy.decide(TrafficObservation(100.0, "test", seen), too_soon) == "NS"
    assert strategy.decide(TrafficObservation(101.0, "test", seen), green_ns(MIN_GREEN_BEFORE_INTERRUPT_S)) == "EW"


def test_a_cleared_green_can_still_end_before_the_gate():
    strategy = on_ns()
    assert strategy.decide(observation(east=1), green_ns(MIN_GREEN_BEFORE_INTERRUPT_S - 1)) == "EW"
    assert strategy.decisions[-1].reason == "queue cleared"


def test_long_queue_interrupts_a_clearing_queue():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(queued=2, approaching=2)
    seen[Approach.SOUTH] = ApproachObservation(queued=3, approaching=2)
    seen[Approach.EAST] = ApproachObservation(queued=MAX_QUEUE)
    strategy = on_ns()
    assert strategy.decide(TrafficObservation(100.0, "test", seen), green_ns(15)) == "EW"
    assert strategy.decisions[-1].reason == "long queue"


def test_other_side_outgrowing_this_one_interrupts_clearing():
    strategy = on_ns()
    mine = 1
    assert strategy.decide(observation(north=mine, east=mine + OUTGROWN_MARGIN + 1), green_ns(15)) == "EW"
    assert strategy.decisions[-1].reason == "outgrown"


def test_moving_vehicles_count_fully_when_deciding_to_interrupt_clearing():
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(approaching=8, near_line=3)
    seen[Approach.SOUTH] = ApproachObservation(approaching=8, near_line=3)
    seen[Approach.EAST] = ApproachObservation(queued=9)
    seen[Approach.WEST] = ApproachObservation(queued=9)
    strategy = on_ns()
    assert strategy.decide(TrafficObservation(100.0, "test", seen), green_ns(15)) == "NS"


def test_just_under_the_outgrown_margin_keeps_clearing():
    strategy = on_ns()
    assert strategy.decide(observation(north=1, east=1 + OUTGROWN_MARGIN), green_ns(15)) == "NS"


def test_does_not_decide_during_minimum_green():
    strategy = on_ns()
    assert strategy.decide(observation(east=20), green_ns(5)) == "NS"
    assert strategy.decisions == []


def test_max_green_ends_a_busy_green_when_anyone_is_waiting():
    strategy = on_ns()
    assert strategy.decide(observation(north=20, east=1), green_ns(45)) == "EW"
    assert strategy.decisions[-1].reason == "max green"


def test_green_holds_past_max_when_nobody_else_is_there():
    assert on_ns().decide(observation(north=20), green_ns(1000)) == "NS"


def test_target_stays_put_during_yellow():
    strategy = on_ns()
    strategy.decide(observation(east=1), green_ns(15))
    yellow = LightSnapshot(Phase.NS_YELLOW, {}, 1)
    assert strategy.decide(observation(north=20), yellow) == "EW"


def test_decision_log_explains_the_switch():
    strategy = on_ns()
    strategy.decide(observation(east=6), green_ns(15))
    decision = strategy.decisions[-1]
    assert (decision.time_s, decision.direction) == (100.0, "EW")
    assert decision.pressure_ew > decision.pressure_ns


def test_perfect_sensor_reports_only_vehicles_before_the_line():
    traffic = Traffic(seed=1, arrivals_per_minute=rates(0, 0))
    lane = traffic.lanes[Approach.NORTH]
    lane.vehicles = [
        Vehicle("past", Approach.NORTH, "sedan", -5, speed=10),
        Vehicle("queued", Approach.NORTH, "sedan", 2, speed=STOPPED_SPEED / 2, stopped_s=30),
        Vehicle("moving", Approach.NORTH, "sedan", 40, speed=10),
    ]
    assert observe(traffic).approaches[Approach.NORTH].near_line == 0
    lane.vehicles[-1].distance = 20
    assert observe(traffic).approaches[Approach.NORTH].near_line == 1
    lane.vehicles[-1].distance = 40
    seen = observe(traffic).approaches[Approach.NORTH]
    assert (seen.queued, seen.approaching, seen.near_line, seen.oldest_wait_s) == (1, 1, 0, 30)
    assert observe(traffic).approaches[Approach.SOUTH] == ApproachObservation()


def test_invalid_observation_is_rejected():
    with pytest.raises(ValueError):
        ApproachObservation(queued=-1)


def run(simulation, seconds):
    for _ in range(int(seconds / TIME_STEP_S)):
        simulation.step()
    return simulation


def test_quiet_road_is_not_starved_by_a_busy_one():
    simulation = run(Simulation(1, "adaptive", rates(20, 1)), 1200)
    east_west = [v for a in (Approach.EAST, Approach.WEST) for v in simulation.traffic.lanes[a].completed]
    assert len(east_west) > 20
    # Max green, then clearance: nobody on the quiet road should wait much beyond that.
    assert max(v.stopped_s for v in east_west) < 45 + 3 + 2 + 15


def test_every_green_respects_max_green_when_both_sides_are_busy():
    simulation = Simulation(2, "adaptive", rates(25, 25))
    longest = 0.0
    for _ in range(int(1200 / TIME_STEP_S)):
        simulation.step()
        if simulation.lights.phase in (Phase.NS_GREEN, Phase.EW_GREEN):
            longest = max(longest, simulation.lights.seconds_in_phase)
    limit = Adaptive().max_green_s
    assert 10 <= longest <= limit + 1 + 2 * TIME_STEP_S  # One second is the observation interval.


def test_adaptive_run_is_reproducible_and_logs_switches():
    first, second = run(Simulation(9, "adaptive"), 300), run(Simulation(9, "adaptive"), 300)
    assert first.controller.strategy.decisions == second.controller.strategy.decisions
    assert first.controller.strategy.decisions


def test_both_strategies_get_identical_arrivals():
    fixed, adaptive = run(Simulation(4, "fixed"), 600), run(Simulation(4, "adaptive"), 600)
    for approach in Approach:
        fixed_lane, adaptive_lane = fixed.traffic.lanes[approach], adaptive.traffic.lanes[approach]
        assert fixed_lane.spawned == adaptive_lane.spawned
        assert fixed_lane.next_arrival_s == adaptive_lane.next_arrival_s


def test_time_lost_counts_slowing_down_as_well_as_stopping():
    summary = summarize(run(Simulation(1, "fixed"), 600).traffic)
    assert summary.mean_time_lost_s > summary.mean_delay_s


def test_delay_includes_time_kept_out_of_a_full_lane():
    simulation = run(Simulation(1, "fixed", rates(40, 40)), 600)
    summary = summarize(simulation.traffic)
    late = [v for lane in simulation.traffic.lanes.values() for v in lane.completed if v.entry_delay_s > 0]
    assert late
    assert summary.unfinished > 0
    assert summary.worst_wait_s >= max(v.delay_s for v in late)


def test_simulation_logic_does_not_load_the_renderer():
    code = "import sys, evaluation.compare; sys.exit('panda3d' in sys.modules or 'ursina' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_a_blind_approach_is_not_read_as_an_empty_road():
    """With east and west unseen, their zeros must not hold north-south green for ever."""
    strategy = Adaptive()
    seen = {
        Approach.NORTH: ApproachObservation(queued=3),
        Approach.SOUTH: ApproachObservation(queued=3),
        Approach.EAST: ApproachObservation(available=False),
        Approach.WEST: ApproachObservation(available=False),
    }
    lights = LightSnapshot(Phase.NS_GREEN, dict(LAMPS[Phase.NS_GREEN]), 30.0)
    assert strategy.decide(TrafficObservation(25.0, "cv", seen), lights) == "EW"
    assert strategy.decisions[-1].reason == "camera unavailable"
    assert strategy.decide(TrafficObservation(50.0, "cv", seen), lights) == "NS"


def test_control_returns_once_every_approach_is_seen():
    strategy = Adaptive()
    blind = {approach: ApproachObservation(available=False) for approach in Approach}
    clear = {approach: ApproachObservation() for approach in Approach}
    clear[Approach.EAST] = ApproachObservation(queued=4)
    lights = LightSnapshot(Phase.NS_GREEN, dict(LAMPS[Phase.NS_GREEN]), 30.0)
    assert strategy.decide(TrafficObservation(0.0, "cv", blind), lights) == "NS"
    assert strategy.decide(TrafficObservation(1.0, "cv", clear), lights) == "EW"
    assert strategy.decisions[-1].reason == "queue cleared"


def saturated_ns(**east):
    seen = {approach: ApproachObservation() for approach in Approach}
    seen[Approach.NORTH] = ApproachObservation(queued=SATURATED_VEHICLES)
    seen[Approach.EAST] = ApproachObservation(**east)
    return TrafficObservation(100.0, "test", seen)


@pytest.mark.parametrize(
    "east",
    [
        {"queued": 1, "oldest_wait_s": MAX_WAIT_S},
        {"queued": MAX_QUEUE},
        {"queued": SATURATED_VEHICLES + OUTGROWN_MARGIN + 1},
    ],
    ids=["long wait", "long queue", "outgrown"],
)
def test_nothing_interrupts_a_green_with_many_vehicles_left_to_clear(east):
    assert on_ns().decide(saturated_ns(**east), green_ns(30)) == "NS"


def test_max_green_still_ends_a_saturated_green():
    strategy = on_ns()
    assert strategy.decide(saturated_ns(queued=1), green_ns(strategy.max_green_s)) == "EW"
    assert strategy.decisions[-1].reason == "max green"


def test_overload_keeps_greens_long_instead_of_spending_them_on_switching():
    simulation = Simulation(2, "adaptive", rates(14, 14))
    switches = 0
    before = simulation.lights.phase
    for _ in range(int(1200 / TIME_STEP_S)):
        simulation.step()
        now = simulation.lights.phase
        switches += now is not before and now in (Phase.NS_GREEN, Phase.EW_GREEN)
        before = now
    assert switches < 1200 / 30  # It was one green every 21 s before.
