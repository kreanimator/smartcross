from controller.controller import TrafficController
from controller.fixed_timer import FixedTimer
from controller.observation import TrafficObservation
from controller.traffic_lights import Phase, Timings, TrafficLights


def observation_at(now_s):
    return TrafficObservation(now_s, "test", {})


def test_fixed_timer_alternates_each_period():
    timer = FixedTimer()  # 20 s of green plus 5 s of clearance for each direction.
    times = (0, 24.9, 25, 49.9, 50)
    assert [timer.decide(observation_at(t), None) for t in times] == ["NS", "NS", "EW", "EW", "NS"]


def test_fixed_timer_can_favour_one_direction():
    timer = FixedTimer(green_ns_s=10, green_ew_s=35)  # NS window 15 s, EW window 40 s.
    times = (0, 14.9, 15, 54.9, 55)
    assert [timer.decide(observation_at(t), None) for t in times] == ["NS", "NS", "EW", "EW", "NS"]


class AlwaysFlips:
    """A misbehaving strategy that asks for the other direction on every call."""

    def __init__(self):
        self.calls = 0

    def decide(self, observation, lights):
        self.calls += 1
        return "NS" if self.calls % 2 else "EW"


def test_strategy_cannot_bypass_the_safety_timings():
    timings = Timings(min_green_s=10, yellow_s=3, all_red_s=2)
    controller = TrafficController(AlwaysFlips(), TrafficLights(timings))
    phases = [controller.update(0.1, observation_at(0)).phase for _ in range(1000)]
    changes = [b for a, b in zip(phases, phases[1:]) if a is not b]
    for phase, next_phase in zip(changes, changes[1:]):
        if phase is Phase.NS_GREEN:
            assert next_phase is Phase.NS_YELLOW
        if phase is Phase.EW_GREEN:
            assert next_phase is Phase.EW_YELLOW
        if phase in (Phase.NS_YELLOW, Phase.EW_YELLOW):
            assert next_phase is Phase.ALL_RED
    assert changes  # The lights did change.


def test_fixed_timer_run_gives_each_direction_green():
    controller = TrafficController(FixedTimer())
    phases = {controller.update(0.1, observation_at(i / 10)).phase for i in range(1000)}
    assert phases == set(Phase)


def test_when_observations_stop_a_fixed_plan_takes_over():
    controller = TrafficController(AlwaysFlips())
    controller.update(0, observation_at(0))
    greens = set()
    for _ in range(int(120 / 0.05)):
        snapshot = controller.update(0.05)
        if snapshot.phase in (Phase.NS_GREEN, Phase.EW_GREEN):
            greens.add(snapshot.phase)
            assert snapshot.seconds_in_phase < 60
    assert greens == {Phase.NS_GREEN, Phase.EW_GREEN}


def test_a_short_gap_in_observations_changes_nothing():
    strategy = AlwaysFlips()
    controller = TrafficController(strategy)
    controller.update(0, observation_at(0))
    fallback_calls = []
    controller.fallback.decide = lambda observation, lights: fallback_calls.append(1) or "NS"
    for _ in range(int(4.9 / 0.05)):
        controller.update(0.05)
    assert fallback_calls == []


class Stuck:
    def __init__(self, direction="NS"):
        self.direction = direction

    def decide(self, observation, lights):
        return self.direction


def run_controller(controller, seconds, start_s=0.0, dt=0.05):
    """Observations arrive once a second. Returns the phases shown, in order, without repeats."""
    phases = []
    for step in range(round(seconds / dt)):
        now_s = start_s + step * dt
        observation = observation_at(now_s) if step % 20 == 0 else None
        phase = controller.update(dt, observation).phase
        if not phases or phases[-1] is not phase:
            phases.append(phase)
    return phases


def test_a_stuck_strategy_loses_control_to_the_fixed_plan():
    controller = TrafficController(Stuck())
    run_controller(controller, 130)
    assert controller.lights.forced_ends == 1
    phases = run_controller(controller, 200, start_s=130)
    assert controller.lights.forced_ends == 1  # The fixed plan never reaches the limit.
    assert phases.count(Phase.NS_GREEN) >= 3
    assert phases.count(Phase.EW_GREEN) >= 3


def test_a_strategy_regains_control_by_asking_for_something_else():
    strategy = Stuck()
    controller = TrafficController(strategy)
    run_controller(controller, 130)
    assert controller.distrusted_request == "NS"
    strategy.direction = "EW"
    run_controller(controller, 40, start_s=130)
    assert controller.distrusted_request is None
    assert controller.lights.get_states().phase is Phase.EW_GREEN
    strategy.direction = "NS"
    run_controller(controller, 40, start_s=170)
    assert controller.lights.get_states().phase is Phase.NS_GREEN
