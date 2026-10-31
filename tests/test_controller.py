from controller.controller import TrafficController
from controller.fixed_timer import FixedTimer
from controller.traffic_lights import Phase, Timings, TrafficLights


def test_fixed_timer_alternates_each_period():
    timer = FixedTimer(period_s=25)
    assert [timer.decide(t) for t in (0, 24.9, 25, 49.9, 50)] == ["NS", "NS", "EW", "EW", "NS"]


class AlwaysFlips:
    """A misbehaving strategy that asks for the other direction on every call."""

    def __init__(self):
        self.calls = 0

    def decide(self, now_s):
        self.calls += 1
        return "NS" if self.calls % 2 else "EW"


def test_strategy_cannot_bypass_the_safety_timings():
    timings = Timings(min_green_s=10, yellow_s=3, all_red_s=2)
    controller = TrafficController(AlwaysFlips(), TrafficLights(timings))
    phases = [controller.update(0.1).phase for _ in range(1000)]
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
    phases = {controller.update(0.1).phase for _ in range(1000)}
    assert phases == set(Phase)
