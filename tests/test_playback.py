import pytest

from simulator.simulation import MAX_CATCH_UP_S, TIME_STEP_S, Playback, Simulation

FRAME_S = 1 / 60


def state_of(simulation):
    lanes = simulation.traffic.lanes.values()
    return simulation.lights, [(v.id, v.model, v.distance, v.speed) for lane in lanes for v in lane.vehicles]


def test_pause_runs_no_steps():
    playback = Playback()
    playback.paused = True
    assert playback.steps_due(10) == 0


def test_resuming_does_not_jump_ahead():
    playback = Playback()
    playback.paused = True
    playback.steps_due(10)
    playback.paused = False
    assert playback.steps_due(FRAME_S) <= 1


@pytest.mark.parametrize("speed", [1, 2, 4, 8])
def test_speed_multiplies_simulated_time(speed):
    playback = Playback()
    playback.speed = speed
    steps = sum(playback.steps_due(FRAME_S) for _ in range(600))
    assert steps * TIME_STEP_S == pytest.approx(10 * speed, abs=TIME_STEP_S * 2)


def test_stall_does_not_cause_a_jump():
    playback = Playback()
    assert playback.steps_due(100) * TIME_STEP_S <= MAX_CATCH_UP_S


def test_run_is_the_same_however_it_is_watched():
    straight = Simulation(seed=5)
    for _ in range(6000):
        straight.step()

    watched = Simulation(seed=5)
    playback = Playback()
    steps = 0
    pattern = [(1, False), (8, False), (2, True), (4, False)]
    frame = 0
    while steps < 6000:
        playback.speed, playback.paused = pattern[frame // 50 % len(pattern)]
        for _ in range(playback.steps_due(FRAME_S)):
            if steps < 6000:
                watched.step()
                steps += 1
        frame += 1

    assert state_of(watched) == state_of(straight)


def test_a_new_simulation_shares_nothing_with_an_earlier_one():
    earlier = Simulation(seed=5)
    for _ in range(3000):
        earlier.step()
    restarted, untouched = Simulation(seed=5), Simulation(seed=5)
    for _ in range(600):
        restarted.step()
        untouched.step()
    assert state_of(restarted) == state_of(untouched)
