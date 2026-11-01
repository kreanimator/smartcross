"""Messaging logic on an in-memory bus, including every fault it can be made to suffer."""

import json
import random

import pytest

from controller.adaptive import Adaptive
from controller.observation import ApproachObservation, TrafficObservation
from controller.traffic_lights import LAMPS, Approach, Lamp, LightSnapshot, Phase, Timings
from evaluation.metrics import summarize
from messaging.bus import InMemoryBroker, InMemoryBus
from messaging.schemas import (
    SCHEMA_VERSION,
    TOPIC_PHASE_REQUEST,
    InvalidMessage,
    PhaseRequest,
    Received,
    decode_lights,
    decode_observation,
    decode_phase_request,
    encode_lights,
    encode_observation,
    encode_phase_request,
)
from messaging.services import STALE_AFTER_S, ControllerService, MessageGuard, RemoteStrategy
from simulator.simulation import TIME_STEP_S, Simulation

RUN = "test-run"
LEGAL_NEXT = {
    Phase.ALL_RED: {Phase.NS_GREEN, Phase.EW_GREEN},
    Phase.NS_GREEN: {Phase.NS_YELLOW},
    Phase.EW_GREEN: {Phase.EW_YELLOW},
    Phase.NS_YELLOW: {Phase.ALL_RED},
    Phase.EW_YELLOW: {Phase.ALL_RED},
}
GREENS = {Phase.NS_GREEN, Phase.EW_GREEN}


def observation(time_s=1.0):
    seen = {a: ApproachObservation(queued=2, approaching=1, near_line=1, oldest_wait_s=3.5) for a in Approach}
    seen[Approach.EAST] = ApproachObservation(average_speed_mps=4.25)
    seen[Approach.WEST] = ApproachObservation(available=False)
    return TrafficObservation(time_s, "perfect_sensor", seen)


def snapshot(phase=Phase.NS_GREEN, seconds=2.5):
    return LightSnapshot(phase, dict(LAMPS[phase]), seconds)


def edited(payload: bytes, change) -> bytes:
    message = json.loads(payload)
    change(message)
    return json.dumps(message).encode()


def connected(seed=1, deliver=None):
    """A controller and a simulation joined by an in-memory broker."""
    broker = InMemoryBroker()
    service = ControllerService(InMemoryBus(broker), RUN, Adaptive())
    strategy = RemoteStrategy(InMemoryBus(broker), RUN, reply_timeout_s=0, recovery_wait_s=0)
    if deliver:
        broker.deliver = deliver
    return broker, service, strategy, Simulation(seed, strategy)


class SafetyWatch:
    """Fails on anything unsafe the lights show, whatever messages reached them."""

    def __init__(self, simulation):
        self.simulation = simulation
        self.lights = simulation.lights
        self.greens = 0
        self.longest_green_s = 0.0
        self.timings = Timings()

    def step(self) -> None:
        self.simulation.step()
        now, before = self.simulation.lights, self.lights
        lit = {a for a, lamp in now.lamps.items() if lamp is not Lamp.RED}
        assert not (lit & {Approach.NORTH, Approach.SOUTH} and lit & {Approach.EAST, Approach.WEST})
        if now.phase is not before.phase:
            assert now.phase in LEGAL_NEXT[before.phase]
            if before.phase in GREENS:
                # The snapshot is from the step before the change, so that step still counts.
                assert before.seconds_in_phase + TIME_STEP_S >= self.timings.min_green_s - 1e-9
            if now.phase in GREENS:
                self.greens += 1
        if now.phase in GREENS:
            self.longest_green_s = max(self.longest_green_s, now.seconds_in_phase)
        self.lights = now

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / TIME_STEP_S)):
            self.step()


def test_observation_survives_the_round_trip():
    received = decode_observation(encode_observation(observation(), RUN))
    assert received.simulation_id == RUN
    assert received.timestamp_sim_s == 1.0
    assert received.body == observation()


def test_lights_survive_the_round_trip():
    received = decode_lights(encode_lights(snapshot(), 7.5, RUN))
    assert received.timestamp_sim_s == 7.5
    assert received.body == snapshot()


def test_phase_request_survives_the_round_trip():
    request = PhaseRequest(3.0, "EW", 9)
    assert decode_phase_request(encode_phase_request(request, RUN)).body == request


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema_version=SCHEMA_VERSION + 1),
        lambda m: m.pop("schema_version"),
        lambda m: m.pop("simulation_id"),
        lambda m: m.update(timestamp_sim_s=-1),
        lambda m: m.update(timestamp_sim_s="soon"),
        lambda m: m.update(timestamp_sim_s=True),
        lambda m: m.pop("source"),
        lambda m: m["approaches"].pop("north"),
        lambda m: m["approaches"].update(diagonal={}),
        lambda m: m["approaches"].update(north=3),
        lambda m: m["approaches"]["north"].update(queued=-1),
        lambda m: m["approaches"]["north"].update(queued=1.5),
        lambda m: m["approaches"]["north"].pop("near_line"),
        lambda m: m["approaches"]["north"].update(oldest_wait_s=-0.1),
        lambda m: m["approaches"]["north"].update(average_speed_mps="fast"),
        lambda m: m["approaches"]["north"].update(available="yes"),
        lambda m: m["approaches"]["north"].pop("available"),
    ],
)
def test_malformed_observations_are_rejected(change):
    with pytest.raises(InvalidMessage):
        decode_observation(edited(encode_observation(observation(), RUN), change))


@pytest.mark.parametrize("payload", [b"", b"not json", b"[1, 2]", b"null", b'{"timestamp_sim_s": NaN}', b"\xff\xfe"])
def test_garbage_is_rejected(payload):
    for decode in (decode_observation, decode_lights, decode_phase_request):
        with pytest.raises(InvalidMessage):
            decode(payload)


def test_infinite_numbers_are_rejected():
    payload = encode_observation(observation(), RUN).replace(b'"oldest_wait_s": 3.5', b'"oldest_wait_s": Infinity')
    with pytest.raises(InvalidMessage):
        decode_observation(payload)


@pytest.mark.parametrize(
    "change",
    [lambda m: m.update(phase="ALL_GREEN"), lambda m: m["lamps"].update(east="GREEN"), lambda m: m.pop("lamps")],
)
def test_lights_that_contradict_themselves_are_rejected(change):
    with pytest.raises(InvalidMessage):
        decode_lights(edited(encode_lights(snapshot(), 1.0, RUN), change))


@pytest.mark.parametrize("direction", ["NORTH", "", None, 5, ["NS"], {"NS": 1}])
def test_unknown_directions_are_rejected(direction):
    payload = edited(encode_phase_request(PhaseRequest(1.0, "NS", 1), RUN), lambda m: m.update(direction=direction))
    with pytest.raises(InvalidMessage):
        decode_phase_request(payload)


def message_at(time_s, simulation_id=RUN):
    return Received(simulation_id, time_s, None)


def test_guard_rejects_repeats_and_older_messages():
    guard = MessageGuard(RUN)
    assert guard.accept(message_at(2.0))
    assert not guard.accept(message_at(2.0))
    assert not guard.accept(message_at(1.0))
    assert guard.accept(message_at(3.0))
    assert guard.rejected == {"repeated or out of order": 2}


def test_guard_rejects_other_runs_without_losing_its_place():
    guard = MessageGuard(RUN)
    assert not guard.accept(message_at(100.0, "another-run"))
    assert guard.accept(message_at(1.0))


def test_guard_rejects_stale_and_future_messages():
    guard = MessageGuard(RUN, max_age_s=5)
    assert not guard.accept(message_at(1.0), now_s=7.0)
    assert not guard.accept(message_at(9.0), now_s=7.0)
    assert guard.accept(message_at(2.0), now_s=7.0)
    assert guard.rejected == {"stale": 1, "from the future": 1}


def test_a_healthy_bus_changes_nothing():
    """The whole point of the design: MQTT adds no change to the control rules."""
    _, _, strategy, simulation = connected(seed=3)
    direct = Simulation(3, Adaptive())
    for _ in range(round(600 / TIME_STEP_S)):
        simulation.step()
        direct.step()
        assert simulation.lights == direct.lights
    assert summarize(simulation.traffic) == summarize(direct.traffic)
    assert [mode for _, mode in strategy.mode_changes] == ["controller"]


def test_with_no_controller_the_lights_run_the_fallback_plan():
    strategy = RemoteStrategy(InMemoryBus(InMemoryBroker()), RUN, reply_timeout_s=0, recovery_wait_s=0)
    watch = SafetyWatch(Simulation(1, strategy))
    watch.run(300)
    assert [mode for _, mode in strategy.mode_changes] == ["fallback"]
    assert watch.greens >= 2 * 300 // 50
    assert watch.longest_green_s < 25


def test_a_broker_outage_holds_then_falls_back_then_recovers():
    broker, _, strategy, simulation = connected()
    watch = SafetyWatch(simulation)
    watch.run(100)
    assert strategy.mode == "controller"

    broker.up = False
    watch.run(1)
    assert strategy.mode == "holding"
    watch.run(STALE_AFTER_S + 1)
    assert strategy.mode == "fallback"
    greens = watch.greens
    watch.run(200)
    assert watch.greens > greens + 4  # Both directions keep being served with nothing connected.

    broker.up = True
    watch.run(2)
    assert strategy.mode == "controller"
    watch.run(100)
    assert [mode for _, mode in strategy.mode_changes] == ["controller", "holding", "fallback", "controller"]


def test_requests_alone_going_missing_is_handled_the_same_way():
    broker, _, strategy, simulation = connected()
    watch = SafetyWatch(simulation)
    watch.run(50)
    broker.deliver = lambda topic, payload: [] if topic == TOPIC_PHASE_REQUEST else [payload]
    watch.run(STALE_AFTER_S + 2)
    assert strategy.mode == "fallback"


def chaos(seed):
    """Drops, repeats and delays messages at random, so they also arrive out of order."""
    rng = random.Random(seed)
    held = []

    def deliver(_topic, payload):
        delivered = []
        roll = rng.random()
        if roll < 0.2:
            pass
        elif roll < 0.4:
            delivered += [payload, payload]
        elif roll < 0.6:
            held.append(payload)
        else:
            delivered.append(payload)
        if held and rng.random() < 0.3:
            delivered.append(held.pop(rng.randrange(len(held))))
        return delivered

    return deliver


@pytest.mark.parametrize("seed", range(5))
def test_lost_repeated_and_reordered_messages_stay_safe(seed):
    per_topic = {}  # So a held message is only ever released on its own topic.

    def deliver(topic, payload):
        return per_topic.setdefault(topic, chaos(seed))(topic, payload)

    _, _, strategy, simulation = connected(seed=seed, deliver=deliver)
    watch = SafetyWatch(simulation)
    watch.run(600)
    assert watch.greens > 5
    assert watch.longest_green_s < 120
    assert "repeated or out of order" in strategy.guard.rejected


def test_a_flood_of_conflicting_requests_cannot_beat_the_state_machine():
    broker, _, strategy, simulation = connected()
    watch = SafetyWatch(simulation)
    bus = InMemoryBus(broker)
    for index in range(round(120 / TIME_STEP_S)):
        now_s = simulation.traffic.now_s
        direction = "NS" if index % 2 else "EW"
        bus.publish(TOPIC_PHASE_REQUEST, encode_phase_request(PhaseRequest(now_s, direction, index), RUN))
        watch.step()


def test_garbage_on_every_topic_changes_nothing():
    broker, service, strategy, simulation = connected(seed=2)
    direct = Simulation(2, Adaptive())
    bus = InMemoryBus(broker)
    for _ in range(round(120 / TIME_STEP_S)):
        for topic in list(broker.subscribers):
            bus.publish(topic, b"garbage")
        simulation.step()
        direct.step()
        assert simulation.lights == direct.lights
    assert strategy.guard.rejected["invalid"] > 0
    assert service.guard.rejected["invalid"] > 0
    assert service.lights_guard.rejected["invalid"] > 0


def test_messages_from_another_run_are_ignored():
    broker, _, strategy, simulation = connected(seed=2)
    direct = Simulation(2, Adaptive())
    bus = InMemoryBus(broker)
    for index in range(round(60 / TIME_STEP_S)):
        request = PhaseRequest(simulation.traffic.now_s + TIME_STEP_S, "EW", index)
        bus.publish(TOPIC_PHASE_REQUEST, encode_phase_request(request, "another-run"))
        simulation.step()
        direct.step()
        assert simulation.lights == direct.lights
    assert strategy.guard.rejected["other simulation"] > 0


def test_a_reply_that_arrives_too_late_is_not_used():
    broker = InMemoryBroker()
    strategy = RemoteStrategy(InMemoryBus(broker), RUN, reply_timeout_s=0, recovery_wait_s=0)
    late = PhaseRequest(1.0, "EW", 1)
    InMemoryBus(broker).publish(TOPIC_PHASE_REQUEST, encode_phase_request(late, RUN))
    strategy.decide(observation(1.0 + STALE_AFTER_S + 1), snapshot())
    assert strategy.last_request is None
    assert strategy.guard.rejected["stale"] == 1
    assert strategy.mode == "fallback"
