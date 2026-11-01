"""Messaging over a real Mosquitto, in a throwaway container. Skipped without Docker."""

import shutil
import socket
import subprocess
import time

import pytest

pytest.importorskip("paho.mqtt.client")

from controller.adaptive import Adaptive  # noqa: E402
from controller.traffic_lights import Lamp  # noqa: E402
from evaluation.metrics import summarize  # noqa: E402
from messaging.bus import MqttBus  # noqa: E402
from messaging.schemas import TOPIC_HEALTH  # noqa: E402
from messaging.services import ControllerService, RemoteStrategy  # noqa: E402
from simulator.perfect_sensor import PerfectSensor
from simulator.simulation import TIME_STEP_S, Simulation  # noqa: E402

IMAGE = "eclipse-mosquitto:2.0.21"
RUN = "broker-test"


def docker(*args) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"no broker on port {port}")


@pytest.fixture
def broker():
    if not shutil.which("docker"):
        pytest.skip("no Docker")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    try:
        container = docker(
            "run", "-d", "-p", f"127.0.0.1:{port}:1883", IMAGE, "mosquitto", "-c", "/mosquitto-no-auth.conf"
        )
    except (subprocess.SubprocessError, OSError) as error:
        pytest.skip(f"cannot start a broker: {error}")
    wait_for_port(port)
    yield container, port
    docker("rm", "-f", container)


@pytest.fixture
def linked(broker):
    container, port = broker
    controller_bus = MqttBus("controller", port=port)
    simulation_bus = MqttBus("simulation", port=port)
    assert controller_bus.wait_connected() and simulation_bus.wait_connected()
    service = ControllerService(controller_bus, RUN, Adaptive())
    time.sleep(0.2)  # Lets the broker register the subscriptions.
    yield container, port, service, simulation_bus, controller_bus
    controller_bus.close()
    simulation_bus.close()


def test_a_run_over_a_real_broker_matches_the_direct_run_exactly(linked):
    _, _, service, bus, _ = linked
    strategy = RemoteStrategy(bus, RUN)
    simulation = Simulation(5, strategy, sensor=PerfectSensor())
    direct = Simulation(5, Adaptive(), sensor=PerfectSensor())
    for _ in range(round(300 / TIME_STEP_S)):
        simulation.step()
        direct.step()
        assert simulation.lights == direct.lights
    assert summarize(simulation.traffic) == summarize(direct.traffic)
    assert [mode for _, mode in strategy.mode_changes] == ["controller"]
    assert service.sent == 301


def test_losing_the_broker_stays_safe_and_control_comes_back(linked):
    container, port, _, bus, controller_bus = linked
    strategy = RemoteStrategy(bus, RUN, reply_timeout_s=2)
    simulation = Simulation(6, strategy, sensor=PerfectSensor())

    def advance(seconds):
        phases = set()
        for _ in range(round(seconds / TIME_STEP_S)):
            simulation.step()
            lit = {a for a, lamp in simulation.lights.lamps.items() if lamp is not Lamp.RED}
            assert len({a.value in ("north", "south") for a in lit}) <= 1
            phases.add(simulation.lights.phase)
        return phases

    advance(60)
    assert strategy.mode == "controller"

    docker("kill", container)
    deadline = time.monotonic() + 10
    while (bus.connected or controller_bus.connected) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not bus.connected and not controller_bus.connected

    phases = advance(120)
    assert strategy.mode == "fallback"
    assert {p.value for p in phases} >= {"NS_GREEN", "EW_GREEN"}

    docker("start", container)
    wait_for_port(port)
    assert bus.wait_connected(15) and controller_bus.wait_connected(15)
    time.sleep(0.3)  # Lets both clients subscribe again.
    advance(10)
    assert strategy.mode == "controller"
    assert [mode for _, mode in strategy.mode_changes] == ["controller", "holding", "fallback", "controller"]


def test_a_dead_connection_is_reported_by_the_broker(broker):
    _, port = broker
    heard = []
    monitor = MqttBus("monitor", port=port)
    assert monitor.wait_connected()
    monitor.subscribe(TOPIC_HEALTH, heard.append)
    time.sleep(0.2)

    victim = MqttBus("victim", port=port)
    assert victim.wait_connected()
    deadline = time.monotonic() + 5
    while not any(b'"online"' in m and b"victim" in m for m in heard) and time.monotonic() < deadline:
        time.sleep(0.01)
    victim.client.loop_stop()
    victim.client.socket().close()  # Dies without saying goodbye.

    deadline = time.monotonic() + 10
    while not any(b'"offline"' in m and b"victim" in m for m in heard) and time.monotonic() < deadline:
        time.sleep(0.01)
    monitor.close()
    assert any(b'"offline"' in m and b"victim" in m for m in heard)
