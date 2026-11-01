"""Runs the controller and the simulation as separate programs talking over MQTT.

    docker compose up -d mosquitto
    python -m messaging.run controller --simulation-id demo
    python -m messaging.run simulation --simulation-id demo --seed 42 --seconds 600

Start the controller first. If it is missing or the broker goes down, the simulation falls back
to its local fixed-time plan and reports when it did.
"""

import argparse
import time
from dataclasses import asdict

from controller.adaptive import Adaptive
from evaluation.metrics import summarize
from messaging.bus import MqttBus
from messaging.services import ControllerService, RemoteStrategy
from simulator.simulation import TIME_STEP_S, Simulation


def run_controller(args) -> None:
    bus = MqttBus("controller", args.host, args.port)
    service = ControllerService(bus, args.simulation_id, Adaptive())
    print(f"controller listening on {args.host}:{args.port} for simulation {args.simulation_id}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"sent {service.sent} requests; rejected {dict(service.guard.rejected)}")
    finally:
        bus.close()


def run_simulation(args) -> None:
    bus = MqttBus("simulation", args.host, args.port)
    if not bus.wait_connected():
        print("no broker yet: starting on the fallback plan")
    strategy = RemoteStrategy(bus, args.simulation_id)
    simulation = Simulation(args.seed, strategy)
    for _ in range(int(args.seconds / TIME_STEP_S)):
        simulation.step()
    bus.close()
    print(f"control modes over time: {strategy.mode_changes}")
    print(f"rejected requests: {dict(strategy.guard.rejected)}")
    print(asdict(summarize(simulation.traffic)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("controller", "simulation"))
    parser.add_argument("--simulation-id", required=True, help="keeps messages from other runs out")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=600, help="simulated seconds")
    args = parser.parse_args()
    (run_controller if args.role == "controller" else run_simulation)(args)


if __name__ == "__main__":
    main()
