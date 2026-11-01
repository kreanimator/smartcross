"""Traffic reaches live control through the cameras alone."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from simulator.simulation import Simulation

ROOT = Path(__file__).parent.parent


def test_a_simulation_needs_a_sensor_to_be_chosen():
    with pytest.raises(TypeError):
        Simulation(1, "adaptive")


def test_the_programs_that_run_the_intersection_never_load_the_exact_counts():
    pytest.importorskip("ultralytics")
    pytest.importorskip("paho.mqtt.client")
    code = (
        "import sys, simulator.world, simulator.viewer, messaging.run, vision.camera_sensor;"
        "print('simulator.perfect_sensor' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_live_window_never_reads_the_simulations_vehicles():
    tree = ast.parse((ROOT / "simulator" / "viewer.py").read_text())
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"vehicles", "lanes"}
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "traffic"
    ]
    assert reads == []
