"""Measures a finished run from ground truth, as defined in docs/TESTING.md."""

from dataclasses import dataclass

import numpy as np

from controller.traffic_lights import Approach
from simulator.traffic import Traffic

NS = (Approach.NORTH, Approach.SOUTH)
EW = (Approach.EAST, Approach.WEST)


@dataclass(frozen=True)
class Summary:
    completed: int
    unfinished: int  # Still on the map or waiting to enter when the run ended.
    throughput_per_hour: float
    mean_time_lost_s: float  # Includes slowing down, so frequent stop-and-go cannot hide.
    mean_delay_s: float
    mean_delay_ns_s: float
    mean_delay_ew_s: float
    p95_delay_s: float
    worst_wait_s: float  # Includes unfinished vehicles, so starvation cannot hide.


def _mean_delay(traffic: Traffic, approaches) -> float:
    delays = [v.delay_s for approach in approaches for v in traffic.lanes[approach].completed]
    return float(np.mean(delays)) if delays else 0.0


def summarize(traffic: Traffic) -> Summary:
    lanes = traffic.lanes.values()
    completed = [v for lane in lanes for v in lane.completed]
    on_map = [v for lane in lanes for v in lane.vehicles]
    outside = [v for lane in lanes for v in lane.waiting_to_enter]

    delays = [v.delay_s for v in completed]
    unfinished_waits = [v.delay_s for v in on_map] + [traffic.now_s - v.scheduled_s for v in outside]
    return Summary(
        completed=len(completed),
        unfinished=len(on_map) + len(outside),
        throughput_per_hour=len(completed) / (traffic.now_s / 3600),
        mean_time_lost_s=float(np.mean([v.entry_delay_s + v.slowdown_s for v in completed])) if completed else 0.0,
        mean_delay_s=float(np.mean(delays)) if delays else 0.0,
        mean_delay_ns_s=_mean_delay(traffic, NS),
        mean_delay_ew_s=_mean_delay(traffic, EW),
        p95_delay_s=float(np.percentile(delays, 95)) if delays else 0.0,
        worst_wait_s=max(delays + unfinished_waits, default=0.0),
    )
