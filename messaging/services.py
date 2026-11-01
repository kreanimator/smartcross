"""The controller and the lights talking over a bus, without changing the control rules.

RemoteStrategy sits on the simulator side as an ordinary strategy, so the lights' state machine
stays the only code that can change a lamp. A message can only ever ask for a direction.

When the controller goes quiet, the lights keep its last valid request for STALE_AFTER_S and
then run a local fixed-time plan, so that no direction is starved. They return to the
controller as soon as a valid request arrives. Every change still goes through the state
machine's yellow, all-red and minimum green.

All ages are measured in simulated seconds.
"""

import queue
import time
from collections import Counter

from controller.fixed_timer import FixedTimer
from controller.observation import TrafficObservation
from controller.traffic_lights import LightSnapshot
from messaging.schemas import (
    TOPIC_LIGHTS,
    TOPIC_OBSERVATION,
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

STALE_AFTER_S = 5.0
REPLY_TIMEOUT_S = 1.0  # Real seconds to wait for each reply while the controller is answering.
RECOVERY_WAIT_S = 0.05  # While it is not, a shorter wait keeps the simulation moving.


class MessageGuard:
    """Accepts only this run's messages, each newer than the last and not too old."""

    def __init__(self, simulation_id: str, max_age_s: float | None = None):
        self.simulation_id = simulation_id
        self.max_age_s = max_age_s
        self.newest_s = None
        self.rejected = Counter()

    def accept(self, message: Received, now_s: float | None = None) -> bool:
        reason = self._problem(message, now_s)
        if reason:
            self.rejected[reason] += 1
            return False
        self.newest_s = message.timestamp_sim_s
        return True

    def _problem(self, message: Received, now_s: float | None) -> str | None:
        if message.simulation_id != self.simulation_id:
            return "other simulation"
        if self.newest_s is not None and message.timestamp_sim_s <= self.newest_s:
            return "repeated or out of order"
        if now_s is not None and message.timestamp_sim_s > now_s:
            return "from the future"
        if now_s is not None and self.max_age_s is not None and now_s - message.timestamp_sim_s > self.max_age_s:
            return "stale"
        return None


class ControllerService:
    """Reads observations and light states, and publishes what the strategy asks for."""

    def __init__(self, bus, simulation_id: str, strategy):
        self.bus = bus
        self.simulation_id = simulation_id
        self.strategy = strategy
        self.guard = MessageGuard(simulation_id)
        self.lights_guard = MessageGuard(simulation_id)
        self.lights: Received | None = None
        self.sent = 0
        bus.subscribe(TOPIC_LIGHTS, self._on_lights)
        bus.subscribe(TOPIC_OBSERVATION, self._on_observation)

    def _on_lights(self, payload: bytes) -> None:
        try:
            message = decode_lights(payload)
        except InvalidMessage:
            self.lights_guard.rejected["invalid"] += 1
            return
        if self.lights_guard.accept(message):
            self.lights = message

    def _on_observation(self, payload: bytes) -> None:
        try:
            message = decode_observation(payload)
        except InvalidMessage:
            self.guard.rejected["invalid"] += 1
            return
        if not self.guard.accept(message):
            return
        if self.lights is None or self.lights.timestamp_sim_s != message.timestamp_sim_s:
            self.guard.rejected["no matching light state"] += 1
            return
        direction = self.strategy.decide(message.body, self.lights.body)
        self.sent += 1
        request = PhaseRequest(message.timestamp_sim_s, direction, self.sent)
        self.bus.publish(TOPIC_PHASE_REQUEST, encode_phase_request(request, self.simulation_id))


class RemoteStrategy:
    """Publishes what the simulator knows and returns the direction the controller asks for."""

    def __init__(
        self,
        bus,
        simulation_id: str,
        fallback=None,
        stale_after_s: float = STALE_AFTER_S,
        reply_timeout_s: float = REPLY_TIMEOUT_S,
        recovery_wait_s: float = RECOVERY_WAIT_S,
    ):
        self.bus = bus
        self.simulation_id = simulation_id
        self.fallback = fallback or FixedTimer()
        self.stale_after_s = stale_after_s
        self.reply_timeout_s = reply_timeout_s
        self.recovery_wait_s = recovery_wait_s
        self.guard = MessageGuard(simulation_id, stale_after_s)
        self.replies = queue.SimpleQueue()
        self.last_request: PhaseRequest | None = None
        self.mode = None
        self.mode_changes: list[tuple[float, str]] = []
        bus.subscribe(TOPIC_PHASE_REQUEST, self.replies.put)

    def decide(self, observation: TrafficObservation, lights: LightSnapshot) -> str:
        now_s = observation.timestamp_sim_s
        if self.bus.connected:
            self.bus.publish(TOPIC_LIGHTS, encode_lights(lights, now_s, self.simulation_id))
            self.bus.publish(TOPIC_OBSERVATION, encode_observation(observation, self.simulation_id))
        answering = self.mode in (None, "controller")
        self._read_replies(now_s, self.reply_timeout_s if answering else self.recovery_wait_s)

        if self.last_request is None or now_s - self.last_request.timestamp_sim_s > self.stale_after_s:
            self._set_mode("fallback", now_s)
            return self.fallback.decide(observation, lights)
        if self.last_request.timestamp_sim_s == now_s:
            self._set_mode("controller", now_s)
        else:
            self._set_mode("holding", now_s)
        return self.last_request.direction

    def _read_replies(self, now_s: float, wait_s: float) -> None:
        """Stops at the reply to this observation, or when the wait runs out."""
        deadline = time.monotonic() + (wait_s if self.bus.connected else 0)
        while True:
            try:
                payload = self.replies.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                return
            try:
                message = decode_phase_request(payload)
            except InvalidMessage:
                self.guard.rejected["invalid"] += 1
                continue
            if self.guard.accept(message, now_s):
                self.last_request = message.body
                if message.timestamp_sim_s == now_s:
                    return

    def _set_mode(self, mode: str, now_s: float) -> None:
        if mode != self.mode:
            self.mode = mode
            self.mode_changes.append((now_s, mode))
