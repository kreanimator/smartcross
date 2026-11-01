"""The MQTT payloads: JSON with a schema version and simulation ID, checked fully on arrival.

Anything malformed raises InvalidMessage, so bad data never reaches the controller or lights.
"""

import json
import math
from dataclasses import dataclass

from controller.observation import ApproachObservation, TrafficObservation
from controller.traffic_lights import GREEN_FOR, LAMPS, Approach, LightSnapshot, Phase

SCHEMA_VERSION = 1

TOPIC_OBSERVATION = "smartcross/observation"
TOPIC_PHASE_REQUEST = "smartcross/controller/phase-request"
TOPIC_LIGHTS = "smartcross/lights/state"
TOPIC_HEALTH = "smartcross/system/health"


class InvalidMessage(ValueError):
    pass


@dataclass(frozen=True)
class PhaseRequest:
    timestamp_sim_s: float
    direction: str
    message_id: int


@dataclass(frozen=True)
class Received:
    simulation_id: str
    timestamp_sim_s: float
    body: object


def _reject_constant(name: str):
    raise InvalidMessage(f"{name} is not a number")


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise InvalidMessage(f"{name} must be a finite number of at least 0, not {value!r}")
    return value


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidMessage(f"{name} must be a whole number of at least 0, not {value!r}")
    return value


def _encode(simulation_id: str, timestamp_sim_s: float, **fields) -> bytes:
    message = {"schema_version": SCHEMA_VERSION, "simulation_id": simulation_id, "timestamp_sim_s": timestamp_sim_s}
    return json.dumps(message | fields).encode()


def _open(payload: bytes) -> dict:
    try:
        message = json.loads(payload, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise InvalidMessage(f"not JSON: {error}") from error
    if not isinstance(message, dict):
        raise InvalidMessage("not a JSON object")
    if message.get("schema_version") != SCHEMA_VERSION:
        raise InvalidMessage(f"unsupported schema version: {message.get('schema_version')!r}")
    if not isinstance(message.get("simulation_id"), str):
        raise InvalidMessage("simulation_id must be a string")
    _number(message.get("timestamp_sim_s"), "timestamp_sim_s")
    return message


def encode_observation(observation: TrafficObservation, simulation_id: str) -> bytes:
    approaches = {
        approach.value: {
            "queued": seen.queued,
            "approaching": seen.approaching,
            "near_line": seen.near_line,
            "oldest_wait_s": seen.oldest_wait_s,
            "average_speed_mps": seen.average_speed_mps,
            "available": seen.available,
        }
        for approach, seen in observation.approaches.items()
    }
    return _encode(simulation_id, observation.timestamp_sim_s, source=observation.source, approaches=approaches)


def decode_observation(payload: bytes) -> Received:
    message = _open(payload)
    source, approaches = message.get("source"), message.get("approaches")
    if not isinstance(source, str):
        raise InvalidMessage("source must be a string")
    if not isinstance(approaches, dict) or set(approaches) != {approach.value for approach in Approach}:
        raise InvalidMessage("approaches must list exactly north, south, east and west")
    decoded = {}
    for approach in Approach:
        seen = approaches[approach.value]
        if not isinstance(seen, dict):
            raise InvalidMessage(f"{approach.value} must be an object")
        speed = seen.get("average_speed_mps")
        if not isinstance(seen.get("available"), bool):
            raise InvalidMessage("available must be true or false")
        decoded[approach] = ApproachObservation(
            queued=_count(seen.get("queued"), "queued"),
            approaching=_count(seen.get("approaching"), "approaching"),
            near_line=_count(seen.get("near_line"), "near_line"),
            oldest_wait_s=_number(seen.get("oldest_wait_s"), "oldest_wait_s"),
            average_speed_mps=None if speed is None else _number(speed, "average_speed_mps"),
            available=seen["available"],
        )
    timestamp = message["timestamp_sim_s"]
    return Received(message["simulation_id"], timestamp, TrafficObservation(timestamp, source, decoded))


def encode_lights(lights: LightSnapshot, timestamp_sim_s: float, simulation_id: str) -> bytes:
    lamps = {approach.value: lamp.value for approach, lamp in lights.lamps.items()}
    return _encode(
        simulation_id, timestamp_sim_s, phase=lights.phase.value, lamps=lamps, seconds_in_phase=lights.seconds_in_phase
    )


def decode_lights(payload: bytes) -> Received:
    message = _open(payload)
    try:
        phase = Phase(message.get("phase"))
    except ValueError as error:
        raise InvalidMessage(f"unknown phase: {message.get('phase')!r}") from error
    if message.get("lamps") != {approach.value: lamp.value for approach, lamp in LAMPS[phase].items()}:
        raise InvalidMessage(f"lamps do not match phase {phase.value}")
    seconds = _number(message.get("seconds_in_phase"), "seconds_in_phase")
    return Received(message["simulation_id"], message["timestamp_sim_s"], LightSnapshot(phase, dict(LAMPS[phase]), seconds))


def encode_phase_request(request: PhaseRequest, simulation_id: str) -> bytes:
    return _encode(simulation_id, request.timestamp_sim_s, direction=request.direction, message_id=request.message_id)


def decode_phase_request(payload: bytes) -> Received:
    message = _open(payload)
    direction = message.get("direction")
    if not isinstance(direction, str) or direction not in GREEN_FOR:
        raise InvalidMessage(f"unknown direction: {direction!r}")
    request = PhaseRequest(message["timestamp_sim_s"], direction, _count(message.get("message_id"), "message_id"))
    return Received(message["simulation_id"], request.timestamp_sim_s, request)


def encode_health(service: str, state: str, timestamp_s: float) -> bytes:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "service": service, "state": state, "timestamp_s": timestamp_s}
    ).encode()
