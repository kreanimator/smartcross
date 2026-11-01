"""Publish and subscribe, over a real MQTT broker or in memory for tests."""

import time
from collections import defaultdict
from collections.abc import Callable

from messaging.schemas import TOPIC_HEALTH, encode_health

QOS = 1  # At least once: the client retries, and duplicates are rejected on arrival.
RECONNECT_MIN_S = 0.1
RECONNECT_MAX_S = 2


class InMemoryBroker:
    """Delivers at once. Tests can set `up`, or `deliver` to drop, repeat or reorder messages."""

    def __init__(self):
        self.up = True
        self.subscribers = defaultdict(list)
        self.deliver = lambda topic, payload: [payload]

    def publish(self, topic: str, payload: bytes) -> None:
        for delivered in self.deliver(topic, payload):
            for handler in list(self.subscribers[topic]):
                handler(delivered)


class InMemoryBus:
    def __init__(self, broker: InMemoryBroker):
        self.broker = broker

    @property
    def connected(self) -> bool:
        return self.broker.up

    def publish(self, topic: str, payload: bytes) -> None:
        if self.connected:
            self.broker.publish(topic, payload)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None:
        self.broker.subscribers[topic].append(handler)


class MqttBus:
    """A broker connection that reconnects by itself and announces its health.

    Handlers run on the client's network thread. The broker reports the service offline if the
    connection dies without saying goodbye.
    """

    def __init__(self, service: str, host: str = "127.0.0.1", port: int = 1883):
        import paho.mqtt.client as mqtt

        self.service = service
        self.handlers = {}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.will_set(TOPIC_HEALTH, encode_health(service, "offline", time.time()), QOS)
        self.client.reconnect_delay_set(RECONNECT_MIN_S, RECONNECT_MAX_S)
        self.client.on_connect = self._on_connect
        self.client.on_message = lambda _client, _userdata, message: self.handlers[message.topic](message.payload)
        self.client.connect_async(host, port)
        self.client.loop_start()

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties) -> None:
        if reason_code.is_failure:
            return
        for topic in self.handlers:
            self.client.subscribe(topic, QOS)
        self._announce("online")

    def _announce(self, state: str):
        return self.client.publish(TOPIC_HEALTH, encode_health(self.service, state, time.time()), QOS)

    @property
    def connected(self) -> bool:
        return self.client.is_connected()

    def wait_connected(self, timeout_s: float = 10) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.connected

    def publish(self, topic: str, payload: bytes) -> None:
        """Drops the message while disconnected: by reconnection it would only be stale."""
        if self.connected:
            self.client.publish(topic, payload, QOS)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None:
        self.handlers[topic] = handler
        if self.connected:
            result, _ = self.client.subscribe(topic, QOS)
            if result:
                raise ConnectionError(f"could not subscribe to {topic}: {result}")

    def close(self) -> None:
        if self.connected:
            self._announce("offline").wait_for_publish(1)
        self.client.disconnect()
        self.client.loop_stop()
