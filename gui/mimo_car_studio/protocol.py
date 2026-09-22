from __future__ import annotations

import json
from typing import Any


PROTOCOL_VERSION = 1
DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 8888


def encode_message(message: dict[str, Any]) -> bytes:
    payload = dict(message)
    payload.setdefault("v", PROTOCOL_VERSION)
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class JsonLineDecoder:
    def __init__(self, maximum_line_bytes: int = 1_000_000) -> None:
        self._buffer = bytearray()
        self.maximum_line_bytes = maximum_line_bytes

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        self._buffer.extend(data)
        if len(self._buffer) > self.maximum_line_bytes:
            self._buffer.clear()
            raise ValueError("received JSON line exceeds safety limit")
        messages: list[dict[str, Any]] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._buffer[:newline]).strip()
            del self._buffer[: newline + 1]
            if not raw:
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue  # skip corrupted frames (RF noise, truncated JSON, etc.)
            if not isinstance(value, dict):
                raise ValueError("protocol frame is not a JSON object")
            messages.append(value)
        return messages
