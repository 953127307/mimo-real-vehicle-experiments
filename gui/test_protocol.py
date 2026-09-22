#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end protocol test: CarLink ↔ MockCarServer.

Validates the full joystick → firmware protocol flow without real hardware.
Run:  python test_protocol.py
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

# Ensure we can import the joystick module
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import the mock server and CarLink
from mock_car_server import MockCar, U_MAX, handle_client  # noqa: E402

PROTOCOL_VERSION = 1
TEST_PORT = 18888  # different from real port to avoid conflicts
TEST_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Re-implement a minimal CarLink for testing (avoids tkinter dependency)
# ---------------------------------------------------------------------------
class TestCarLink:
    """Minimal CarLink clone for headless testing."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._seq = 0
        self._buffer = bytearray()
        self.connected = False

    def connect(self, host: str = TEST_HOST, port: int = TEST_PORT) -> None:
        self._sock = socket.create_connection((host, port), timeout=3.0)
        self._sock.settimeout(0.3)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buffer.clear()
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def send(self, payload: dict) -> int:
        self._seq += 1
        payload.setdefault("v", PROTOCOL_VERSION)
        payload.setdefault("seq", self._seq)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")
        if self._sock:
            self._sock.sendall(data)
        return self._seq

    def recv_line(self, timeout: float = 2.0) -> dict | None:
        """Read one JSON line, return parsed dict or None on timeout."""
        if not self._sock:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            nl = self._buffer.find(b"\n")
            if nl >= 0:
                raw = bytes(self._buffer[:nl]).strip()
                del self._buffer[: nl + 1]
                if not raw:
                    continue
                try:
                    return json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
            try:
                data = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return None
            if not data:
                return None
            self._buffer.extend(data)
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0


def check(condition: bool, label: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def run_tests() -> bool:
    print("=" * 60)
    print("  MIMO Car Protocol Test Suite")
    print("=" * 60)

    # ---- Start mock server ----
    print("\n[1] Starting mock car server…")
    car = MockCar()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TEST_HOST, TEST_PORT))
    server.listen(1)

    def serve():
        conn, addr = server.accept()
        handle_client(conn, addr, car)

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    time.sleep(0.2)
    print("  Mock server ready")

    # ---- Test 1: Hello handshake ----
    print("\n[2] Hello handshake…")
    client = TestCarLink()
    client.connect(TEST_HOST, TEST_PORT)
    check(client.connected, "TCP connected")

    client.send({"cmd": "hello"})
    resp = client.recv_line(timeout=2.0)
    check(resp is not None, "Received response")
    check(resp and resp.get("type") == "hello", f"Response is hello (type={resp.get('type') if resp else 'None'})")
    check(resp and resp.get("device", "").startswith("MIMO"), "Device name matches")
    check(resp and resp.get("config", {}).get("u_max") == U_MAX,
          f"u_max = {U_MAX:.1f}V")

    # ---- Test 2: Start manual mode ----
    print("\n[3] Start MANUAL mode…")
    client.send({"cmd": "start", "mode": "manual"})
    resp = client.recv_line(timeout=2.0)
    check(resp is not None, "Received ack")
    check(resp and resp.get("type") == "ack", "Response is ack")
    check(resp and resp.get("message") == "start_requested", "Message = start_requested")
    check(car.mode == "manual", f"Mock car mode = manual (actual: {car.mode})")
    check(car.armed, f"Mock car armed = True (actual: {car.armed})")

    # ---- Test 3: Send manual voltages ----
    print("\n[4] Send manual voltages…")
    test_cases = [
        (U_MAX, U_MAX, "Full forward"),
        (-U_MAX, -U_MAX, "Full reverse"),
        (U_MAX, -U_MAX, "Spin right (u_R=+, u_L=-)"),
        (-U_MAX, U_MAX, "Spin left (u_R=-, u_L=+)"),
        (0.25 * U_MAX, U_MAX, "Forward + right turn"),
        (0.0, 0.0, "Stop"),
        (10.0, 10.0, "Clamp to uMax"),
        (-10.0, -10.0, "Clamp to -uMax"),
    ]

    for u_r, u_l, desc in test_cases:
        client.send({"cmd": "manual", "u_r": u_r, "u_l": u_l})
        resp = client.recv_line(timeout=1.0)
        check(resp is not None, f"{desc}: got ack")
        check(resp and resp.get("message") == "manual_updated",
              f"{desc}: manual_updated")

        expected_r = max(-U_MAX, min(U_MAX, u_r))
        expected_l = max(-U_MAX, min(U_MAX, u_l))
        check(abs(car.u_r - expected_r) < 0.01,
              f"{desc}: u_R clamped ({car.u_r:.1f} vs {expected_r:.1f})")
        check(abs(car.u_l - expected_l) < 0.01,
              f"{desc}: u_L clamped ({car.u_l:.1f} vs {expected_l:.1f})")
        time.sleep(0.02)

    # ---- Test 4: Keepalive hello ----
    print("\n[5] Keepalive hello…")
    client.send({"cmd": "hello"})
    resp = client.recv_line(timeout=2.0)
    check(resp is not None, "Keepalive hello response")
    check(resp and resp.get("type") == "hello", "Type = hello")
    check(car.mode == "manual", f"Mode still manual after keepalive (actual: {car.mode})")

    # ---- Test 5: Stop ----
    print("\n[6] Stop…")
    client.send({"cmd": "stop"})
    resp = client.recv_line(timeout=2.0)
    check(resp is not None, "Stop ack")
    check(resp and resp.get("message") == "stop_requested", "stop_requested")
    check(car.mode == "idle", f"Mode = idle after stop (actual: {car.mode})")
    check(not car.armed, "Not armed after stop")

    # ---- Test 6: Error handling ----
    print("\n[7] Error handling…")
    client.send({"cmd": "unknown_thing"})
    resp = client.recv_line(timeout=2.0)
    check(resp is not None, "Error response received")
    check(resp and resp.get("type") == "error", "Type = error")
    check(resp and resp.get("message") == "unknown_command", "unknown_command")

    client.send({"cmd": "start", "mode": "nonexistent"})
    resp = client.recv_line(timeout=2.0)
    check(resp is not None, "Invalid mode response")
    check(resp and resp.get("type") == "error", "Type = error for bad mode")
    check(resp and resp.get("message") == "invalid_mode", "invalid_mode")

    # ---- Test 7: Rapid fire manual commands (simulate 25Hz control loop) ----
    print("\n[8] Rapid-fire manual commands (25 Hz × 1 second)…")
    client.send({"cmd": "start", "mode": "manual"})
    client.recv_line(timeout=1.0)

    received = 0
    for i in range(25):
        client.send({"cmd": "manual", "u_r": 1.0 + i * 0.1, "u_l": 1.0 + i * 0.05})
        # Don't wait for ack for every command (real firmware sends them all)
        time.sleep(1.0 / 25)
    # Drain all acks
    for _ in range(30):
        r = client.recv_line(timeout=0.2)
        if r and r.get("type") == "ack":
            received += 1
    check(received >= 20, f"At least 20/25 manual acks received (got {received})")
    check(car.u_r > 0, f"Final u_R > 0 ({car.u_r:.1f})")
    check(car.u_l > 0, f"Final u_L > 0 ({car.u_l:.1f})")
    check(car.mode == "manual", "Still in manual mode")
    check(car.armed, "Still armed")

    # ---- Cleanup ----
    client.send({"cmd": "stop"})
    client.recv_line(timeout=1.0)
    client.disconnect()
    server.close()

    # ---- Summary ----
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed"
          + (f", {FAIL} FAILED" if FAIL else " — ALL PASS"))
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
