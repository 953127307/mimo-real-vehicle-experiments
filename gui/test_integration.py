#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration test: actual CarLink + voltage logic against mock server.

Tests the exact same code paths the GUI uses, without needing a display.
Run:  python test_integration.py
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mock_car_server import MockCar, handle_client  # noqa: E402
# Import the ACTUAL CarLink from the joystick app (no tkinter dependency)
# CarLink only uses stdlib modules, so it can be imported headlessly
from joystick_car import CarLink, CONTROL_HZ, GEARS, U_MAX_DEFAULT  # noqa: E402

TEST_HOST = "127.0.0.1"
TEST_PORT = 18889

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


def compute_voltage(pressed: set[str], gear_index: int,
                    u_max: float = U_MAX_DEFAULT) -> tuple[float, float]:
    """Replicate the exact voltage computation from JoystickApp._control_tick()."""
    g = GEARS[gear_index]
    v_max = min(u_max, g["voltage"])

    v = 0.0
    omega = 0.0
    if "up" in pressed:
        v += 1.0
    if "down" in pressed:
        v -= 1.0
    if "left" in pressed:
        omega += 1.0
    if "right" in pressed:
        omega -= 1.0

    u_r = max(-v_max, min(v_max, v_max * (v + omega)))
    u_l = max(-v_max, min(v_max, v_max * (v - omega)))
    return u_r, u_l


def run_tests() -> bool:
    print("=" * 60)
    print("  Integration Test: Real CarLink + Voltage Logic")
    print("=" * 60)

    # ---- Start mock server ----
    print("\n[1] Starting mock server…")
    car = MockCar()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TEST_HOST, TEST_PORT))
    server.listen(1)

    def serve():
        conn, addr = server.accept()
        handle_client(conn, addr, car)

    threading.Thread(target=serve, daemon=True).start()
    time.sleep(0.2)

    # ---- Connect using REAL CarLink ----
    print("\n[2] Connecting with real CarLink…")
    link = CarLink()
    link.connect_wifi(TEST_HOST, TEST_PORT)
    check(link.connected, "CarLink connected")

    # ---- Hello handshake ----
    print("\n[3] Hello handshake…")
    link.send({"cmd": "hello"})
    time.sleep(0.3)
    msgs = link.drain()
    hello = None
    for m in msgs:
        if m.get("type") == "hello":
            hello = m
    check(hello is not None, "Received hello")
    u_max = hello.get("config", {}).get("u_max", U_MAX_DEFAULT) if hello else U_MAX_DEFAULT
    check(u_max == 4.0, f"u_max from hello = {u_max}V")

    # ---- Enter MANUAL mode ----
    print("\n[4] Enter MANUAL mode…")
    link.send({"cmd": "start", "mode": "manual"})
    time.sleep(0.2)
    msgs = link.drain()
    started = any(m.get("message") == "start_requested" for m in msgs)
    check(started, "start_requested ack received")
    check(car.mode == "manual" and car.armed, "Mock car in MANUAL + armed")

    # ---- Test voltage computation + sending ----
    print("\n[5] Voltage computation + send test…")

    test_inputs = [
        # (pressed_keys, gear, desc, expected_ur, expected_ul)
        ({"up"}, 3, "Up only, 4 V", 4.0, 4.0),
        ({"up"}, 0, "Up only, 1 V", 1.0, 1.0),
        ({"up"}, 1, "Up only, 2 V", 2.0, 2.0),
        ({"up"}, 2, "Up only, 3 V", 3.0, 3.0),
        ({"down"}, 3, "Down only, 4 V", -4.0, -4.0),
        ({"left"}, 3, "Left only, 4 V", 4.0, -4.0),
        ({"right"}, 3, "Right only, 4 V", -4.0, 4.0),
        ({"up", "left"}, 3, "Forward+Left, 4 V", 4.0, 0.0),
        ({"up", "right"}, 3, "Forward+Right, 4 V", 0.0, 4.0),
        (set(), 3, "No keys (stop)", 0.0, 0.0),
        ({"up", "down"}, 3, "Up+Down (cancel)", 0.0, 0.0),
        ({"left", "right"}, 3, "Left+Right (cancel)", 0.0, 0.0),
    ]

    for pressed, gear_idx, desc, exp_r, exp_l in test_inputs:
        u_r, u_l = compute_voltage(pressed, gear_idx, u_max)

        # Send via CarLink
        link.send({"cmd": "manual", "u_r": u_r, "u_l": u_l})
        time.sleep(0.05)

        check(abs(u_r - exp_r) < 0.01,
              f"{desc}: u_R={u_r:.2f} (expected {exp_r:.2f})")
        check(abs(u_l - exp_l) < 0.01,
              f"{desc}: u_L={u_l:.2f} (expected {exp_l:.2f})")
        check(abs(car.u_r - u_r) < 0.01,
              f"{desc}: mock u_R matches ({car.u_r:.2f})")
        check(abs(car.u_l - u_l) < 0.01,
              f"{desc}: mock u_L matches ({car.u_l:.2f})")

    # ---- Rapid 25 Hz test (simulating real control loop) ----
    print("\n[6] 25 Hz control loop simulation (2 seconds)…")
    gear_idx = 3  # 4 V
    sequence = [
        ({"up"}, 0.5),
        ({"up", "right"}, 0.5),
        ({"up"}, 0.5),
        (set(), 0.5),
    ]

    total_sent = 0
    total_read = 0
    for pressed, duration in sequence:
        phase_end = time.monotonic() + duration
        while time.monotonic() < phase_end:
            u_r, u_l = compute_voltage(pressed, gear_idx, u_max)
            link.send({"cmd": "manual", "u_r": u_r, "u_l": u_l})
            total_sent += 1
            # Drain any messages (non-blocking)
            msgs = link.drain()
            total_read += len(msgs)

            # Sleep to maintain 25 Hz
            time.sleep(1.0 / CONTROL_HZ)

    check(total_sent >= 45, f"Sent >= 45 commands in 2s (sent {total_sent})")
    check(total_sent <= 60, f"Sent <= 60 commands (25 Hz cap, sent {total_sent})")
    check(car.mode == "manual", "Still in manual mode")
    check(car.armed, "Still armed")

    # ---- Stop ----
    print("\n[7] Stop + disconnect…")
    link.send({"cmd": "stop"})
    time.sleep(0.2)
    link.drain()
    check(car.mode == "idle", "Mode returned to idle")
    check(not car.armed, "Not armed after stop")

    link.disconnect()
    check(not link.connected, "CarLink disconnected")
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
