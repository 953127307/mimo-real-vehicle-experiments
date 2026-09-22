#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mock MIMO car firmware server — for testing the joystick app without real hardware.

Listens on TCP port 8888, responds to the JSON-line protocol exactly as the real
ESP32 firmware would: hello, start/stop, manual voltage, keep-alive.

Run:  python mock_car_server.py
Then connect the joystick app to 127.0.0.1:8888.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time

PROTOCOL_VERSION = 1
REGRESSOR_BASIS = "tanh_delta_0.01_gyro_gate"
HOST = "0.0.0.0"
PORT = 8888
U_MAX = 4.0


class MockCar:
    """Simulates the ESP32 firmware's TCP command handling."""

    def __init__(self) -> None:
        self.mode = "idle"
        self.fault = "none"
        self.armed = False
        self.u_r = 0.0
        self.u_l = 0.0
        self.motor_deadzone_v = 0.0
        self.derivative_tau = 0.025
        self.gyro_tau = 0.15
        self.gyro_deadband = 0.040
        self.gyro_blend = 0.75
        self.collection_period = 0.01
        self.integral_window = 0.20
        self.manual_snapshots = False
        self.imu_ok = True
        self.ina_ok = True
        self.imu_calibrated = False
        self.imu_calibration_count = 0
        self.drop_ack_once: set[str] = set()
        self.tau1 = 0.040
        self.tau2 = 0.015
        self.u_max = 3.0
        self.theory_max_position_error = 0.20
        self.theory_max_heading_error = math.radians(60.0)
        self.theory_max_z2 = 2.0
        self.theory_max_z3 = 2.0
        self.theory_safety_grace = 0.75
        self.motion_kx = 2.0
        self.motion_ky = 6.0
        self.motion_kth = 3.0
        self.motion_max_v = 0.10
        self.motion_max_omega = 1.00
        self.motion_max_accel = 0.20
        self.motion_max_angular_accel = 2.50
        self.outer_max_v = 0.13
        self.outer_max_omega = 1.80
        self.motion_max_target_error = 0.20
        self.pose = [0.0, 0.0, 0.0]
        self.motion_reference = self.pose.copy()
        self.velocity = [0.0, 0.0]
        self._last_model_time = time.monotonic()
        self._seq = 0
        self._last_rx = time.monotonic()
        self._lock = threading.Lock()

    def handle(self, line: str) -> str | None:
        """Process one JSON command, return reply line or None."""
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            return json.dumps({"v": PROTOCOL_VERSION, "type": "error",
                               "seq": 0, "message": "invalid_json"}) + "\n"

        version = doc.get("v", 0)
        if version != PROTOCOL_VERSION:
            return json.dumps({"v": PROTOCOL_VERSION, "type": "error",
                               "seq": doc.get("seq", 0),
                               "message": "unsupported_protocol"}) + "\n"

        cmd = doc.get("cmd", "")
        seq = doc.get("seq", 0)

        with self._lock:
            self._last_rx = time.monotonic()

            if cmd == "hello":
                return self._hello(seq)
            elif cmd == "stop":
                self.mode = "idle"
                self.armed = False
                self.fault = "none"
                self.u_r = 0.0
                self.u_l = 0.0
                return self._ack(seq, "stop_requested")
            elif cmd == "start":
                mode = doc.get("mode", "")
                if mode not in ("manual", "monitor", "collect", "theory"):
                    return self._err(seq, "invalid_mode")
                self.mode = mode
                self.armed = mode in ("manual", "collect", "theory")
                self.u_r = 0.0
                self.u_l = 0.0
                print(f"  [MOCK] mode={mode} armed={self.armed}")
                return self._ack(seq, "start_requested")
            elif cmd == "manual":
                u_r = doc.get("u_r", 0.0)
                u_l = doc.get("u_l", 0.0)
                self.u_r = max(-U_MAX, min(U_MAX, float(u_r)))
                self.u_l = max(-U_MAX, min(U_MAX, float(u_l)))
                if abs(self.u_r) <= self.motor_deadzone_v:
                    self.u_r = 0.0
                if abs(self.u_l) <= self.motor_deadzone_v:
                    self.u_l = 0.0
                return self._ack(seq, "manual_updated")
            elif cmd == "zero_pose":
                self.pose = [0.0, 0.0, 0.0]
                self.motion_reference = self.pose.copy()
                return self._ack(seq, "zero_pose_requested")
            elif cmd == "set_pose":
                return self._ack(seq, "set_pose_requested")
            elif cmd == "calibrate_imu":
                if self.armed:
                    return self._err(seq, "stop_before_calibration")
                self.imu_calibration_count += 1
                self.imu_calibrated = self.imu_ok
                return self._ack(seq, "calibration_requested")
            elif cmd == "configure":
                if "derivative_tau" in doc:
                    self.derivative_tau = float(doc["derivative_tau"])
                if "gyro_tau" in doc:
                    self.gyro_tau = float(doc["gyro_tau"])
                if "collection_period" in doc:
                    self.collection_period = float(doc["collection_period"])
                if "integral_window" in doc:
                    self.integral_window = float(doc["integral_window"])
                if "tau1" in doc:
                    self.tau1 = float(doc["tau1"])
                if "tau2" in doc:
                    self.tau2 = float(doc["tau2"])
                if "manual_snapshots" in doc:
                    self.manual_snapshots = bool(doc["manual_snapshots"])
                for key in (
                    "u_max",
                    "motor_deadzone_v",
                    "gyro_deadband",
                    "gyro_blend",
                    "theory_max_position_error",
                    "theory_max_heading_error",
                    "theory_max_z2",
                    "theory_max_z3",
                    "theory_safety_grace",
                    "motion_kx",
                    "motion_ky",
                    "motion_kth",
                    "motion_max_v",
                    "motion_max_omega",
                    "motion_max_accel",
                    "motion_max_angular_accel",
                    "outer_max_v",
                    "outer_max_omega",
                    "motion_max_target_error",
                ):
                    if key in doc:
                        setattr(self, key, float(doc[key]))
                return self._ack(seq, "configuration_updated")
            elif cmd == "set_weights":
                return self._ack(seq, "weights_accepted")
            else:
                return self._err(seq, "unknown_command")

    def _hello(self, seq: int) -> str:
        doc = {
            "v": PROTOCOL_VERSION,
            "type": "hello",
            "seq": seq,
            "device": "MIMO differential-drive car (MOCK)",
            "firmware": "mock-1.0.0",
            "regressor_basis": REGRESSOR_BASIS,
            "boot_id": 1,
            "reset_reason": "power_on",
            "uptime_ms": int(time.monotonic() * 1000),
            "ip": "127.0.0.1",
            "ap_ip": "127.0.0.1",
            "mode": self.mode,
            "fault": self.fault,
            "weights_valid": True,
            "geometry": {
                "wheel_radius_m": 0.0325,
                "track_width_m": 0.2035,
                "ticks_per_rev": 1320,
            },
            "config": {
                "u_max": U_MAX,
                "motor_deadzone_v": self.motor_deadzone_v,
                "gyro_deadband": self.gyro_deadband,
                "theory_max_position_error": self.theory_max_position_error,
                "theory_max_heading_error": self.theory_max_heading_error,
                "theory_max_z2": self.theory_max_z2,
                "theory_max_z3": self.theory_max_z3,
                "motion_kx": self.motion_kx,
                "motion_ky": self.motion_ky,
                "motion_kth": self.motion_kth,
                "motion_max_v": self.motion_max_v,
                "motion_max_omega": self.motion_max_omega,
                "motion_max_accel": self.motion_max_accel,
                "motion_max_angular_accel": self.motion_max_angular_accel,
                "outer_max_v": self.outer_max_v,
                "outer_max_omega": self.outer_max_omega,
                "motion_max_target_error": self.motion_max_target_error,
                "telemetry_period": 0.02,
                "run_duration": 30.0,
                "current_limit": 1.3,
                "battery_min": 9.6,
            },
        }
        return json.dumps(doc, separators=(",", ":")) + "\n"

    def _ack(self, seq: int, message: str) -> str | None:
        if message in self.drop_ack_once:
            self.drop_ack_once.remove(message)
            return None
        return json.dumps({"v": PROTOCOL_VERSION, "type": "ack",
                           "seq": seq, "message": message}) + "\n"

    def _err(self, seq: int, message: str) -> str:
        return json.dumps({"v": PROTOCOL_VERSION, "type": "error",
                           "seq": seq, "message": message}) + "\n"

    def telemetry(self) -> str:
        with self._lock:
            now = time.monotonic()
            dt = min(0.1, max(0.0, now - self._last_model_time))
            self._last_model_time = now
            target_v = 0.025 * 0.5 * (self.u_r + self.u_l)
            target_omega = 0.025 * (self.u_r - self.u_l) / 0.2035
            alpha = min(1.0, dt / 0.12)
            self.velocity[0] += alpha * (target_v - self.velocity[0])
            self.velocity[1] += alpha * (target_omega - self.velocity[1])
            self.pose[2] += self.velocity[1] * dt
            self.pose[0] += self.velocity[0] * math.cos(self.pose[2]) * dt
            self.pose[1] += self.velocity[0] * math.sin(self.pose[2]) * dt
            wheel_right = self.velocity[0] + 0.5 * 0.2035 * self.velocity[1]
            wheel_left = self.velocity[0] - 0.5 * 0.2035 * self.velocity[1]
            doc = {
                "v": PROTOCOL_VERSION,
                "type": "telemetry",
                "seq": self._seq,
                "t_us": int(time.monotonic() * 1e6),
                "mode": self.mode,
                "fault": self.fault,
                "armed": self.armed,
                "weights_valid": True,
                "dropped": 0,
                "loop_us": 120,
                "s": {
                    "pose": self.pose.copy(),
                    "velocity_raw": [target_v, target_omega],
                    "velocity": self.velocity.copy(),
                    "wheel_r": wheel_right,
                    "wheel_l": wheel_left,
                    "gyro_z": self.velocity[1],
                    "imu_ok": self.imu_ok,
                    "imu_calibrated": self.imu_calibrated,
                    "ina_ok": self.ina_ok,
                    "current_raw": [
                        abs(self.u_r) * 0.032,
                        abs(self.u_l) * 0.032,
                    ],
                    "current": [abs(self.u_r) * 0.03, abs(self.u_l) * 0.03],
                },
                "c": {
                    "reference": self.pose.copy(),
                    "motion_reference": self.motion_reference.copy(),
                    "pose_error": [0.0, 0.0, 0.0],
                    "alpha1": [0.0, 0.0],
                    "beta1": [0.0, 0.0],
                    "beta1_dot": [0.0, 0.0],
                    "alpha2": [0.0, 0.0],
                    "beta2": [0.0, 0.0],
                    "beta2_dot": [0.0, 0.0],
                    "z2": [0.0, 0.0],
                    "z3": [0.0, 0.0],
                    "uc": [0.0, 0.0],
                    "u": [self.u_r, self.u_l],
                },
            }
            self._seq += 1
        return json.dumps(doc, separators=(",", ":")) + "\n"


def handle_client(conn: socket.socket, addr: tuple, car: MockCar) -> None:
    """Serve one TCP client."""
    print(f"\n[MOCK] Client connected: {addr}")
    conn.settimeout(0.5)
    buf = bytearray()
    last_telemetry = time.monotonic()

    try:
        while True:
            try:
                data = conn.recv(4096)
            except socket.timeout:
                data = b""
            except OSError:
                break

            if not data:
                # check if client is still alive
                try:
                    conn.sendall(b"")
                except OSError:
                    break
                # Send periodic telemetry for realism
                now = time.monotonic()
                if now - last_telemetry > 0.05:
                    try:
                        conn.sendall(car.telemetry().encode("utf-8"))
                    except OSError:
                        break
                    last_telemetry = now
                continue

            buf.extend(data)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                raw = bytes(buf[:nl]).strip()
                del buf[: nl + 1]
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                print(f"  [MOCK] RX: {line[:120]}")
                reply = car.handle(line)
                if reply:
                    try:
                        conn.sendall(reply.encode("utf-8"))
                    except OSError:
                        return
                    # Print voltages if it's a manual command
                    if '"cmd":"manual"' in line:
                        print(f"  [MOCK] -> MOTORS: u_R={car.u_r:+.2f}V  u_L={car.u_l:+.2f}V  "
                              f"mode={car.mode} armed={car.armed}")

    except Exception as e:
        print(f"  [MOCK] Client error: {e}")
    finally:
        print(f"  [MOCK] Client disconnected: {addr}")
        try:
            conn.close()
        except OSError:
            pass


def main() -> None:
    print("=" * 60)
    print("  MOCK MIMO Car Firmware Server")
    print(f"  Listening on {HOST}:{PORT}")
    print("  Connect the joystick app to 127.0.0.1:8888")
    print("=" * 60)

    car = MockCar()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(4)

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, car),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[MOCK] Shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    main()
