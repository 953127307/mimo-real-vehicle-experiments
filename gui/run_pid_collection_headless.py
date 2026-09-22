#!/usr/bin/env python3
"""Collect one firmware integral-snapshot dataset without opening Tk."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from joystick_car import (
    COLLECTION_DERIVATIVE_TAU_S,
    CURRENT_FILTER_TAU_S,
    GYRO_RATE_FILTER_TAU_S,
    MANUAL_VOLTAGE_SLEW_V_PER_S,
    PID_COLLECTION_PRESETS,
    PID_COLLECTION_WARMUP_S,
    PidPathController,
    CarLink,
    pid_collection_reference,
    persistent_results_dir,
    snapshot_dataset_cache_path,
)
from mimo_car_studio.synthesis import IntegralSnapshotDataset


HOST = "192.168.4.1"
PORT = 8888
PRESET_NAME = "中型双纽线（0.56 m × 0.24 m，55 s）"
SPEED_SCALE = 1.5
REQUIRED_FIRMWARE = "1.7.1"
INTEGRAL_WINDOW_S = 0.10
MOTOR_DEADZONE_V = 0.5
CONTROL_PERIOD_S = 1.0 / 25.0
TELEMETRY_TIMEOUT_S = 0.75
TELEMETRY_ABORT_S = 2.0
MAX_RADIUS_M = 0.70
MAX_TRACKING_ERROR_M = 0.35
COMMAND_TIMEOUT_S = 5.0

SNAPSHOT_FIELDS = {
    "zdot2": 2,
    "y2": 8,
    "x3": 2,
    "zdot3": 2,
    "y3": 8,
    "x4": 2,
}


def wait_for_message(link: CarLink, predicate, timeout_s: float = COMMAND_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for message in link.drain():
            if predicate(message):
                return message
        if not link.connected:
            raise ConnectionError(link.status_text)
        time.sleep(0.01)
    raise TimeoutError("firmware response timeout")


def send_and_expect(link: CarLink, payload: dict[str, Any], expected_ack: str) -> None:
    sequence = link.send(payload)
    response = wait_for_message(
        link,
        lambda message: int(message.get("seq", -1)) == sequence
        and message.get("type") in ("ack", "error"),
    )
    if response.get("type") != "ack" or response.get("message") != expected_ack:
        raise RuntimeError(
            f"{payload.get('cmd')} rejected: {response.get('message', '?')}"
        )


def hello(link: CarLink) -> dict[str, Any]:
    sequence = link.send({"cmd": "hello"})
    return wait_for_message(
        link,
        lambda message: int(message.get("seq", -1)) == sequence
        and message.get("type") == "hello",
        8.0,
    )


def parse_live_state(message: dict[str, Any]):
    if message.get("type") != "telemetry":
        return None
    sensors = message.get("s")
    if not isinstance(sensors, dict):
        return None
    pose = np.asarray(sensors.get("pose"), dtype=float)
    velocity = np.asarray(sensors.get("velocity"), dtype=float)
    bus_voltage = np.asarray(sensors.get("bus_voltage"), dtype=float)
    if (
        pose.shape != (3,)
        or velocity.shape != (2,)
        or bus_voltage.shape != (3,)
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(bus_voltage))
    ):
        return None
    return pose, velocity, bus_voltage, bool(sensors.get("ina_ok", False))


def append_snapshot(
    dataset: IntegralSnapshotDataset,
    segment: int,
    message: dict[str, Any],
) -> bool:
    snapshot = message.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("kind") != "integral_v3_sgn_gyro_gate":
        raise RuntimeError(f"unexpected snapshot kind: {snapshot.get('kind')!r}")
    row: dict[str, np.ndarray] = {}
    for key, size in SNAPSHOT_FIELDS.items():
        value = np.asarray(snapshot.get(key), dtype=float)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid snapshot field {key}")
        row[key] = value
    for key in ("velocity_raw", "current_raw"):
        value = np.asarray(snapshot.get(key), dtype=float)
        if value.shape == (2,) and np.all(np.isfinite(value)):
            row[key] = value
    dataset.append_integral(
        row,
        t_us=int(message.get("t_us", 0)),
        window_s=float(snapshot.get("window_s", 0.0)),
        segment=segment,
    )
    return True


def save_dataset(dataset: IntegralSnapshotDataset) -> tuple[Path, Path]:
    output_dir = persistent_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"integral_pid_new_hardware_K{len(dataset)}_{stamp}.npz"
    dataset.save(output_path)

    cache_path = snapshot_dataset_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    dataset.save(temporary)
    temporary.replace(cache_path)
    return output_path, cache_path


def run_collection(
    preset_name: str = PRESET_NAME,
    speed_scale: float = SPEED_SCALE,
) -> tuple[IntegralSnapshotDataset, Path, Path]:
    preset = dict(PID_COLLECTION_PRESETS[preset_name])
    preset["nu"] *= speed_scale
    duration = float(preset["duration"])
    dataset = IntegralSnapshotDataset()
    segment = dataset.start_segment(
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "firmware": REQUIRED_FIRMWARE,
            "path": preset_name,
            "duration_s": duration,
            "u_max_v": 4.0,
            "motor_deadzone_v": MOTOR_DEADZONE_V,
            "integral_window_s": INTEGRAL_WINDOW_S,
            "speed_scale": speed_scale,
            "omega_source": "encoder",
            "source": "firmware_integral_v3_sgn_gyro_gate",
            "runner": "run_pid_collection_headless.py",
        }
    )
    controller = PidPathController()
    link = CarLink()
    latest_pose: np.ndarray | None = None
    latest_velocity: np.ndarray | None = None
    latest_bus = np.zeros(3, dtype=float)
    last_live_time = 0.0
    last_u = np.zeros(2, dtype=float)
    last_report = 0.0
    start = 0.0
    last_control = 0.0
    completed = False

    try:
        link.connect_wifi(HOST, PORT)
        device = hello(link)
        firmware = str(device.get("firmware", "?"))
        if firmware != REQUIRED_FIRMWARE:
            raise RuntimeError(
                f"firmware {REQUIRED_FIRMWARE} required, got {firmware}"
            )
        if str(device.get("fault", "none")) not in ("", "none"):
            raise RuntimeError(f"pre-existing firmware fault: {device.get('fault')}")
        print(
            f"CONNECTED firmware={firmware} weights_valid={device.get('weights_valid')}",
            flush=True,
        )
        send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        send_and_expect(
            link,
            {
                "cmd": "configure",
                "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
                "current_tau": CURRENT_FILTER_TAU_S,
                "motor_deadzone_v": MOTOR_DEADZONE_V,
                "gyro_tau": GYRO_RATE_FILTER_TAU_S,
                "integral_window": INTEGRAL_WINDOW_S,
                "manual_snapshots": True,
            },
            "configuration_updated",
        )
        send_and_expect(link, {"cmd": "zero_pose"}, "zero_pose_requested")
        send_and_expect(link, {"cmd": "start", "mode": "manual"}, "start_requested")

        ready_deadline = time.monotonic() + 3.0
        next_ready_zero = 0.0
        while time.monotonic() < ready_deadline and latest_pose is None:
            now = time.monotonic()
            if now >= next_ready_zero:
                link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
                next_ready_zero = now + 0.20
            for message in link.drain():
                state = parse_live_state(message)
                if state is not None:
                    latest_pose, latest_velocity, latest_bus, ina_ok = state
                    last_live_time = time.monotonic()
                    if not ina_ok:
                        raise RuntimeError("INA3221 is not ready")
            time.sleep(0.01)
        if latest_pose is None or latest_velocity is None:
            raise TimeoutError("no fresh telemetry before collection")

        controller.reset()
        start = time.monotonic()
        last_control = start
        last_report = start
        print(
            f"STARTED preset={preset_name} duration={duration:.0f}s "
            f"Tw={INTEGRAL_WINDOW_S:.2f}s bus={latest_bus.tolist()}",
            flush=True,
        )
        while True:
            now = time.monotonic()
            elapsed = now - start
            for message in link.drain():
                fault = str(message.get("fault", "none"))
                if fault not in ("", "none", "?"):
                    raise RuntimeError(f"firmware fault: {fault}")
                state = parse_live_state(message)
                if state is None:
                    continue
                latest_pose, latest_velocity, latest_bus, ina_ok = state
                last_live_time = now
                if not ina_ok:
                    raise RuntimeError("INA3221 became invalid")
                if elapsed >= PID_COLLECTION_WARMUP_S:
                    append_snapshot(dataset, segment, message)

            if not link.connected:
                raise ConnectionError(link.status_text)
            age = now - last_live_time
            if age > TELEMETRY_ABORT_S:
                raise TimeoutError(f"telemetry stalled for {age:.2f}s")
            if age > TELEMETRY_TIMEOUT_S:
                link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
                time.sleep(0.01)
                continue
            if elapsed >= duration:
                completed = True
                break

            if now - last_control >= CONTROL_PERIOD_S:
                dt = max(1.0e-3, min(0.2, now - last_control))
                reference = pid_collection_reference(elapsed, preset)
                u_r, u_l, tracking = controller.step(
                    latest_pose, latest_velocity, reference, dt, 4.0
                )
                max_step = MANUAL_VOLTAGE_SLEW_V_PER_S * dt
                requested = np.asarray((u_r, u_l), dtype=float)
                applied = np.clip(requested, last_u - max_step, last_u + max_step)
                tracking_error = math.hypot(tracking["e_x"], tracking["e_y"])
                radius = float(np.linalg.norm(latest_pose[:2]))
                if radius > MAX_RADIUS_M:
                    raise RuntimeError(f"workspace radius exceeded: {radius:.3f}m")
                if elapsed > PID_COLLECTION_WARMUP_S and tracking_error > MAX_TRACKING_ERROR_M:
                    raise RuntimeError(
                        f"tracking error exceeded: {tracking_error:.3f}m"
                    )
                link.send(
                    {"cmd": "manual", "u_r": float(applied[0]), "u_l": float(applied[1])}
                )
                last_u = applied
                last_control = now

            if now - last_report >= 2.0:
                print(
                    f"t={elapsed:5.1f}/{duration:.0f}s K={len(dataset):3d} "
                    f"pose=({latest_pose[0]:+.3f},{latest_pose[1]:+.3f},"
                    f"{math.degrees(latest_pose[2]):+.1f}deg) "
                    f"u=({last_u[0]:+.2f},{last_u[1]:+.2f}) "
                    f"bus_min={float(np.min(latest_bus)):.2f}V",
                    flush=True,
                )
                last_report = now
            time.sleep(0.005)
    finally:
        try:
            link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
            time.sleep(0.05)
            send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        except Exception:
            pass
        try:
            send_and_expect(
                link,
                {"cmd": "configure", "manual_snapshots": False},
                "configuration_updated",
            )
        except Exception:
            pass
        link.disconnect()

    if not completed:
        raise RuntimeError("collection did not complete")
    if len(dataset) < 300:
        raise RuntimeError(f"too few integral windows: K={len(dataset)}")
    output_path, cache_path = save_dataset(dataset)
    return dataset, output_path, cache_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required safety gate; the vehicle will move",
    )
    parser.add_argument(
        "--preset", choices=tuple(PID_COLLECTION_PRESETS), default=PRESET_NAME
    )
    parser.add_argument("--speed-scale", type=float, default=SPEED_SCALE)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required because this command moves the vehicle")
    if not math.isfinite(args.speed_scale) or not 0.25 <= args.speed_scale <= 2.0:
        parser.error("--speed-scale must be finite and between 0.25 and 2.0")
    dataset, output_path, cache_path = run_collection(
        args.preset, args.speed_scale
    )
    matrices = dataset.matrices()
    summary = {
        "samples": len(dataset),
        "segments": dataset.segment_count,
        "rank_g2": int(np.linalg.matrix_rank(np.vstack((matrices["zdot2"], matrices["y2"])))),
        "rank_g3": int(np.linalg.matrix_rank(np.vstack((matrices["zdot3"], matrices["y3"])))),
        "output": str(output_path),
        "cache": str(cache_path),
    }
    print("COLLECTION_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
