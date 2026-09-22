#!/usr/bin/env python3
"""Run the sampling-PID external baseline (paper case E4) without the GUI.

The controller is the excitation PidPathController used for data collection
(PC-side, 25 Hz, pose P loop + wheel-voltage PI).  Per user decision
2026-09-17 its voltage ceiling is unified with the other cases at the common
4 V bound (base_limit_frac=1.0); the collection path keeps its own 0.60
default.  The task is the campaign's center-start circle (firmware
ref_shape=3 semantics): the reference point starts on the circle at (R, 0)
with heading pi/2 and full speed at t=0, while the car is zeroed at the
circle center.  Runs are logged in the same comparison-trace schema as E1-E3
so the frozen-campaign pipeline can ingest them as case "E4".
"""

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
    PathReference,
    PidPathController,
    CarLink,
    persistent_results_dir,
)


HOST = "192.168.4.1"
PORT = 8888
REQUIRED_FIRMWARE = "1.7.4"
CIRCLE_RADIUS_M = 0.40
CIRCLE_NU_RAD_S = 0.30
CIRCLE_SPEED_MPS = CIRCLE_RADIUS_M * CIRCLE_NU_RAD_S
RUN_DURATION_S = 2.0 * math.pi / CIRCLE_NU_RAD_S + 3.0
HOLD_DURATION_S = 0.0
# Declared task parameters, kept identical to the E1-E3 campaign for the
# frozen-pipeline consistency checks (the PID baseline itself uses no
# preview; the horizon is recorded because it defines the SS window).
OUTER_PREVIEW_HORIZON_S = 1.3
MOTOR_DEADZONE_V = 0.5
U_MAX_V = 4.0
CONTROL_PERIOD_S = 1.0 / 25.0
TELEMETRY_TIMEOUT_S = 0.75
TELEMETRY_ABORT_S = 2.0
KEEPALIVE_PERIOD_S = 1.5
COMMAND_TIMEOUT_S = 5.0
MAX_RADIUS_M = 0.75
SAFETY_GRACE_S = 8.0
MAX_POSITION_ERROR_M = 0.80
TRAJECTORY_NAME = "匀速圆（圆心起步，R 0.40 m，0.12 m/s，1 圈）"
BASELINE_NAME = "sampling PID baseline (data-collection excitation controller)"


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


def center_start_circle_reference(elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Campaign ref_shape=3 reference: center start, full speed from t=0.

    q_r(t) = (R cos vt, R sin vt), theta_r = pi/2 + vt, v = R nu, omega = nu.
    """
    t = max(0.0, float(elapsed_s))
    phase = CIRCLE_NU_RAD_S * t
    pose = np.asarray(
        (
            CIRCLE_RADIUS_M * math.cos(phase),
            CIRCLE_RADIUS_M * math.sin(phase),
            math.pi / 2.0 + phase,
        ),
        dtype=float,
    )
    velocity = np.asarray((CIRCLE_SPEED_MPS, CIRCLE_NU_RAD_S), dtype=float)
    return pose, velocity


def body_frame_error(
    reference: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    delta = reference[:2] - pose[:2]
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    heading_error = (
        float(reference[2]) - float(pose[2]) + math.pi
    ) % (2.0 * math.pi) - math.pi
    return np.asarray(
        (
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            heading_error,
        ),
        dtype=float,
    )


def parse_live_state(message: dict[str, Any]):
    """Decode both manual-mode 'state' frames and full 'telemetry' frames.

    Manual mode with manual_snapshots=False streams lightweight 50 Hz state
    frames whose fields sit at the top level ('pose', 'vel', 'cur', 'u');
    the full telemetry frame nests them under 's'.
    """
    mtype = message.get("type")
    if mtype == "telemetry":
        sensors = message.get("s")
        if not isinstance(sensors, dict):
            return None
        pose_source, velocity_source = sensors.get("pose"), sensors.get("velocity")
        current_source, voltage_source = None, None
        bus_source = sensors.get("bus_voltage")
        ina_ok = bool(sensors.get("ina_ok", False))
    elif mtype == "state":
        pose_source, velocity_source = message.get("pose"), message.get("vel")
        current_source, voltage_source = message.get("cur"), message.get("u")
        bus_source = message.get("bus_voltage")
        ina_ok = bool(message.get("ina_ok", False))
    else:
        return None
    pose = np.asarray(pose_source, dtype=float)
    velocity = np.asarray(velocity_source, dtype=float)
    current = (
        np.asarray(current_source, dtype=float)
        if current_source is not None
        else np.zeros(2, dtype=float)
    )
    applied_u = (
        np.asarray(voltage_source, dtype=float)
        if voltage_source is not None
        else np.zeros(2, dtype=float)
    )
    bus_voltage = np.asarray(bus_source, dtype=float)
    if (
        pose.shape != (3,)
        or velocity.shape != (2,)
        or current.shape != (2,)
        or applied_u.shape != (2,)
        or bus_voltage.shape != (3,)
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(applied_u))
        or not np.all(np.isfinite(bus_voltage))
    ):
        return None
    return (
        pose,
        velocity,
        current,
        applied_u,
        bus_voltage,
        ina_ok,
        int(message.get("t_us", 0)),
    )


def baseline_config() -> dict[str, Any]:
    """Task description stored in runtime_config_json for the E4 pipeline."""
    return {
        "baseline": "sampling_pid",
        "comparison_case": 4.0,
        "control_hz": 25.0,
        "hold_duration": HOLD_DURATION_S,
        "motor_deadzone_v": MOTOR_DEADZONE_V,
        "outer_capture_radius_m": 0.010,
        "outer_blend_radius_m": 0.030,
        "outer_kp": 0.8,
        "outer_ktheta": 0.7,
        "outer_preview_horizon": OUTER_PREVIEW_HORIZON_S,
        "outer_vbar": 0.5,
        "pid_base_limit_frac": 1.0,
        "pid_outer_ktheta": 3.2,
        "pid_outer_kx": 1.8,
        "pid_outer_ky": 4.0,
        "pid_wheel_ff_v_per_mps": 42.0,
        "pid_wheel_ki": 2.0,
        "pid_wheel_kp": 14.0,
        "ref_nu_rad_s": CIRCLE_NU_RAD_S,
        "ref_radius_m": CIRCLE_RADIUS_M,
        "ref_shape": 3.0,
        "r2": 0.0,
        "r3": 0.0,
        "run_duration": RUN_DURATION_S,
        "u_max": U_MAX_V,
    }


def export_trace(
    trial: int,
    reason: str,
    rows: list[dict[str, Any]],
    firmware: str,
    config: dict[str, Any],
) -> Path | None:
    if not rows:
        return None
    payload: dict[str, Any] = {
        "schema": np.asarray([3], dtype=np.int64),
        "t": np.asarray([row["t"] for row in rows], dtype=float),
        "reason": np.asarray([reason]),
        "trajectory": np.asarray([TRAJECTORY_NAME]),
        "comparison_case": np.asarray(["E4"]),
        "comparison_name": np.asarray([BASELINE_NAME]),
        "initial_nonlinear_compensation_enabled": np.asarray([False]),
        "trial": np.asarray([trial], dtype=np.int64),
        "firmware": np.asarray([firmware]),
        "runtime_config_json": np.asarray(
            [json.dumps(config, ensure_ascii=False, sort_keys=True)]
        ),
        "weights_json": np.asarray(
            [
                json.dumps(
                    {"used_by_controller": False, "note": "external PID baseline"},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ]
        ),
    }
    for key in rows[0]:
        if key != "t":
            payload[key] = np.asarray([row[key] for row in rows], dtype=float)
    target_dir = persistent_results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"theory_trace_e4_{stamp}_trial{trial}_{reason}.npz"
    np.savez_compressed(target, **payload)
    return target


def run_trial(trial: int) -> tuple[dict[str, Any], Path | None]:
    controller = PidPathController()
    link = CarLink()
    rows: list[dict[str, Any]] = []
    reason = "host_error"
    firmware = "?"
    config = baseline_config()
    completed = False
    t0_us: int | None = None
    last_u = np.zeros(2, dtype=float)
    last_command = np.zeros(2, dtype=float)

    try:
        link.connect_wifi(HOST, PORT)
        device = hello(link)
        if str(device.get("fault", "none")) == "network_lost":
            # A session killed without a clean FIN latches this benign
            # connectivity fault; the firmware clears it on the next stop.
            send_and_expect(link, {"cmd": "stop"}, "stop_requested")
            time.sleep(0.20)
            device = hello(link)
        firmware = str(device.get("firmware", "?"))
        if firmware != REQUIRED_FIRMWARE:
            raise RuntimeError(
                f"firmware {REQUIRED_FIRMWARE} required, got {firmware}"
            )
        if str(device.get("fault", "none")) not in ("", "none"):
            raise RuntimeError(f"pre-existing firmware fault: {device.get('fault')}")
        geometry = device.get("geometry", {})
        try:
            track_width = float(geometry.get("track_width_m", 0.2035))
        except (TypeError, ValueError):
            track_width = 0.2035
        config["track_width_m"] = track_width
        print(
            f"[E4 trial {trial}] connected · fault={device.get('fault')} · "
            f"weights_valid={device.get('weights_valid')} · "
            f"track_width={track_width:.4f} m",
            flush=True,
        )
        send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        time.sleep(0.20)
        send_and_expect(
            link,
            {
                "cmd": "configure",
                "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
                "current_tau": CURRENT_FILTER_TAU_S,
                "motor_deadzone_v": MOTOR_DEADZONE_V,
                "gyro_tau": GYRO_RATE_FILTER_TAU_S,
                "manual_snapshots": False,
            },
            "configuration_updated",
        )
        time.sleep(0.15)
        send_and_expect(link, {"cmd": "zero_pose"}, "zero_pose_requested")
        time.sleep(0.15)
        send_and_expect(link, {"cmd": "start", "mode": "manual"}, "start_requested")
        print(f"[E4 trial {trial}] STARTED · motors may move", flush=True)

        latest_pose: np.ndarray | None = None
        latest_velocity: np.ndarray | None = None
        last_live_time = 0.0
        start = time.monotonic()
        last_control = 0.0
        next_keepalive = start + KEEPALIVE_PERIOD_S
        next_report = start + 2.0
        deadline = start + RUN_DURATION_S + 8.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            elapsed = now - start
            if now >= next_keepalive:
                link.send({"cmd": "hello"})
                next_keepalive = now + KEEPALIVE_PERIOD_S
            for message in link.drain():
                fault = str(message.get("fault", "none"))
                if fault not in ("", "none", "?"):
                    raise RuntimeError(f"firmware fault: {fault}")
                state = parse_live_state(message)
                if state is None:
                    continue
                latest_pose, latest_velocity, latest_current, applied_u, _bus, ina_ok, timestamp_us = state
                last_live_time = now
                if not ina_ok:
                    raise RuntimeError("INA3221 became invalid")
                if t0_us is None:
                    t0_us = timestamp_us
                t = max(0.0, (timestamp_us - t0_us) * 1.0e-6)
                reference_pose, reference_velocity = (
                    center_start_circle_reference(t)
                )
                pose_error = body_frame_error(reference_pose, latest_pose)
                rows.append(
                    {
                        "t": t,
                        "pose": latest_pose.copy(),
                        "velocity": latest_velocity.copy(),
                        "current": latest_current.copy(),
                        "reference": reference_pose,
                        "motion_reference": reference_pose,
                        "pose_error": pose_error,
                        "alpha1": reference_velocity,
                        "beta1": last_command.copy(),
                        "beta1_dot": np.zeros(2, dtype=float),
                        "alpha2": np.zeros(2, dtype=float),
                        "beta2": np.zeros(2, dtype=float),
                        "beta2_dot": np.zeros(2, dtype=float),
                        "z2": latest_velocity - last_command,
                        "z3": np.zeros(2, dtype=float),
                        "uc": last_u.copy(),
                        "u": applied_u.copy(),
                    }
                )
            if not link.connected:
                reason = "connection_lost"
                break
            if latest_pose is None or latest_velocity is None:
                if elapsed > TELEMETRY_ABORT_S:
                    raise TimeoutError("no telemetry after start")
                link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
                time.sleep(0.01)
                continue
            age = now - last_live_time
            if age > TELEMETRY_ABORT_S:
                reason = f"telemetry_stall_{age:.2f}s"
                raise TimeoutError(reason)
            if age > TELEMETRY_TIMEOUT_S:
                link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
                time.sleep(0.01)
                continue
            if elapsed >= RUN_DURATION_S:
                completed = True
                reason = "complete"
                link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
                break

            if now - last_control >= CONTROL_PERIOD_S:
                dt = max(1.0e-3, min(0.2, now - last_control))
                reference_pose, _velocity = center_start_circle_reference(elapsed)
                u_r, u_l, tracking = controller.step(
                    latest_pose,
                    latest_velocity,
                    PathReference(
                        pose=tuple(reference_pose),
                        velocity=(CIRCLE_SPEED_MPS, CIRCLE_NU_RAD_S),
                    ),
                    dt, U_MAX_V, track_width, base_limit_frac=1.0,
                )
                last_command = np.asarray(
                    (tracking["v_command"], tracking["omega_command"]),
                    dtype=float,
                )
                max_step = MANUAL_VOLTAGE_SLEW_V_PER_S * dt
                requested = np.asarray((u_r, u_l), dtype=float)
                applied = np.clip(requested, last_u - max_step, last_u + max_step)
                radius = float(np.linalg.norm(latest_pose[:2]))
                if radius > MAX_RADIUS_M:
                    raise RuntimeError(f"workspace radius exceeded: {radius:.3f}m")
                position_error = float(
                    np.linalg.norm(latest_pose[:2] - reference_pose[:2])
                )
                if elapsed > SAFETY_GRACE_S and position_error > MAX_POSITION_ERROR_M:
                    raise RuntimeError(
                        f"position error exceeded: {position_error:.3f}m"
                    )
                link.send(
                    {"cmd": "manual", "u_r": float(applied[0]), "u_l": float(applied[1])}
                )
                last_u = applied
                last_control = now

            if now - next_report >= 2.0:
                reference_pose, _ = center_start_circle_reference(elapsed)
                position_error = float(
                    np.linalg.norm(latest_pose[:2] - reference_pose[:2])
                )
                print(
                    f"[E4 trial {trial}] t={elapsed:5.1f}/{RUN_DURATION_S:.1f}s · "
                    f"pos_err={position_error:.3f}m · "
                    f"u=({last_u[0]:+.2f},{last_u[1]:+.2f})",
                    flush=True,
                )
                next_report = now + 2.0
            time.sleep(0.005)
        else:
            reason = "timeout"
        if not completed and reason == "host_error":
            reason = "incomplete"
    except KeyboardInterrupt:
        reason = "operator_interrupt"
        raise
    except Exception as exc:
        if reason == "host_error":
            reason = f"host_error_{type(exc).__name__}"
        print(f"[E4 trial {trial}] ERROR: {exc}", flush=True)
    finally:
        try:
            link.send({"cmd": "manual", "u_r": 0.0, "u_l": 0.0})
            time.sleep(0.05)
            send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        except Exception:
            pass
        link.disconnect()

    result = summarize(rows, completed)
    result.update({"case": "E4", "trial": trial, "reason": reason})
    exported = export_trace(trial, reason, rows, firmware, config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if exported is not None:
        print(f"TRACE={exported}", flush=True)
    return result, exported


def summarize(rows: list[dict[str, Any]], completed: bool) -> dict[str, Any]:
    if not rows:
        return {"completed": False, "samples": 0}
    pose_error = np.asarray([row["pose_error"] for row in rows], dtype=float)
    voltage = np.asarray([row["u"] for row in rows], dtype=float)
    t = np.asarray([row["t"] for row in rows], dtype=float)
    cruise = (t >= 8.0) & (t <= RUN_DURATION_S - OUTER_PREVIEW_HORIZON_S)
    if np.any(cruise):
        cruise_rmse = float(
            np.sqrt(np.mean(np.sum(pose_error[cruise, :2] ** 2, axis=1)))
        )
    else:
        cruise_rmse = float("nan")
    return {
        "completed": bool(completed),
        "samples": len(rows),
        "duration_s": float(rows[-1]["t"]),
        "position_rmse_m": float(
            np.sqrt(np.mean(np.sum(pose_error[:, :2] ** 2, axis=1)))
        ),
        "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
        "cruise_position_rmse_m": cruise_rmse,
        "max_abs_voltage_v": float(np.max(np.abs(voltage))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required safety gate; the vehicle will move",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--start-trial", type=int, default=1)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required because this command moves the vehicle")
    if not 1 <= args.trials <= 5:
        parser.error("--trials must be between 1 and 5")
    all_results = []
    for offset in range(args.trials):
        trial = args.start_trial + offset
        result, _path = run_trial(trial)
        all_results.append(result)
        if not result.get("completed"):
            print("STOPPING_REPETITIONS_AFTER_INCOMPLETE_TRIAL", flush=True)
            break
        if offset + 1 < args.trials:
            time.sleep(2.0)
    print("RUN_SUMMARY=" + json.dumps(all_results, ensure_ascii=False), flush=True)
    return 0 if all(item.get("completed") for item in all_results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
