#!/usr/bin/env python3
"""Run E1/E2/E3 real-car comparison trials without opening the Tk GUI."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from joystick_car import (
    CarLink,
    PID_BENCHMARK_DEFAULTS,
    TRAJECTORY_PRESETS,
    THEORY_SAFETY_CONSECUTIVE_FRAMES,
    load_validated_weights_cache,
    persistent_results_dir,
    theory_config_for_preset,
    validated_weights_cache_path,
)


HOST = "192.168.4.1"
PORT = 8888
PRESET = "匀速圆（圆心起步，R 0.40 m，0.12 m/s，1 圈）"
# Keep these values identical to the current manuscript, firmware defaults,
# and all three comparison cases.
E1_TAU1 = 0.040
E1_TAU2 = 0.015
E1_OUTER_KP = 0.80
E1_OUTER_VBAR = 0.50
E1_OUTER_KTHETA = 0.70
E1_OUTER_PREVIEW_HORIZON = 1.20
E1_OUTER_CAPTURE_RADIUS = 0.010
E1_OUTER_BLEND_RADIUS = 0.030
E1_MOTOR_DEADZONE_V = 0.5
REQUIRED_FIRMWARE = "1.7.4"
COMMAND_TIMEOUT_S = 5.0
KEEPALIVE_PERIOD_S = 1.5


def wait_for_message(
    link: CarLink,
    predicate,
    timeout_s: float = COMMAND_TIMEOUT_S,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for message in link.drain():
            if predicate(message):
                return message
        if not link.connected:
            raise ConnectionError(link.status_text)
        time.sleep(0.01)
    raise TimeoutError("firmware response timeout")


def send_and_expect(
    link: CarLink,
    payload: dict[str, Any],
    expected_ack: str,
) -> None:
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
    )


def without_initial_nonlinear_compensation(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Keep the Z-feedback columns and zero only the six sigma columns."""
    modified = dict(payload)
    for key in ("w2", "w3"):
        matrix = np.asarray(payload[key], dtype=float).reshape(2, 8).copy()
        matrix[:, 2:] = 0.0
        modified[key] = matrix.reshape(-1).tolist()
    return modified


def comparison_config(
    case: str,
    adaptive_r2: float | None = None,
    adaptive_r3: float | None = None,
    preset: str = PRESET,
    motion_max_target_error: float | None = None,
    theory_max_position_error: float | None = None,
) -> dict[str, float]:
    # E3 is the adaptive-only ablation: it uses the same data-driven
    # controller as E1, but its initial nonlinear compensation is removed.
    # Firmware case 3 is the separate cascaded-PID benchmark and is not the
    # paper's E3 experiment.
    case_id = {"E1": 1, "E2": 2, "E3": 1}[case]
    r2 = 0.5 if adaptive_r2 is None else adaptive_r2
    r3 = 0.5 if adaptive_r3 is None else adaptive_r3
    if case == "E2":
        r2 = r3 = 0.0
    config = theory_config_for_preset(
        preset,
        tau1=E1_TAU1,
        tau2=E1_TAU2,
        outer_kp=E1_OUTER_KP,
        outer_vbar=E1_OUTER_VBAR,
        outer_ktheta=E1_OUTER_KTHETA,
        outer_preview_horizon=E1_OUTER_PREVIEW_HORIZON,
        outer_capture_radius=E1_OUTER_CAPTURE_RADIUS,
        outer_blend_radius=E1_OUTER_BLEND_RADIUS,
        motor_deadzone_v=E1_MOTOR_DEADZONE_V,
        r2=r2,
        r3=r3,
        comparison_case=case_id,
        **PID_BENCHMARK_DEFAULTS,
    )
    if motion_max_target_error is not None:
        limit = float(motion_max_target_error)
        if not math.isfinite(limit) or not 0.01 <= limit <= 1.0:
            raise ValueError("motion reference safety bound must be within 0.01–1.0 m")
        config["motion_max_target_error"] = limit
    if theory_max_position_error is not None:
        limit = float(theory_max_position_error)
        if not math.isfinite(limit) or not 0.01 <= limit <= 1.0:
            raise ValueError("theory position safety bound must be within 0.01–1.0 m")
        config["theory_max_position_error"] = limit
    return config


def telemetry_row(message: dict[str, Any], t0_us: int | None) -> tuple[dict, int]:
    sensors = message.get("s")
    control = message.get("c")
    if not isinstance(sensors, dict) or not isinstance(control, dict):
        raise ValueError("missing telemetry blocks")
    timestamp_us = int(message.get("t_us", 0))
    if t0_us is None or timestamp_us < t0_us:
        t0_us = timestamp_us
    row = {
        "t": max(0.0, (timestamp_us - t0_us) * 1.0e-6),
        "pose": np.asarray(sensors["pose"], dtype=float),
        "velocity": np.asarray(sensors["velocity"], dtype=float),
        "current": np.asarray(sensors["current"], dtype=float),
        "reference": np.asarray(control["reference"], dtype=float),
        "motion_reference": np.asarray(
            control.get("motion_reference", control["reference"]), dtype=float
        ),
        "pose_error": np.asarray(control["pose_error"], dtype=float),
        "alpha1": np.asarray(control["alpha1"], dtype=float),
        "beta1": np.asarray(control["beta1"], dtype=float),
        "beta1_dot": np.asarray(control["beta1_dot"], dtype=float),
        "alpha2": np.asarray(control["alpha2"], dtype=float),
        "beta2": np.asarray(control["beta2"], dtype=float),
        "beta2_dot": np.asarray(control["beta2_dot"], dtype=float),
        "z2": np.asarray(control["z2"], dtype=float),
        "z3": np.asarray(control["z3"], dtype=float),
        "uc": np.asarray(control["uc"], dtype=float),
        "u": np.asarray(control["u"], dtype=float),
    }
    expected = {
        "pose": 3,
        "velocity": 2,
        "current": 2,
        "reference": 3,
        "motion_reference": 3,
        "pose_error": 3,
        "alpha1": 2,
        "beta1": 2,
        "beta1_dot": 2,
        "alpha2": 2,
        "beta2": 2,
        "beta2_dot": 2,
        "z2": 2,
        "z3": 2,
        "uc": 2,
        "u": 2,
    }
    for key, size in expected.items():
        value = row[key]
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid {key}")
    return row, t0_us


def safety_violation(row: dict[str, Any], config: dict[str, float]) -> str | None:
    if row["t"] < config["theory_safety_grace"]:
        return None
    motion_error = float(
        np.linalg.norm(row["motion_reference"][:2] - row["reference"][:2])
    )
    position_error = float(np.linalg.norm(row["pose_error"][:2]))
    heading_error = abs(float(row["pose_error"][2]))
    z2_norm = float(np.linalg.norm(row["z2"]))
    z3_norm = float(np.linalg.norm(row["z3"]))
    if motion_error > config["motion_max_target_error"]:
        return f"motion_reference_error={motion_error:.4f}m"
    if position_error > config["theory_max_position_error"]:
        return f"position_error={position_error:.4f}m"
    if heading_error > config["theory_max_heading_error"]:
        return f"heading_error={math.degrees(heading_error):.2f}deg"
    if z2_norm > config["theory_max_z2"]:
        return f"z2_norm={z2_norm:.4f}"
    if z3_norm > config["theory_max_z3"]:
        return f"z3_norm={z3_norm:.4f}"
    return None


def metrics(rows: list[dict[str, Any]], completed: bool) -> dict[str, Any]:
    if not rows:
        return {"completed": False, "samples": 0}
    pose = np.asarray([row["pose"] for row in rows], dtype=float)
    reference = np.asarray([row["reference"] for row in rows], dtype=float)
    delta = reference[:, :2] - pose[:, :2]
    cosine = np.cos(pose[:, 2])
    sine = np.sin(pose[:, 2])
    heading_error = (
        reference[:, 2] - pose[:, 2] + np.pi
    ) % (2.0 * np.pi) - np.pi
    pose_error = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            heading_error,
        )
    )
    z2 = np.asarray([row["z2"] for row in rows], dtype=float)
    z3 = np.asarray([row["z3"] for row in rows], dtype=float)
    voltage = np.asarray([row["u"] for row in rows], dtype=float)
    return {
        "completed": bool(completed),
        "samples": len(rows),
        "duration_s": float(rows[-1]["t"]),
        "position_rmse_m": float(
            np.sqrt(np.mean(np.sum(pose_error[:, :2] ** 2, axis=1)))
        ),
        "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
        "z2_norm_rms": float(
            np.sqrt(np.mean(np.sum(z2 ** 2, axis=1)))
        ),
        "z3_norm_rms": float(
            np.sqrt(np.mean(np.sum(z3 ** 2, axis=1)))
        ),
        "max_abs_voltage_v": float(np.max(np.abs(voltage))),
    }


def export_trace(
    case: str,
    trial: int,
    reason: str,
    rows: list[dict[str, Any]],
    config: dict[str, float],
    firmware: str,
    weights_payload: dict[str, Any] | None,
    preset: str,
    adaptive_only: bool = False,
) -> Path | None:
    if not rows:
        return None
    payload: dict[str, Any] = {
        "schema": np.asarray([3], dtype=np.int64),
        "t": np.asarray([row["t"] for row in rows], dtype=float),
        "reason": np.asarray([reason]),
        "trajectory": np.asarray([preset]),
        "comparison_case": np.asarray([case]),
        "comparison_name": np.asarray(
            [
                (
                    "adaptive only (zero initial nonlinear weights)"
                    if (adaptive_only or case == "E3")
                    else {
                        "E1": "proposed adaptive data-driven",
                        "E2": "data-driven without adaptation",
                        "E3": "adaptive-only ablation (zero initial nonlinear weights)",
                    }[case]
                )
            ]
        ),
        "initial_nonlinear_compensation_enabled": np.asarray(
            [not (adaptive_only or case == "E3")], dtype=np.bool_
        ),
        "trial": np.asarray([trial], dtype=np.int64),
        "firmware": np.asarray([firmware]),
        "runtime_config_json": np.asarray(
            [json.dumps(config, ensure_ascii=False, sort_keys=True)]
        ),
        "weights_json": np.asarray(
            [json.dumps(weights_payload or {}, ensure_ascii=False, sort_keys=True)]
        ),
    }
    for key in rows[0]:
        if key != "t":
            payload[key] = np.asarray([row[key] for row in rows], dtype=float)
    target_dir = persistent_results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    case_slug = (
        f"{case.lower()}_adaptive_only"
        if (adaptive_only or case == "E3")
        else case.lower()
    )
    target = target_dir / f"theory_trace_{case_slug}_{stamp}_trial{trial}_{reason}.npz"
    np.savez_compressed(target, **payload)
    return target


def run_trial(
    case: str,
    trial: int,
    adaptive_r2: float | None = None,
    adaptive_r3: float | None = None,
    preset: str = PRESET,
    adaptive_only: bool = False,
    motion_max_target_error: float | None = None,
    theory_max_position_error: float | None = None,
    weights_payload_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    if adaptive_only and case not in ("E1", "E3"):
        raise ValueError("adaptive-only ablation is only valid for E1 or E3")
    run_label = (
        "E3-adaptive-only" if case == "E3"
        else "E1-adaptive-only" if adaptive_only
        else case
    )
    config = comparison_config(
        case, adaptive_r2, adaptive_r3, preset, motion_max_target_error,
        theory_max_position_error,
    )
    weights_payload = None
    if case in ("E1", "E2", "E3"):
        if weights_payload_override is None:
            weights_payload, _document = load_validated_weights_cache(
                validated_weights_cache_path()
            )
        else:
            weights_payload = dict(weights_payload_override)
        if adaptive_only or case == "E3":
            weights_payload = without_initial_nonlinear_compensation(
                weights_payload
            )
    link = CarLink()
    rows: list[dict[str, Any]] = []
    reason = "host_error"
    firmware = "?"
    completed = False
    t0_us: int | None = None
    consecutive_safety = 0
    try:
        link.connect_wifi(HOST, PORT)
        device = hello(link)
        firmware = str(device.get("firmware", "?"))
        if firmware != REQUIRED_FIRMWARE:
            raise RuntimeError(
                f"firmware {REQUIRED_FIRMWARE} required, got {firmware}"
            )
        print(
            f"[{run_label} trial {trial}] connected · fault={device.get('fault')} · "
            f"weights_valid={device.get('weights_valid')}",
            flush=True,
        )
        send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        time.sleep(0.20)
        send_and_expect(link, {"cmd": "configure", **config},
                        "configuration_updated")
        time.sleep(0.15)
        if weights_payload is not None:
            send_and_expect(
                link,
                {"cmd": "set_weights", **weights_payload},
                "weights_accepted",
            )
            time.sleep(0.15)
        send_and_expect(link, {"cmd": "zero_pose"}, "zero_pose_requested")
        time.sleep(0.15)
        send_and_expect(
            link, {"cmd": "start", "mode": "theory"}, "start_requested"
        )
        print(f"[{run_label} trial {trial}] STARTED · motors may move", flush=True)

        deadline = time.monotonic() + (
            config["run_duration"] + config.get("hold_duration", 0.0) + 8.0
        )
        next_keepalive = time.monotonic() + KEEPALIVE_PERIOD_S
        next_report = time.monotonic() + 2.0
        entered_theory = False
        last_fault = "none"
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_keepalive:
                link.send({"cmd": "hello"})
                next_keepalive = now + KEEPALIVE_PERIOD_S
            messages = link.drain()
            for message in messages:
                mode = str(message.get("mode", ""))
                fault = str(message.get("fault", "none"))
                if fault not in ("", "none", "?"):
                    last_fault = fault
                if message.get("type") == "telemetry" and mode == "theory":
                    entered_theory = True
                    try:
                        row, t0_us = telemetry_row(message, t0_us)
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append(row)
                    violation = safety_violation(row, config)
                    consecutive_safety = (
                        consecutive_safety + 1 if violation else 0
                    )
                    if consecutive_safety >= THEORY_SAFETY_CONSECUTIVE_FRAMES:
                        reason = "host_safety_" + (violation or "unknown")
                        send_and_expect(link, {"cmd": "stop"}, "stop_requested")
                        raise RuntimeError(reason)
                if entered_theory and mode == "idle":
                    reason = "complete" if last_fault == "none" else f"fault_{last_fault}"
                    completed = last_fault == "none"
                    break
            if entered_theory and reason != "host_error":
                break
            if not link.connected:
                reason = "connection_lost"
                break
            if now >= next_report and rows:
                last = rows[-1]
                print(
                    f"[{run_label} trial {trial}] t={last['t']:.1f}s · "
                    f"pos={np.linalg.norm(last['pose_error'][:2]):.4f}m · "
                    f"|z2|={np.linalg.norm(last['z2']):.3f} · "
                    f"|z3|={np.linalg.norm(last['z3']):.3f}",
                    flush=True,
                )
                next_report = now + 2.0
            time.sleep(0.01)
        else:
            reason = "timeout"
        if reason != "complete":
            try:
                send_and_expect(link, {"cmd": "stop"}, "stop_requested")
            except Exception:
                pass
    except KeyboardInterrupt:
        reason = "operator_interrupt"
        try:
            send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        except Exception:
            pass
        raise
    except Exception as exc:
        if not reason.startswith("host_safety_"):
            reason = f"host_error_{type(exc).__name__}"
        print(f"[{run_label} trial {trial}] ERROR: {exc}", flush=True)
        try:
            send_and_expect(link, {"cmd": "stop"}, "stop_requested")
        except Exception:
            pass
    finally:
        link.disconnect()

    result = metrics(rows, completed)
    result.update({"case": run_label, "trial": trial, "reason": reason})
    exported = export_trace(
        case, trial, reason, rows, config, firmware, weights_payload, preset,
        adaptive_only,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if exported is not None:
        print(f"TRACE={exported}", flush=True)
    return result, exported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("E1", "E2", "E3"), required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--start-trial", type=int, default=1)
    parser.add_argument("--r2", type=float)
    parser.add_argument("--r3", type=float)
    parser.add_argument(
        "--adaptive-only",
        action="store_true",
        help="keep W[:,0:2] feedback gains but zero W[:,2:8] nonlinear weights",
    )
    parser.add_argument(
        "--theory-max-position-error",
        type=float,
        help="theoretical position-error safety bound in metres "
             "(default 0.2; the official batch sets 0.5)",
    )
    parser.add_argument(
        "--motion-max-target-error",
        type=float,
        help="temporary motion-reference safety bound in metres",
    )
    parser.add_argument(
        "--preset", choices=tuple(TRAJECTORY_PRESETS), default=PRESET
    )
    args = parser.parse_args()
    if not 1 <= args.trials <= 5:
        parser.error("--trials must be between 1 and 5")
    if args.case != "E1" and (args.r2 is not None or args.r3 is not None):
        parser.error("--r2 and --r3 are only valid for E1")
    if args.adaptive_only and args.case != "E1":
        parser.error("--adaptive-only is only valid for E1")
    for name, value in (("--r2", args.r2), ("--r3", args.r3)):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"{name} must be finite and nonnegative")
    theory_max_position_error = args.theory_max_position_error
    if theory_max_position_error is not None:
        if not 0.01 <= theory_max_position_error <= 1.0:
            parser.error("--theory-max-position-error must be within 0.01-1.0 m")
        if math.isfinite(theory_max_position_error) is not True:
            parser.error("--theory-max-position-error must be finite")
    all_results = []
    for offset in range(args.trials):
        trial = args.start_trial + offset
        result, _path = run_trial(
            args.case, trial, args.r2, args.r3, args.preset,
            args.adaptive_only, args.motion_max_target_error,
            theory_max_position_error,
            weights_payload_override=None,
        )
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
