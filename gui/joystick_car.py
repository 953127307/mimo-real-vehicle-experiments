#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MIMO Car Keyboard Joystick — desktop controller for the differential-drive car.

Arrow keys  → drive the car (differential steering)
Space       → cycle through fixed voltage levels (1 V / 2 V / 3 V / 4 V)
ESC         → emergency stop + exit
Enter       → connect / disconnect
Tab         → toggle WiFi / USB serial mode

Protocol: JSON-lines over TCP (default 192.168.4.1:8888) or USB serial (460800 baud).
The app sends a keep-alive `hello` every 2 s so the firmware never latches NETWORK_LOST.
Releasing all direction keys immediately sends zero voltage (dead-man's switch).

Build:  pyinstaller --onefile --windowed --name MIMO-Joystick joystick_car.py
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import queue
import socket
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mimo_car_studio.synthesis import (
    IntegralSnapshotDataset,
    QualificationReport,
    REGRESSOR_BASIS,
    RawTimeSeriesDataset,
    SnapshotDataset,
    SynthesisResult,
    qualify_synthesis,
    route_ii_threshold,
    synthesize_dataset,
)

# ---------------------------------------------------------------------------
# Protocol constants (keep in sync with firmware)
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = 1
DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 8888
BAUD = 460800
KEEPALIVE_S = 2.0
CONNECT_HELLO_TIMEOUT_S = 12.0
CONNECT_STAGE_TIMEOUT_S = 8.0
CONNECT_RETRY_INTERVAL_S = 0.50
U_MAX_DEFAULT = 4.0  # volts, matches firmware hard motor-command ceiling
CONTROL_HZ = 25       # command send rate (Hz)
UI_REFRESH_MS = 40
PLOT_REFRESH_MS = 250
PLOT_WINDOW_S = 20.0
PLOT_MAX_SAMPLES = 1500
LOG_MAX_LINES = 500
MANUAL_VOLTAGE_SLEW_V_PER_S = 12.0
COLLECTION_DERIVATIVE_TAU_S = 0.10
GYRO_RATE_FILTER_TAU_S = 0.05
GYRO_RATE_DEADBAND_DEFAULT_RAD_S = 0.040
CURRENT_FILTER_TAU_S = 0.040
PID_COLLECTION_WARMUP_S = 3.0
INTEGRAL_SNAPSHOT_WINDOW_S = 0.20
PID_TELEMETRY_TIMEOUT_S = 0.75
PID_TELEMETRY_ABORT_S = 2.0
PID_COLLECTION_BASE_LIMIT_FRAC = 0.60
# User-selected PID pose-feedback source: use the MPU6050 yaw rate only.
# Encoder yaw remains available for a later direction-calibration workflow.
PID_COLLECTION_GYRO_BLEND = 1.0
QUALITY_UPDATE_SAMPLES = 100
THEORY_MAX_SAMPLES = 5000
THEORY_PLOT_WINDOW_S = 60.0
WORKFLOW_TIMEOUT_S = 4.0
WORKFLOW_SETTLE_MS = 120
WORKFLOW_SENSOR_READY_TIMEOUT_S = 2.0
WORKFLOW_SENSOR_READY_POLL_MS = 100
# Firmware averages 600 stationary MPU6050 samples at 3 ms/sample.  Leave a
# margin before the next command so no control command can arm the car while
# its yaw bias is still being estimated.
GYRO_CALIBRATION_SETTLE_MS = 2400
STOP_RETRY_INTERVAL_S = 0.25
THEORY_STOP_GRACE_S = 0.25
THEORY_MAX_POSITION_ERROR_M = 0.20
THEORY_MAX_HEADING_ERROR_RAD = math.radians(60.0)
THEORY_MAX_Z2_NORM = 2.00
THEORY_MAX_Z3_NORM = 2.00
THEORY_SAFETY_CONSECUTIVE_FRAMES = 3
THEORY_SAFETY_GRACE_S = 0.75
THEORY_BRINGUP_U_MAX_V = 4.0
MOTOR_DEADZONE_DEFAULT_V = 0.5
SYNTHESIS_DEFAULTS = {
    "kappa2_v": 1.0,
    "kappa2_w": 1.0,
    "kappa3_v": 2.0,
    "kappa3_w": 2.0,
    "epsilon2": 0.8,
    "epsilon3": 1.5,
    "dbar2": 0.004,
    "dbar3": 0.008,
}
THEORY_FILTER_DEFAULTS = {
    "tau1": 0.040,
    "tau2": 0.015,
}
GUIDANCE_DEFAULTS = {
    # Previewed polar capture; keep identical to the manuscript simulation.
    "outer_kp": 0.80,
    "outer_vbar": 0.50,
    "outer_ktheta": 0.70,
    "outer_preview_horizon": 1.40,
    "outer_capture_radius": 0.010,
    "outer_blend_radius": 0.030,
}
MOTION_REFERENCE_DEFAULTS = {
    "motion_kx": 2.0,
    "motion_ky": 6.0,
    "motion_kth": 3.0,
    "motion_max_v": 0.10,
    "motion_max_omega": 1.00,
    "motion_max_accel": 0.20,
    "motion_max_angular_accel": 2.50,
    "outer_max_v": 0.30,
    "outer_max_omega": 3.00,
    "motion_max_target_error": 0.20,
}
RUNTIME_CONTROL_DEFAULTS = {
    "motor_deadzone_v": MOTOR_DEADZONE_DEFAULT_V,
    # THEORY 自适应学习率：实车默认 0.5；用户仍可设为 0 关闭在线更新。
    # 固件自适应律 mhat_dot = r*(z_driven*sigma - delta*mhat)。
    "r2": 0.5,
    "r3": 0.5,
}
PID_BENCHMARK_DEFAULTS = {
    # E3: same previewed polar-capture outer loop and DSC filters, with a conventional
    # wheel-velocity -> current-reference -> motor-current PID cascade.
    "pid_velocity_ff": 0.45,
    "pid_velocity_kp": 0.25,
    "pid_velocity_ki": 0.10,
    "pid_velocity_kd": 0.0,
    "pid_current_kp": 10.0,
    "pid_current_ki": 20.0,
    "pid_current_kd": 0.0,
    "pid_voltage_ff": 20.0,
    "pid_current_ref_max": 0.30,
    "pid_derivative_tau": 0.020,
}
COMPARISON_CASES = {
    "E1 完整数据驱动（自适应）": {
        "case_id": 1,
        "short_name": "E1",
        "requires_weights": True,
        "description": "Route II 权重 + 在线自适应",
    },
    "E2 数据驱动（关闭自适应）": {
        "case_id": 2,
        "short_name": "E2",
        "requires_weights": True,
        "description": "相同 Route II 权重，固件强制 r₂=r₃=0",
    },
    "E3 级联 PID（速度–电流双级）": {
        "case_id": 3,
        "short_name": "E3",
        "requires_weights": False,
        "description": "同一外环和滤波器，内环改为速度–电流级联 PID",
    },
}
# 论文 tp13_k20_k30 战役的固定参数，逐项取自冻结轨迹的 runtime_config_json
# （T_p=1.3 s 为该战役实值，不是 GUI 默认的 1.40 s；勿随其它战役改动）。
PAPER_CAMPAIGN = {
    "preset": "匀速圆（圆心起步，R 0.40 m，0.12 m/s，1 圈）",
    "tau1": 0.040,
    "tau2": 0.015,
    "outer_kp": 0.80,
    "outer_vbar": 0.50,
    "outer_ktheta": 0.70,
    "outer_preview_horizon": 1.30,
    "outer_capture_radius": 0.010,
    "outer_blend_radius": 0.030,
    "motor_deadzone_v": 0.5,
    "r2": 0.5,
    "r3": 0.5,
    "theory_max_position_error": 0.60,
    "theory_max_heading_error": math.radians(60.0),
    "theory_max_z2": 2.0,
    "theory_max_z3": 2.0,
    "motion_max_target_error": 0.60,
}
PAPER_CASE_LABELS = {
    "E1": "E1 完整数据驱动（自适应）",
    "E2": "E2 数据驱动（关闭自适应）",
    "E3": "E3 自适应消融（初始非线性补偿清零）",
}
# E4 采样 PID 基线：PC 端 25 Hz 位姿 P 环 + 轮压 PI，base_limit_frac=1.0。
PAPER_E4_RUN_DURATION_S = 2.0 * math.pi / 0.30 + 3.0
PAPER_E4_CONTROL_PERIOD_S = 1.0 / 25.0
PAPER_E4_CIRCLE_RADIUS_M = 0.40
PAPER_E4_CIRCLE_NU_RAD_S = 0.30
PAPER_E4_CIRCLE_SPEED_MPS = 0.12
# K460 训练集用的 PID 采集预设与窗口（0.56 m × 0.24 m 双纽线 55 s、T_w=0.10 s、
# 1.5× 速度倍率——与冻结数据集元数据及 main.tex L766-771 一致）。
PAPER_COLLECTION_PATH_NAME = "中型双纽线（0.56 m × 0.24 m，55 s）"
PAPER_COLLECTION_SPEED_SCALE = "1.5"
PAPER_COLLECTION_WINDOW_S = "0.10"
THEORY_SAFETY_DEFAULTS = {
    "theory_max_position_error": THEORY_MAX_POSITION_ERROR_M,
    "theory_max_heading_error_deg": math.degrees(THEORY_MAX_HEADING_ERROR_RAD),
    "theory_max_z2": THEORY_MAX_Z2_NORM,
    "theory_max_z3": THEORY_MAX_Z3_NORM,
}
# 全部可调参数集中在"参数调节"弹窗中编辑（主界面不放置输入框）。
SYNTHESIS_PARAMETER_LABELS = (
    ("kappa2_v", "κ2v(选定)"),
    ("kappa2_w", "κ2ω(选定)"),
    ("kappa3_v", "κ3v(选定)"),
    ("kappa3_w", "κ3ω(选定)"),
    ("epsilon2", "ε₂"),
    ("epsilon3", "ε₃"),
    ("dbar2", "d̄₂(设计)"),
    ("dbar3", "d̄₃(设计)"),
)
RUNTIME_PARAMETER_LABELS = (
    ("tau1", "τ₁(s)"),
    ("tau2", "τ₂(s)"),
    ("outer_kp", "外环 kρ"),
    ("outer_vbar", "外环 v̄c(m/s)"),
    ("outer_ktheta", "外环 kα"),
    ("outer_preview_horizon", "预瞄 Tp(s)"),
    ("outer_capture_radius", "捕获半径 rs(m)"),
    ("outer_blend_radius", "渐消边界 rb(m)"),
    ("motor_deadzone_v", "死区±(V)"),
    ("r2", "r₂(自适应率)"),
    ("r3", "r₃(自适应率)"),
    ("theory_max_position_error", "位置界(m)"),
    ("theory_max_heading_error_deg", "航向界(°)"),
    ("theory_max_z2", "||z₂||界"),
    ("theory_max_z3", "||z₃||界"),
)
PID_PARAMETER_LABELS = (
    ("pid_velocity_ff", "速度前馈 A/(m/s)"),
    ("pid_velocity_kp", "速度 Kp"),
    ("pid_velocity_ki", "速度 Ki"),
    ("pid_velocity_kd", "速度 Kd"),
    ("pid_current_kp", "电流 Kp"),
    ("pid_current_ki", "电流 Ki"),
    ("pid_current_kd", "电流 Kd"),
    ("pid_voltage_ff", "电压前馈 V/(m/s)"),
    ("pid_current_ref_max", "电流参考限幅(A)"),
    ("pid_derivative_tau", "PID微分滤波(s)"),
)
ALL_PARAMETER_LABELS = (
    SYNTHESIS_PARAMETER_LABELS + RUNTIME_PARAMETER_LABELS + PID_PARAMETER_LABELS
)
ALL_PARAMETER_DEFAULTS = (
    SYNTHESIS_DEFAULTS | THEORY_FILTER_DEFAULTS | GUIDANCE_DEFAULTS
    | RUNTIME_CONTROL_DEFAULTS
    | THEORY_SAFETY_DEFAULTS | PID_BENCHMARK_DEFAULTS
)
VALIDATED_WEIGHTS_SCHEMA = 8
PAPER_QUALIFICATION_METHOD = "firmware_integral_v3_sgn_gyro_gate_route_ii_v1"
VALIDATED_WEIGHTS_FILENAME = "last_validated_weights.json"
SNAPSHOT_CACHE_FILENAME = "last_integral_dataset_v6_sgn_gyro_gate.npz"
SNAPSHOT_CACHE_INTERVAL_MS = 2000


def validated_weights_cache_path() -> Path:
    """Return a stable per-user path that also works from a one-file EXE."""
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home() / ".mimo_car"
    return base / "MIMO-Car" / VALIDATED_WEIGHTS_FILENAME


def snapshot_dataset_cache_path() -> Path:
    """Return the persistent cache path for the most recent snapshots."""
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home() / ".mimo_car"
    return base / "MIMO-Car" / SNAPSHOT_CACHE_FILENAME


def persistent_results_dir() -> Path:
    """Return a user-visible export directory outside PyInstaller temp files."""
    return Path.home() / "Documents" / "MIMO-Car" / "closed_loop_results"


def _validated_weights_digest(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_upload_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("权重载荷格式无效")
    normalized = dict(payload)
    if normalized.get("regressor_basis") != REGRESSOR_BASIS:
        raise ValueError("权重回归基与当前 sgn 固件不兼容")
    if normalized.get("route") != "route_ii":
        raise ValueError("权重不是由 Route II 综合生成")
    for key in ("w2", "w3"):
        values = np.asarray(normalized.get(key), dtype=float)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{key} 必须包含 16 个有限数值")
        normalized[key] = values.tolist()
    for key in (
        "kappa2", "kappa3", "kappa_min2", "kappa_min3",
        "sigma_margin2", "sigma_margin3", "lambda2", "lambda3",
        "epsilon2", "epsilon3", "dbar2", "dbar3",
        "certificate_max2", "certificate_max3", "xi_max2", "xi_max3",
    ):
        value = float(normalized.get(key))
        if not math.isfinite(value):
            raise ValueError(f"{key} 不是有限数值")
        normalized[key] = value
    for key in ("rank2", "rank3"):
        value = int(normalized.get(key))
        if value != 10:
            raise ValueError(f"{key} 未达到满秩 10")
        normalized[key] = value
    for block in (2, 3):
        kappa = normalized[f"kappa{block}"]
        kappa_min = normalized[f"kappa_min{block}"]
        if kappa <= kappa_min:
            raise ValueError(
                f"kappa{block}={kappa:g} 必须严格大于最小值 {kappa_min:g}"
            )
        if normalized[f"sigma_margin{block}"] <= 0.0:
            raise ValueError(f"Route II 块 {block} 的投影数据裕度非正")
        if not math.isclose(
            normalized[f"lambda{block}"], -kappa,
            rel_tol=1.0e-9, abs_tol=1.0e-12,
        ):
            raise ValueError("固件兼容字段与 Route II kappa 不一致")
        if not math.isclose(
            normalized[f"certificate_max{block}"],
            normalized[f"xi_max{block}"],
            rel_tol=1.0e-9, abs_tol=1.0e-12,
        ):
            raise ValueError("Route II 证书与固件兼容字段不一致")
    if normalized["epsilon2"] <= 0.0 or normalized["epsilon3"] <= 0.0:
        raise ValueError("epsilon 必须为正")
    if normalized["dbar2"] <= 0.0 or normalized["dbar3"] <= 0.0:
        raise ValueError("论文假设要求 dbar 必须为正")
    if normalized["xi_max2"] > 0.0 or normalized["xi_max3"] > 0.0:
        raise ValueError("权重未通过 Route II 范数证书校验")
    for key in ("kappa2_v", "kappa2_w", "kappa3_v", "kappa3_w"):
        if key in normalized:
            value = float(normalized[key])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} 必须为正")
            normalized[key] = value
    return normalized


def save_validated_weights_cache(
    path: str | Path,
    result: SynthesisResult,
    source_dataset: str | None = None,
    qualification: QualificationReport | None = None,
) -> dict[str, Any]:
    if not result.valid:
        raise ValueError("只能保存通过论文 Route II 条件的权重")
    if qualification is not None:
        if qualification.method != PAPER_QUALIFICATION_METHOD:
            raise ValueError("诊断方法与当前论文条件流程不一致")
        for block, design, kappa in (
            (qualification.block2, result.dbar2, result.kappa2),
            (qualification.block3, result.dbar3, result.kappa3),
        ):
            if not math.isclose(
                block.design_dbar, design, rel_tol=1.0e-9, abs_tol=1.0e-12
            ):
                raise ValueError("诊断中的设计扰动界与综合结果不一致")
            if not math.isclose(
                block.kappa, kappa, rel_tol=1.0e-9, abs_tol=1.0e-12
            ):
                raise ValueError("诊断中的 kappa 与综合结果不一致")
    # IMU 机制已取消：ω 全部来自编码器，gyro_deadband 不再是数据定义，
    # operating_config 保留字段仅为兼容旧权重文件（恒 0）。
    document: dict[str, Any] = {
        "schema": VALIDATED_WEIGHTS_SCHEMA,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": int(result.samples),
        "source_dataset": source_dataset or "",
        "operating_config": {"gyro_deadband": 0.0},
        "payload": _validate_upload_payload(result.upload_payload()),
        "validation": {
            "block2": {
                "rank": int(result.block2.rank),
                "rows": int(result.block2.rows),
                "match_residual": float(result.block2.match_residual),
                "xi_max": float(result.block2.xi_max),
                "kappa": float(result.block2.kappa),
                "kappa_min": float(result.block2.kappa_min),
                "spectral_margin": float(result.block2.spectral_margin),
            },
            "block3": {
                "rank": int(result.block3.rank),
                "rows": int(result.block3.rows),
                "match_residual": float(result.block3.match_residual),
                "xi_max": float(result.block3.xi_max),
                "kappa": float(result.block3.kappa),
                "kappa_min": float(result.block3.kappa_min),
                "spectral_margin": float(result.block3.spectral_margin),
            },
        },
        "qualification": (
            qualification.as_dict() if qualification is not None else None
        ),
    }
    document["sha256"] = _validated_weights_digest(document)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(target)
    return document


def load_validated_weights_cache(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("已验证权重文件格式无效")
    if document.get("schema") != VALIDATED_WEIGHTS_SCHEMA:
        raise ValueError("已验证权重文件版本不受支持")
    expected_digest = document.get("sha256")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if not isinstance(expected_digest, str) or expected_digest != _validated_weights_digest(unsigned):
        raise ValueError("已验证权重文件完整性校验失败")
    samples = int(document.get("samples", 0))
    if samples < 10:
        raise ValueError("已验证权重的样本数不足")
    operating_config = document.get("operating_config")
    if not isinstance(operating_config, dict):
        raise ValueError("缺少权重对应的陀螺角速度阈值")
    try:
        validate_gyro_deadband(operating_config["gyro_deadband"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("权重对应的陀螺角速度阈值无效") from exc
    validation = document.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("缺少权重校验摘要")
    for name in ("block2", "block3"):
        block = validation.get(name)
        if not isinstance(block, dict):
            raise ValueError(f"缺少 {name} 校验摘要")
        if int(block.get("rank", 0)) != 10 or int(block.get("rows", 0)) != 10:
            raise ValueError(f"{name} 未通过满秩校验")
        residual = float(block.get("match_residual", math.inf))
        xi_max = float(block.get("xi_max", math.inf))
        kappa = float(block.get("kappa", -math.inf))
        kappa_min = float(block.get("kappa_min", math.inf))
        spectral_margin = float(block.get("spectral_margin", -math.inf))
        if not math.isfinite(residual) or residual > 1.0e-6:
            raise ValueError(f"{name} 未通过匹配残差校验")
        if not math.isfinite(xi_max) or xi_max > 0.0:
            raise ValueError(f"{name} 未通过 Route II 范数证书校验")
        if not (math.isfinite(kappa) and math.isfinite(kappa_min)) or kappa <= kappa_min:
            raise ValueError(f"{name} 的 kappa 未严格超过最小值")
        if not math.isfinite(spectral_margin) or spectral_margin <= 0.0:
            raise ValueError(f"{name} 未通过 Route II 投影数据条件")
    qualification = document.get("qualification")
    if qualification is not None:
        if not isinstance(qualification, dict):
            raise ValueError("权重诊断摘要格式无效")
        if qualification.get("method") != PAPER_QUALIFICATION_METHOD:
            raise ValueError("权重不是由当前 Route II 积分快照流程生成")
    payload = _validate_upload_payload(document.get("payload"))
    return payload, document
TRAJECTORY_PRESETS = {
    "低速直线往返（约 0.6 m 行程）": {
        "ref_a": 0.06,
        "ref_b": 0.0,
        "ref_nu": 0.20,
        "run_duration": 35.0,
    },
    "小型双纽线（0.20 m × 0.10 m）": {
        "ref_a": 0.20,
        "ref_b": 0.10,
        "ref_nu": 0.20,
        "run_duration": 35.0,
    },
    "中型双纽线（0.35 m × 0.18 m）": {
        "ref_a": 0.35,
        "ref_b": 0.18,
        "ref_nu": 0.16,
        "run_duration": 45.0,
    },
    "大圆（R 0.40 m，约 0.12 m/s，1 圈）": {
        "ref_a": 0.40,
        "ref_b": 0.0,
        "ref_nu": 0.30,
        "ref_shape": 1.0,
        "run_duration": 23.0,
    },
    "四叶草（R 0.35 m，约 0.07–0.14 m/s，1 圈）": {
        "ref_a": 0.35,
        "ref_b": 0.0,
        "ref_nu": 0.20,
        "ref_shape": 2.0,
        "run_duration": 33.5,
    },
    # 匀速圆、圆心起步：圆心为车上电零点位姿，q_r(0)=(R,0) 在车正前方，
    # 固件 ref_shape=3：从 t=0 开始以恒定速度完成一整圈，并继续沿圆周
    # 运行 3 s，随后由自主停止逻辑在当前圆周位置直接停车。
    "匀速圆（圆心起步，R 0.40 m，0.12 m/s，1 圈）": {
        "ref_a": 0.40,
        "ref_b": 0.0,
        "ref_nu": 0.30,
        "ref_shape": 3.0,
        "run_duration": 2 * math.pi / 0.30 + 3.0,
        "hold_duration": 0.0,
    },
}
PID_COLLECTION_PRESETS = {
    "小型双纽线（0.36 m × 0.16 m，45 s）": {
        "a": 0.18,
        "b": 0.08,
        "nu": 0.24,
        "duration": 45.0,
    },
    "中型双纽线（0.56 m × 0.24 m，55 s）": {
        "a": 0.28,
        "b": 0.12,
        "nu": 0.20,
        "duration": 55.0,
    },
    "大圆（R 0.40 m，0.12 m/s，1 圈 26 s）": {
        "a": 0.40,
        "b": 0.0,
        "nu": 0.30,
        "shape": "circle",
        "duration": 26.0,
    },
}
SNAPSHOT_FIELDS = {
    "zdot2": 2,
    "y2": 8,
    "x3": 2,
    "zdot3": 2,
    "y3": 8,
    "x4": 2,
}


def driver_axes(pressed: set[str]) -> tuple[float, float]:
    """Return normalized forward and turning commands from held direction keys."""
    forward = float("up" in pressed) - float("down" in pressed)
    turning = float("left" in pressed) - float("right" in pressed)
    return forward, turning


def wrap_angle(value: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PathReference:
    pose: tuple[float, float, float]
    velocity: tuple[float, float]


def pid_collection_reference(
    elapsed_s: float,
    preset: dict[str, float],
) -> PathReference:
    """Return a smoothly started, heading-aligned reference.

    Shape is selected by preset["shape"]: "circle" (x=R sin νt, y=R(1−cos νt),
    constant v=Rν, ω=ν — matches the THEORY large-circle preset) or the
    default figure-eight lemniscate.
    """
    t = max(0.0, float(elapsed_s))
    a = float(preset["a"])
    b = float(preset["b"])
    nu = float(preset["nu"])
    shape = str(preset.get("shape", "lemniscate"))
    ramp_s = PID_COLLECTION_WARMUP_S
    q = min(1.0, t / ramp_s)
    window = q**3 * (10.0 - 15.0 * q + 6.0 * q**2)
    window_dot = (
        (30.0 * q**2 - 60.0 * q**3 + 30.0 * q**4) / ramp_s
        if q < 1.0
        else 0.0
    )
    if q < 1.0:
        effective_time = ramp_s * (2.5 * q**4 - 3.0 * q**5 + q**6)
    else:
        effective_time = t - 0.5 * ramp_s
    phase = nu * effective_time
    phase_dot = nu * window
    phase_ddot = nu * window_dot

    if shape == "circle":
        raw_x = a * math.sin(phase)
        raw_y = a * (1.0 - math.cos(phase))
        raw_x_phase = a * math.cos(phase)
        raw_y_phase = a * math.sin(phase)
        raw_x_phase2 = -a * math.sin(phase)
        raw_y_phase2 = a * math.cos(phase)
    else:
        raw_x = a * math.sin(phase)
        raw_y = b * math.sin(2.0 * phase)
        raw_x_phase = a * math.cos(phase)
        raw_y_phase = 2.0 * b * math.cos(2.0 * phase)
        raw_x_phase2 = -a * math.sin(phase)
        raw_y_phase2 = -4.0 * b * math.sin(2.0 * phase)
    raw_dx = raw_x_phase * phase_dot
    raw_dy = raw_y_phase * phase_dot
    raw_ddx = raw_x_phase2 * phase_dot**2 + raw_x_phase * phase_ddot
    raw_ddy = raw_y_phase2 * phase_dot**2 + raw_y_phase * phase_ddot

    # Rotate the curve so its initial tangent agrees with the zeroed car
    # heading (phase=0 tangent angle: 0 for the circle, atan2(2b, a) for the
    # lemniscate).
    if shape == "circle":
        rotation = 0.0
    else:
        rotation = -math.atan2(2.0 * b, a)
    c = math.cos(rotation)
    s = math.sin(rotation)

    def rotate(x_value: float, y_value: float) -> tuple[float, float]:
        return c * x_value - s * y_value, s * x_value + c * y_value

    x_ref, y_ref = rotate(raw_x, raw_y)
    dx_ref, dy_ref = rotate(raw_dx, raw_dy)
    ddx_ref, ddy_ref = rotate(raw_ddx, raw_ddy)
    speed_squared = dx_ref**2 + dy_ref**2
    speed = math.sqrt(speed_squared)
    if speed_squared > 1.0e-8:
        theta_ref = math.atan2(dy_ref, dx_ref)
        omega_ref = (dx_ref * ddy_ref - dy_ref * ddx_ref) / speed_squared
    else:
        theta_ref = 0.0
        omega_ref = 0.0
    return PathReference(
        pose=(x_ref, y_ref, theta_ref),
        velocity=(speed, omega_ref),
    )


class PidPathController:
    """Cascaded pose/velocity PID used only for open-weight data collection."""

    OUTER_KX = 1.8
    OUTER_KY = 4.0
    OUTER_KTH = 3.2
    WHEEL_FF_V_PER_MPS = 42.0
    WHEEL_KP = 14.0
    WHEEL_KI = 2.0
    WHEEL_STATIC_V = 0.20
    WHEEL_STATIC_SPEED_MPS = 0.006
    WHEEL_INTEGRAL_LIMIT = 0.25

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.wheel_integral = np.zeros(2, dtype=float)

    def step(
        self,
        pose: np.ndarray,
        velocity: np.ndarray,
        reference: PathReference,
        dt: float,
        u_max: float,
        track_width_m: float = 0.2035,
        base_limit_frac: float = PID_COLLECTION_BASE_LIMIT_FRAC,
    ) -> tuple[float, float, dict[str, float]]:
        dt = float(np.clip(dt, 1.0e-3, 0.2))
        x_ref, y_ref, theta_ref = reference.pose
        v_ref, omega_ref = reference.velocity
        dx = x_ref - float(pose[0])
        dy = y_ref - float(pose[1])
        theta = float(pose[2])
        c = math.cos(theta)
        s = math.sin(theta)
        e_x = c * dx + s * dy
        e_y = -s * dx + c * dy
        e_theta = wrap_angle(theta_ref - theta)

        v_command = (
            v_ref * math.cos(e_theta)
            + self.OUTER_KX * e_x
        )
        omega_command = (
            omega_ref
            + self.OUTER_KY * e_y
            + self.OUTER_KTH * e_theta
        )
        v_command = float(np.clip(v_command, -0.10, 0.13))
        omega_command = float(np.clip(omega_command, -1.8, 1.8))

        half_track = 0.5 * track_width_m
        wheel_target = np.asarray(
            (v_command + half_track * omega_command,
             v_command - half_track * omega_command),
            dtype=float,
        )
        wheel_measured = np.asarray(
            (float(velocity[0]) + half_track * float(velocity[1]),
             float(velocity[0]) - half_track * float(velocity[1])),
            dtype=float,
        )
        wheel_error = wheel_target - wheel_measured
        proposed_integral = np.clip(
            self.wheel_integral + wheel_error * dt,
            -self.WHEEL_INTEGRAL_LIMIT,
            self.WHEEL_INTEGRAL_LIMIT,
        )
        static_compensation = self.WHEEL_STATIC_V * np.sign(wheel_target) * (
            np.abs(wheel_target) >= self.WHEEL_STATIC_SPEED_MPS
        )
        raw_voltage = (
            self.WHEEL_FF_V_PER_MPS * wheel_target
            + self.WHEEL_KP * wheel_error
            + self.WHEEL_KI * proposed_integral
            + static_compensation
        )
        base_limit = u_max * base_limit_frac
        peak_voltage = float(np.max(np.abs(raw_voltage)))
        saturation_scale = min(1.0, base_limit / max(peak_voltage, 1.0e-9))
        output_voltage = raw_voltage * saturation_scale
        if saturation_scale >= 1.0:
            self.wheel_integral = proposed_integral
        right = float(output_voltage[0])
        left = float(output_voltage[1])
        return right, left, {
            "e_x": e_x,
            "e_y": e_y,
            "e_theta": e_theta,
            "v_command": v_command,
            "omega_command": omega_command,
        }


def body_frame_error(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """论文 E4 行记录用的机体系参考-位姿误差（与冻结 runner 一致）。"""
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


def center_start_circle_reference(
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """战役 ref_shape=3 参考：圆心起步、t=0 即全速。

    q_r(t) = (R cos νt, R sin νt)，theta_r = π/2 + νt，v = 0.12 m/s，
    ω = ν = 0.30 rad/s；与冻结 E4 runner 的 PC 侧参考逐点一致。
    """
    t = max(0.0, float(elapsed_s))
    phase = PAPER_E4_CIRCLE_NU_RAD_S * t
    pose = np.asarray(
        (
            PAPER_E4_CIRCLE_RADIUS_M * math.cos(phase),
            PAPER_E4_CIRCLE_RADIUS_M * math.sin(phase),
            math.pi / 2.0 + phase,
        ),
        dtype=float,
    )
    velocity = np.asarray(
        (PAPER_E4_CIRCLE_SPEED_MPS, PAPER_E4_CIRCLE_NU_RAD_S), dtype=float
    )
    return pose, velocity


def compute_manual_voltages(
    pressed: set[str],
    u_max: float,
    voltage_level: float,
) -> tuple[float, float]:
    """Convert driver intent to pure differential-drive wheel voltages."""
    forward, turning = driver_axes(pressed)
    if forward == 0.0 and turning == 0.0:
        return 0.0, 0.0

    base_limit = min(max(0.0, float(voltage_level)), max(0.0, float(u_max)))
    base_right = float(np.clip(base_limit * (forward + turning), -base_limit, base_limit))
    base_left = float(np.clip(base_limit * (forward - turning), -base_limit, base_limit))
    return base_right, base_left


def stacked_matrix_quality(
    zdot: np.ndarray, y: np.ndarray
) -> tuple[int, float, float]:
    """Return rank, smallest singular value, and condition of G=[Zdot;Y]."""
    matrix = np.vstack((np.asarray(zdot, dtype=float), np.asarray(y, dtype=float)))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(singular_values > 1.0e-9))
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    condition = (
        float(singular_values[0] / sigma_min)
        if rank == matrix.shape[0] and sigma_min > 0.0
        else float("inf")
    )
    return rank, sigma_min, condition


def should_collect_snapshot(
    operation_mode: str,
    controller_ready: bool = True,
) -> bool:
    """Accept synthesis data only from the settled PID path collector."""
    return operation_mode == "pid_collect" and controller_ready


def without_zero_voltage_samples(
    dataset: SnapshotDataset | RawTimeSeriesDataset,
    tolerance: float = 1.0e-6,
) -> tuple[SnapshotDataset, int]:
    """Return samples whose two-channel applied-voltage vector is nonzero."""
    matrices = dataset.matrices()
    keep = np.max(np.abs(matrices["x4"]), axis=0) > tolerance
    filtered = SnapshotDataset()
    for index in np.flatnonzero(keep):
        filtered.append(
            {key: matrices[key][:, index] for key in SnapshotDataset.REQUIRED}
        )
    return filtered, int(keep.size - np.count_nonzero(keep))


def comparison_case_runtime(
    name: str,
    requested_r2: float,
    requested_r3: float,
) -> dict[str, Any]:
    """Return normalized runtime settings for an E1/E2/E3 experiment."""
    if name not in COMPARISON_CASES:
        raise ValueError(f"未知对比方案：{name}")
    r2 = float(requested_r2)
    r3 = float(requested_r3)
    if not math.isfinite(r2) or r2 < 0.0:
        raise ValueError("r₂（速度块自适应率）必须为不小于 0 的有限数值")
    if not math.isfinite(r3) or r3 < 0.0:
        raise ValueError("r₃（电流块自适应率）必须为不小于 0 的有限数值")
    spec = dict(COMPARISON_CASES[name])
    if int(spec["case_id"]) != 1:
        r2 = 0.0
        r3 = 0.0
    return {**spec, "name": name, "r2": r2, "r3": r3}


def without_initial_nonlinear_compensation(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """论文 E3 消融：保留 w2/w3 的 Z 反馈列，仅把六个 sigma 列清零。

    与冻结战役 run_comparison_headless.py 的实现逐字一致：固件仍运行
    case 1 完整控制器，消融完全由上传的权重载荷定义。
    """
    modified = dict(payload)
    for key in ("w2", "w3"):
        matrix = np.asarray(payload[key], dtype=float).reshape(2, 8).copy()
        matrix[:, 2:] = 0.0
        modified[key] = matrix.reshape(-1).tolist()
    return modified


def theory_config_for_preset(
    name: str,
    *,
    tau1: float = THEORY_FILTER_DEFAULTS["tau1"],
    tau2: float = THEORY_FILTER_DEFAULTS["tau2"],
    outer_kp: float = GUIDANCE_DEFAULTS["outer_kp"],
    outer_vbar: float = GUIDANCE_DEFAULTS["outer_vbar"],
    outer_ktheta: float = GUIDANCE_DEFAULTS["outer_ktheta"],
    outer_preview_horizon: float = GUIDANCE_DEFAULTS["outer_preview_horizon"],
    outer_capture_radius: float = GUIDANCE_DEFAULTS["outer_capture_radius"],
    outer_blend_radius: float = GUIDANCE_DEFAULTS["outer_blend_radius"],
    motor_deadzone_v: float = MOTOR_DEADZONE_DEFAULT_V,
    r2: float = RUNTIME_CONTROL_DEFAULTS["r2"],
    r3: float = RUNTIME_CONTROL_DEFAULTS["r3"],
    theory_max_position_error: float = THEORY_MAX_POSITION_ERROR_M,
    theory_max_heading_error: float = THEORY_MAX_HEADING_ERROR_RAD,
    theory_max_z2: float = THEORY_MAX_Z2_NORM,
    theory_max_z3: float = THEORY_MAX_Z3_NORM,
    comparison_case: int = 1,
    pid_velocity_ff: float = PID_BENCHMARK_DEFAULTS["pid_velocity_ff"],
    pid_velocity_kp: float = PID_BENCHMARK_DEFAULTS["pid_velocity_kp"],
    pid_velocity_ki: float = PID_BENCHMARK_DEFAULTS["pid_velocity_ki"],
    pid_velocity_kd: float = PID_BENCHMARK_DEFAULTS["pid_velocity_kd"],
    pid_current_kp: float = PID_BENCHMARK_DEFAULTS["pid_current_kp"],
    pid_current_ki: float = PID_BENCHMARK_DEFAULTS["pid_current_ki"],
    pid_current_kd: float = PID_BENCHMARK_DEFAULTS["pid_current_kd"],
    pid_voltage_ff: float = PID_BENCHMARK_DEFAULTS["pid_voltage_ff"],
    pid_current_ref_max: float = PID_BENCHMARK_DEFAULTS["pid_current_ref_max"],
    pid_derivative_tau: float = PID_BENCHMARK_DEFAULTS["pid_derivative_tau"],
) -> dict[str, float]:
    """Build the firmware configuration for a named THEORY trajectory."""
    if name not in TRAJECTORY_PRESETS:
        raise ValueError(f"unknown trajectory preset: {name}")
    if not math.isfinite(tau1) or not math.isfinite(tau2):
        raise ValueError("τ₁ 和 τ₂ 必须为有限数值")
    if tau1 < 0.005 or tau2 < 0.005 or tau1 > 1.0 or tau2 > 1.0:
        raise ValueError("τ₁ 和 τ₂ 必须位于 0.005–1.0 s")
    guidance_parameters = {
        "outer_kp": float(outer_kp),
        "outer_vbar": float(outer_vbar),
        "outer_ktheta": float(outer_ktheta),
        "outer_preview_horizon": float(outer_preview_horizon),
        "outer_capture_radius": float(outer_capture_radius),
        "outer_blend_radius": float(outer_blend_radius),
    }
    if not all(math.isfinite(value) for value in guidance_parameters.values()):
        raise ValueError("外环参数必须为有限数值")
    if not 0.0 < guidance_parameters["outer_kp"] <= 20.0:
        raise ValueError("kρ 必须位于 0–20（不含 0）")
    if not 0.0 < guidance_parameters["outer_vbar"] <= 0.50:
        raise ValueError("v̄c 必须位于 0–0.50 m/s（不含 0）")
    if not 0.0 < guidance_parameters["outer_ktheta"] <= 20.0:
        raise ValueError("kα 必须位于 0–20（不含 0）")
    if not 0.0 <= guidance_parameters["outer_preview_horizon"] <= 10.0:
        raise ValueError("预瞄时域 Tp 必须位于 0–10 s")
    if guidance_parameters["outer_preview_horizon"] >= float(
        TRAJECTORY_PRESETS[name]["run_duration"]
    ):
        raise ValueError("有限时域轨迹必须满足 Tp < Tf")
    if not 0.0 < guidance_parameters["outer_capture_radius"] <= 1.0:
        raise ValueError("捕获半径 rs 必须位于 0–1 m（不含 0）")
    if not (
        guidance_parameters["outer_capture_radius"]
        < guidance_parameters["outer_blend_radius"] <= 1.0
    ):
        raise ValueError("渐消边界必须满足 rs < rb ≤ 1 m")
    if not math.isfinite(r2) or r2 < 0.0:
        raise ValueError("r₂（速度块自适应率）必须为不小于 0 的有限数值")
    if not math.isfinite(r3) or r3 < 0.0:
        raise ValueError("r₃（电流块自适应率）必须为不小于 0 的有限数值")
    comparison_case_value = float(comparison_case)
    if comparison_case_value not in (1.0, 2.0, 3.0):
        raise ValueError("对比方案编号必须为 E1、E2 或 E3")
    comparison_case = int(comparison_case_value)
    pid_parameters = {
        "pid_velocity_ff": float(pid_velocity_ff),
        "pid_velocity_kp": float(pid_velocity_kp),
        "pid_velocity_ki": float(pid_velocity_ki),
        "pid_velocity_kd": float(pid_velocity_kd),
        "pid_current_kp": float(pid_current_kp),
        "pid_current_ki": float(pid_current_ki),
        "pid_current_kd": float(pid_current_kd),
        "pid_voltage_ff": float(pid_voltage_ff),
        "pid_current_ref_max": float(pid_current_ref_max),
        "pid_derivative_tau": float(pid_derivative_tau),
    }
    for key, value in pid_parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
        if key == "pid_current_ref_max":
            if not 0.05 <= value <= 1.0:
                raise ValueError("PID 电流参考限幅必须位于 0.05–1.0 A")
        elif key == "pid_derivative_tau":
            if not 0.001 <= value <= 1.0:
                raise ValueError("PID 微分滤波时间常数必须位于 0.001–1.0 s")
        elif value < 0.0:
            raise ValueError(f"{key} 必须不小于 0")
    motor_deadzone_v = validate_motor_deadzone(
        motor_deadzone_v,
        max_voltage=THEORY_BRINGUP_U_MAX_V,
    )
    safety_limits = validate_theory_safety_limits(
        theory_max_position_error,
        theory_max_heading_error,
        theory_max_z2,
        theory_max_z3,
    )
    return {
        **TRAJECTORY_PRESETS[name],
        # 轨迹形状必须显式下发：固件 updateNumber 只更新下发的字段，
        # 缺省会残留上一次运行（如大圆 ref_shape=1）的形状。
        "ref_shape": float(TRAJECTORY_PRESETS[name].get("ref_shape", 0.0)),
        "u_max": THEORY_BRINGUP_U_MAX_V,
        "motor_deadzone_v": motor_deadzone_v,
        "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
        "gyro_tau": GYRO_RATE_FILTER_TAU_S,
        "velocity_tau": 0.030,
        "current_tau": CURRENT_FILTER_TAU_S,
        "tau1": tau1,
        "tau2": tau2,
        **guidance_parameters,
        "r2": r2,
        "r3": r3,
        "comparison_case": float(comparison_case),
        **pid_parameters,
        **MOTION_REFERENCE_DEFAULTS,
        **safety_limits,
        "theory_safety_grace": THEORY_SAFETY_GRACE_S,
    }


def validate_motor_deadzone(
    value: float,
    *,
    max_voltage: float = THEORY_BRINGUP_U_MAX_V,
) -> float:
    """Validate the symmetric final motor-voltage dead-zone threshold."""
    value = float(value)
    max_voltage = float(max_voltage)
    if not math.isfinite(value) or not math.isfinite(max_voltage):
        raise ValueError("电压死区必须为有限数值")
    if max_voltage <= 0.0 or not 0.0 <= value < max_voltage:
        raise ValueError(f"电压死区必须满足 0 ≤ Ud < {max_voltage:g} V")
    return value


def validate_gyro_deadband(value: float) -> float:
    """Validate the filtered yaw-rate threshold used by the firmware gate."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("陀螺角速度阈值必须位于 0 到 1 rad/s")
    return value


def validate_theory_safety_limits(
    max_position_error: float,
    max_heading_error: float,
    max_z2: float,
    max_z3: float,
) -> dict[str, float]:
    """Validate THEORY safety limits using the firmware's accepted units."""
    max_position_error = float(max_position_error)
    max_heading_error = float(max_heading_error)
    max_z2 = float(max_z2)
    max_z3 = float(max_z3)
    values = (max_position_error, max_heading_error, max_z2, max_z3)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("THEORY 越界阈值必须为有限数值")
    if not 0.01 <= max_position_error <= 1.0:
        raise ValueError("位置误差上限必须位于 0.01 到 1.0 m")
    if not 0.05 <= max_heading_error <= 3.14:
        raise ValueError("航向误差上限必须位于 2.9° 到 179.9°")
    if max_z2 <= 0.0 or max_z3 <= 0.0:
        raise ValueError("z₂、z₃ 范数上限必须严格大于 0")
    return {
        "theory_max_position_error": max_position_error,
        "theory_max_heading_error": max_heading_error,
        "theory_max_z2": max_z2,
        "theory_max_z3": max_z3,
    }

# ---------------------------------------------------------------------------
# Serial support (optional — graceful degradation if pyserial not installed)
# ---------------------------------------------------------------------------
try:
    import serial
    from serial.tools import list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def find_serial_port() -> str | None:
    if not HAS_SERIAL:
        return None
    ports = list_ports.comports()
    candidates = [p.device for p in ports if p.device]
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Fixed manual voltage levels
# ---------------------------------------------------------------------------
GEARS = [
    {"label": "1 V 档", "voltage": 1.0, "color": "#4fc3f7"},
    {"label": "2 V 档", "voltage": 2.0, "color": "#81c784"},
    {"label": "3 V 档", "voltage": 3.0, "color": "#ffb74d"},
    {"label": "4 V 档", "voltage": 4.0, "color": "#e57373"},
]


# ---------------------------------------------------------------------------
# Car link (WiFi TCP or USB serial)
# ---------------------------------------------------------------------------
class CarLink:
    """Thread-safe connection to the car firmware."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._ser = None
        self._use_serial = False
        self._host = DEFAULT_HOST
        self._port = DEFAULT_PORT
        self._serial_port = ""
        self._recv_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._seq = 0
        self.messages: queue.Queue[dict] = queue.Queue(maxsize=5000)
        self._buffer = bytearray()
        self.connected = False
        self.status_text = "Disconnected"
        self._last_send_error: str = ""
        self._send_ok_count: int = 0
        self._send_fail_count: int = 0
        # Updated by the reader thread as soon as a live state frame arrives.
        # This is intentionally independent of Tk's main thread: a slow
        # matplotlib redraw must not look like a firmware telemetry outage.
        self._last_live_state_rx = 0.0

    # --- public API ---

    def connect_wifi(self, host: str = DEFAULT_HOST,
                     port: int = DEFAULT_PORT) -> None:
        self.disconnect()
        self._use_serial = False
        self._host = host
        self._port = port
        self._stop.clear()
        self._buffer.clear()
        self._last_live_state_rx = 0.0
        self._sock = socket.create_connection((host, port), timeout=4.0)
        self._sock.settimeout(0.2)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.connected = True
        self.status_text = f"WiFi {host}:{port}"
        self._start_reader()

    def connect_serial(self, port: str) -> None:
        if not HAS_SERIAL:
            raise RuntimeError("pyserial not installed — run: pip install pyserial")
        self.disconnect()
        self._use_serial = True
        self._serial_port = port
        self._stop.clear()
        self._buffer.clear()
        self._last_live_state_rx = 0.0
        # CRITICAL: dsrdtr=False prevents DTR/RTS from being set during open(),
        # which would reset the ESP32. Also set timeout=0.2 (same as TCP).
        try:
            self._ser = serial.Serial()
            self._ser.port = port
            self._ser.baudrate = BAUD
            self._ser.timeout = 0.2
            self._ser.dsrdtr = False
            self._ser.open()
        except Exception:
            self._ser = serial.Serial(port, BAUD, timeout=0.2)
            self._ser.dsrdtr = False
        self.connected = True
        self.status_text = f"USB {port}"
        self._start_reader()

    def disconnect(self) -> None:
        self._stop.set()
        self.connected = False
        self.status_text = "Disconnected"
        sock, self._sock = self._sock, None
        if sock:
            try:
                sock.close()
            except OSError:
                pass
        ser, self._ser = self._ser, None
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        # Wait for old reader thread to fully exit so it can't corrupt
        # a new connection (it checks self._stop in a 0.2s recv() loop).
        old_thread = self._recv_thread
        self._recv_thread = None
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=1.0)
        while not self.messages.empty():
            try:
                self.messages.get_nowait()
            except queue.Empty:
                break
        self._buffer.clear()
        self._last_live_state_rx = 0.0

    @property
    def last_live_state_rx(self) -> float:
        """Host monotonic time of the newest telemetry/state frame received."""
        return self._last_live_state_rx

    def send(self, payload: dict) -> int:
        payload = dict(payload)
        if "seq" in payload:
            sequence = int(payload["seq"])
            self._seq = max(self._seq, sequence)
        else:
            self._seq += 1
            sequence = self._seq
            payload["seq"] = sequence
        payload.setdefault("v", PROTOCOL_VERSION)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")
        ok = False
        with self._send_lock:
            if self._use_serial:
                if self._ser and self._ser.is_open:
                    try:
                        self._ser.write(data)
                        ok = True
                    except Exception as e:
                        self._last_send_error = str(e)
                else:
                    self._last_send_error = "serial port closed"
            else:
                if self._sock:
                    try:
                        self._sock.sendall(data)
                        ok = True
                    except OSError as e:
                        self._last_send_error = str(e)
                        self.connected = False
                        self.status_text = f"Send failed: {e}"
                else:
                    self._last_send_error = "no socket"
        if ok:
            self._send_ok_count += 1
        else:
            self._send_fail_count += 1
        return sequence

    def drain(self) -> list[dict]:
        result = []
        while True:
            try:
                result.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return result

    # --- internals ---

    def _start_reader(self) -> None:
        self._recv_thread = threading.Thread(
            target=self._reader, name="car-reader", daemon=True)
        self._recv_thread.start()
        time.sleep(0.3)
        self.drain()

    def _reader(self) -> None:
        while not self._stop.is_set():
            if self._use_serial:
                if not self._ser or not self._ser.is_open:
                    break
                try:
                    data = self._ser.read(4096)
                except Exception:
                    continue
                if not data:   # timeout, not disconnect — keep reading
                    continue
            else:
                if not self._sock:
                    break
                try:
                    data = self._sock.recv(65536)
                except socket.timeout:
                    continue
                except (OSError, Exception) as exc:
                    if not self._stop.is_set():
                        self.status_text = f"Receive failed: {exc}"
                    data = b""
                if not data:
                    if not self._stop.is_set():
                        self.connected = False
                        if not self.status_text.startswith("Receive failed:"):
                            self.status_text = "Firmware closed the TCP connection"
                    # 立即关闭本端 socket，确保 FIN 送到固件：固件是单客户端
                    # server（tcpClientFd 占着不 accept 新连接），本端 socket
                    # 悬空会让下一次重连永远握手超时（表现为"必须重开 EXE"）。
                    sock, self._sock = self._sock, None
                    if sock:
                        try:
                            sock.close()
                        except OSError:
                            pass
                    break

            self._buffer.extend(data)
            while True:
                nl = self._buffer.find(b"\n")
                if nl < 0:
                    break
                raw = bytes(self._buffer[:nl]).strip()
                del self._buffer[: nl + 1]
                if not raw:
                    continue
                try:
                    msg = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(msg, dict):
                    received_at = time.monotonic()
                    msg["_rx_monotonic"] = received_at
                    if msg.get("type") in ("telemetry", "state"):
                        self._last_live_state_rx = received_at
                    try:
                        self.messages.put_nowait(msg)
                    except queue.Full:
                        try:
                            self.messages.get_nowait()
                        except queue.Empty:
                            pass
                        self.messages.put_nowait(msg)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class JoystickApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MIMO Car Joystick")
        initial_width = min(1500, max(1180, self.winfo_screenwidth() - 80))
        initial_height = min(860, max(680, self.winfo_screenheight() - 100))
        self.geometry(f"{initial_width}x{initial_height}")
        self.minsize(1100, 650)
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")

        # --- state ---
        self.link = CarLink()
        self._pressed: set[str] = set()          # currently held keys
        self._space_down = False
        self._gear_index = 0                     # 0..3
        self._u_max = U_MAX_DEFAULT
        self._motor_deadzone_v = MOTOR_DEADZONE_DEFAULT_V
        self._motor_deadzone_supported = False
        self._gyro_deadband_supported = False
        self._theory_safety_supported = False
        self._motion_reference_supported = False
        self._guidance_supported = False
        self._comparison_modes_supported = False
        self._active_theory_safety = {
            "theory_max_position_error": THEORY_MAX_POSITION_ERROR_M,
            "theory_max_heading_error": THEORY_MAX_HEADING_ERROR_RAD,
            "theory_max_z2": THEORY_MAX_Z2_NORM,
            "theory_max_z3": THEORY_MAX_Z3_NORM,
            "theory_safety_grace": THEORY_SAFETY_GRACE_S,
            "motion_max_target_error": MOTION_REFERENCE_DEFAULTS[
                "motion_max_target_error"
            ],
        }
        self._control_job: str | None = None     # after() job id
        self._keepalive_job: str | None = None
        self._connect_job: str | None = None   # after() job id for _connect_tick

        # connection state machine (non-blocking)
        # 0=idle, 1=hello, 2=clear fault, 3=start manual, 4=armed
        self._connecting_stage = 0
        self._connect_deadline = 0.0
        self._last_hello_time = 0.0  # throttle hello retries
        self._connect_pending_sequence: int | None = None
        self._connect_last_send_time = 0.0
        self._connect_manual_phase = ""
        self._fw_version = "?"
        self._fw_mode = "?"
        self._fw_fault = "?"
        self._last_boot_id: str | None = None
        self._reported_fault = "none"
        self._latest_loop_us = 0
        self._latest_dropped_frames = 0
        self._operation_mode = "manual"
        # 运行中的 PID 采集参考（含速度倍率缩放）；未启动采集时为 None。
        self._pid_active_preset: dict[str, float] | None = None

        # Confirmed command workflow for THEORY entry/exit.
        self._workflow_kind: str | None = None
        self._workflow_steps: deque[dict] = deque()
        self._workflow_current: dict | None = None
        self._workflow_deadline = 0.0
        self._workflow_last_stop_retry = 0.0
        self._pending_theory_duration = 0.0
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence: int | None = None

        # last sent voltages for display
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._cmd_count = 0          # incrementing counter to prove commands are sent
        self._last_control_time = time.monotonic()
        self._collection_ready = False
        self._pid_controller = PidPathController()
        self._pid_started_at = 0.0
        self._pid_pause_started_at: float | None = None
        self._pid_paused_s = 0.0
        self._pid_duration = 0.0
        self._pid_reference_name = next(iter(PID_COLLECTION_PRESETS))
        self._latest_pose = np.zeros(3, dtype=float)
        self._latest_velocity = np.zeros(2, dtype=float)
        self._latest_sensor_time = 0.0
        self._latest_imu_ok = False
        self._latest_imu_calibrated = False
        self._latest_ina_ok = False
        self._latest_gyro_z = 0.0
        self._latest_wheel_right = 0.0
        self._latest_wheel_left = 0.0
        self._track_width_m = 0.2035
        # 论文按钮所需：最近一帧状态里的电流、实际施加电压与时间戳
        # （原状态处理函数不保存这三项，E4 轨迹导出需要它们）。
        self._latest_current = np.zeros(2, dtype=float)
        self._latest_applied_u = np.zeros(2, dtype=float)
        self._latest_state_us = 0
        # E3 消融标记：置 True 时导出 initial_nonlinear_compensation_enabled=False，
        # 且权重上传 payload 的 w2/w3 非线性补偿列已清零。
        self._paper_adaptive_only = False
        # E4 采样 PID 基线的导出缓冲（与 run_pid_baseline_headless.py 行一致）。
        self._e4_rows: list[dict] = []
        self._e4_t0_us: int | None = None
        self._e4_last_command = np.zeros(2, dtype=float)
        self._e4_last_u = np.zeros(2, dtype=float)
        self._e4_last_state_us = 0
        self._e4_completed = False
        self._pid_last_metrics: dict[str, float] = {}
        self._pid_pose_trace: deque[np.ndarray] = deque(maxlen=PLOT_MAX_SAMPLES)
        self._pid_trace_times: deque[float] = deque(maxlen=PLOT_MAX_SAMPLES)

        # log ring buffer
        self._log_lines: list[str] = []

        # Each row is one complete non-overlapping firmware integral window.
        # The PC never differentiates velocity, yaw rate, or current.
        self._dataset = IntegralSnapshotDataset()
        self._active_collection_segment: int | None = None
        self._integral_window = INTEGRAL_SNAPSHOT_WINDOW_S
        self._dataset_path: Path | None = None
        self._snapshot_cache_path = snapshot_dataset_cache_path()
        self._snapshot_cache_dirty = False
        self._snapshot_cache_job: str | None = None
        self._synthesis: SynthesisResult | None = None
        self._qualification: QualificationReport | None = None
        self._validated_weights_payload: dict[str, Any] | None = None
        self._validated_weights_info: dict[str, Any] | None = None
        self._validated_weights_path = validated_weights_cache_path()
        self._plot_rows: deque[dict[str, np.ndarray]] = deque(maxlen=PLOT_MAX_SAMPLES)
        self._plot_times: deque[float] = deque(maxlen=PLOT_MAX_SAMPLES)
        self._snapshot_t0_us: int | None = None
        self._snapshot_fallback_t = 0.0
        self._plot_dirty = False
        self._plot_canvas: FigureCanvasTkAgg | None = None
        self._plot_scroll_canvas: tk.Canvas | None = None
        self._plot_figure: Figure | None = None
        self._plot_axes = None
        self._plot_axis_positions = None
        self._plot_job: str | None = None
        self._plot_view = "synthesis"
        self._collection_plot_mode = "trajectory"
        self._theory_plot_mode = "trajectory"
        self._collection_trajectory_artists: dict[str, Any] | None = None
        self._plot_switch_button: tk.Button | None = None
        self._plot_mode_buttons: dict[str, tk.Button] = {}
        self._theory_plot_mode_buttons: dict[str, tk.Button] = {}
        self._regressor_scroll_frame: tk.Frame | None = None
        self._regressor_scroll_canvas: tk.Canvas | None = None
        self._regressor_scroll_window: int | None = None
        self._regressor_figure: Figure | None = None
        self._regressor_axes = None
        self._regressor_canvas: FigureCanvasTkAgg | None = None

        # Closed-loop THEORY telemetry retained independently from snapshots.
        self._theory_rows: deque[dict[str, np.ndarray | float]] = deque(
            maxlen=THEORY_MAX_SAMPLES
        )
        self._theory_t0_us: int | None = None
        self._theory_exported_path: Path | None = None
        self._theory_safety_count = 0
        self._theory_safety_reason: str | None = None
        self._active_comparison_name = next(iter(COMPARISON_CASES))
        self._active_comparison_case = "E1"
        self._active_theory_config: dict[str, float] = {}
        self._active_theory_weights_json = "{}"

        # --- build UI ---
        self._build_ui()
        self._restore_cached_dataset()
        self._restore_validated_weights()

        # --- bind keys ---
        # A single bind_all path avoids processing the same physical key in
        # both the toplevel and the global Tk bind tags.
        for key in ("<Up>", "<Down>", "<Left>", "<Right>",
                    "<space>", "<Escape>", "<Return>", "<Tab>"):
            self.bind_all(key, self._on_action_key, add="+")

        # Releases implement the direction dead-man switch and suppress OS
        # auto-repeat for voltage-level switching.
        for key in ("<KeyRelease-Up>", "<KeyRelease-Down>",
                    "<KeyRelease-Left>", "<KeyRelease-Right>"):
            self.bind_all(key, self._on_dir_release, add="+")
        self.bind_all("<KeyRelease-space>", self._on_space_release, add="+")

        # handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # start UI refresh timer
        self._refresh_ui()
        self._plot_job = self.after(PLOT_REFRESH_MS, self._refresh_plot)

        # auto-focus the root window so keys work immediately
        self.after(200, self.focus_set)

        self._log("App started — press Enter to connect")

    # ==================================================================
    # Logging
    # ==================================================================

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self._log_lines.append(line)
        removed_oldest = len(self._log_lines) > LOG_MAX_LINES
        if removed_oldest:
            self._log_lines.pop(0)
        if not hasattr(self, "_log_text"):
            return

        # Follow new entries only while the user is already at the bottom.
        # This keeps a manually selected historical position stable.
        was_at_bottom = self._log_text.yview()[1] >= 0.995
        self._log_text.configure(state=tk.NORMAL)
        if removed_oldest:
            self._log_text.delete("1.0", "2.0")
        self._log_text.insert(tk.END, line + "\n")
        self._log_text.configure(state=tk.DISABLED)
        if was_at_bottom:
            self._log_text.see(tk.END)

    def _scroll_log(self, event) -> str:
        """Scroll the log under the pointer without changing driving focus."""
        steps = -int(event.delta / 120) if event.delta else 0
        if steps == 0:
            steps = -1 if event.delta > 0 else 1
        self._log_text.yview_scroll(3 * steps, "units")
        return "break"

    def _replace_plot_history(
        self, matrices: dict[str, np.ndarray],
        window_s: np.ndarray | None = None,
    ) -> int:
        """Load only the columns that survived offline snapshot reconstruction."""
        column_counts: list[int] = []
        for key, expected_rows in SNAPSHOT_FIELDS.items():
            value = np.asarray(matrices[key], dtype=float)
            if value.ndim != 2 or value.shape[0] != expected_rows:
                raise ValueError(f"{key} matrix shape is invalid")
            column_counts.append(int(value.shape[1]))
        processed_count = min(column_counts)
        if processed_count <= 0:
            raise ValueError("offline reconstruction produced no snapshots")

        if window_s is None:
            window_s = np.full(processed_count, INTEGRAL_SNAPSHOT_WINDOW_S)
        window_s = np.asarray(window_s, dtype=float).reshape(-1)
        if window_s.size != processed_count:
            raise ValueError("window_s length does not match snapshot count")
        time_axis = np.concatenate(([0.0], np.cumsum(window_s)))
        self._plot_rows.clear()
        self._plot_times.clear()
        first = max(0, processed_count - PLOT_MAX_SAMPLES)
        for index in range(first, processed_count):
            row = {key: matrices[key][:, index].copy() for key in SNAPSHOT_FIELDS}
            for key in ("velocity_raw", "current_raw"):
                if key in matrices:
                    value = np.asarray(matrices[key], dtype=float)
                    if value.shape == (2, processed_count):
                        row[key] = value[:, index].copy()
            self._plot_rows.append(row)
            self._plot_times.append(float(time_axis[index + 1]))
        return processed_count

    def _restore_cached_dataset(self) -> None:
        if not self._snapshot_cache_path.exists():
            return
        try:
            dataset = IntegralSnapshotDataset.load(self._snapshot_cache_path)
            matrices = dataset.matrices()
            snapshot_count = len(dataset)
            processed_count = self._replace_plot_history(
                matrices, dataset.window_seconds
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self._log(f"Snapshot cache rejected: {exc}")
            return
        self._dataset = dataset
        self._dataset_path = None
        self._snapshot_cache_dirty = False
        self._snapshot_t0_us = None
        self._snapshot_fallback_t = (
            self._plot_times[-1] if self._plot_times else 0.0
        )
        duration = (
            self._plot_times[-1] - self._plot_times[0]
            if len(self._plot_times) > 1
            else 0.0
        )
        count_text = f"{snapshot_count}"
        self._sample_var.set(
            f"积分快照：{count_text}  ·  {dataset.segment_count} 段 · 已恢复 · 图窗 {duration:.1f} s"
        )
        self._plot_view = "synthesis"
        self._plot_dirty = True
        self._update_quality_display()
        self._log(
            "Restored last snapshot dataset: "
            f"integral={snapshot_count}, processed={processed_count}"
        )

    def _schedule_snapshot_cache(self) -> None:
        self._snapshot_cache_dirty = True
        if self._snapshot_cache_job is None:
            self._snapshot_cache_job = self.after(
                SNAPSHOT_CACHE_INTERVAL_MS,
                self._flush_snapshot_cache,
            )

    def _flush_snapshot_cache(self, force: bool = False) -> None:
        if self._snapshot_cache_job is not None:
            try:
                self.after_cancel(self._snapshot_cache_job)
            except tk.TclError:
                pass
            self._snapshot_cache_job = None
        if not self._snapshot_cache_dirty and not force:
            return
        if len(self._dataset) == 0:
            self._snapshot_cache_dirty = False
            return
        target = self._snapshot_cache_path
        temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._dataset.save(temporary)
            temporary.replace(target)
            self._snapshot_cache_dirty = False
            self._log(f"Auto-saved snapshot dataset: K={len(self._dataset)}")
        except (OSError, ValueError) as exc:
            self._log(f"Could not auto-save snapshot dataset: {exc}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _delete_snapshot_cache(self) -> None:
        if self._snapshot_cache_job is not None:
            try:
                self.after_cancel(self._snapshot_cache_job)
            except tk.TclError:
                pass
            self._snapshot_cache_job = None
        self._snapshot_cache_dirty = False
        try:
            self._snapshot_cache_path.unlink(missing_ok=True)
        except OSError as exc:
            self._log(f"Could not delete snapshot cache: {exc}")

    def _restore_validated_weights(self) -> None:
        if not self._validated_weights_path.exists():
            return
        try:
            payload, document = load_validated_weights_cache(
                self._validated_weights_path
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._synthesis_status_var.set("上次 PASS 文件无效")
            self._synthesis_status_label.configure(fg="#e57373")
            self._log(f"Validated weights cache rejected: {exc}")
            return
        self._validated_weights_payload = payload
        self._validated_weights_info = document
        for key in SYNTHESIS_DEFAULTS:
            if key in payload and key in self._synthesis_vars:
                self._synthesis_vars[key].set(f"{float(payload[key]):g}")
        kappa2_text = (
            f"κ₂=({float(payload['kappa2_v']):.6g},"
            f"{float(payload['kappa2_w']):.6g})"
            if "kappa2_v" in payload
            else f"κ₂={float(payload['kappa2']):.6g}"
        )
        kappa3_text = (
            f"κ₃=({float(payload['kappa3_v']):.6g},"
            f"{float(payload['kappa3_w']):.6g})"
            if "kappa3_v" in payload
            else f"κ₃={float(payload['kappa3']):.6g}"
        )
        self._kappa_min_var.set(
            "Route II 已缓存："
            f"{kappa2_text} > κ₂,min={float(payload['kappa_min2']):.6g}  |  "
            f"{kappa3_text} > κ₃,min={float(payload['kappa_min3']):.6g}"
        )
        samples = int(document["samples"])
        saved_at = str(document.get("saved_at", ""))
        self._synthesis_status_var.set(
            f"上次 PASS · K={samples} · 可直接 THEORY"
        )
        self._synthesis_status_label.configure(fg="#81c784")
        self._log(
            f"Restored Route-II validated weights: K={samples}, saved={saved_at}"
        )

    def _remember_validated_weights(
        self,
        result: SynthesisResult,
        qualification: QualificationReport | None,
    ) -> None:
        source = str(self._dataset_path) if self._dataset_path is not None else None
        self._validated_weights_payload = _validate_upload_payload(
            result.upload_payload()
        )
        self._validated_weights_info = {
            "samples": int(result.samples),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dataset": source or "",
            "operating_config": {"gyro_deadband": 0.0},
        }
        try:
            self._validated_weights_info = save_validated_weights_cache(
                self._validated_weights_path,
                result,
                source,
                qualification,
            )
            self._log(
                "Validated weights cached for future THEORY sessions"
            )
        except OSError as exc:
            self._log(f"Could not persist validated weights cache: {exc}")

    def _clear_validated_weights(self) -> None:
        if (
            self._validated_weights_payload is None
            and not self._validated_weights_path.exists()
        ):
            return
        if not messagebox.askyesno(
            "清除已验证权重",
            "确认删除上一次通过校验的权重？\n删除后必须重新综合 PASS 才能启动 THEORY。",
        ):
            return
        self._validated_weights_payload = None
        self._validated_weights_info = None
        self._synthesis = None
        try:
            self._validated_weights_path.unlink(missing_ok=True)
        except OSError as exc:
            self._log(f"Could not delete validated weights cache: {exc}")
        self._synthesis_status_var.set("已清除 PASS 权重")
        self._synthesis_status_label.configure(fg="#888888")
        self._update_theory_start_state()
        self._log("Validated weights cleared explicitly")

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        bg = "#1a1a2e"
        fg = "#e0e0e0"

        # -- title bar --
        title_frame = tk.Frame(self, bg="#0f3460", height=44)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="🎮  MIMO Car Joystick", font=("Segoe UI", 14, "bold"),
                 fg="#ffffff", bg="#0f3460").pack(side=tk.LEFT, padx=16, pady=8)
        self._conn_light = tk.Canvas(title_frame, width=14, height=14,
                                      bg="#0f3460", highlightthickness=0)
        self._conn_light.pack(side=tk.RIGHT, padx=16, pady=8)
        self._conn_dot = self._conn_light.create_oval(2, 2, 12, 12, fill="#f44336",
                                                       outline="")

        body = tk.Frame(self, bg=bg)
        body.pack(fill=tk.BOTH, expand=True)
        control_panel = tk.Frame(body, bg=bg, width=540)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        control_panel.pack_propagate(False)
        plot_panel = tk.Frame(body, bg="#111122")
        plot_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        # -- connection row --
        conn_frame = tk.Frame(control_panel, bg=bg)
        conn_frame.pack(fill=tk.X, padx=16, pady=(12, 0))

        tk.Label(conn_frame, text="Connection:", font=("Segoe UI", 9),
                 fg=fg, bg=bg).pack(side=tk.LEFT)

        self._mode_var = tk.StringVar(value="wifi")
        tk.Radiobutton(conn_frame, text="WiFi", variable=self._mode_var, value="wifi",
                       font=("Segoe UI", 9), fg=fg, bg=bg, selectcolor=bg,
                       activebackground=bg, activeforeground=fg,
                       command=self._on_mode_change).pack(side=tk.LEFT, padx=(8, 0))
        tk.Radiobutton(conn_frame, text="USB", variable=self._mode_var, value="usb",
                       font=("Segoe UI", 9), fg=fg, bg=bg, selectcolor=bg,
                       activebackground=bg, activeforeground=fg,
                       command=self._on_mode_change).pack(side=tk.LEFT, padx=(8, 0))

        self._host_var = tk.StringVar(value=DEFAULT_HOST)
        self._port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self._serial_var = tk.StringVar(value=find_serial_port() or "COM3")

        # WiFi fields
        self._wifi_frame = tk.Frame(conn_frame, bg=bg)
        self._wifi_frame.pack(side=tk.LEFT, padx=(16, 0))
        tk.Label(self._wifi_frame, text="Host:", font=("Segoe UI", 9),
                 fg=fg, bg=bg).pack(side=tk.LEFT)
        self._host_entry = tk.Entry(self._wifi_frame, textvariable=self._host_var,
                                     width=13, font=("Consolas", 9),
                                     bg="#2a2a4a", fg=fg, insertbackground=fg,
                                     relief=tk.FLAT)
        self._host_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(self._wifi_frame, text="Port:", font=("Segoe UI", 9),
                 fg=fg, bg=bg).pack(side=tk.LEFT)
        self._port_entry = tk.Entry(self._wifi_frame, textvariable=self._port_var,
                                     width=6, font=("Consolas", 9),
                                     bg="#2a2a4a", fg=fg, insertbackground=fg,
                                     relief=tk.FLAT)
        self._port_entry.pack(side=tk.LEFT, padx=(4, 0))

        # USB fields (hidden by default)
        self._usb_frame = tk.Frame(conn_frame, bg=bg)
        tk.Label(self._usb_frame, text="Port:", font=("Segoe UI", 9),
                 fg=fg, bg=bg).pack(side=tk.LEFT)
        self._serial_entry = tk.Entry(self._usb_frame, textvariable=self._serial_var,
                                       width=10, font=("Consolas", 9),
                                       bg="#2a2a4a", fg=fg, insertbackground=fg,
                                       relief=tk.FLAT)
        self._serial_entry.pack(side=tk.LEFT, padx=(4, 0))

        self._connect_btn = tk.Button(conn_frame, text="Connect", font=("Segoe UI", 9, "bold"),
                                       fg="#ffffff", bg="#0f3460", relief=tk.FLAT,
                                       activebackground="#1a5276", activeforeground="#ffffff",
                                       padx=12, pady=2, command=self._toggle_connect)
        self._connect_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # -- separator --
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=16, pady=12)

        # -- gear indicator --
        self._gear_frame = tk.Frame(control_panel, bg=bg)
        self._gear_frame.pack(fill=tk.X, padx=16, pady=(4, 0))

        tk.Label(self._gear_frame, text="人工电压档:", font=("Segoe UI", 9),
                 fg=fg, bg=bg).pack(side=tk.LEFT)

        self._gear_canvas = tk.Canvas(self._gear_frame, width=180, height=36,
                                       bg=bg, highlightthickness=0)
        self._gear_canvas.pack(side=tk.LEFT, padx=(12, 0))
        self._gear_text = self._gear_canvas.create_text(
            90, 18, text="", font=("Segoe UI", 12, "bold"), fill="#ffffff")

        tk.Label(self._gear_frame, text="空格切换 1/2/3/4 V",
                 font=("Segoe UI", 8), fg="#888888", bg=bg).pack(side=tk.RIGHT)

        # -- PID path data collection --
        collection_frame = tk.Frame(control_panel, bg=bg)
        collection_frame.pack(fill=tk.X, padx=16, pady=(2, 0))
        self._collection_status_var = tk.StringVar(
            value="PID 路径控制采集"
        )
        tk.Label(
            collection_frame,
            textvariable=self._collection_status_var,
            font=("Segoe UI", 9, "bold"),
            fg="#81c784",
            bg=bg,
        ).pack(side=tk.LEFT)

        pid_frame = tk.Frame(control_panel, bg=bg)
        pid_frame.pack(fill=tk.X, padx=16, pady=(4, 0))
        self._pid_path_var = tk.StringVar(value=next(iter(PID_COLLECTION_PRESETS)))
        self._pid_path_box = ttk.Combobox(
            pid_frame,
            textvariable=self._pid_path_var,
            values=tuple(PID_COLLECTION_PRESETS),
            state="readonly",
            width=31,
            font=("Segoe UI", 8),
        )
        self._pid_path_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._pid_start_button = tk.Button(
            pid_frame,
            text="启动 PID 采集",
            font=("Segoe UI", 8, "bold"),
            fg="#777777",
            bg="#2a2a4a",
            activebackground="#1a5276",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self._start_pid_collection,
        )
        self._pid_start_button.pack(side=tk.LEFT, padx=(6, 0))
        self._pid_stop_button = tk.Button(
            pid_frame,
            text="停止采集",
            font=("Segoe UI", 8),
            fg="#dddddd",
            bg="#7f1d1d",
            activebackground="#a52a2a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._stop_pid_collection,
        )
        self._pid_stop_button.pack(side=tk.LEFT, padx=(4, 0))

        pid_cfg_frame = tk.Frame(control_panel, bg=bg)
        pid_cfg_frame.pack(fill=tk.X, padx=16, pady=(4, 0))
        tk.Label(
            pid_cfg_frame, text="速度倍率", font=("Segoe UI", 8),
            fg="#bbbbbb", bg=bg,
        ).pack(side=tk.LEFT)
        self._pid_speed_var = tk.StringVar(value="1.5")
        tk.Entry(
            pid_cfg_frame, textvariable=self._pid_speed_var, width=5,
            font=("Consolas", 9), bg="#1e1e2e", fg="#eeeeee",
            insertbackground="#eeeeee", relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(4, 12))
        tk.Label(
            pid_cfg_frame, text="积分窗口 T_w (s)", font=("Segoe UI", 8),
            fg="#bbbbbb", bg=bg,
        ).pack(side=tk.LEFT)
        self._pid_window_var = tk.StringVar(value="0.10")
        tk.Entry(
            pid_cfg_frame, textvariable=self._pid_window_var, width=6,
            font=("Consolas", 9), bg="#1e1e2e", fg="#eeeeee",
            insertbackground="#eeeeee", relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # -- direction pad --
        tk.Label(
            control_panel,
            text="人工驾驶（仅控制，不记录权重合成数据）",
            font=("Segoe UI", 8),
            fg="#888888",
            bg=bg,
        ).pack(pady=(7, 0))
        self._dpad_frame = tk.Frame(control_panel, bg=bg)
        self._dpad_frame.pack(pady=(10, 6))
        self._draw_dpad()

        # -- voltage display --
        volt_frame = tk.Frame(control_panel, bg=bg)
        volt_frame.pack(fill=tk.X, padx=16, pady=(4, 0))

        self._volt_var = tk.StringVar(value="u_R =  0.00 V    u_L =  0.00 V")
        volt_label = tk.Label(volt_frame, textvariable=self._volt_var,
                 font=("Consolas", 13, "bold"), fg="#4fc3f7", bg=bg)
        volt_label.pack()

        # -- cmd count (proves commands are flowing) --
        self._cmd_var = tk.StringVar(value="")
        tk.Label(volt_frame, textvariable=self._cmd_var,
                 font=("Consolas", 8), fg="#555555", bg=bg).pack()

        self._sensor_var = tk.StringVar(
            value="IMU:等待数据  gyro=+0.000  vR=+0.000  vL=+0.000 m/s"
        )
        self._sensor_label = tk.Label(
            volt_frame,
            textvariable=self._sensor_var,
            font=("Consolas", 8),
            fg="#888888",
            bg=bg,
        )
        self._sensor_label.pack()

        # -- key state indicators --
        keys_frame = tk.Frame(control_panel, bg=bg)
        keys_frame.pack(fill=tk.X, padx=16, pady=(6, 0))
        tk.Label(keys_frame, text="Keys:", font=("Segoe UI", 8),
                 fg="#888888", bg=bg).pack(side=tk.LEFT)

        self._key_labels: dict[str, tk.Label] = {}
        for name in ("up", "down", "left", "right"):
            lbl = tk.Label(keys_frame, text=name.upper(), font=("Consolas", 9, "bold"),
                           fg="#444444", bg="#111122", width=6, padx=2, pady=0)
            lbl.pack(side=tk.LEFT, padx=2)
            self._key_labels[name] = lbl

        self._key_count_label = tk.Label(keys_frame, text="",
                                         font=("Consolas", 8), fg="#555555", bg=bg)
        self._key_count_label.pack(side=tk.RIGHT)

        # -- synthesis snapshot controls --
        data_frame = tk.Frame(control_panel, bg=bg)
        data_frame.pack(fill=tk.X, padx=16, pady=(8, 0))
        self._sample_var = tk.StringVar(value="积分快照：0")
        tk.Label(data_frame, textvariable=self._sample_var,
                 font=("Segoe UI", 9, "bold"), fg="#81c784", bg=bg).pack(side=tk.LEFT)
        tk.Button(data_frame, text="保存 NPZ", font=("Segoe UI", 8),
                  fg="#dddddd", bg="#2a2a4a", relief=tk.FLAT,
                  activebackground="#3a3a5a", activeforeground="#ffffff",
                  padx=7, command=self._save_dataset).pack(side=tk.RIGHT, padx=(0, 6))
        tk.Button(data_frame, text="加载 NPZ", font=("Segoe UI", 8),
                  fg="#dddddd", bg="#2a2a4a", relief=tk.FLAT,
                  activebackground="#3a3a5a", activeforeground="#ffffff",
                  padx=7, command=self._load_dataset).pack(side=tk.RIGHT, padx=(0, 6))
        tk.Button(data_frame, text="清空", font=("Segoe UI", 8),
                  fg="#dddddd", bg="#2a2a4a", relief=tk.FLAT,
                  activebackground="#3a3a5a", activeforeground="#ffffff",
                  padx=7, command=self._clear_dataset).pack(side=tk.RIGHT, padx=(0, 6))

        self._quality_var = tk.StringVar(
            value="数据质量：G₂ 0/10  ·  G₃ 0/10"
        )
        self._quality_label = tk.Label(
            control_panel,
            textvariable=self._quality_var,
            font=("Consolas", 8),
            fg="#888888",
            bg=bg,
            anchor=tk.W,
        )
        self._quality_label.pack(fill=tk.X, padx=16, pady=(3, 0))

        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=16, pady=(7, 5)
        )

        # -- offline synthesis and THEORY validation --
        synthesis_frame = tk.Frame(control_panel, bg=bg)
        synthesis_frame.pack(fill=tk.X, padx=16)
        tk.Label(
            synthesis_frame,
            text="权重综合",
            font=("Segoe UI", 9, "bold"),
            fg=fg,
            bg=bg,
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        # 全部可调参数（综合 + THEORY 运行）在"参数调节"弹窗中编辑，主界面不放输入框。
        self._synthesis_vars = {
            key: tk.StringVar(value=f"{ALL_PARAMETER_DEFAULTS[key]:g}")
            for key, _label in ALL_PARAMETER_LABELS
        }

        self._kappa_min_var = tk.StringVar(
            value="Route II 最小值：κ₂,min=--  κ₃,min=--（选定 κ 必须严格更大）"
        )
        tk.Label(
            control_panel,
            textvariable=self._kappa_min_var,
            font=("Consolas", 7),
            fg="#64b5f6",
            bg=bg,
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=16, pady=(3, 0))

        self._disturbance_diagnostics_var = tk.StringVar(
            value="附加诊断（不参与 PASS）：综合后显示留出残差与分段指标"
        )
        tk.Label(
            control_panel,
            textvariable=self._disturbance_diagnostics_var,
            font=("Consolas", 7),
            fg="#888888",
            bg=bg,
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=16, pady=(3, 0))

        synthesis_actions = tk.Frame(control_panel, bg=bg)
        synthesis_actions.pack(fill=tk.X, padx=16, pady=(4, 0))
        self._synthesize_button = tk.Button(
            synthesis_actions,
            text="计算并校验权重",
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#0f3460",
            activebackground="#1a5276",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._synthesize_weights,
        )
        self._synthesize_button.pack(side=tk.LEFT)
        self._synthesis_params_button = tk.Button(
            synthesis_actions,
            text="参数调节",
            font=("Segoe UI", 8),
            fg="#cccccc",
            bg="#2a2a4a",
            activebackground="#3a3a5a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._open_synthesis_dialog,
        )
        self._synthesis_params_button.pack(side=tk.LEFT, padx=(6, 0))
        self._clear_validated_weights_button = tk.Button(
            synthesis_actions,
            text="清除 PASS",
            font=("Segoe UI", 8),
            fg="#cccccc",
            bg="#2a2a4a",
            activebackground="#4a2a3a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._clear_validated_weights,
        )
        self._clear_validated_weights_button.pack(side=tk.LEFT, padx=(6, 0))
        self._apply_deadzone_button = tk.Button(
            synthesis_actions,
            text="应用死区",
            font=("Segoe UI", 8),
            fg="#777777",
            bg="#2a2a4a",
            activebackground="#3a3a5a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self._apply_motor_deadzone,
        )
        self._apply_deadzone_button.pack(side=tk.LEFT, padx=(6, 0))
        self._synthesis_status_var = tk.StringVar(value="尚未计算")
        self._synthesis_status_label = tk.Label(
            synthesis_actions,
            textvariable=self._synthesis_status_var,
            font=("Consolas", 8),
            fg="#888888",
            bg=bg,
        )
        self._synthesis_status_label.pack(side=tk.LEFT, padx=(8, 0))

        theory_frame = tk.Frame(control_panel, bg=bg)
        theory_frame.pack(fill=tk.X, padx=16, pady=(5, 0))
        tk.Label(
            theory_frame,
            text="验证轨迹",
            font=("Segoe UI", 8),
            fg="#aaaaaa",
            bg=bg,
        ).pack(side=tk.LEFT)
        self._trajectory_var = tk.StringVar(value=next(iter(TRAJECTORY_PRESETS)))
        self._trajectory_box = ttk.Combobox(
            theory_frame,
            textvariable=self._trajectory_var,
            values=tuple(TRAJECTORY_PRESETS),
            state="readonly",
            width=30,
            font=("Segoe UI", 8),
        )
        self._trajectory_box.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        comparison_frame = tk.Frame(control_panel, bg=bg)
        comparison_frame.pack(fill=tk.X, padx=16, pady=(4, 0))
        tk.Label(
            comparison_frame,
            text="实验方案",
            font=("Segoe UI", 8),
            fg="#aaaaaa",
            bg=bg,
        ).pack(side=tk.LEFT)
        self._comparison_case_var = tk.StringVar(value=next(iter(COMPARISON_CASES)))
        self._comparison_case_box = ttk.Combobox(
            comparison_frame,
            textvariable=self._comparison_case_var,
            values=tuple(COMPARISON_CASES),
            state="readonly",
            width=30,
            font=("Segoe UI", 8),
        )
        self._comparison_case_box.pack(
            side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True
        )
        self._comparison_case_box.bind(
            "<<ComboboxSelected>>", self._on_comparison_case_changed
        )

        theory_actions = tk.Frame(control_panel, bg=bg)
        theory_actions.pack(fill=tk.X, padx=16, pady=(4, 0))
        self._theory_start_button = tk.Button(
            theory_actions,
            text="上传权重并启动 THEORY",
            font=("Segoe UI", 8, "bold"),
            fg="#777777",
            bg="#2a2a4a",
            activebackground="#3a3a5a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self._start_theory_validation,
        )
        self._theory_start_button.pack(side=tk.LEFT)
        self._theory_stop_button = tk.Button(
            theory_actions,
            text="停止 / 返回人工",
            font=("Segoe UI", 8),
            fg="#dddddd",
            bg="#7f1d1d",
            activebackground="#a52a2a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._return_to_manual,
        )
        self._theory_stop_button.pack(side=tk.RIGHT)
        self._theory_status_var = tk.StringVar(value="THEORY 未启动")
        tk.Label(
            control_panel,
            textvariable=self._theory_status_var,
            font=("Segoe UI", 8),
            fg="#888888",
            bg=bg,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=16, pady=(3, 0))

        # 论文实验面板：tp13_k20_k30 冻结参数一键复现（E1-E4 + 数据采集）。
        paper_panel = tk.Frame(control_panel, bg=bg)
        paper_panel.pack(fill=tk.X, padx=16, pady=(6, 0))
        tk.Label(
            paper_panel,
            text="论文实验（冻结参数：圆 R 0.40 m · 0.12 m/s · Tp=1.3 s · 死区 ±0.5 V）",
            font=("Segoe UI", 8, "bold"),
            fg="#64b5f6",
            bg=bg,
            anchor=tk.W,
        ).pack(fill=tk.X)
        paper_row1 = tk.Frame(paper_panel, bg=bg)
        paper_row1.pack(fill=tk.X, pady=(3, 0))
        self._paper_collect_button = tk.Button(
            paper_row1,
            text="采集数据（论文双纽线 K460）",
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#0f6b4f",
            activebackground="#12866a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._start_paper_collection,
        )
        self._paper_collect_button.pack(side=tk.LEFT)
        self._paper_e4_button = tk.Button(
            paper_row1,
            text="E4 采样PID基线",
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#8a5a00",
            activebackground="#a87000",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=self._start_paper_e4,
        )
        self._paper_e4_button.pack(side=tk.RIGHT)
        paper_row2 = tk.Frame(paper_panel, bg=bg)
        paper_row2.pack(fill=tk.X, pady=(3, 0))
        self._paper_case_buttons: dict[str, tk.Button] = {}
        for case_key, case_text, case_color in (
            ("E1", "E1 完整（自适应）", "#283593"),
            ("E2", "E2 关自适应", "#283593"),
            ("E3", "E3 消融（补偿清零）", "#283593"),
        ):
            button = tk.Button(
                paper_row2,
                text=case_text,
                font=("Segoe UI", 8, "bold"),
                fg="#ffffff",
                bg=case_color,
                activebackground="#3f51b5",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                command=lambda key=case_key: self._start_paper_theory(key),
            )
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
            self._paper_case_buttons[case_key] = button
        tk.Label(
            paper_panel,
            text="E1–E3 使用最近一次 PASS 权重；E3 自动把权重补偿列清零。"
                 "轨迹与安全参数固定为 tp13_k20_k30 战役实值。",
            font=("Segoe UI", 7),
            fg="#777777",
            bg=bg,
            anchor=tk.W,
            wraplength=560,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(2, 0))


        # -- log area --
        log_frame = tk.Frame(control_panel, bg="#0d0d1a")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))
        self._log_scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL)
        self._log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text = tk.Text(
            log_frame,
            font=("Consolas", 10),
            fg="#c4c4c4",
            bg="#0d0d1a",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
            yscrollcommand=self._log_scrollbar.set,
            padx=7,
            pady=5,
        )
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._log_scrollbar.configure(command=self._log_text.yview)
        self._log_text.bind("<MouseWheel>", self._scroll_log)
        if self._log_lines:
            self._log_text.configure(state=tk.NORMAL)
            self._log_text.insert(tk.END, "\n".join(self._log_lines) + "\n")
            self._log_text.configure(state=tk.DISABLED)
            self._log_text.see(tk.END)

        # -- status bar --
        status_frame = tk.Frame(control_panel, bg="#0d0d1a")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_var = tk.StringVar(value="Disconnected · Press Enter to connect")
        tk.Label(status_frame, textvariable=self._status_var,
                 font=("Segoe UI", 8), fg="#888888", bg="#0d0d1a",
                 anchor=tk.W, padx=12, pady=4).pack(fill=tk.X)

        # -- shortcut hints --
        hints_frame = tk.Frame(control_panel, bg=bg)
        hints_frame.pack(fill=tk.X, padx=16, pady=(4, 8))

        hints = [
            ("↑←↓→", "Drive"),
            ("␣", "Gear"),
            ("Esc", "Stop"),
            ("Enter", "Connect"),
            ("Tab", "WiFi/USB"),
        ]
        for key, desc in hints:
            row = tk.Frame(hints_frame, bg=bg)
            row.pack(side=tk.LEFT, padx=(0, 16))
            tk.Label(row, text=key, font=("Segoe UI", 9, "bold"),
                     fg="#cccccc", bg=bg).pack(side=tk.LEFT)
            tk.Label(row, text=f" {desc}", font=("Segoe UI", 8),
                     fg="#666666", bg=bg).pack(side=tk.LEFT)

        self._build_plot_panel(plot_panel)

    def _draw_dpad(self) -> None:
        """Draw a directional pad showing which keys are pressed."""
        bg = "#1a1a2e"
        w = 140
        h = 140
        cx, cy = w // 2, h // 2
        btn_w, btn_h = 40, 40
        gap = 4

        self._dpad_canvas = tk.Canvas(self._dpad_frame, width=w, height=h,
                                       bg=bg, highlightthickness=0)
        self._dpad_canvas.pack()

        arrows = {}
        arrows["up"] = (cx - btn_w//2, cy - btn_h - gap,
                        cx + btn_w//2, cy - gap)
        arrows["down"] = (cx - btn_w//2, cy + gap,
                          cx + btn_w//2, cy + btn_h + gap)
        arrows["left"] = (cx - btn_w - gap, cy - btn_h//2,
                          cx - gap, cy + btn_h//2)
        arrows["right"] = (cx + gap, cy - btn_h//2,
                           cx + btn_w + gap, cy + btn_h//2)

        self._dpad_rects: dict[str, int] = {}
        for tag, (x1, y1, x2, y2) in arrows.items():
            r = self._dpad_canvas.create_rectangle(
                x1, y1, x2, y2, fill="#16213e", outline="#2a2a4a", width=1)
            symbols = {"up": "▲", "down": "▼", "left": "◀", "right": "▶"}
            self._dpad_canvas.create_text(
                (x1 + x2) // 2, (y1 + y2) // 2,
                text=symbols[tag], font=("Segoe UI", 12),
                fill="#555555", tags=("arrow_text",))
            self._dpad_rects[tag] = r

        self._dpad_canvas.create_text(cx, cy, text="STOP",
                                       font=("Segoe UI", 7), fill="#444444")

    def _set_dpad_active(self, direction: str, active: bool) -> None:
        if not hasattr(self, "_dpad_rects"):
            return
        rect_id = self._dpad_rects.get(direction)
        if rect_id is None:
            return
        color = "#e94560" if active else "#16213e"
        self._dpad_canvas.itemconfig(rect_id, fill=color)
        # Also update the key state label
        if hasattr(self, "_key_labels") and direction in self._key_labels:
            lbl = self._key_labels[direction]
            lbl.configure(fg="#4fc3f7" if active else "#444444",
                          bg="#1a1a3e" if active else "#111122")

    # ==================================================================
    # Keyboard handling (robust multi-layer approach)
    # ==================================================================

    def _on_action_key(self, event: tk.Event) -> None:
        """bind_all handler for specific action keys — fires for ALL widgets.
        Entry widgets are filtered so text editing still works."""
        # Let Entry widgets handle keys normally (text editing)
        widget = event.widget
        if isinstance(widget, tk.Entry):
            return
        # Also check if widget is a descendant of an Entry (ttk edge cases)
        try:
            if isinstance(widget, tk.Entry) or widget.winfo_class() == "Entry":
                return
        except Exception:
            pass

        name = event.keysym
        if name in ("Up", "Down", "Left", "Right"):
            direction = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}[name]
            if direction not in self._pressed:
                self._pressed.add(direction)
                self._set_dpad_active(direction, True)
            return
        if name == "space":
            if not self._space_down:
                self._space_down = True
                self._cycle_gear()
            return
        if name == "Escape":
            self._emergency_stop()
            return
        if name == "Return":
            self._toggle_connect()
            return
        if name == "Tab":
            self._cycle_mode()
            return

    def _on_dir_release(self, event: tk.Event) -> None:
        """bind_all handler for direction key release."""
        widget = event.widget
        if isinstance(widget, tk.Entry):
            return
        try:
            if widget.winfo_class() == "Entry":
                return
        except Exception:
            pass

        name = event.keysym
        direction_map = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}
        direction = direction_map.get(name)
        if direction:
            self._pressed.discard(direction)
            self._set_dpad_active(direction, False)

    def _on_space_release(self, _event: tk.Event) -> None:
        self._space_down = False

    def _cycle_gear(self) -> None:
        self._gear_index = (self._gear_index + 1) % len(GEARS)
        self._update_gear_display()

    def _update_gear_display(self) -> None:
        g = GEARS[self._gear_index]
        self._gear_canvas.itemconfig(
            self._gear_text,
            text=f"⚡ {g['label']}",
            fill=g["color"])

    # ==================================================================
    # Control loop
    # ==================================================================

    def _control_tick(self) -> None:
        """Run either pure manual driving or PID path data collection."""
        if not self.link.connected:
            self._schedule_control()
            return
        if self._operation_mode not in ("manual", "pid_collect", "paper_e4"):
            return

        now = time.monotonic()
        dt = min(0.2, max(0.0, now - self._last_control_time))
        self._last_control_time = now
        if self._operation_mode == "pid_collect":
            self._pid_collection_tick(now, dt)
        elif self._operation_mode == "paper_e4":
            self._paper_e4_tick(now, dt)
        else:
            self._manual_control_tick(dt)
        self._schedule_control()

    def _manual_control_tick(self, dt: float) -> None:
        """Send direction-key commands without collecting synthesis data."""
        g = GEARS[self._gear_index]
        forward, turning = driver_axes(self._pressed)
        driver_active = forward != 0.0 or turning != 0.0
        self._collection_ready = False
        target_r, target_l = compute_manual_voltages(
            self._pressed,
            self._u_max,
            g["voltage"],
        )

        if driver_active:
            max_step = MANUAL_VOLTAGE_SLEW_V_PER_S * dt
            u_r = float(np.clip(target_r, self._last_ur - max_step,
                                self._last_ur + max_step))
            u_l = float(np.clip(target_l, self._last_ul - max_step,
                                self._last_ul + max_step))
        else:
            u_r = 0.0
            u_l = 0.0
        self._send_manual_voltage(u_r, u_l)
        n = len(self._pressed)
        self._cmd_var.set(f"cmds: {self._cmd_count}  voltage: "
                          f"{GEARS[self._gear_index]['voltage']:.0f} V  "
                          f"keys held: {n}")
        self._collection_status_var.set("人工驾驶 · 不记录权重合成数据")
        if hasattr(self, "_key_count_label"):
            self._key_count_label.configure(text=f"held: {n}")

    def _pid_elapsed(self, now: float | None = None) -> float:
        """Return active PID trajectory time, excluding safe telemetry pauses."""
        current = time.monotonic() if now is None else float(now)
        if self._pid_started_at <= 0.0:
            return 0.0
        active_until = (
            self._pid_pause_started_at
            if self._pid_pause_started_at is not None
            else current
        )
        return max(0.0, active_until - self._pid_started_at - self._pid_paused_s)

    def _pause_pid_clock(self, now: float, since: float | None = None) -> None:
        if self._pid_pause_started_at is None:
            pause_start = float(now) if since is None else float(since)
            self._pid_pause_started_at = min(
                float(now), max(self._pid_started_at, pause_start)
            )

    def _resume_pid_clock(self, now: float) -> None:
        if self._pid_pause_started_at is None:
            return
        self._pid_paused_s += max(0.0, float(now) - self._pid_pause_started_at)
        self._pid_pause_started_at = None

    def _pid_collection_tick(self, now: float, dt: float) -> None:
        """Track the collection path using only the PID-generated voltages."""
        processed_age = now - self._latest_sensor_time
        newest_transport_time = max(
            self._latest_sensor_time,
            self.link.last_live_state_rx,
        )
        transport_age = now - newest_transport_time
        if (
            self._latest_sensor_time <= 0.0
            or processed_age > PID_TELEMETRY_TIMEOUT_S
        ):
            self._pause_pid_clock(
                now,
                self._latest_sensor_time if self._latest_sensor_time > 0.0 else now,
            )
            self._collection_ready = False
            self._send_manual_voltage(0.0, 0.0)
            if (
                newest_transport_time > 0.0
                and transport_age <= PID_TELEMETRY_TIMEOUT_S
            ):
                self._collection_status_var.set(
                    "遥测已收到 · 界面正在追赶绘图积压 · 电机暂时停止"
                )
                return
            self._collection_status_var.set("等待固件实时位姿 · 电机保持停止")
            wall_elapsed = now - self._pid_started_at
            if (
                wall_elapsed > PID_TELEMETRY_ABORT_S
                and newest_transport_time > 0.0
                and transport_age > PID_TELEMETRY_ABORT_S
            ):
                self.after(0, self._stop_pid_collection)
            return
        if not self._latest_ina_ok:
            self._pause_pid_clock(now)
            self._collection_ready = False
            self._send_manual_voltage(0.0, 0.0)
            self._collection_status_var.set("INA 无效 · 电机保持停止 · 等待固件锁存故障")
            return

        self._resume_pid_clock(now)
        elapsed = self._pid_elapsed(now)
        if elapsed >= self._pid_duration > 0.0:
            self._send_manual_voltage(0.0, 0.0)
            self.after(0, self._stop_pid_collection)
            return

        preset = (
            self._pid_active_preset
            or PID_COLLECTION_PRESETS[self._pid_reference_name]
        )
        reference = pid_collection_reference(elapsed, preset)
        base_r, base_l, metrics = self._pid_controller.step(
            self._latest_pose,
            self._latest_velocity,
            reference,
            dt,
            self._u_max,
            self._track_width_m,
        )
        max_step = MANUAL_VOLTAGE_SLEW_V_PER_S * dt
        u_r = float(np.clip(base_r, self._last_ur - max_step,
                            self._last_ur + max_step))
        u_l = float(np.clip(base_l, self._last_ul - max_step,
                            self._last_ul + max_step))
        self._collection_ready = elapsed >= PID_COLLECTION_WARMUP_S
        self._pid_last_metrics = metrics
        self._send_manual_voltage(u_r, u_l)
        self._cmd_var.set(
            f"PID {elapsed:4.1f}/{self._pid_duration:.0f}s  "
            f"e=({metrics['e_x']:+.2f},{metrics['e_y']:+.2f},"
            f"{metrics['e_theta']:+.2f})  gyro={self._latest_gyro_z:+.2f}"
        )
        if self._collection_ready:
            self._collection_status_var.set("PID 路径采集中 · 仅记录 PID 控制电压")
        else:
            self._collection_status_var.set(
                f"PID路径预热 · {elapsed:.1f}/{PID_COLLECTION_WARMUP_S:.0f} s · 不记录"
            )

    def _send_manual_voltage(self, u_r: float, u_l: float) -> None:
        self._last_ur = float(u_r)
        self._last_ul = float(u_l)
        self._cmd_count += 1
        self.link.send({"cmd": "manual", "u_r": self._last_ur, "u_l": self._last_ul})
        self._volt_var.set(
            f"u_R = {self._last_ur: 6.2f} V    u_L = {self._last_ul: 6.2f} V"
        )

    def _schedule_control(self) -> None:
        if self._control_job:
            self.after_cancel(self._control_job)
        interval_ms = int(1000 / CONTROL_HZ)
        self._control_job = self.after(interval_ms, self._control_tick)

    def _start_control_loop(self) -> None:
        self._control_tick()

    def _stop_connect_tick(self) -> None:
        if self._connect_job:
            self.after_cancel(self._connect_job)
            self._connect_job = None

    def _stop_control_loop(self) -> None:
        if self._control_job:
            self.after_cancel(self._control_job)
            self._control_job = None

    # ==================================================================
    # Keep-alive
    # ==================================================================

    def _keepalive_tick(self) -> None:
        if not self.link.connected:
            # 运行中 TCP 断开：reader 已关闭 socket。立即把界面切回
            # 断链状态，避免"界面显示已连接却控制不了"，并保证下一次
            # Connect 直接走全新握手（旧 socket 悬空会让重连必然超时）。
            if self._connecting_stage >= 4 or self._operation_mode != "idle":
                self._on_disconnected("运行中连接断开")
            return
        self.link.send({"cmd": "hello"})
        self._keepalive_job = self.after(int(KEEPALIVE_S * 1000), self._keepalive_tick)

    def _start_keepalive(self) -> None:
        self._keepalive_tick()

    def _stop_keepalive(self) -> None:
        if self._keepalive_job:
            try:
                self.after_cancel(self._keepalive_job)
            except tk.TclError:
                pass  # job already fired (e.g. watchdog inside _keepalive_tick)
            self._keepalive_job = None

    # ==================================================================
    # Connection actions (non-blocking state machine)
    # ==================================================================

    def _requested_motor_deadzone(self) -> float:
        return validate_motor_deadzone(
            self._synthesis_vars["motor_deadzone_v"].get(),
            max_voltage=THEORY_BRINGUP_U_MAX_V,
        )

    def _requested_theory_safety_limits(self) -> dict[str, float]:
        heading_deg = float(
            self._synthesis_vars["theory_max_heading_error_deg"].get()
        )
        return validate_theory_safety_limits(
            self._synthesis_vars["theory_max_position_error"].get(),
            math.radians(heading_deg),
            self._synthesis_vars["theory_max_z2"].get(),
            self._synthesis_vars["theory_max_z3"].get(),
        )

    def _toggle_connect(self) -> None:
        # The reader can mark the socket dead just before Tk repaints the
        # still-visible Disconnect button.  A non-idle connection stage means
        # that click is still a disconnect request, never a reconnect request.
        if self.link.connected or self._connecting_stage != 0:
            self._do_disconnect()
        else:
            self._do_connect()

    def _do_connect(self) -> None:
        """Initiate connection. The rest is driven by _connect_tick()."""
        try:
            self._motor_deadzone_v = self._requested_motor_deadzone()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("死区配置错误", str(exc))
            return
        try:
            if self._mode_var.get() == "usb":
                port = self._serial_var.get().strip()
                if not port:
                    messagebox.showwarning("No Port", "Enter a serial port (e.g. COM3).")
                    return
                self.link.connect_serial(port)
            else:
                host = self._host_var.get().strip()
                port = int(self._port_var.get().strip())
                self.link.connect_wifi(host, port)

            self._log(f"Socket open → {self.link.status_text}")
            self._status_var.set(f"Connected to {self.link.status_text} · sending hello…")

            self._connecting_stage = 1
            # USB serial open resets the ESP32 (driver asserts DTR). Give it time
            # to boot before the first hello; WiFi connects immediately.
            self._connect_deadline = time.monotonic() + CONNECT_HELLO_TIMEOUT_S
            self._last_hello_time = 0.0
            self._connect_pending_sequence = None
            self._connect_last_send_time = 0.0
            self._connect_manual_phase = ""
            self._connect_btn.config(text="…", bg="#555555", state=tk.DISABLED)

            # drive via periodic tick (non-blocking, will send hello from tick)
            self._connect_tick()

        except Exception as e:
            self.link.disconnect()
            self._log(f"Connect FAILED: {e}")
            messagebox.showerror("Connection Failed", str(e))

    def _schedule_connect_retry(self) -> None:
        """Schedule next _connect_tick, cancelling any previous pending call."""
        if self._connect_job:
            self.after_cancel(self._connect_job)
        self._connect_job = self.after(100, self._connect_tick)

    def _fail_connect_handshake(self, detail: str) -> None:
        self._log(f"Handshake timeout: {detail}")
        self.link.disconnect()
        self._on_disconnected()
        messagebox.showerror(
            "连接超时",
            f"固件未完成连接确认：{detail}\n已自动重试多次，请再次连接。",
        )

    def _connect_tick(self) -> None:
        """Non-blocking connection state machine — poll messages until armed."""
        # Already armed — stop polling
        if self._connecting_stage >= 4:
            return

        if not self.link.connected:
            self._log("Link died during handshake")
            self._on_disconnected()
            return

        now = time.monotonic()
        msgs = self.link.drain()

        if self._connecting_stage == 1:
            # Send hello every 2s until FW responds (USB ESP32 may be booting)
            if now - self._last_hello_time > 1.0:
                self.link.send({"cmd": "hello"})
                self._last_hello_time = now
            # waiting for hello
            for m in msgs:
                if m.get("type") == "hello":
                    self._fw_version = m.get("firmware", "?")
                    self._fw_mode = m.get("mode", "?")
                    self._fw_fault = m.get("fault", "?")
                    firmware_basis = str(m.get("regressor_basis", "") or "")
                    if firmware_basis != REGRESSOR_BASIS:
                        self._fail_connect_handshake(
                            "固件回归基不兼容；请烧录配套 sgn 固件"
                        )
                        return
                    boot_id = str(m.get("boot_id", "") or "")
                    reset_reason = str(
                        m.get("reset_reason", "unknown") or "unknown"
                    )
                    try:
                        uptime_ms = int(m.get("uptime_ms", 0) or 0)
                    except (TypeError, ValueError):
                        uptime_ms = 0
                    if (
                        boot_id
                        and self._last_boot_id is not None
                        and boot_id != self._last_boot_id
                    ):
                        self._log(
                            f"⚠ Firmware reboot detected: reset={reset_reason} "
                            f"uptime={uptime_ms / 1000.0:.1f}s"
                        )
                    if boot_id:
                        self._last_boot_id = boot_id
                    cfg = m.get("config", {})
                    self._motor_deadzone_supported = (
                        "motor_deadzone_v" in cfg
                    )
                    self._gyro_deadband_supported = "gyro_deadband" in cfg
                    self._theory_safety_supported = all(
                        key in cfg
                        for key in (
                            "theory_max_position_error",
                            "theory_max_heading_error",
                            "theory_max_z2",
                            "theory_max_z3",
                        )
                    )
                    self._motion_reference_supported = all(
                        key in cfg for key in MOTION_REFERENCE_DEFAULTS
                    )
                    self._guidance_supported = all(
                        key in cfg for key in GUIDANCE_DEFAULTS
                    )
                    self._comparison_modes_supported = all(
                        key in cfg
                        for key in ("comparison_case", *PID_BENCHMARK_DEFAULTS)
                    )
                    if not self._motor_deadzone_supported:
                        self._log(
                            "⚠ 当前固件不支持由 EXE 下发电压死区；"
                            "请烧录新版固件"
                        )
                    if not self._gyro_deadband_supported:
                        self._log(
                            "⚠ 当前固件不支持陀螺角速度阈值；请烧录新版固件"
                        )
                    if not self._theory_safety_supported:
                        self._log(
                            "⚠ 当前固件不支持由 EXE 下发 THEORY 越界阈值；"
                            "请烧录新版固件"
                        )
                    if not self._motion_reference_supported:
                        self._log(
                            "⚠ 当前固件不支持闭环运动参考生成器；"
                            "必须烧录 1.4.0 或更高版本后才能启动 THEORY"
                        )
                    if not self._guidance_supported:
                        self._log(
                            "⚠ 当前固件不支持论文预瞄极坐标捕获外环；"
                            "必须烧录 1.7.0 或更高版本后才能启动 THEORY"
                        )
                    if not self._comparison_modes_supported:
                        self._log(
                            "⚠ 当前固件不支持 E3 板载级联 PID；"
                            "E1/E2 仍可运行，E3 需烧录 1.5.0 或更高版本"
                        )
                    try:
                        advertised_u_max = float(
                            cfg.get("u_max", U_MAX_DEFAULT)
                        )
                    except (TypeError, ValueError):
                        advertised_u_max = U_MAX_DEFAULT
                    self._u_max = max(
                        0.2, min(U_MAX_DEFAULT, advertised_u_max)
                    )
                    if advertised_u_max > U_MAX_DEFAULT:
                        self._log(
                            f"PC hard voltage cap: firmware advertised "
                            f"{advertised_u_max:.1f}V → {U_MAX_DEFAULT:.1f}V"
                        )
                    geometry = m.get("geometry", {})
                    self._track_width_m = float(
                        geometry.get("track_width_m", self._track_width_m)
                    )
                    self._log(
                        f"Hello: FW={self._fw_version} mode={self._fw_mode} "
                        f"fault={self._fw_fault} uMax={self._u_max:.1f}V "
                        f"reset={reset_reason} uptime={uptime_ms / 1000.0:.1f}s"
                    )

                    # Firmware faults are latched. A stop request is the
                    # deliberate acknowledgement that clears the latch.
                    self._connect_pending_sequence = self.link.send({"cmd": "stop"})
                    self._connect_last_send_time = now
                    self._connecting_stage = 2
                    self._connect_deadline = now + CONNECT_STAGE_TIMEOUT_S
                    self._status_var.set("正在停车并清除已锁存故障…")
                    break
            else:
                # not yet — retry
                if now > self._connect_deadline:
                    self._fail_connect_handshake("等待 hello")
                    return
                self._schedule_connect_retry()
                return

        if self._connecting_stage == 2:
            # waiting for stop/clear-fault ack
            for m in msgs:
                if (
                    m.get("type") == "ack"
                    and m.get("message") == "stop_requested"
                    and (
                        self._connect_pending_sequence is None
                        or m.get("seq") == self._connect_pending_sequence
                    )
                ):
                    self._log("Stop acknowledged · latched fault clear requested")
                    self._connect_pending_sequence = self.link.send(
                        {
                            "cmd": "configure",
                            "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
                            "current_tau": CURRENT_FILTER_TAU_S,
                            "motor_deadzone_v": self._motor_deadzone_v,
                            "manual_snapshots": False,
                        }
                    )
                    self._connect_last_send_time = now
                    self._connect_manual_phase = "configure"
                    self._connecting_stage = 3
                    self._connect_deadline = now + CONNECT_STAGE_TIMEOUT_S
                    self._status_var.set("正在配置稳定通信模式…")
                    self._schedule_connect_retry()
                    return
                elif m.get("type") == "error":
                    err = m.get("message", "?")
                    self._log(f"Fault clear rejected: {err}")
                    self.link.disconnect()
                    self._on_disconnected()
                    messagebox.showerror("固件错误", f"无法清除已锁存故障：{err}")
                    return
            if now > self._connect_deadline:
                self._fail_connect_handshake("等待停车/清故障确认")
                return
            if now - self._connect_last_send_time >= CONNECT_RETRY_INTERVAL_S:
                self.link.send(
                    {"cmd": "stop", "seq": self._connect_pending_sequence}
                )
                self._connect_last_send_time = now
            self._schedule_connect_retry()
            return

        if self._connecting_stage == 3:
            # Configure while stopped, then enter MANUAL. Both commands are
            # retried with the same sequence because ACK frames may be delayed
            # behind telemetry on a busy ESP32 link.
            for m in msgs:
                if (
                    m.get("type") == "ack"
                    and m.get("seq") == self._connect_pending_sequence
                    and self._connect_manual_phase == "configure"
                    and m.get("message") == "configuration_updated"
                ):
                    self._connect_pending_sequence = self.link.send(
                        {"cmd": "start", "mode": "manual"}
                    )
                    self._connect_last_send_time = now
                    self._connect_manual_phase = "start"
                    self._connect_deadline = now + CONNECT_STAGE_TIMEOUT_S
                    self._status_var.set("正在进入人工控制模式…")
                    self._schedule_connect_retry()
                    return
                if (
                    m.get("type") == "ack"
                    and m.get("seq") == self._connect_pending_sequence
                    and self._connect_manual_phase == "start"
                    and m.get("message") == "start_requested"
                ):
                    self._log("Mode → MANUAL (armed)")
                    self._connecting_stage = 4
                    self._on_connected()
                    return
                elif m.get("type") == "error":
                    err = m.get("message", "?")
                    self._log(f"Start rejected: {err}")
                    self.link.disconnect()
                    self._on_disconnected()
                    messagebox.showerror("Firmware Error", f"Cannot enter manual mode: {err}")
                    return
            if now > self._connect_deadline:
                detail = (
                    "等待通信配置确认"
                    if self._connect_manual_phase == "configure"
                    else "等待人工模式确认"
                )
                self._fail_connect_handshake(detail)
                return
            if now - self._connect_last_send_time >= CONNECT_RETRY_INTERVAL_S:
                if self._connect_manual_phase == "configure":
                    payload = {
                        "cmd": "configure",
                        "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
                        "current_tau": CURRENT_FILTER_TAU_S,
                        "motor_deadzone_v": self._motor_deadzone_v,
                        "manual_snapshots": False,
                    }
                else:
                    payload = {"cmd": "start", "mode": "manual"}
                payload["seq"] = self._connect_pending_sequence
                self.link.send(payload)
                self._connect_last_send_time = now
            self._schedule_connect_retry()
            return

    def _do_disconnect(self) -> None:
        self._connecting_stage = 0
        self._motor_deadzone_supported = False
        self._gyro_deadband_supported = False
        self._theory_safety_supported = False
        self._motion_reference_supported = False
        self._guidance_supported = False
        self._comparison_modes_supported = False
        self._reset_workflow()
        self._operation_mode = "idle"
        self._stop_connect_tick()
        self._stop_control_loop()
        self._stop_keepalive()
        try:
            self.link.send({"cmd": "stop"})
        except Exception:
            pass
        time.sleep(0.1)
        self.link.disconnect()
        self._pressed.clear()
        for d in ("up", "down", "left", "right"):
            self._set_dpad_active(d, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._cmd_count = 0
        self._volt_var.set("u_R =  0.00 V    u_L =  0.00 V")
        self._cmd_var.set("")
        self._log("Disconnected")
        self._on_disconnected()

    def _on_connected(self) -> None:
        self._stop_connect_tick()
        self._reported_fault = "none"
        self._fw_fault = "none"
        self._operation_mode = "manual"
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._collection_ready = False
        self._last_control_time = time.monotonic()
        self._connect_btn.config(text="Disconnect", bg="#c0392b",
                                 activebackground="#e74c3c", state=tk.NORMAL)
        self._conn_dot_color("#4caf50")
        self._status_var.set(
            f"Connected · {self.link.status_text} · FW={self._fw_version} "
            f"· uMax={self._u_max:.1f}V · 死区=±{self._motor_deadzone_v:g}V "
            "· MANUAL"
        )
        self._update_gear_display()
        self._cmd_count = 0
        self._log(
            f"Stable manual mode configured · snapshots off · "
            f"derivative τ={COLLECTION_DERIVATIVE_TAU_S:.2f}s · "
            f"dead-zone=±{self._motor_deadzone_v:g}V"
        )
        self._start_keepalive()
        self._start_control_loop()
        self._update_theory_start_state()
        self._update_pid_start_state()
        # Ensure the window has focus for keyboard events
        self.focus_set()
        self._log("READY — arrow keys drive, Space switches 1/2/3/4 V")

    def _on_disconnected(self, unexpected_reason: str | None = None) -> None:
        retain_theory_plot = bool(self._theory_rows) and (
            self._operation_mode in ("theory", "theory_complete")
            or self._plot_view == "theory"
        )
        self._connecting_stage = 0
        self._motor_deadzone_supported = False
        self._gyro_deadband_supported = False
        self._theory_safety_supported = False
        self._motion_reference_supported = False
        self._guidance_supported = False
        self._comparison_modes_supported = False
        self._stop_connect_tick()
        self._stop_control_loop()
        self._stop_keepalive()
        self._discard_active_collection_segment("通信中断")
        self._flush_snapshot_cache(force=True)
        self._reset_workflow()
        self._operation_mode = "idle"
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence = None
        self._plot_view = "theory" if retain_theory_plot else "synthesis"
        self._plot_dirty = True
        self._stop_connect_tick()
        self._connect_btn.config(text="Connect", bg="#0f3460",
                                 activebackground="#1a5276", state=tk.NORMAL)
        self._conn_dot_color("#f44336")
        if unexpected_reason:
            self._status_var.set(f"连接已中断：{unexpected_reason} · 请重新连接")
        else:
            self._status_var.set("Disconnected · Press Enter to connect")
        self._gear_canvas.itemconfig(self._gear_text, text="", fill="#ffffff")
        self._latest_sensor_time = 0.0
        self._latest_imu_ok = False
        self._latest_imu_calibrated = False
        self._latest_ina_ok = False
        self._update_sensor_display()
        self._update_theory_start_state()
        self._update_pid_start_state()

    def _emergency_stop(self) -> None:
        """Stop motors + disconnect (ESC key)."""
        self._connecting_stage = 0
        self._reset_workflow()
        self._operation_mode = "idle"
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence = None
        self._stop_connect_tick()
        self._stop_control_loop()
        self._stop_keepalive()
        try:
            self.link.send({"cmd": "stop"})
        except Exception:
            pass
        time.sleep(0.1)
        self.link.disconnect()
        self._pressed.clear()
        for d in ("up", "down", "left", "right"):
            self._set_dpad_active(d, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._cmd_count = 0
        self._volt_var.set("u_R =  0.00 V    u_L =  0.00 V")
        self._cmd_var.set("")
        self._on_disconnected()
        self._status_var.set("⚠ Emergency stop")
        self._log("⚠ EMERGENCY STOP")

    def _cycle_mode(self) -> None:
        if self.link.connected:
            return
        current = self._mode_var.get()
        self._mode_var.set("usb" if current == "wifi" else "wifi")
        self._on_mode_change()

    def _on_mode_change(self) -> None:
        is_wifi = self._mode_var.get() == "wifi"
        if is_wifi:
            self._wifi_frame.pack(side=tk.LEFT, padx=(16, 0))
            self._usb_frame.pack_forget()
        else:
            self._wifi_frame.pack_forget()
            self._usb_frame.pack(side=tk.LEFT, padx=(16, 0))

    def _conn_dot_color(self, color: str) -> None:
        self._conn_light.itemconfig(self._conn_dot, fill=color)

    # ==================================================================
    # PID collection, offline synthesis, and confirmed THEORY workflow
    # ==================================================================

    def _apply_motor_deadzone(self) -> None:
        if not self.link.connected or self._connecting_stage < 4:
            messagebox.showwarning("尚未连接", "请先连接小车。")
            return
        if not self._motor_deadzone_supported or not self._gyro_deadband_supported:
            messagebox.showerror(
                "固件不兼容",
                "当前固件不支持 EXE 下发的电压死区或陀螺角速度阈值，请先烧录新版固件。",
            )
            return
        if self._workflow_kind is not None or self._operation_mode != "manual":
            messagebox.showwarning(
                "当前不可调整",
                "只能在人工控制且没有其他控制流程运行时调整电压死区。",
            )
            return
        try:
            value = self._requested_motor_deadzone()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("死区配置错误", str(exc))
            return
        self._motor_deadzone_v = value
        self._stop_control_loop()
        self._pressed.clear()
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._operation_mode = "transition"
        self._workflow_kind = "apply_motor_deadzone"
        self._workflow_steps = deque(
            (
                {"name": "停车", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": "下发电压死区",
                 "payload": {
                     "cmd": "configure",
                     "motor_deadzone_v": value,
                     "manual_snapshots": False,
                 },
                 "ack": "configuration_updated"},
                {"name": "恢复人工控制",
                 "payload": {"cmd": "start", "mode": "manual"},
                 "ack": "start_requested"},
            )
        )
        self._status_var.set(f"正在应用对称电压死区 ±{value:g} V…")
        self._send_next_workflow_step()

    def _update_pid_start_state(self) -> None:
        if not hasattr(self, "_pid_start_button"):
            return
        sensor_fresh = (
            self._latest_sensor_time > 0.0
            and time.monotonic() - self._latest_sensor_time
            <= PID_TELEMETRY_TIMEOUT_S
        )
        enabled = (
            self.link.connected
            and self._connecting_stage >= 4
            and self._workflow_kind is None
            and self._operation_mode == "manual"
            and sensor_fresh
            and self._latest_ina_ok
        )
        self._pid_start_button.configure(
            state=tk.NORMAL if enabled else tk.DISABLED,
            fg="#ffffff" if enabled else "#777777",
            bg="#0f6b4f" if enabled else "#2a2a4a",
        )

    def _start_pid_collection(self) -> None:
        if not self.link.connected or self._connecting_stage < 4:
            messagebox.showwarning("尚未连接", "请先连接小车。")
            return
        if not self._motor_deadzone_supported or not self._gyro_deadband_supported:
            messagebox.showerror(
                "固件不兼容",
                "当前固件不支持 EXE 下发的电压死区或陀螺角速度阈值，请先烧录新版固件。",
            )
            return
        if self._operation_mode != "manual" or self._workflow_kind is not None:
            return
        sensor_age = time.monotonic() - self._latest_sensor_time
        if self._latest_sensor_time <= 0.0 or sensor_age > PID_TELEMETRY_TIMEOUT_S:
            messagebox.showerror("传感器未就绪", "尚未收到新鲜的实时传感器数据。")
            return
        if not self._latest_ina_ok:
            messagebox.showerror(
                "电流传感器未就绪",
                "INA3221 当前无效，不能开始用于权重综合的数据采集。",
            )
            return
        preset_name = self._pid_path_var.get()
        preset = PID_COLLECTION_PRESETS.get(preset_name)
        if preset is None:
            messagebox.showerror("路径配置错误", "未找到所选 PID 采集路径。")
            return
        try:
            motor_deadzone_v = self._requested_motor_deadzone()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("运行配置错误", str(exc))
            return
        self._motor_deadzone_v = motor_deadzone_v
        try:
            speed_scale = float(self._pid_speed_var.get())
            integral_window = float(self._pid_window_var.get())
        except ValueError:
            messagebox.showerror("运行配置错误", "速度倍率和积分窗口必须是数字。")
            return
        if not 0.5 <= speed_scale <= 4.0:
            messagebox.showerror("运行配置错误", "速度倍率需在 0.5–4.0 之间。")
            return
        if not 0.05 <= integral_window <= 1.0:
            messagebox.showerror(
                "运行配置错误", "积分窗口需在 0.05–1.0 s 之间（固件允许范围）。"
            )
            return
        self._integral_window = integral_window
        preset = dict(preset)
        preset["nu"] *= speed_scale
        self._pid_active_preset = preset
        confirmed = messagebox.askokcancel(
            "启动 PID 路径采集",
            f"采集路径：{preset_name}\n"
            f"运行时间：{preset['duration']:.0f} s\n"
            f"速度倍率：{speed_scale:g}×（nu {PID_COLLECTION_PRESETS[self._pid_path_var.get()]['nu']:g} → {preset['nu']:g}）\n"
            f"积分窗口 T_w：{integral_window:g} s（快照率 {1.0 / integral_window:.0f} Hz）\n\n"
            f"最终电压死区：±{motor_deadzone_v:g} V\n"
            f"ω 来源：编码器差速（IMU 机制已取消）\n"
            "小车将自动清零位姿并沿路径运行，人工方向键在采集期间无效。\n"
            "请放在空旷地面，并确保可立即按 Esc 急停。",
        )
        if not confirmed:
            return
        self._active_collection_segment = self._dataset.start_segment(
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "firmware": self._fw_version,
                "path": preset_name,
                "duration_s": float(preset["duration"]),
                "track_width_m": float(self._track_width_m),
                "u_max_v": float(self._u_max),
                "motor_deadzone_v": motor_deadzone_v,
                "integral_window_s": integral_window,
                "speed_scale": speed_scale,
                "omega_source": "encoder",
                "source": "firmware_integral_v3_sgn_gyro_gate",
            }
        )
        self._synthesis = None
        self._qualification = None
        self._disturbance_diagnostics_var.set(
            "扰动诊断：正在采集新数据，采集结束后重新综合"
        )
        self._snapshot_t0_us = None
        self._snapshot_fallback_t = 0.0
        # Keep the accumulated raw dataset, but start a fresh live plot window
        # for this segment so restored timestamps cannot run backwards.
        self._plot_rows.clear()
        self._plot_times.clear()
        self._stop_control_loop()
        self._pressed.clear()
        for direction in ("up", "down", "left", "right"):
            self._set_dpad_active(direction, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._latest_sensor_time = 0.0
        self._collection_ready = False
        self._operation_mode = "transition"
        self._pid_reference_name = preset_name
        self._pid_duration = float(preset["duration"])
        self._pid_pause_started_at = None
        self._pid_paused_s = 0.0
        self._pid_pose_trace.clear()
        self._pid_trace_times.clear()
        self._plot_view = "collection"
        self._collection_plot_mode = "trajectory"
        self._plot_dirty = True
        self._workflow_kind = "start_pid_collect"
        self._workflow_steps = deque(
            (
                {"name": "停车", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": "配置采集滤波",
                 "payload": {
                     "cmd": "configure",
                     "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
                     "current_tau": CURRENT_FILTER_TAU_S,
                     "motor_deadzone_v": motor_deadzone_v,
                     "gyro_tau": GYRO_RATE_FILTER_TAU_S,
                     "integral_window": self._integral_window,
                     "manual_snapshots": True,
                 },
                 "ack": "configuration_updated"},
                {"name": "位姿清零", "payload": {"cmd": "zero_pose"},
                 "ack": "zero_pose_requested"},
                {"name": "启动 PID 采集底层输出",
                 "payload": {"cmd": "start", "mode": "manual"},
                 "ack": "start_requested",
                 "requires_sensor_ready": True},
            )
        )
        self._collection_status_var.set("正在准备 PID 路径采集…")
        self._status_var.set("正在停车、清零位姿并启动 PID 路径采集…")
        self._update_pid_start_state()
        self._update_theory_start_state()
        self._send_next_workflow_step()

    def _stop_pid_collection(self) -> None:
        if not self.link.connected or self._connecting_stage < 4:
            return
        if self._operation_mode not in ("pid_collect", "transition"):
            return
        if self._workflow_kind == "stop_pid_collect":
            return
        self._stop_control_loop()
        self._pressed.clear()
        self._collection_ready = False
        self._reset_workflow()
        self._operation_mode = "transition"
        self._workflow_kind = "stop_pid_collect"
        self._workflow_steps = deque(
            (
                {"name": "停止 PID 采集", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": "关闭采集快照流",
                 "payload": {"cmd": "configure", "manual_snapshots": False},
                 "ack": "configuration_updated"},
                {"name": "恢复人工控制",
                 "payload": {"cmd": "start", "mode": "manual"},
                 "ack": "start_requested"},
            )
        )
        self._collection_status_var.set("正在停车并返回人工控制…")
        self._send_next_workflow_step()

    # ==================================================================
    # Paper E4: external sampling-PID baseline (PC-side 25 Hz controller)
    # ==================================================================

    def _start_paper_e4(self) -> None:
        """Run the paper E4 baseline: PidPathController at 25 Hz, 4 V authority."""
        if not self.link.connected or self._connecting_stage < 4:
            messagebox.showwarning("尚未连接", "请先连接小车。")
            return
        if not self._motor_deadzone_supported or not self._gyro_deadband_supported:
            messagebox.showerror(
                "固件不兼容",
                "当前固件不支持 EXE 下发的电压死区，请先烧录新版固件。",
            )
            return
        if self._operation_mode != "manual" or self._workflow_kind is not None:
            return
        sensor_age = time.monotonic() - self._latest_sensor_time
        if self._latest_sensor_time <= 0.0 or sensor_age > PID_TELEMETRY_TIMEOUT_S:
            messagebox.showerror("传感器未就绪", "尚未收到新鲜的实时传感器数据。")
            return
        if not self._latest_ina_ok:
            messagebox.showerror(
                "电流传感器未就绪",
                "INA3221 当前无效，不能启动论文 E4 基线。",
            )
            return
        motor_deadzone_v = float(PAPER_CAMPAIGN["motor_deadzone_v"])
        confirmed = messagebox.askokcancel(
            "启动论文 E4 采样 PID 基线",
            "控制器：PC 端 25 Hz 位姿 P 环 + 轮压 PI"
            "（采集激励控制器，base_limit_frac=1.0，电压 4 V 满权）\n"
            f"参考轨迹：{PAPER_CAMPAIGN['preset']}（圆心起步，t=0 即全速）\n"
            f"运行时间：{PAPER_E4_RUN_DURATION_S:.3f} s（1 圈 + 3.0 s）\n"
            f"控制周期：{PAPER_E4_CONTROL_PERIOD_S * 1000:.0f} ms\n"
            f"最终电压死区：±{motor_deadzone_v:g} V\n"
            "小车将自动清零位姿并跟踪 PC 生成的圆参考；\n"
            "请放在空旷地面，并确保可立即按 Esc 急停。",
        )
        if not confirmed:
            return
        self._motor_deadzone_v = motor_deadzone_v
        self._stop_control_loop()
        self._pressed.clear()
        for direction in ("up", "down", "left", "right"):
            self._set_dpad_active(direction, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._latest_sensor_time = 0.0
        self._e4_rows = []
        self._e4_t0_us = None
        self._e4_last_command = np.zeros(2, dtype=float)
        self._e4_last_u = np.zeros(2, dtype=float)
        self._e4_last_state_us = 0
        self._e4_completed = False
        self._pid_controller.reset()
        self._pid_pose_trace.clear()
        self._pid_trace_times.clear()
        self._plot_view = "collection"
        self._collection_plot_mode = "trajectory"
        self._plot_dirty = True
        self._operation_mode = "transition"
        self._workflow_kind = "start_paper_e4"
        self._workflow_steps = deque(
            (
                {"name": "停车", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": "配置 E4 滤波",
                 "payload": {
                     "cmd": "configure",
                     "derivative_tau": COLLECTION_DERIVATIVE_TAU_S,
                     "current_tau": CURRENT_FILTER_TAU_S,
                     "motor_deadzone_v": motor_deadzone_v,
                     "gyro_tau": GYRO_RATE_FILTER_TAU_S,
                     "manual_snapshots": False,
                 },
                 "ack": "configuration_updated"},
                {"name": "位姿清零", "payload": {"cmd": "zero_pose"},
                 "ack": "zero_pose_requested"},
                {"name": "启动 E4 底层人工模式",
                 "payload": {"cmd": "start", "mode": "manual"},
                 "ack": "start_requested",
                 "requires_sensor_ready": True},
            )
        )
        self._collection_status_var.set("正在准备论文 E4 采样 PID 基线…")
        self._status_var.set("正在停车、清零位姿并启动论文 E4 基线…")
        self._update_theory_start_state()
        self._send_next_workflow_step()

    def _stop_paper_e4(self) -> None:
        if not self.link.connected or self._connecting_stage < 4:
            return
        if self._operation_mode not in ("paper_e4", "transition"):
            return
        if self._workflow_kind == "stop_paper_e4":
            return
        self._stop_control_loop()
        self._pressed.clear()
        self._collection_ready = False
        self._reset_workflow()
        if self._e4_rows and not self._e4_completed:
            self._export_paper_e4_trace("stopped")
        self._operation_mode = "transition"
        self._workflow_kind = "stop_paper_e4"
        self._workflow_steps = deque(
            (
                {"name": "停止论文 E4", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": "恢复人工通信配置",
                 "payload": {"cmd": "configure", "manual_snapshots": False},
                 "ack": "configuration_updated"},
                {"name": "恢复人工控制",
                 "payload": {"cmd": "start", "mode": "manual"},
                 "ack": "start_requested"},
            )
        )
        self._collection_status_var.set("正在停车并返回人工控制…")
        self._send_next_workflow_step()

    def _export_paper_e4_trace(self, reason: str) -> Path | None:
        """Export E4 rows in the frozen comparison-trace schema (schema 3)."""
        if not self._e4_rows:
            return None
        rows = list(self._e4_rows)
        config = {
            "baseline": "sampling_pid",
            "comparison_case": 4.0,
            "control_hz": 1.0 / PAPER_E4_CONTROL_PERIOD_S,
            "hold_duration": 0.0,
            "motor_deadzone_v": float(PAPER_CAMPAIGN["motor_deadzone_v"]),
            "outer_capture_radius_m": PAPER_CAMPAIGN["outer_capture_radius"],
            "outer_blend_radius_m": PAPER_CAMPAIGN["outer_blend_radius"],
            "outer_kp": PAPER_CAMPAIGN["outer_kp"],
            "outer_ktheta": PAPER_CAMPAIGN["outer_ktheta"],
            "outer_preview_horizon": PAPER_CAMPAIGN["outer_preview_horizon"],
            "outer_vbar": PAPER_CAMPAIGN["outer_vbar"],
            "pid_base_limit_frac": 1.0,
            "track_width_m": float(self._track_width_m),
            "ref_nu_rad_s": PAPER_E4_CIRCLE_NU_RAD_S,
            "ref_radius_m": PAPER_E4_CIRCLE_RADIUS_M,
            "ref_shape": 3.0,
            "r2": 0.0,
            "r3": 0.0,
            "run_duration": PAPER_E4_RUN_DURATION_S,
            "u_max": float(self._u_max),
        }
        payload: dict[str, Any] = {
            "schema": np.asarray([3], dtype=np.int64),
            "t": np.asarray([row["t"] for row in rows], dtype=float),
            "reason": np.asarray([reason]),
            "trajectory": np.asarray([PAPER_CAMPAIGN["preset"]]),
            "comparison_case": np.asarray(["E4"]),
            "comparison_name": np.asarray(
                ["sampling PID baseline (data-collection excitation controller)"]
            ),
            "initial_nonlinear_compensation_enabled": np.asarray([False]),
            "firmware": np.asarray([self._fw_version]),
            "runtime_config_json": np.asarray(
                [json.dumps(config, ensure_ascii=False, sort_keys=True)]
            ),
            "weights_json": np.asarray(
                [
                    json.dumps(
                        {"used_by_controller": False,
                         "note": "external PID baseline"},
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
        target = target_dir / (
            f"theory_trace_e4_gui_{time.strftime('%Y%m%d_%H%M%S')}_{reason}.npz"
        )
        np.savez_compressed(target, **payload)
        self._log(f"E4 trace exported: {target.name} ({reason}, {len(rows)} rows)")
        return target

    def _paper_e4_tick(self, now: float, dt: float) -> None:
        """25 Hz PC-side PID tracking of the paper center-start circle."""
        processed_age = now - self._latest_sensor_time
        newest_transport_time = max(
            self._latest_sensor_time,
            self.link.last_live_state_rx,
        )
        transport_age = now - newest_transport_time
        if (
            self._latest_sensor_time <= 0.0
            or processed_age > PID_TELEMETRY_TIMEOUT_S
        ):
            self._pause_pid_clock(
                now,
                self._latest_sensor_time if self._latest_sensor_time > 0.0 else now,
            )
            self._collection_ready = False
            self._send_manual_voltage(0.0, 0.0)
            if (
                newest_transport_time > 0.0
                and transport_age <= PID_TELEMETRY_TIMEOUT_S
            ):
                self._collection_status_var.set(
                    "遥测已收到 · 界面正在追赶绘图积压 · 电机暂时停止"
                )
                return
            self._collection_status_var.set("等待固件实时位姿 · 电机保持停止")
            wall_elapsed = now - self._pid_started_at
            if (
                wall_elapsed > PID_TELEMETRY_ABORT_S
                and newest_transport_time > 0.0
                and transport_age > PID_TELEMETRY_ABORT_S
            ):
                self.after(0, self._stop_paper_e4)
            return
        if not self._latest_ina_ok:
            self._pause_pid_clock(now)
            self._collection_ready = False
            self._send_manual_voltage(0.0, 0.0)
            self._collection_status_var.set("INA 无效 · 电机保持停止 · 等待固件锁存故障")
            return

        self._resume_pid_clock(now)
        elapsed = self._pid_elapsed(now)
        if elapsed >= PAPER_E4_RUN_DURATION_S > 0.0:
            self._send_manual_voltage(0.0, 0.0)
            self._e4_completed = True
            self._export_paper_e4_trace("complete")
            self.after(0, self._stop_paper_e4)
            return

        reference_pose, _reference_velocity = center_start_circle_reference(elapsed)
        base_r, base_l, metrics = self._pid_controller.step(
            self._latest_pose,
            self._latest_velocity,
            PathReference(
                pose=tuple(reference_pose),
                velocity=(
                    PAPER_E4_CIRCLE_SPEED_MPS,
                    PAPER_E4_CIRCLE_NU_RAD_S,
                ),
            ),
            dt,
            self._u_max,
            self._track_width_m,
            base_limit_frac=1.0,
        )
        max_step = MANUAL_VOLTAGE_SLEW_V_PER_S * dt
        u_r = float(np.clip(base_r, self._last_ur - max_step,
                            self._last_ur + max_step))
        u_l = float(np.clip(base_l, self._last_ul - max_step,
                            self._last_ul + max_step))
        self._e4_last_command = np.asarray(
            (metrics["v_command"], metrics["omega_command"]), dtype=float
        )
        self._e4_last_u = np.asarray((u_r, u_l), dtype=float)
        self._collection_ready = elapsed >= PID_COLLECTION_WARMUP_S
        self._send_manual_voltage(u_r, u_l)
        position_error = float(
            np.linalg.norm(self._latest_pose[:2] - reference_pose[:2])
        )
        radius = float(np.linalg.norm(self._latest_pose[:2]))
        if radius > 0.75 or (
            elapsed > 8.0 and position_error > 0.80
        ):
            self._send_manual_voltage(0.0, 0.0)
            self._export_paper_e4_trace("safety")
            self._collection_status_var.set(
                "论文 E4 安全边界触发 · 电机已停止 · 正在返回人工控制"
            )
            self.after(0, self._stop_paper_e4)
            return
        self._cmd_var.set(
            f"E4 {elapsed:4.1f}/{PAPER_E4_RUN_DURATION_S:.1f}s  "
            f"e=({metrics['e_x']:+.2f},{metrics['e_y']:+.2f},"
            f"{metrics['e_theta']:+.2f})  gyro={self._latest_gyro_z:+.2f}"
        )
        self._collection_status_var.set(
            "论文 E4 采样 PID 基线运行中 · PC 端 25 Hz · 记录对比轨迹"
        )

    def _record_paper_e4_row(self, message: dict) -> None:
        """Append one comparison-schema row from a fresh live-state frame."""
        try:
            timestamp_us = int(message.get("t_us", 0))
        except (TypeError, ValueError):
            return
        if timestamp_us <= 0 or timestamp_us <= self._e4_last_state_us:
            return
        if self._e4_t0_us is None:
            self._e4_t0_us = timestamp_us
        elapsed_us = timestamp_us - self._e4_t0_us
        t = max(0.0, elapsed_us * 1.0e-6)
        if t >= PAPER_E4_RUN_DURATION_S:
            # 与冻结 runner 一致：到 run_duration 为止，停止流程期间不再记录。
            return
        self._e4_last_state_us = timestamp_us
        try:
            current = np.asarray(message.get("cur"), dtype=float)
            applied_u = np.asarray(message.get("u"), dtype=float)
        except (TypeError, ValueError):
            current = self._latest_current.copy()
            applied_u = self._latest_applied_u.copy()
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            current = self._latest_current.copy()
        if applied_u.shape != (2,) or not np.all(np.isfinite(applied_u)):
            applied_u = self._latest_applied_u.copy()
        self._latest_current = current
        self._latest_applied_u = applied_u
        reference_pose, reference_velocity = center_start_circle_reference(t)
        pose_error = body_frame_error(reference_pose, self._latest_pose)
        self._e4_rows.append(
            {
                "t": t,
                "pose": self._latest_pose.copy(),
                "velocity": self._latest_velocity.copy(),
                "current": current,
                "reference": reference_pose,
                "motion_reference": reference_pose,
                "pose_error": pose_error,
                "alpha1": reference_velocity,
                "beta1": self._e4_last_command.copy(),
                "beta1_dot": np.zeros(2, dtype=float),
                "alpha2": np.zeros(2, dtype=float),
                "beta2": np.zeros(2, dtype=float),
                "beta2_dot": np.zeros(2, dtype=float),
                "z2": self._latest_velocity - self._e4_last_command,
                "z3": np.zeros(2, dtype=float),
                "uc": self._e4_last_u.copy(),
                "u": applied_u,
            }
        )
        trace_time = self._pid_elapsed(time.monotonic())
        if not self._pid_trace_times or trace_time > self._pid_trace_times[-1]:
            self._pid_pose_trace.append(self._latest_pose.copy())
            self._pid_trace_times.append(trace_time)
        self._plot_view = "collection"
        self._plot_dirty = True


    def _update_theory_start_state(self) -> None:
        if not hasattr(self, "_theory_start_button"):
            return
        case_name = self._comparison_case_var.get()
        case_spec = COMPARISON_CASES.get(
            case_name,
            COMPARISON_CASES[next(iter(COMPARISON_CASES))],
        )
        requires_weights = bool(case_spec["requires_weights"])
        comparison_supported = (
            int(case_spec["case_id"]) != 3 or self._comparison_modes_supported
        )
        fault_active = self._fw_fault not in ("", "none", "?")
        ready_mode = self._operation_mode in ("manual", "theory_complete") or (
            self._operation_mode == "idle" and fault_active
        )
        enabled = (
            self.link.connected
            and self._connecting_stage >= 4
            and self._motion_reference_supported
            and self._guidance_supported
            and comparison_supported
            and (
                not requires_weights
                or self._validated_weights_payload is not None
            )
            and self._workflow_kind is None
            and ready_mode
        )
        short_name = str(case_spec["short_name"])
        if self._operation_mode == "theory_complete":
            button_text = f"再次运行 {short_name}"
        elif fault_active:
            button_text = f"复位故障并启动 {short_name}"
        else:
            button_text = (
                f"上传权重并启动 {short_name}"
                if requires_weights else f"启动 {short_name} 级联 PID"
            )
        self._theory_start_button.configure(
            text=button_text,
            state=tk.NORMAL if enabled else tk.DISABLED,
            fg="#ffffff" if enabled else "#777777",
            bg="#0f6b4f" if enabled else "#2a2a4a",
        )

    def _on_comparison_case_changed(self, _event: tk.Event | None = None) -> None:
        case_name = self._comparison_case_var.get()
        spec = COMPARISON_CASES.get(case_name)
        if spec is None:
            return
        if int(spec["case_id"]) == 3 and not self._comparison_modes_supported:
            self._theory_status_var.set(
                "E3 需要支持板载级联 PID 的 1.5.0 或更高版本固件"
            )
        else:
            self._theory_status_var.set(
                f"{spec['short_name']} 已选择 · {spec['description']}"
            )
        self._update_theory_start_state()

    def _reset_workflow(self) -> None:
        self._workflow_kind = None
        self._workflow_steps.clear()
        self._workflow_current = None
        self._workflow_deadline = 0.0
        self._workflow_last_stop_retry = 0.0

    @staticmethod
    def _validate_parameter_value(key: str, value: float) -> str:
        """Return an error message if the parameter value is out of range."""
        if key in SYNTHESIS_DEFAULTS:
            return f"{key} 必须严格大于 0" if value <= 0.0 else ""
        if key in ("tau1", "tau2"):
            if not 0.005 <= value <= 1.0:
                return f"{key} 必须位于 0.005–1.0 s"
            return ""
        if key in GUIDANCE_DEFAULTS:
            if key == "outer_preview_horizon":
                return "预瞄时域必须位于 0–10 s" if not 0.0 <= value <= 10.0 else ""
            if key in ("outer_capture_radius", "outer_blend_radius"):
                return "距离参数必须位于 0–1 m（不含 0）" if not 0.0 < value <= 1.0 else ""
            upper = 0.50 if key == "outer_vbar" else 20.0
            if not 0.0 < value <= upper:
                return f"{key} 必须位于 0–{upper:g}（不含 0）"
            return ""
        if key == "motor_deadzone_v":
            if not 0.0 <= value < THEORY_BRINGUP_U_MAX_V:
                return f"电压死区必须满足 0 ≤ Ud < {THEORY_BRINGUP_U_MAX_V:g} V"
            return ""
        if key in ("r2", "r3"):
            return f"{key} 必须为不小于 0 的有限数值" if value < 0.0 else ""
        if key in PID_BENCHMARK_DEFAULTS:
            if key == "pid_current_ref_max":
                if not 0.05 <= value <= 1.0:
                    return "电流参考限幅必须位于 0.05–1.0 A"
                return ""
            if key == "pid_derivative_tau":
                if not 0.001 <= value <= 1.0:
                    return "微分滤波时间常数必须位于 0.001–1.0 s"
                return ""
            return f"{key} 必须不小于 0" if value < 0.0 else ""
        if key == "theory_max_position_error":
            if not 0.01 <= value <= 1.0:
                return "位置界必须位于 0.01–1.0 m"
            return ""
        if key == "theory_max_heading_error_deg":
            if not 0.05 <= value <= 179.9:
                return "航向界必须位于 0.05–179.9°"
            return ""
        if key in ("theory_max_z2", "theory_max_z3"):
            return f"{key} 必须严格大于 0" if value <= 0.0 else ""
        return ""

    def _open_synthesis_dialog(self) -> None:
        """Pop-up dialog for all tunable parameters, grouped by category."""
        dialog = tk.Toplevel(self)
        dialog.title("参数调节")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        bg = "#1a1a2e"
        fg = "#dddddd"
        tk.Label(
            dialog,
            text="参数调节 · 确定后写回，综合与 THEORY 启动时生效",
            font=("Segoe UI", 9, "bold"),
            fg="#64b5f6",
            bg=bg,
        ).grid(row=0, column=0, columnspan=4, padx=12, pady=(10, 4), sticky=tk.W)
        entry_vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value=self._synthesis_vars[key].get())
            for key, _label in ALL_PARAMETER_LABELS
        }
        grid_row = 1
        for section_title, labels in (
            ("权重综合参数", SYNTHESIS_PARAMETER_LABELS),
            ("THEORY 运行参数", RUNTIME_PARAMETER_LABELS),
            ("E3 级联 PID 参数", PID_PARAMETER_LABELS),
        ):
            tk.Label(
                dialog,
                text=section_title,
                font=("Segoe UI", 9, "bold"),
                fg="#e6a23c",
                bg=bg,
            ).grid(
                row=grid_row, column=0, columnspan=4,
                padx=12, pady=(6, 2), sticky=tk.W,
            )
            grid_row += 1
            for index, (key, label) in enumerate(labels):
                row, column = divmod(index, 2)
                tk.Label(
                    dialog,
                    text=label,
                    font=("Segoe UI", 9),
                    fg="#aaaaaa",
                    bg=bg,
                ).grid(
                    row=grid_row + row, column=column * 2,
                    padx=(12, 4), pady=3, sticky=tk.E,
                )
                tk.Entry(
                    dialog,
                    textvariable=entry_vars[key],
                    width=10,
                    font=("Consolas", 9),
                    bg="#2a2a4a",
                    fg=fg,
                    insertbackground=fg,
                    relief=tk.FLAT,
                ).grid(
                    row=grid_row + row, column=column * 2 + 1,
                    padx=(0, 12), pady=3, sticky=tk.W,
                )
            grid_row += (len(labels) + 1) // 2
        tk.Label(
            dialog,
            text="κ 需严格大于综合后显示的 κ_min（κ₂,min / κ₃,min）；综合参数 ε、d̄ 必须为正。",
            font=("Segoe UI", 8),
            fg="#888888",
            bg=bg,
        ).grid(
            row=grid_row, column=0, columnspan=4,
            padx=12, pady=(4, 2), sticky=tk.W,
        )
        grid_row += 1

        def apply_parameters() -> None:
            parsed: dict[str, float] = {}
            for key, variable in entry_vars.items():
                try:
                    value = float(variable.get())
                except ValueError:
                    messagebox.showerror(
                        "参数调节", f"{key} 必须是数字，当前值无效。",
                        parent=dialog,
                    )
                    return
                error = self._validate_parameter_value(key, value)
                if error:
                    messagebox.showerror("参数调节", f"{key}: {error}。", parent=dialog)
                    return
                parsed[key] = value
            if parsed["outer_blend_radius"] <= parsed["outer_capture_radius"]:
                messagebox.showerror(
                    "参数调节", "渐消边界必须满足 rb > rs。", parent=dialog
                )
                return
            for key, value in parsed.items():
                self._synthesis_vars[key].set(f"{value:g}")
            dialog.destroy()

        button_frame = tk.Frame(dialog, bg=bg)
        button_frame.grid(
            row=grid_row, column=0, columnspan=4, pady=(8, 12)
        )
        tk.Button(
            button_frame,
            text="确定",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#0f3460",
            activebackground="#1a5276",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=apply_parameters,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            button_frame,
            text="取消",
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#2a2a4a",
            activebackground="#3a3a5a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            command=dialog.destroy,
        ).pack(side=tk.LEFT)

    def _synthesize_weights(self) -> None:
        if len(self._dataset) < 10:
            messagebox.showwarning(
                "数据不足",
                "论文中的 10 行增广数据矩阵至少需要 10 个积分快照窗口。",
            )
            return
        try:
            synthesis_dataset = self._dataset
            parameters = {
                key: float(variable.get())
                for key, variable in self._synthesis_vars.items()
                if key in (
                    "kappa2_v", "kappa2_w", "kappa3_v", "kappa3_w",
                    "epsilon2", "epsilon3", "dbar2", "dbar3",
                )
            }
            for prefix in ("kappa2", "kappa3"):
                parameters[prefix] = (
                    parameters.pop(f"{prefix}_v"),
                    parameters.pop(f"{prefix}_w"),
                )
            # IMU 机制已取消：ω 全部来自编码器差速，不再校验陀螺角速度阈值。
            matrices = synthesis_dataset.matrices()
            threshold2 = route_ii_threshold(
                matrices["zdot2"], matrices["y2"],
                parameters["dbar2"], parameters["epsilon2"],
            )
            threshold3 = route_ii_threshold(
                matrices["zdot3"], matrices["y3"],
                parameters["dbar3"], parameters["epsilon3"],
            )
            def threshold_text(value: float) -> str:
                return f"{value:.6g}" if math.isfinite(value) else "不可行"
            self._kappa_min_var.set(
                "Route II 最小值："
                f"κ₂,min={threshold_text(threshold2.kappa_min)}  "
                f"κ₃,min={threshold_text(threshold3.kappa_min)}  |  "
                f"σ裕度=({threshold2.spectral_margin:.3e}, "
                f"{threshold3.spectral_margin:.3e})"
            )
            result = synthesize_dataset(synthesis_dataset, **parameters)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self._synthesis = None
            self._qualification = None
            self._disturbance_diagnostics_var.set(
                "扰动诊断：计算失败，请检查设计参数和采集数据"
            )
            suffix = " · 上次 PASS 仍可用" if self._validated_weights_payload else ""
            self._synthesis_status_var.set(f"FAIL · 计算失败{suffix}")
            self._synthesis_status_label.configure(
                fg="#ffb74d" if self._validated_weights_payload else "#e57373"
            )
            self._update_theory_start_state()
            messagebox.showerror("权重综合失败", str(exc))
            return

        qualification: QualificationReport | None = None
        if len(self._dataset) >= 20:
            try:
                qualification = qualify_synthesis(synthesis_dataset, result)
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                self._log(f"Optional engineering diagnostics unavailable: {exc}")

        self._synthesis = result
        self._qualification = qualification
        kappa2_text = (
            f"κ₂=({result.kappa2_v:.6g},{result.kappa2_w:.6g})"
            if result.kappa2_v is not None
            else f"κ₂={result.kappa2:.6g}"
        )
        kappa3_text = (
            f"κ₃=({result.kappa3_v:.6g},{result.kappa3_w:.6g})"
            if result.kappa3_v is not None
            else f"κ₃={result.kappa3:.6g}"
        )
        self._kappa_min_var.set(
            "Route II："
            f"{kappa2_text} > κ₂,min={result.block2.kappa_min:.6g}，"
            f"||W₂||F={result.block2.weight_norm:.3g}  |  "
            f"{kappa3_text} > κ₃,min={result.block3.kappa_min:.6g}，"
            f"||W₃||F={result.block3.weight_norm:.3g}"
        )
        if qualification is not None:
            q2 = qualification.block2
            q3 = qualification.block3
            self._disturbance_diagnostics_var.set(
                "附加诊断（不参与PASS）残差RMS/max: "
                f"G₂ {q2.residual_rms:.2e}/{q2.residual_max:.2e}, "
                f"G₃ {q3.residual_rms:.2e}/{q3.residual_max:.2e}  |  "
                "κ/κmin: "
                f"G₂ {q2.kappa:.3g}/{q2.kappa_min:.3g}, "
                f"G₃ {q3.kappa:.3g}/{q3.kappa_min:.3g}"
            )
        else:
            self._disturbance_diagnostics_var.set(
                "附加诊断未计算（至少需要 20 个窗口）；PASS 仅按论文条件判定"
            )
        accepted = result.valid
        verdict = "PASS" if accepted else "FAIL"
        previous_available = (
            not accepted and self._validated_weights_payload is not None
        )
        color = "#81c784" if accepted else "#ffb74d" if previous_available else "#e57373"
        retained = " · 上次 PASS 仍可用" if previous_available else ""
        self._synthesis_status_var.set(
            f"{verdict} · K={result.samples} · "
            f"G₂ {result.block2.rank}/10 cert={result.block2.xi_max:.2e}"
            f" · G₃ {result.block3.rank}/10 cert={result.block3.xi_max:.2e}"
            f"{retained}"
        )
        self._synthesis_status_label.configure(fg=color)
        self._log(
            f"Synthesis {verdict}: K={result.samples}, "
            f"method={PAPER_QUALIFICATION_METHOD}, "
            f"match=({result.block2.match_residual:.2e}, "
            f"{result.block3.match_residual:.2e}), "
            f"certificate=({result.block2.xi_max:.3e}, "
            f"{result.block3.xi_max:.3e}), "
            f"kappa/min=({result.kappa2:.4g}/{result.block2.kappa_min:.4g}, "
            f"{result.kappa3:.4g}/{result.block3.kappa_min:.4g}), "
            f"||W||F=({result.block2.weight_norm:.3e}, "
            f"{result.block3.weight_norm:.3e})"
        )
        if qualification is not None:
            self._log(
                "Engineering diagnostics only (not PASS gates): "
                f"Route-II sigma margin=({q2.spectral_margin:.3e}, "
                f"{q3.spectral_margin:.3e}), "
                f"held-out residual rms=({q2.residual_rms:.3e}, {q3.residual_rms:.3e}), "
                f"held-out residual max=({q2.residual_max:.3e}, {q3.residual_max:.3e}), "
                f"split drift=({q2.split_weight_drift:.3f}, "
                f"{q3.split_weight_drift:.3f}), "
                f"cross eig=({q2.cross_eigen_max:.3f}, "
                f"{q3.cross_eigen_max:.3f}), "
                f"direction=({q2.physical_direction_ok}/"
                f"{q2.controller_direction_ok}, "
                f"{q3.physical_direction_ok}/"
                f"{q3.controller_direction_ok})"
            )
        if accepted:
            self._remember_validated_weights(result, qualification)
            if self._dataset_path is not None:
                weights_path = self._dataset_path.with_suffix(".weights.json")
                try:
                    weights_path.write_text(
                        json.dumps(
                            {
                                "payload": result.upload_payload(),
                                "operating_config": {"gyro_deadband": 0.0},
                                "qualification": (
                                    qualification.as_dict()
                                    if qualification is not None
                                    else None
                                ),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    self._log(f"Validated weights saved → {weights_path.name}")
                except OSError as exc:
                    self._log(f"Could not save weights JSON: {exc}")
        self._update_theory_start_state()
        if not accepted:
            paper_reasons: list[str] = []
            for label, block in (
                ("G₂", result.block2),
                ("G₃", result.block3),
            ):
                if block.rank != block.rows:
                    paper_reasons.append(
                        f"{label}: rank(G)={block.rank}/{block.rows}，不满足满行秩"
                    )
                if block.match_residual > 1.0e-6:
                    paper_reasons.append(
                        f"{label}: 匹配方程残差={block.match_residual:.3e}"
                    )
                if block.spectral_margin <= 0.0:
                    paper_reasons.append(
                        f"{label}: σmin(N)-χ={block.spectral_margin:.3e}<=0"
                    )
                if block.kappa <= block.kappa_min:
                    paper_reasons.append(
                        f"{label}: κ={block.kappa:.6g} 必须严格大于 "
                        f"κmin={block.kappa_min:.6g}"
                    )
                if block.xi_max > 0.0:
                    paper_reasons.append(
                        f"{label}: Route II 证书={block.xi_max:.3e}>0"
                    )
            messagebox.showwarning(
                "论文条件未通过",
                "\n".join(paper_reasons)
                + "\n已禁止上传。PASS 仅依据论文 Route II 的满行秩、"
                "投影数据裕度、κ 下界、匹配方程和范数证书。",
            )

    def _start_paper_collection(self) -> None:
        """Data collection on the K460 campaign lemniscate: preset + 1.5× + T_w=0.10 s."""
        if self._pid_path_var.get() != PAPER_COLLECTION_PATH_NAME:
            self._pid_path_var.set(PAPER_COLLECTION_PATH_NAME)
        self._pid_speed_var.set(PAPER_COLLECTION_SPEED_SCALE)
        self._pid_window_var.set(PAPER_COLLECTION_WINDOW_S)
        self._start_pid_collection()

    def _start_paper_theory(self, case_key: str) -> None:
        """Start the paper tp13_k20_k30 campaign E1/E2/E3 with frozen parameters.

        与通用 THEORY 入口的差别：轨迹、外环、滤波、PID 与安全参数全部取自
        PAPER_CAMPAIGN（冻结战役实值，不读参数弹窗）；E3 不使用固件 case 3
        （那是旧级联 PID），而是上传清零非线性补偿列的 case 1 权重 payload。
        """
        if not self.link.connected or self._connecting_stage < 4:
            messagebox.showwarning("尚未连接", "请先连接小车。")
            return
        if (
            not self._motor_deadzone_supported
            or not self._gyro_deadband_supported
            or not self._theory_safety_supported
            or not self._motion_reference_supported
            or not self._guidance_supported
        ):
            messagebox.showerror(
                "固件不兼容",
                "当前固件不支持论文预瞄极坐标捕获外环或完整 THEORY 安全配置，"
                "请先烧录 1.7.0 或更高版本固件。",
            )
            return
        if self._workflow_kind is not None:
            return
        recovery_fault = (
            self._fw_fault if self._fw_fault not in ("", "none", "?") else None
        )
        ready_mode = self._operation_mode in ("manual", "theory_complete") or (
            self._operation_mode == "idle" and recovery_fault is not None
        )
        if not ready_mode:
            return
        if self._validated_weights_payload is None:
            messagebox.showerror(
                "缺少 PASS 权重",
                "论文 E1–E3 需要 tp13_k20_k30 的 PASS 权重；"
                "请先加载数据并完成权重综合。",
            )
            return
        label = PAPER_CASE_LABELS[case_key]
        if case_key == "E2":
            case_id, r2, r3 = 2, 0.0, 0.0
        else:
            case_id, r2, r3 = 1, PAPER_CAMPAIGN["r2"], PAPER_CAMPAIGN["r3"]
        if case_key == "E3":
            weights_for_run = without_initial_nonlinear_compensation(
                self._validated_weights_payload
            )
        else:
            weights_for_run = dict(self._validated_weights_payload)
        config = theory_config_for_preset(
            PAPER_CAMPAIGN["preset"],
            tau1=PAPER_CAMPAIGN["tau1"],
            tau2=PAPER_CAMPAIGN["tau2"],
            outer_kp=PAPER_CAMPAIGN["outer_kp"],
            outer_vbar=PAPER_CAMPAIGN["outer_vbar"],
            outer_ktheta=PAPER_CAMPAIGN["outer_ktheta"],
            outer_preview_horizon=PAPER_CAMPAIGN["outer_preview_horizon"],
            outer_capture_radius=PAPER_CAMPAIGN["outer_capture_radius"],
            outer_blend_radius=PAPER_CAMPAIGN["outer_blend_radius"],
            motor_deadzone_v=PAPER_CAMPAIGN["motor_deadzone_v"],
            r2=r2,
            r3=r3,
            comparison_case=case_id,
            theory_max_position_error=PAPER_CAMPAIGN["theory_max_position_error"],
            theory_max_heading_error=PAPER_CAMPAIGN["theory_max_heading_error"],
            theory_max_z2=PAPER_CAMPAIGN["theory_max_z2"],
            theory_max_z3=PAPER_CAMPAIGN["theory_max_z3"],
            **PID_BENCHMARK_DEFAULTS,
        )
        # 冻结战役实值：目标偏差自动停车阈值为 0.6 m（GUI 默认 0.20 m）。
        config["motion_max_target_error"] = PAPER_CAMPAIGN[
            "motion_max_target_error"
        ]
        ablation_notice = (
            (
                "E3 消融：上传的权重已把 w₂/w₃ 的初始非线性补偿列清零"
                "（σ 列保留、自适应照常 r₂=r₃=0.5），固件仍运行 case 1 完整控制器；"
                "导出记录会标记 initial_nonlinear_compensation_enabled=False。\n"
            )
            if case_key == "E3"
            else ""
        )
        recovery_notice = (
            f"当前锁存故障：{recovery_fault}\n"
            "启动流程将首先停车并清除故障；若故障条件仍存在，固件会再次停车。\n\n"
            if recovery_fault is not None
            else ""
        )
        confirmation_text = (
            recovery_notice
            + f"论文实验：{label}\n"
            + f"预设轨迹：{PAPER_CAMPAIGN['preset']}（ref_shape=3，Tp=1.3 s）\n"
            + f"运行时间：{config['run_duration']:.3f} s\n\n"
            + "预瞄极坐标捕获外环："
            + f"kρ={config['outer_kp']:g}，v̄c={config['outer_vbar']:g} m/s，"
            + f"kα={config['outer_ktheta']:g}，rs={config['outer_capture_radius']:g} m，"
            + f"rb={config['outer_blend_radius']:g} m\n"
            + f"DSC 滤波：τ₁={config['tau1']:g} s，τ₂={config['tau2']:g} s\n"
            + f"自适应率：r₂={config['r2']:g}，r₃={config['r3']:g}\n"
            + f"κ₂={float(weights_for_run['kappa2']):.6g}，"
            + f"κ₃={float(weights_for_run['kappa3']):.6g}\n"
            + f"最终电压死区：±{config['motor_deadzone_v']:g} V\n"
            + f"自动停车：位置>{config['theory_max_position_error']:.2f} m，"
            + f"航向>{math.degrees(config['theory_max_heading_error']):.0f}°，"
            + f"||z₂||>{config['theory_max_z2']:g}，"
            + f"||z₃||>{config['theory_max_z3']:g}\n"
            + ablation_notice
            + "随后将依次停车、下发配置、上传权重、清零位姿并启动。\n"
            + "请将小车静止放在空旷地面，并确保可立即按 Esc 急停。"
        )
        if not messagebox.askokcancel(
            f"启动论文实验 {case_key}", confirmation_text
        ):
            return

        self._motor_deadzone_v = float(PAPER_CAMPAIGN["motor_deadzone_v"])
        self._active_theory_safety = {
            key: float(config[key])
            for key in (
                "theory_max_position_error",
                "theory_max_heading_error",
                "theory_max_z2",
                "theory_max_z3",
                "theory_safety_grace",
                "motion_max_target_error",
            )
        }
        self._stop_control_loop()
        self._pressed.clear()
        for direction in ("up", "down", "left", "right"):
            self._set_dpad_active(direction, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._volt_var.set("u_R =  0.00 V    u_L =  0.00 V")
        self._paper_adaptive_only = case_key == "E3"
        self._operation_mode = "transition"
        self._active_comparison_name = label
        self._active_comparison_case = case_key
        self._active_theory_config = {
            key: float(value) for key, value in config.items()
        }
        self._active_theory_weights_json = json.dumps(
            weights_for_run, ensure_ascii=False, sort_keys=True
        )
        self._pending_theory_duration = float(config["run_duration"]) + float(
            config.get("hold_duration", 0.0)
        )
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence = None
        self._plot_view = "theory"
        self._theory_plot_mode = "trajectory"
        self._theory_rows.clear()
        self._theory_t0_us = None
        self._theory_exported_path = None
        self._theory_safety_count = 0
        self._theory_safety_reason = None
        self._plot_dirty = True
        self._workflow_kind = "start_theory"
        self._workflow_steps = deque(
            (
                {"name": "停车", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": f"下发论文 {case_key} 配置",
                 "payload": {"cmd": "configure", **config},
                 "ack": "configuration_updated"},
                {"name": "上传权重",
                 "payload": {"cmd": "set_weights", **weights_for_run},
                 "ack": "weights_accepted"},
                {"name": "位姿清零", "payload": {"cmd": "zero_pose"},
                 "ack": "zero_pose_requested"},
                {"name": f"启动论文 {case_key}",
                 "payload": {"cmd": "start", "mode": "theory"},
                 "ack": "start_requested",
                 "requires_sensor_ready": True},
            )
        )
        self._theory_status_var.set(f"正在准备论文实验 {case_key}…")
        self._update_theory_start_state()
        self._send_next_workflow_step()

    def _start_theory_validation(self) -> None:
        if not self.link.connected or self._connecting_stage < 4:
            messagebox.showwarning("尚未连接", "请先连接小车。")
            return
        if (
            not self._motor_deadzone_supported
            or not self._gyro_deadband_supported
            or not self._theory_safety_supported
            or not self._motion_reference_supported
            or not self._guidance_supported
        ):
            messagebox.showerror(
                "固件不兼容",
                "当前固件不支持论文预瞄极坐标捕获外环或完整 THEORY 安全配置，"
                "请先烧录 1.7.0 或更高版本固件。",
            )
            return
        if self._workflow_kind is not None:
            return
        recovery_fault = (
            self._fw_fault if self._fw_fault not in ("", "none", "?") else None
        )
        ready_mode = self._operation_mode in ("manual", "theory_complete") or (
            self._operation_mode == "idle" and recovery_fault is not None
        )
        if not ready_mode:
            return
        preset_name = self._trajectory_var.get()
        case_name = self._comparison_case_var.get()
        try:
            tau1 = float(self._synthesis_vars["tau1"].get())
            tau2 = float(self._synthesis_vars["tau2"].get())
            motor_deadzone_v = self._requested_motor_deadzone()
            case = comparison_case_runtime(
                case_name,
                float(self._synthesis_vars["r2"].get()),
                float(self._synthesis_vars["r3"].get()),
            )
            if int(case["case_id"]) == 3 and not self._comparison_modes_supported:
                raise ValueError("E3 级联 PID 需要 1.5.0 或更高版本固件")
            if bool(case["requires_weights"]) and self._validated_weights_payload is None:
                raise ValueError(
                    "E1/E2 需要 PASS 权重；请先加载或采集数据并完成权重综合"
                )
            pid_parameters = {
                key: float(self._synthesis_vars[key].get())
                for key in PID_BENCHMARK_DEFAULTS
            }
            safety_limits = self._requested_theory_safety_limits()
            # IMU 机制已取消：ω 全部来自编码器差速，不再校验陀螺阈值。
            config = theory_config_for_preset(
                preset_name,
                tau1=tau1,
                tau2=tau2,
                outer_kp=float(self._synthesis_vars["outer_kp"].get()),
                outer_vbar=float(self._synthesis_vars["outer_vbar"].get()),
                outer_ktheta=float(self._synthesis_vars["outer_ktheta"].get()),
                outer_preview_horizon=float(
                    self._synthesis_vars["outer_preview_horizon"].get()
                ),
                outer_capture_radius=float(
                    self._synthesis_vars["outer_capture_radius"].get()
                ),
                outer_blend_radius=float(
                    self._synthesis_vars["outer_blend_radius"].get()
                ),
                motor_deadzone_v=motor_deadzone_v,
                r2=float(case["r2"]),
                r3=float(case["r3"]),
                comparison_case=int(case["case_id"]),
                **pid_parameters,
                **safety_limits,
            )
        except ValueError as exc:
            messagebox.showerror("实验配置错误", str(exc))
            return
        requires_weights = bool(case["requires_weights"])
        short_name = str(case["short_name"])
        if int(case["case_id"]) == 3:
            controller_details = (
                "内环：速度 PID → 电流参考 → 电流 PID → 电压\n"
                f"速度 PID：FF={config['pid_velocity_ff']:g}，"
                f"Kp={config['pid_velocity_kp']:g}，"
                f"Ki={config['pid_velocity_ki']:g}，"
                f"Kd={config['pid_velocity_kd']:g}\n"
                f"电流 PID：Kp={config['pid_current_kp']:g}，"
                f"Ki={config['pid_current_ki']:g}，"
                f"Kd={config['pid_current_kd']:g}，"
                f"Iref,max={config['pid_current_ref_max']:g} A\n"
            )
        else:
            assert self._validated_weights_payload is not None
            controller_details = (
                f"自适应率：r₂={config['r2']:g}，r₃={config['r3']:g}\n"
                f"Route II 权重：κ₂="
                f"{float(self._validated_weights_payload['kappa2']):.6g}，"
                f"κ₃={float(self._validated_weights_payload['kappa3']):.6g}\n"
            )
        recovery_notice = (
            f"当前锁存故障：{recovery_fault}\n"
            "启动流程将首先停车并清除故障；若故障条件仍存在，固件会再次停车。\n\n"
            if recovery_fault is not None
            else ""
        )
        weight_notice = (
            "将使用最近一次通过全部校验的 PASS 权重。\n"
            if requires_weights
            else "E3 不读取数据驱动权重。\n"
        )
        workflow_notice = (
            "随后将依次停车、下发配置、上传权重、清零位姿并启动。"
            if requires_weights
            else "随后将依次停车、下发配置、清零位姿并启动。"
        )
        confirmation_text = (
            recovery_notice
            + f"实验方案：{case_name}\n"
            + f"方案定义：{case['description']}\n"
            + f"预设轨迹：{preset_name}\n"
            + f"运行时间：{config['run_duration']:.0f} s"
            + (
                f" + 保持段 {config['hold_duration']:g} s\n"
                if config.get("hold_duration", 0.0) > 0.0
                else "\n"
            )
            + "\n"
            + "预瞄极坐标捕获外环："
            + f"kρ={config['outer_kp']:g}，v̄c={config['outer_vbar']:g} m/s，"
            + f"kα={config['outer_ktheta']:g}\n"
            + f"Tp={config['outer_preview_horizon']:g} s，"
            + f"rs={config['outer_capture_radius']:g} m，"
            + f"rb={config['outer_blend_radius']:g} m\n"
            + f"工程安全限幅：|v_c|≤{config['outer_max_v']:g} m/s，"
            + f"|ω_c|≤{config['outer_max_omega']:g} rad/s；"
            + f"目标偏差>{config['motion_max_target_error']:g} m 自动停车\n"
            + f"DSC 滤波：τ₁={tau1:g} s，τ₂={tau2:g} s\n"
            + "ω 来源：编码器差速（IMU 机制已取消）\n"
            + controller_details
            + f"最终电压死区：±{motor_deadzone_v:g} V\n"
            + f"架空/低速放行电压：Umax={config['u_max']:.1f} V\n"
            + f"自动停车：位置>{config['theory_max_position_error']:.2f} m，"
            + f"航向>{math.degrees(config['theory_max_heading_error']):.0f}°，"
            + f"||z₂||>{config['theory_max_z2']:g}，"
            + f"||z₃||>{config['theory_max_z3']:g}\n"
            + weight_notice
            + "请将小车静止放在空旷地面，并确保可立即按 Esc 急停。\n"
            + workflow_notice
        )
        confirmed = messagebox.askokcancel(
            (
                f"复位并启动 {short_name}"
                if recovery_fault is not None
                else f"启动 {short_name} 对比实验"
            ),
            confirmation_text,
        )
        if not confirmed:
            return
        # 通用 THEORY 入口不走论文 E3 消融 payload，恢复正常导出标记。
        self._paper_adaptive_only = False

        self._motor_deadzone_v = motor_deadzone_v
        self._active_theory_safety = {
            key: float(config[key])
            for key in (
                "theory_max_position_error",
                "theory_max_heading_error",
                "theory_max_z2",
                "theory_max_z3",
                "theory_safety_grace",
                "motion_max_target_error",
            )
        }

        self._stop_control_loop()
        self._pressed.clear()
        for direction in ("up", "down", "left", "right"):
            self._set_dpad_active(direction, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._volt_var.set("u_R =  0.00 V    u_L =  0.00 V")
        self._operation_mode = "transition"
        self._active_comparison_name = case_name
        self._active_comparison_case = short_name
        self._active_theory_config = {
            key: float(value) for key, value in config.items()
        }
        self._active_theory_weights_json = json.dumps(
            self._validated_weights_payload if requires_weights else {},
            ensure_ascii=False,
            sort_keys=True,
        )
        self._pending_theory_duration = float(config["run_duration"]) + float(
            config.get("hold_duration", 0.0)
        )
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence = None
        self._plot_view = "theory"
        self._theory_plot_mode = "trajectory"
        self._theory_rows.clear()
        self._theory_t0_us = None
        self._theory_exported_path = None
        self._theory_safety_count = 0
        self._theory_safety_reason = None
        self._plot_dirty = True
        self._workflow_kind = "start_theory"
        workflow_steps = [
            {"name": "停车", "payload": {"cmd": "stop"},
             "ack": "stop_requested"},
            {"name": f"下发 {short_name} 配置",
             "payload": {"cmd": "configure", **config},
             "ack": "configuration_updated"},
        ]
        if requires_weights:
            assert self._validated_weights_payload is not None
            workflow_steps.append(
                {"name": "上传权重",
                 "payload": {
                     "cmd": "set_weights",
                     **self._validated_weights_payload,
                 },
                 "ack": "weights_accepted"}
            )
        workflow_steps.extend(
            (
                {"name": "位姿清零", "payload": {"cmd": "zero_pose"},
                 "ack": "zero_pose_requested"},
                {"name": f"启动 {short_name}",
                 "payload": {"cmd": "start", "mode": "theory"},
                 "ack": "start_requested",
                 "requires_sensor_ready": True},
            )
        )
        self._workflow_steps = deque(workflow_steps)
        self._theory_status_var.set(f"正在准备 {short_name}…")
        self._update_theory_start_state()
        self._send_next_workflow_step()

    def _return_to_manual(self) -> None:
        if not self.link.connected or self._connecting_stage < 4:
            return
        if self._operation_mode == "pid_collect":
            self._stop_pid_collection()
            return
        if self._operation_mode == "paper_e4":
            self._send_manual_voltage(0.0, 0.0)
            self._stop_paper_e4()
            return
        self._stop_control_loop()
        self._pressed.clear()
        for direction in ("up", "down", "left", "right"):
            self._set_dpad_active(direction, False)
        self._reset_workflow()
        self._operation_mode = "transition"
        self._workflow_kind = "return_manual"
        self._workflow_steps = deque(
            (
                {"name": "停止 THEORY", "payload": {"cmd": "stop"},
                 "ack": "stop_requested"},
                {"name": "恢复人工通信配置",
                 "payload": {"cmd": "configure", "manual_snapshots": False},
                 "ack": "configuration_updated"},
                {"name": "返回人工控制",
                 "payload": {"cmd": "start", "mode": "manual"},
                 "ack": "start_requested"},
            )
        )
        self._theory_status_var.set("正在停止并返回人工控制…")
        self._send_next_workflow_step()

    def _send_next_workflow_step(self) -> None:
        if self._workflow_kind is None:
            return
        if self._workflow_current is not None:
            return
        if not self._workflow_steps:
            self._finish_workflow()
            return
        step = self._workflow_steps.popleft()
        if step.get("requires_sensor_ready"):
            now = time.monotonic()
            sensor_fresh = (
                self._latest_sensor_time > 0.0
                and now - self._latest_sensor_time <= PID_TELEMETRY_TIMEOUT_S
            )
            sensor_ready = (
                sensor_fresh
                and self._latest_ina_ok
            )
            if not sensor_ready:
                wait_started = float(step.setdefault("_sensor_wait_started", now))
                if now - wait_started < WORKFLOW_SENSOR_READY_TIMEOUT_S:
                    self._workflow_steps.appendleft(step)
                    waiting_text = "等待 INA3221 遥测就绪…"
                    if self._workflow_kind == "start_pid_collect":
                        self._collection_status_var.set(waiting_text)
                    else:
                        self._theory_status_var.set(waiting_text)
                    self.after(
                        WORKFLOW_SENSOR_READY_POLL_MS,
                        self._send_next_workflow_step,
                    )
                    return
                if not sensor_fresh:
                    reason = "传感器状态已过期，启动前未收到新鲜遥测。"
                else:
                    reason = "INA3221 未就绪，已禁止启动电机。"
                self._abort_workflow(reason)
                return
        sequence = self.link.send(dict(step["payload"]))
        self._workflow_current = {**step, "seq": sequence}
        self._workflow_deadline = time.monotonic() + WORKFLOW_TIMEOUT_S
        if step["payload"].get("cmd") == "stop":
            self._workflow_last_stop_retry = time.monotonic()
        if self._workflow_kind in ("start_pid_collect", "stop_pid_collect"):
            self._collection_status_var.set(f"正在执行：{step['name']}…")
        elif self._workflow_kind == "apply_motor_deadzone":
            self._status_var.set(f"正在执行：{step['name']}…")
        else:
            self._theory_status_var.set(f"正在执行：{step['name']}…")
        self._log(f"Workflow → {step['name']} (seq={sequence})")

    def _handle_workflow_message(self, message: dict) -> bool:
        current = self._workflow_current
        if current is None or message.get("seq") != current["seq"]:
            return False
        if message.get("type") == "error":
            self._abort_workflow(
                f"{current['name']}被固件拒绝：{message.get('message', '?')}"
            )
            return True
        if message.get("type") != "ack":
            return False
        if message.get("message") != current["ack"]:
            self._abort_workflow(
                f"{current['name']}返回了意外确认：{message.get('message', '?')}"
            )
            return True
        self._log(f"Workflow OK ← {current['name']}")
        self._workflow_current = None
        self._workflow_deadline = 0.0
        settle_ms = int(current.get("settle_ms", WORKFLOW_SETTLE_MS))
        self.after(settle_ms, self._send_next_workflow_step)
        return True

    def _finish_workflow(self) -> None:
        kind = self._workflow_kind
        self._reset_workflow()
        if kind == "start_pid_collect":
            self._reported_fault = "none"
            self._operation_mode = "pid_collect"
            self._pid_started_at = time.monotonic()
            self._pid_pause_started_at = None
            self._pid_paused_s = 0.0
            self._last_control_time = self._pid_started_at
            self._collection_ready = False
            self._pid_controller.reset()
            self._pid_last_metrics = {}
            self._status_var.set(
                f"PID 路径采集运行中 · {self._pid_reference_name} · Esc 急停"
            )
            self._collection_status_var.set("等待实时位姿 · 电机保持停止")
            self._start_control_loop()
            self._log(f"PID collection started: {self._pid_reference_name}")
        elif kind == "start_paper_e4":
            self._reported_fault = "none"
            self._operation_mode = "paper_e4"
            self._pid_started_at = time.monotonic()
            self._pid_pause_started_at = None
            self._pid_paused_s = 0.0
            self._last_control_time = self._pid_started_at
            self._collection_ready = False
            self._pid_controller.reset()
            self._status_var.set(
                "论文 E4 采样 PID 基线运行中 · PC 端 25 Hz · Esc 急停"
            )
            self._collection_status_var.set("等待实时位姿 · 电机保持停止")
            self._start_control_loop()
            self._log("Paper E4 sampling-PID baseline started")
        elif kind == "start_theory":
            self._reported_fault = "none"
            self._fw_fault = "none"
            self._operation_mode = "theory"
            self._theory_stop_deadline = (
                time.monotonic()
                + self._pending_theory_duration
                + THEORY_STOP_GRACE_S
            )
            self._theory_last_stop_send = 0.0
            self._theory_stop_sequence = None
            self._theory_status_var.set(
                f"{self._active_comparison_case} 运行中 · Esc 可急停"
            )
            self._status_var.set(
                f"Connected · {self.link.status_text} · "
                f"{self._active_comparison_case} · "
                f"{self._trajectory_var.get()}"
            )
            self._log(
                f"{self._active_comparison_case} comparison run started: "
                f"{self._active_comparison_name}"
            )
        elif kind == "apply_motor_deadzone":
            self._reported_fault = "none"
            self._fw_fault = "none"
            self._operation_mode = "manual"
            self._latest_sensor_time = 0.0
            self._status_var.set(
                f"Connected · {self.link.status_text} · FW={self._fw_version} "
                f"· 死区=±{self._motor_deadzone_v:g}V · MANUAL"
            )
            self._start_control_loop()
            self.focus_set()
            self._log(
                f"Motor dead-zone applied from EXE: "
                f"±{self._motor_deadzone_v:g} V"
            )
        elif kind in ("return_manual", "stop_pid_collect", "stop_paper_e4"):
            self._reported_fault = "none"
            self._fw_fault = "none"
            self._operation_mode = "manual"
            self._paper_adaptive_only = False
            self._theory_stop_deadline = 0.0
            self._theory_last_stop_send = 0.0
            self._theory_stop_sequence = None
            self._collection_ready = False
            if kind == "stop_pid_collect":
                self._active_collection_segment = None
            self._latest_sensor_time = 0.0
            # Keep the most recent completed run inspectable until the user
            # explicitly selects or starts another data view.
            if kind in ("stop_pid_collect", "stop_paper_e4") and (
                self._plot_rows or self._pid_pose_trace
            ):
                self._plot_view = "collection"
            elif kind == "return_manual" and self._theory_rows:
                self._plot_view = "theory"
            else:
                self._plot_view = "synthesis"
            self._plot_dirty = True
            if kind == "stop_pid_collect":
                self._flush_snapshot_cache(force=True)
                self._collection_status_var.set("PID采集已停止 · 人工控制不记录数据")
            elif kind == "stop_paper_e4":
                self._collection_status_var.set(
                    "论文 E4 已停止 · 轨迹 NPZ 已导出 · 人工控制已恢复"
                )
            else:
                self._theory_status_var.set("THEORY 已停止 · 人工控制已恢复")
            self._status_var.set(
                f"Connected · {self.link.status_text} · FW={self._fw_version} "
                f"· uMax={self._u_max:.1f}V · MANUAL"
            )
            self._start_control_loop()
            self.focus_set()
            self._log("Mode → MANUAL")
        self._update_pid_start_state()
        self._update_theory_start_state()

    def _abort_workflow(self, reason: str) -> None:
        try:
            self.link.send({"cmd": "stop"})
        except Exception:
            pass
        self._discard_active_collection_segment(reason)
        self._reset_workflow()
        self._operation_mode = "idle"
        self._plot_view = "synthesis"
        self._plot_dirty = True
        self._theory_status_var.set(f"控制流程未启动 · {reason}")
        self._collection_status_var.set(f"PID采集未启动 · {reason}")
        self._status_var.set(f"⚠ {reason}")
        self._log(f"Workflow aborted: {reason}")
        self._update_pid_start_state()
        self._update_theory_start_state()
        messagebox.showerror("控制流程中止", reason)

    # ==================================================================
    # Weight-synthesis snapshot plots
    # ==================================================================

    def _accept_live_state(self, message: dict) -> None:
        """Cache the newest pose and velocity for the PC-side PID collector."""
        if message.get("type") == "telemetry":
            sensors = message.get("s", {})
            pose = sensors.get("pose") if isinstance(sensors, dict) else None
            velocity = sensors.get("velocity") if isinstance(sensors, dict) else None
        elif message.get("type") == "state":
            sensors = message
            pose = message.get("pose")
            velocity = message.get("vel")
        else:
            return
        try:
            pose_array = np.asarray(pose, dtype=float)
            velocity_array = np.asarray(velocity, dtype=float)
        except (TypeError, ValueError):
            return
        if (
            pose_array.shape != (3,)
            or velocity_array.shape != (2,)
            or not np.all(np.isfinite(pose_array))
            or not np.all(np.isfinite(velocity_array))
        ):
            return
        self._latest_pose = pose_array
        self._latest_velocity = velocity_array
        if isinstance(sensors, dict):
            self._latest_imu_ok = bool(sensors.get("imu_ok", False))
            self._latest_imu_calibrated = bool(
                sensors.get("imu_calibrated", False)
            )
            self._latest_ina_ok = bool(sensors.get("ina_ok", False))
            for key, attribute in (
                ("gyro_z", "_latest_gyro_z"),
                ("wheel_r", "_latest_wheel_right"),
                ("wheel_l", "_latest_wheel_left"),
            ):
                try:
                    value = float(sensors.get(key, getattr(self, attribute)))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    setattr(self, attribute, value)
        now = time.monotonic()
        try:
            received_at = float(message.get("_rx_monotonic", now))
        except (TypeError, ValueError):
            received_at = now
        if not math.isfinite(received_at) or received_at <= 0.0:
            received_at = now
        self._latest_sensor_time = min(now, received_at)
        if isinstance(sensors, dict):
            try:
                current_array = np.asarray(sensors.get("cur"), dtype=float)
                applied_array = np.asarray(sensors.get("u"), dtype=float)
            except (TypeError, ValueError):
                current_array = None
                applied_array = None
            if (
                current_array is not None
                and current_array.shape == (2,)
                and np.all(np.isfinite(current_array))
            ):
                self._latest_current = current_array
            if (
                applied_array is not None
                and applied_array.shape == (2,)
                and np.all(np.isfinite(applied_array))
            ):
                self._latest_applied_u = applied_array
            try:
                self._latest_state_us = int(message.get("t_us", 0))
            except (TypeError, ValueError):
                pass
        self._update_sensor_display()
        self._update_pid_start_state()
        if self._operation_mode == "paper_e4":
            self._record_paper_e4_row(message)
        if self._operation_mode == "pid_collect":
            trace_time = self._pid_elapsed(now)
            if not self._pid_trace_times or trace_time > self._pid_trace_times[-1]:
                self._pid_pose_trace.append(pose_array.copy())
                self._pid_trace_times.append(trace_time)
            self._plot_view = "collection"
            self._plot_dirty = True

    def _update_sensor_display(self) -> None:
        """Show the exact sensor sources used by the PID pose estimator."""
        ina_text = "OK" if self._latest_ina_ok else "FAIL"
        self._sensor_var.set(
            f"INA:{ina_text}  ω=编码器差速（IMU 已取消）  "
            f"gyro={self._latest_gyro_z:+.3f}  "
            f"vR={self._latest_wheel_right:+.3f}  "
            f"vL={self._latest_wheel_left:+.3f} m/s"
        )
        self._sensor_label.configure(
            fg="#81c784" if self._latest_ina_ok else "#ef5350"
        )

    def _accept_snapshot(self, message: dict) -> None:
        snapshot = message.get("snapshot") if message.get("type") == "telemetry" else message
        if not isinstance(snapshot, dict):
            return
        if not should_collect_snapshot(
            self._operation_mode,
            self._collection_ready,
        ):
            return
        if snapshot.get("kind") != "integral_v3_sgn_gyro_gate":
            self._log(
                "Ignored incompatible snapshot; integral_v3_sgn_gyro_gate is required"
            )
            return
        try:
            row = {
                key: np.asarray(snapshot.get(key), dtype=float)
                for key in SNAPSHOT_FIELDS
            }
            for key, expected_length in SNAPSHOT_FIELDS.items():
                value = row[key]
                if value.shape != (expected_length,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{key} shape/value invalid")
        except (TypeError, ValueError) as exc:
            self._log(f"Ignored invalid snapshot: {exc}")
            return
        for key in ("velocity_raw", "current_raw"):
            try:
                value = np.asarray(snapshot.get(key), dtype=float)
            except (TypeError, ValueError):
                continue
            if value.shape == (2,) and np.all(np.isfinite(value)):
                row[key] = value

        timestamp_us = int(message.get("t_us", 0) or 0)
        if timestamp_us <= 0 or self._active_collection_segment is None:
            self._log("Ignored integral snapshot without timestamp/active segment")
            return
        try:
            self._dataset.append_integral(
                row,
                t_us=timestamp_us,
                window_s=float(snapshot.get("window_s", 0.0)),
                segment=self._active_collection_segment,
            )
        except ValueError as exc:
            self._log(f"Ignored invalid integral snapshot: {exc}")
            return
        if timestamp_us > 0:
            if self._snapshot_t0_us is None or timestamp_us < self._snapshot_t0_us:
                self._snapshot_t0_us = timestamp_us
            sample_time = (timestamp_us - self._snapshot_t0_us) * 1.0e-6
            self._snapshot_fallback_t = sample_time
        else:
            self._snapshot_fallback_t += 0.02
            sample_time = self._snapshot_fallback_t

        self._plot_rows.append(row)
        self._plot_times.append(sample_time)
        self._schedule_snapshot_cache()
        self._plot_dirty = True
        duration = self._plot_times[-1] - self._plot_times[0] if len(self._plot_times) > 1 else 0.0
        self._sample_var.set(
            f"积分快照：{len(self._dataset)} · {self._dataset.segment_count} 段"
            f" · 图窗 {duration:.1f} s"
        )
        count = len(self._dataset)
        if count <= 10 or count % QUALITY_UPDATE_SAMPLES == 0:
            self._update_quality_display()

    def _update_quality_display(self) -> None:
        count = len(self._dataset)
        if count == 0:
            self._quality_var.set("数据质量：G₂ 0/10  ·  G₃ 0/10")
            self._quality_label.configure(fg="#888888")
            return
        try:
            matrices = self._dataset.matrices()
            diagnostics = self._dataset.diagnostics()
            rank2, sigma2, condition2 = stacked_matrix_quality(
                matrices["zdot2"], matrices["y2"]
            )
            rank3, sigma3, condition3 = stacked_matrix_quality(
                matrices["zdot3"], matrices["y3"]
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._quality_var.set(f"数据质量计算失败：{exc}")
            self._quality_label.configure(fg="#e57373")
            return

        def condition_text(value: float) -> str:
            return f"{value:.1e}" if np.isfinite(value) else "inf"

        self._quality_var.set(
            f"G₂ {rank2}/10  σmin={sigma2:.2e}  κ={condition_text(condition2)}"
            f"   |   G₃ {rank3}/10  σmin={sigma3:.2e}  κ={condition_text(condition3)}"
            f"   |   段={diagnostics.segments} T窗={diagnostics.median_dt_s:.3f}s"
        )
        full_rank = rank2 == 10 and rank3 == 10
        well_conditioned = (
            full_rank
            and condition2 < 1.0e6
            and condition3 < 1.0e6
            and diagnostics.max_dt_s <= 0.25
        )
        color = "#81c784" if well_conditioned else "#ffb74d" if full_rank else "#e57373"
        self._quality_label.configure(fg=color)

    def _clear_dataset(self) -> None:
        self._dataset.clear()
        self._delete_snapshot_cache()
        self._dataset_path = None
        self._synthesis = None
        self._qualification = None
        self._active_collection_segment = None
        self._plot_rows.clear()
        self._plot_times.clear()
        self._pid_pose_trace.clear()
        self._pid_trace_times.clear()
        self._plot_view = "synthesis"
        self._collection_plot_mode = "trajectory"
        self._snapshot_t0_us = None
        self._snapshot_fallback_t = 0.0
        self._plot_dirty = True
        self._sample_var.set("积分快照：0")
        self._disturbance_diagnostics_var.set(
            "附加诊断（不参与 PASS）：综合后显示留出残差与分段指标"
        )
        self._kappa_min_var.set(
            "Route II 最小值：κ₂,min=--  κ₃,min=--（选定 κ 必须严格更大）"
        )
        if self._validated_weights_payload is not None:
            samples = int((self._validated_weights_info or {}).get("samples", 0))
            self._synthesis_status_var.set(
                f"当前数据已清空 · 上次 PASS K={samples} 仍可用"
            )
            self._synthesis_status_label.configure(fg="#81c784")
        else:
            self._synthesis_status_var.set("尚未计算")
            self._synthesis_status_label.configure(fg="#888888")
        self._update_quality_display()
        self._update_theory_start_state()
        self._log("PID collection synthesis dataset cleared")

    def _save_dataset(self) -> None:
        count = len(self._dataset)
        if count == 0:
            messagebox.showwarning("No Data", "尚未收到可用于权重合成的快照。")
            return
        result_dir = persistent_results_dir()
        result_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"integral_pid_K{count}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        path = filedialog.asksaveasfilename(
            title="保存人工驾驶合成数据",
            initialdir=result_dir,
            initialfile=default_name,
            defaultextension=".npz",
            filetypes=[("NumPy synthesis dataset", "*.npz")],
        )
        if not path:
            return
        try:
            self._dataset.save(path)
            self._dataset_path = Path(path)
            self._log(
                f"Saved {len(self._dataset)} integral windows "
                f"in {self._dataset.segment_count} segments → {Path(path).name}"
            )
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def _load_dataset(self) -> None:
        result_dir = persistent_results_dir()
        result_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="加载 PID 路径采集合成数据",
            initialdir=result_dir,
            filetypes=[("NumPy synthesis dataset", "*.npz")],
        )
        if not path:
            return
        try:
            dataset = IntegralSnapshotDataset.load(path)
            matrices = dataset.matrices()
            snapshot_count = len(dataset)
            processed_count = self._replace_plot_history(
                matrices, dataset.window_seconds
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("加载失败", str(exc))
            return

        self._dataset = dataset
        self._dataset_path = Path(path)
        self._snapshot_cache_dirty = True
        self._synthesis = None
        self._qualification = None
        self._disturbance_diagnostics_var.set(
            "扰动诊断：已加载新数据，等待 Route II 重新综合"
        )
        self._kappa_min_var.set(
            "Route II 最小值：点击“计算并校验权重”后由当前数据计算"
        )
        self._active_collection_segment = None
        self._snapshot_t0_us = None
        self._snapshot_fallback_t = self._plot_times[-1] if self._plot_times else 0.0
        duration = (
            self._plot_times[-1] - self._plot_times[0]
            if len(self._plot_times) > 1
            else 0.0
        )
        count_text = f"{snapshot_count}"
        self._sample_var.set(
            f"积分快照：{count_text} · {dataset.segment_count} 段 · 图窗 {duration:.1f} s"
        )
        suffix = " · 上次 PASS 仍可用" if self._validated_weights_payload else ""
        self._synthesis_status_var.set(f"已加载 · 请计算权重{suffix}")
        self._synthesis_status_label.configure(fg="#ffb74d")
        self._plot_view = "synthesis"
        self._plot_dirty = True
        self._update_quality_display()
        self._update_theory_start_state()
        self._log(
            f"Loaded integral={snapshot_count}, processed={processed_count} "
            f"← {Path(path).name}"
        )
        self._flush_snapshot_cache(force=True)

    def _mark_theory_complete(self) -> None:
        if self._operation_mode != "theory":
            return
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence = None
        safety_reason = self._theory_safety_reason
        self._operation_mode = "idle" if safety_reason else "theory_complete"
        self._plot_view = "theory"
        self._plot_dirty = True
        exported = self._export_theory_trace("safety" if safety_reason else "complete")
        if safety_reason:
            self._theory_status_var.set(
                f"{self._active_comparison_case} 安全停车：{safety_reason}"
                " · 禁止直接复测"
            )
            self._status_var.set(
                f"{self._active_comparison_case} 安全门触发 · 电机已停止"
                " · 请先检查导出数据"
            )
        else:
            self._theory_status_var.set(
                f"{self._active_comparison_case} 轨迹运行结束 · 可直接再次验证"
            )
            self._status_var.set(
                f"{self._active_comparison_case} 验证已完成 · 电机已停止"
                " · 最终轨迹已保留"
            )
        if exported is not None:
            self._log(f"THEORY trace exported → {exported.name}")
        self._update_theory_start_state()

    def _export_theory_trace(self, reason: str) -> Path | None:
        if self._theory_exported_path is not None:
            return self._theory_exported_path
        if not self._theory_rows:
            return None
        rows = list(self._theory_rows)
        keys = tuple(key for key in rows[0] if key != "t")
        payload: dict[str, Any] = {
            "schema": np.asarray([3], dtype=np.int64),
            "t": np.asarray([float(row["t"]) for row in rows], dtype=float),
            "reason": np.asarray([reason]),
            "trajectory": np.asarray([self._trajectory_var.get()]),
            "comparison_case": np.asarray([self._active_comparison_case]),
            "comparison_name": np.asarray([self._active_comparison_name]),
            # 生成器 peek_trace 需要该字段区分论文 E3 消融（补偿列清零）。
            "initial_nonlinear_compensation_enabled": np.asarray(
                [not getattr(self, "_paper_adaptive_only", False)], dtype=np.bool_
            ),
            "firmware": np.asarray([self._fw_version]),
            "runtime_config_json": np.asarray(
                [json.dumps(self._active_theory_config, ensure_ascii=False)]
            ),
            "weights_json": np.asarray([self._active_theory_weights_json]),
        }
        for key in keys:
            payload[key] = np.asarray([row[key] for row in rows], dtype=float)
        target_dir = persistent_results_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        case_slug = self._active_comparison_case.lower()
        if getattr(self, "_paper_adaptive_only", False):
            case_slug = f"{case_slug}_adaptive_only"
        target = target_dir / (
            f"theory_trace_{case_slug}_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{reason}.npz"
        )
        np.savez_compressed(target, **payload)
        self._theory_exported_path = target
        return target

    def _trigger_theory_safety_stop(self, reason: str) -> None:
        if self._operation_mode != "theory" or self._theory_safety_reason is not None:
            return
        self._theory_safety_reason = reason
        self._theory_status_var.set(f"安全门触发：{reason} · 正在硬停车")
        self._status_var.set(f"⚠ THEORY 安全停车：{reason}")
        self._log(f"⚠ THEORY safety stop: {reason}")
        self._export_theory_trace("safety")
        try:
            self._theory_stop_sequence = self.link.send({"cmd": "stop"})
        except Exception as exc:
            self._log(f"Safety stop send failed: {exc}")
        self._theory_stop_deadline = time.monotonic()
        self._theory_last_stop_send = 0.0

    def _accept_theory_telemetry(self, message: dict) -> None:
        mode = str(message.get("mode", ""))
        if mode != "theory":
            if self._operation_mode == "theory" and mode == "idle":
                # Redundant stop: firmware should already have stopped at its
                # autonomous deadline, but this also exercises the immediate
                # hard-stop path before the UI reports completion.
                try:
                    self.link.send({"cmd": "stop"})
                except Exception:
                    pass
                self._mark_theory_complete()
            return
        sensors = message.get("s")
        control = message.get("c")
        if message.get("type") == "state":
            sensors = {
                "pose": message.get("pose"),
                "velocity": message.get("vel"),
                "current": message.get("cur"),
            }
            control = {
                "reference": message.get("reference"),
                "motion_reference": message.get("motion_reference"),
                "pose_error": message.get("e"),
                "alpha1": message.get("alpha1"),
                "beta1": message.get("beta1"),
                "beta1_dot": message.get("beta1_dot"),
                "alpha2": message.get("alpha2"),
                "beta2": message.get("beta2"),
                "beta2_dot": message.get("beta2_dot"),
                "z2": message.get("z2"),
                "z3": message.get("z3"),
                "uc": message.get("uc"),
                "u": message.get("u"),
            }
        if not isinstance(sensors, dict) or not isinstance(control, dict):
            return
        try:
            reference = np.asarray(control.get("reference"), dtype=float)
            arrays = {
                "pose": np.asarray(sensors.get("pose"), dtype=float),
                "velocity": np.asarray(sensors.get("velocity"), dtype=float),
                "current": np.asarray(sensors.get("current"), dtype=float),
                "reference": reference,
                "motion_reference": np.asarray(
                    control.get("motion_reference", reference), dtype=float
                ),
                "pose_error": np.asarray(control.get("pose_error"), dtype=float),
                "alpha1": np.asarray(control.get("alpha1"), dtype=float),
                "beta1": np.asarray(control.get("beta1"), dtype=float),
                "beta1_dot": np.asarray(control.get("beta1_dot"), dtype=float),
                "alpha2": np.asarray(control.get("alpha2"), dtype=float),
                "beta2": np.asarray(control.get("beta2"), dtype=float),
                "beta2_dot": np.asarray(control.get("beta2_dot"), dtype=float),
                "z2": np.asarray(control.get("z2"), dtype=float),
                "z3": np.asarray(control.get("z3"), dtype=float),
                "uc": np.asarray(control.get("uc"), dtype=float),
                "u": np.asarray(control.get("u"), dtype=float),
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
            for key, length in expected.items():
                value = arrays[key]
                if value.shape != (length,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{key} invalid")
        except (TypeError, ValueError) as exc:
            self._log(f"Ignored invalid THEORY telemetry: {exc}")
            return

        timestamp_us = int(message.get("t_us", 0) or 0)
        if self._theory_t0_us is None or timestamp_us < self._theory_t0_us:
            self._theory_t0_us = timestamp_us
        elapsed = max(0.0, (timestamp_us - self._theory_t0_us) * 1.0e-6)
        self._theory_rows.append({"t": elapsed, **arrays})
        self._plot_view = "theory"
        self._plot_dirty = True
        position_error = float(np.linalg.norm(arrays["pose_error"][:2]))
        motion_target_error = float(
            np.linalg.norm(
                arrays["motion_reference"][:2] - arrays["reference"][:2]
            )
        )
        heading_error = abs(float(arrays["pose_error"][2]))
        z2_norm = float(np.linalg.norm(arrays["z2"]))
        z3_norm = float(np.linalg.norm(arrays["z3"]))
        self._theory_status_var.set(
            f"{self._active_comparison_case} 运行中 · t={elapsed:.1f} s · "
            f"运动参考误差={position_error:.3f} m"
        )
        limits = self._active_theory_safety
        if elapsed >= limits["theory_safety_grace"]:
            violation = None
            if motion_target_error > limits["motion_max_target_error"]:
                violation = (
                    f"运动参考偏离目标 {motion_target_error:.3f} m > "
                    f"{limits['motion_max_target_error']:.3f} m"
                )
            elif position_error > limits["theory_max_position_error"]:
                violation = (
                    f"实车偏离运动参考 {position_error:.3f} m > "
                    f"{limits['theory_max_position_error']:.3f} m"
                )
            elif heading_error > limits["theory_max_heading_error"]:
                violation = (
                    f"航向误差 {math.degrees(heading_error):.1f}° > "
                    f"{math.degrees(limits['theory_max_heading_error']):.1f}°"
                )
            elif z2_norm > limits["theory_max_z2"]:
                violation = (
                    f"||z2||={z2_norm:.3f} > {limits['theory_max_z2']:.3f}"
                )
            elif z3_norm > limits["theory_max_z3"]:
                violation = (
                    f"||z3||={z3_norm:.3f} > {limits['theory_max_z3']:.3f}"
                )
            self._theory_safety_count = (
                self._theory_safety_count + 1 if violation is not None else 0
            )
            if self._theory_safety_count >= THEORY_SAFETY_CONSECUTIVE_FRAMES:
                self._trigger_theory_safety_stop(violation or "连续越界")

    def _build_plot_panel(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg="#111122")
        header.pack(fill=tk.X, padx=10, pady=(4, 0))
        self._plot_header_var = tk.StringVar(value="权重合成数据 · 实时 20 s")
        tk.Label(header, textvariable=self._plot_header_var,
                 font=("Segoe UI", 10, "bold"), fg="#dddddd",
                 bg="#111122").pack(side=tk.LEFT)
        self._plot_header_detail_var = tk.StringVar(value="X₄, X₃, Ż₂, Y₂, Ż₃, Y₃")
        tk.Label(header, textvariable=self._plot_header_detail_var,
                 font=("Segoe UI", 8), fg="#777788",
                 bg="#111122").pack(side=tk.RIGHT)
        mode_frame = tk.Frame(header, bg="#111122")
        for mode, text in (
            ("trajectory", "轨迹图"),
            ("data", "六矩阵"),
            ("regressors", "回归元素"),
        ):
            button = tk.Button(
                mode_frame,
                text=text,
                command=lambda value=mode: self._set_collection_plot_mode(value),
                font=("Microsoft YaHei UI", 8, "bold"),
                fg="#dddddd",
                bg="#29293d",
                activeforeground="#ffffff",
                activebackground="#14805f",
                relief=tk.FLAT,
                bd=0,
                padx=9,
                pady=3,
                cursor="hand2",
            )
            button.pack(side=tk.LEFT, padx=(4, 0))
            self._plot_mode_buttons[mode] = button
        self._plot_switch_button = self._plot_mode_buttons["data"]
        self._plot_mode_frame = mode_frame

        theory_mode_frame = tk.Frame(header, bg="#111122")
        for mode, text in (
            ("trajectory", "轨迹图"),
            ("signals", "控制信号"),
        ):
            button = tk.Button(
                theory_mode_frame,
                text=text,
                command=lambda value=mode: self._set_theory_plot_mode(value),
                font=("Microsoft YaHei UI", 8, "bold"),
                fg="#dddddd",
                bg="#29293d",
                activeforeground="#ffffff",
                activebackground="#14805f",
                relief=tk.FLAT,
                bd=0,
                padx=9,
                pady=3,
                cursor="hand2",
            )
            button.pack(side=tk.LEFT, padx=(4, 0))
            self._theory_plot_mode_buttons[mode] = button
        self._theory_plot_mode_frame = theory_mode_frame

        plot_body = tk.Frame(parent, bg="#111122")
        plot_body.pack(fill=tk.BOTH, expand=True)
        self._plot_body = plot_body
        # THEORY 控制信号页含 8 个子图（含 α₂/β₂），页面放不下时用滚轮滚动。
        plot_scroll = tk.Canvas(
            plot_body, bg="#111122", highlightthickness=0, bd=0,
        )
        plot_bar = tk.Scrollbar(
            plot_body, orient=tk.VERTICAL, command=plot_scroll.yview,
        )
        plot_scroll.configure(yscrollcommand=plot_bar.set)
        plot_bar.pack(side=tk.RIGHT, fill=tk.Y)
        plot_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        plot_inner = tk.Frame(plot_scroll, bg="#111122")
        plot_window = plot_scroll.create_window(
            (0, 0), window=plot_inner, anchor="nw"
        )
        figure = Figure(figsize=(10.2, 13.0), dpi=100, facecolor="#111122")
        axes = figure.subplots(5, 2)
        figure.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.05,
                               hspace=0.55, wspace=0.28)
        self._plot_figure = figure
        self._plot_axes = axes
        self._plot_axis_positions = tuple(
            axis.get_position().frozen() for axis in axes.flat
        )
        self._plot_canvas = FigureCanvasTkAgg(figure, master=plot_inner)
        plot_widget = self._plot_canvas.get_tk_widget()
        plot_widget.pack(fill=tk.BOTH, expand=True)
        plot_widget.configure(height=1300)

        def update_plot_scroll(_event=None) -> None:
            plot_scroll.configure(scrollregion=plot_scroll.bbox("all"))

        def resize_plot_width(event) -> None:
            plot_scroll.itemconfigure(plot_window, width=event.width)

        def scroll_main_plot(event) -> None:
            direction = -1 if event.delta > 0 else 1
            plot_scroll.yview_scroll(3 * direction, "units")

        plot_inner.bind("<Configure>", update_plot_scroll)
        plot_scroll.bind("<Configure>", resize_plot_width)
        plot_scroll.bind("<MouseWheel>", scroll_main_plot)
        plot_widget.bind("<MouseWheel>", scroll_main_plot)
        self._plot_scroll_canvas = plot_scroll

        regressor_frame = tk.Frame(plot_body, bg="#111122")
        regressor_scroll = tk.Canvas(
            regressor_frame,
            bg="#111122",
            highlightthickness=0,
            bd=0,
        )
        regressor_bar = tk.Scrollbar(
            regressor_frame,
            orient=tk.VERTICAL,
            command=regressor_scroll.yview,
        )
        regressor_scroll.configure(yscrollcommand=regressor_bar.set)
        regressor_bar.pack(side=tk.RIGHT, fill=tk.Y)
        regressor_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        regressor_inner = tk.Frame(regressor_scroll, bg="#111122")
        regressor_window = regressor_scroll.create_window(
            (0, 0), window=regressor_inner, anchor="nw"
        )
        regressor_figure = Figure(
            figsize=(10.0, 18.0), dpi=100, facecolor="#111122"
        )
        regressor_axes = regressor_figure.subplots(8, 2)
        regressor_figure.subplots_adjust(
            left=0.08, right=0.975, top=0.985, bottom=0.035,
            hspace=0.62, wspace=0.24,
        )
        regressor_canvas = FigureCanvasTkAgg(
            regressor_figure, master=regressor_inner
        )
        regressor_widget = regressor_canvas.get_tk_widget()
        regressor_widget.pack(fill=tk.BOTH, expand=True)
        regressor_widget.configure(height=1800)

        def update_regressor_scroll(_event=None) -> None:
            regressor_scroll.configure(scrollregion=regressor_scroll.bbox("all"))

        def resize_regressor_width(event) -> None:
            regressor_scroll.itemconfigure(regressor_window, width=event.width)

        def scroll_regressors(event) -> None:
            if self._collection_plot_mode != "regressors":
                return
            direction = -1 if event.delta > 0 else 1
            regressor_scroll.yview_scroll(3 * direction, "units")

        regressor_inner.bind("<Configure>", update_regressor_scroll)
        regressor_scroll.bind("<Configure>", resize_regressor_width)
        regressor_scroll.bind("<MouseWheel>", scroll_regressors)
        regressor_widget.bind("<MouseWheel>", scroll_regressors)
        self._regressor_scroll_frame = regressor_frame
        self._regressor_scroll_canvas = regressor_scroll
        self._regressor_scroll_window = regressor_window
        self._regressor_figure = regressor_figure
        self._regressor_axes = regressor_axes
        self._regressor_canvas = regressor_canvas
        self._plot_dirty = True
        self._draw_snapshot_plots()
        self._plot_dirty = False

    def _refresh_plot(self, force: bool = False) -> None:
        try:
            if (self._plot_canvas is not None and (force or self._plot_dirty)
                    and self._plot_axes is not None):
                try:
                    self._draw_snapshot_plots()
                    self._plot_dirty = False
                except Exception as exc:
                    self._log(f"Plot update failed: {exc}")
        finally:
            self._plot_job = self.after(PLOT_REFRESH_MS, self._refresh_plot)

    def _draw_snapshot_plots(self) -> None:
        self._update_plot_switch_button()
        if (
            self._plot_view == "collection"
            and self._collection_plot_mode == "regressors"
        ):
            self._show_regressor_canvas()
            self._draw_regressor_element_plots()
            return
        self._show_main_plot_canvas()
        if self._plot_view == "theory":
            if self._theory_plot_mode == "signals":
                self._draw_theory_signal_plots()
            else:
                self._draw_theory_plots()
            return
        if (
            self._plot_view == "collection"
            and self._collection_plot_mode == "trajectory"
        ):
            self._draw_collection_trajectory()
            return
        self._restore_plot_grid()
        if self._plot_view == "collection":
            state_text = (
                "实时 20 s"
                if self._operation_mode == "pid_collect"
                else "采集结果（已保留）"
            )
            self._plot_header_var.set(
                f"PID 路径采集 · 权重合成六矩阵 · {state_text}"
            )
        else:
            self._plot_header_var.set("权重合成数据 · 实时 20 s")
        self._plot_header_detail_var.set("X₄, X₃, Ż₂, Y₂, Ż₃, Y₃")
        axes = tuple(self._plot_axes.flat)
        specs = (
            ("x4", r"$X_4=U$ · applied voltage", (r"$u_R$", r"$u_L$"), "V"),
            ("x3", r"$X_3=I$ · armature current", (r"$i_R$", r"$i_L$"), "A"),
            ("zdot2", r"$\dot{Z}_2$ · velocity block", (r"$\dot Z_{2,1}$", r"$\dot Z_{2,2}$"), ""),
            ("y2", r"$Y_2=[Z_2;\Sigma_2]$ · regressors", (), ""),
            ("zdot3", r"$\dot{Z}_3$ · current block", (r"$\dot Z_{3,1}$", r"$\dot Z_{3,2}$"), ""),
            ("y3", r"$Y_3=[Z_3;\Sigma_3]$ · regressors", (), ""),
        )
        colors = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7",
                  "#F0E442", "#0072B2", "#D55E00", "#999999")

        for ax, (key, title, labels, ylabel) in zip(axes, specs):
            ax.clear()
            ax.set_facecolor("#17172b")
            ax.set_title(title, color="#eeeeee", fontsize=9)
            ax.set_xlabel("t (s)", color="#bbbbbb", fontsize=8)
            if ylabel:
                ax.set_ylabel(ylabel, color="#bbbbbb", fontsize=8)
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#555566")
            ax.grid(True, color="#35354a", linestyle=":", linewidth=0.6)
            ax.axhline(0.0, color="#777788", linewidth=0.6)

        if self._plot_rows:
            times = np.asarray(self._plot_times, dtype=float)
            first = max(0, int(np.searchsorted(times, times[-1] - PLOT_WINDOW_S)))
            times = times[first:]
            rows = list(self._plot_rows)[first:]
            for ax, (key, _title, labels, _ylabel) in zip(axes, specs):
                matrix = np.asarray([row[key] for row in rows], dtype=float).T
                for index in range(matrix.shape[0]):
                    if labels:
                        label = labels[index]
                    elif index < 2:
                        label = rf"$Z_{{{key[-1]},{index + 1}}}$"
                    else:
                        label = rf"$\Sigma_{{{key[-1]},{index - 1}}}$"
                    ax.plot(times, matrix[index], color=colors[index % len(colors)],
                            linewidth=1.0 if index < 2 else 0.75, label=label)
                ax.legend(loc="upper right", fontsize=6, ncol=2, frameon=False,
                          labelcolor="#cccccc")
                if times.size > 1:
                    ax.set_xlim(times[0], times[-1])

        # 六矩阵只占前 6 个轴；4×2 网格的剩余轴在此模式下隐藏。
        for extra_axis in axes[6:]:
            extra_axis.clear()
            extra_axis.set_visible(False)

        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _set_collection_plot_mode(self, mode: str) -> None:
        if self._plot_view != "collection" or mode not in {
            "trajectory", "data", "regressors"
        }:
            return
        if self._operation_mode == "pid_collect" and mode != "trajectory":
            return
        self._collection_plot_mode = mode
        self._plot_dirty = True
        self._update_plot_switch_button()

    def _toggle_collection_plot(self) -> None:
        """Backward-compatible trajectory/data toggle used by older callers."""
        target = "data" if self._collection_plot_mode == "trajectory" else "trajectory"
        self._set_collection_plot_mode(target)

    def _set_theory_plot_mode(self, mode: str) -> None:
        if self._plot_view != "theory" or mode not in {"trajectory", "signals"}:
            return
        self._theory_plot_mode = mode
        self._plot_dirty = True
        self._update_plot_switch_button()

    def _update_plot_switch_button(self) -> None:
        if not self._plot_mode_buttons:
            return
        self._plot_mode_frame.pack_forget()
        self._theory_plot_mode_frame.pack_forget()
        if self._plot_view == "theory":
            for mode, button in self._theory_plot_mode_buttons.items():
                active = mode == self._theory_plot_mode
                button.configure(
                    bg="#0f6b4f" if active else "#29293d",
                    fg="#ffffff" if active else "#bbbbbb",
                )
            self._theory_plot_mode_frame.pack(side=tk.RIGHT, padx=(8, 10))
            return
        if self._plot_view != "collection":
            return
        if self._operation_mode == "pid_collect":
            self._collection_plot_mode = "trajectory"
            return
        for mode, button in self._plot_mode_buttons.items():
            active = mode == self._collection_plot_mode
            button.configure(
                bg="#0f6b4f" if active else "#29293d",
                fg="#ffffff" if active else "#bbbbbb",
            )
        self._plot_mode_frame.pack(side=tk.RIGHT, padx=(8, 10))

    def _show_main_plot_canvas(self) -> None:
        if self._regressor_scroll_frame is not None:
            self._regressor_scroll_frame.pack_forget()
        if self._plot_scroll_canvas is not None:
            if not self._plot_scroll_canvas.winfo_manager():
                self._plot_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _show_regressor_canvas(self) -> None:
        if self._plot_scroll_canvas is not None:
            self._plot_scroll_canvas.pack_forget()
        if self._regressor_scroll_frame is not None:
            if not self._regressor_scroll_frame.winfo_manager():
                self._regressor_scroll_frame.pack(fill=tk.BOTH, expand=True)

    def _draw_regressor_element_plots(self) -> None:
        if self._regressor_axes is None:
            return
        finished = self._operation_mode != "pid_collect"
        state_text = "采集结果（已保留）" if finished else "实时 20 s"
        self._plot_header_var.set(f"PID 回归项来源信号 · {state_text}")
        times = np.asarray(self._plot_times, dtype=float)
        rows = list(self._plot_rows)
        if times.size:
            first = max(0, int(np.searchsorted(times, times[-1] - PLOT_WINDOW_S)))
            times = times[first:]
            rows = rows[first:]

        y2 = np.asarray([row["y2"] for row in rows], dtype=float).T if rows else None
        x3 = np.asarray([row["x3"] for row in rows], dtype=float).T if rows else None
        y3 = np.asarray([row["y3"] for row in rows], dtype=float).T if rows else None

        def optional_source(key: str):
            if not rows or any(key not in row for row in rows):
                return None
            value = np.asarray([row[key] for row in rows], dtype=float).T
            return value if value.shape == (2, len(rows)) else None

        velocity_raw = optional_source("velocity_raw")
        current_raw = optional_source("current_raw")
        raw_available = velocity_raw is not None and current_raw is not None
        source_note = "已叠加滤波前 / 滤波后" if raw_available else "仅有滤波后；需烧录配套固件显示滤波前"
        self._plot_header_detail_var.set(
            f"显示每个回归项构造前的来源信号 · {source_note} · 鼠标滚轮查看全部"
        )

        panels = None
        if y2 is not None and x3 is not None and y3 is not None:
            velocity_filtered = y2[2:4]
            current_filtered = x3
            beta1 = velocity_filtered - y2[0:2]
            beta2 = current_filtered - y3[0:2]
            beta1_dot = y2[6:8]
            beta2_dot = y3[6:8]
            half_track = 0.5 * self._track_width_m
            wheel_right_filtered = velocity_filtered[0] + half_track * velocity_filtered[1]
            wheel_left_filtered = velocity_filtered[0] - half_track * velocity_filtered[1]
            wheel_right_raw = (
                None
                if velocity_raw is None
                else velocity_raw[0] + half_track * velocity_raw[1]
            )
            wheel_left_raw = (
                None
                if velocity_raw is None
                else velocity_raw[0] - half_track * velocity_raw[1]
            )

            def filtered_pair(raw, filtered, symbol: str):
                curves = []
                if raw is not None:
                    curves.append((raw, f"滤波前 {symbol}", "#B9B9C9", "-"))
                curves.append((filtered, f"滤波后 {symbol}", "#56B4E9", "-"))
                return curves

            def dynamic_surface(raw, filtered, beta, symbol: str, beta_label: str):
                return filtered_pair(raw, filtered, symbol) + [
                    (beta, beta_label, "#009E73", "--")
                ]

            panels = (
                (
                    (
                        r"$Z_{2,1}=v-\beta_{1,1}$ · 来源：$v,\beta_{1,1}$",
                        dynamic_surface(
                            None if velocity_raw is None else velocity_raw[0],
                            velocity_filtered[0], beta1[0], "v", r"$\beta_{1,1}$",
                        ),
                    ),
                    (
                        r"$Z_{3,1}=i_R-\beta_{2,1}$ · 来源：$i_R,\beta_{2,1}$",
                        dynamic_surface(
                            None if current_raw is None else current_raw[0],
                            current_filtered[0], beta2[0], r"$i_R$", r"$\beta_{2,1}$",
                        ),
                    ),
                ),
                (
                    (
                        r"$Z_{2,2}=\omega-\beta_{1,2}$ · 来源：$\omega,\beta_{1,2}$",
                        dynamic_surface(
                            None if velocity_raw is None else velocity_raw[1],
                            velocity_filtered[1], beta1[1], r"$\omega$", r"$\beta_{1,2}$",
                        ),
                    ),
                    (
                        r"$Z_{3,2}=i_L-\beta_{2,2}$ · 来源：$i_L,\beta_{2,2}$",
                        dynamic_surface(
                            None if current_raw is None else current_raw[1],
                            current_filtered[1], beta2[1], r"$i_L$", r"$\beta_{2,2}$",
                        ),
                    ),
                ),
                (
                    (r"$\Sigma_{2,1}=v$ · 来源：纵向速度", filtered_pair(None if velocity_raw is None else velocity_raw[0], velocity_filtered[0], "v")),
                    (r"$\Sigma_{3,1}=i_R$ · 来源：右电机电流", filtered_pair(None if current_raw is None else current_raw[0], current_filtered[0], r"$i_R$")),
                ),
                (
                    (r"$\Sigma_{2,2}=\omega$ · 来源：车体角速度", filtered_pair(None if velocity_raw is None else velocity_raw[1], velocity_filtered[1], r"$\omega$")),
                    (r"$\Sigma_{3,2}=i_L$ · 来源：左电机电流", filtered_pair(None if current_raw is None else current_raw[1], current_filtered[1], r"$i_L$")),
                ),
                (
                    (r"$\Sigma_{2,3}=\mathrm{sgn}(v)$ · 变换前输入：$v$", filtered_pair(None if velocity_raw is None else velocity_raw[0], velocity_filtered[0], "v")),
                    (r"$\Sigma_{3,3}=v_R$ · 来源：右轮线速度", filtered_pair(wheel_right_raw, wheel_right_filtered, r"$v_R$")),
                ),
                (
                    (r"$\Sigma_{2,4}=\mathrm{sgn}(\omega)$ · 变换前输入：$\omega$", filtered_pair(None if velocity_raw is None else velocity_raw[1], velocity_filtered[1], r"$\omega$")),
                    (r"$\Sigma_{3,4}=v_L$ · 来源：左轮线速度", filtered_pair(wheel_left_raw, wheel_left_filtered, r"$v_L$")),
                ),
                (
                    (r"$\Sigma_{2,5}=\dot\beta_{1,1}$ · 直接回归信号", [(beta1_dot[0], r"$\dot\beta_{1,1}$", "#E69F00", "-")]),
                    (r"$\Sigma_{3,5}=\dot\beta_{2,1}$ · 直接回归信号", [(beta2_dot[0], r"$\dot\beta_{2,1}$", "#E69F00", "-")]),
                ),
                (
                    (r"$\Sigma_{2,6}=\dot\beta_{1,2}$ · 直接回归信号", [(beta1_dot[1], r"$\dot\beta_{1,2}$", "#E69F00", "-")]),
                    (r"$\Sigma_{3,6}=\dot\beta_{2,2}$ · 直接回归信号", [(beta2_dot[1], r"$\dot\beta_{2,2}$", "#E69F00", "-")]),
                ),
            )

        for row_index in range(8):
            for column_index in range(2):
                axis = self._regressor_axes[row_index, column_index]
                axis.clear()
                axis.set_facecolor("#17172b")
                title = "等待来源信号"
                curves = []
                if panels is not None:
                    title, curves = panels[row_index][column_index]
                axis.set_title(
                    title,
                    color="#eeeeee",
                    fontsize=9,
                    pad=5,
                    fontfamily="Microsoft YaHei",
                )
                axis.set_xlabel("t (s)", color="#bbbbbb", fontsize=7)
                axis.tick_params(colors="#aaaaaa", labelsize=7)
                for spine in axis.spines.values():
                    spine.set_color("#555566")
                axis.grid(True, color="#35354a", linestyle=":", linewidth=0.6)
                axis.axhline(0.0, color="#777788", linewidth=0.55)
                for values, label, color, linestyle in curves:
                    axis.plot(
                        times,
                        values,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.1,
                    )
                if curves:
                    axis.legend(
                        loc="upper left",
                        fontsize=7,
                        framealpha=0.25,
                        labelcolor="#dddddd",
                        prop={"family": "Microsoft YaHei", "size": 7},
                    )
                if times.size > 1:
                    axis.set_xlim(times[0], times[-1])
        if self._regressor_canvas is not None:
            self._regressor_canvas.draw_idle()
        if self._regressor_scroll_canvas is not None:
            self._regressor_scroll_canvas.configure(
                scrollregion=self._regressor_scroll_canvas.bbox("all")
            )

    def _restore_plot_grid(self) -> None:
        if self._plot_axes is None or self._plot_axis_positions is None:
            return
        for axis, position in zip(self._plot_axes.flat, self._plot_axis_positions):
            axis.set_visible(True)
            axis.set_position(position)
            axis.set_aspect("auto", adjustable="box")

    def _draw_collection_trajectory(self) -> None:
        state_text = (
            "二维实时轨迹"
            if self._operation_mode == "pid_collect"
            else "最终轨迹（已保留）"
        )
        self._plot_header_var.set(f"PID 路径采集 · {state_text}")
        self._plot_header_detail_var.set(
            "参考轨迹、实测轨迹、小车位置与朝向 · 其余图在采集结束后开放"
            if self._operation_mode == "pid_collect"
            else "参考轨迹、实测轨迹、小车位置与朝向 · 可切换查看采集结果"
        )
        axes = tuple(self._plot_axes.flat)
        axis = axes[0]
        preset = (
            self._pid_active_preset
            or PID_COLLECTION_PRESETS[self._pid_reference_name]
        )
        duration = self._pid_duration or float(preset["duration"])
        reference_key = (self._pid_reference_name, float(duration))
        artists = self._collection_trajectory_artists
        reusable = bool(
            artists
            and artists.get("axis") is axis
            and artists.get("reference_key") == reference_key
            and artists.get("measured") in axis.lines
            and artists.get("body") in axis.patches
            and artists.get("reference_marker") in axis.collections
        )
        if reusable:
            assert artists is not None
            reference_xy = artists["reference_xy"]
        else:
            reference_times = np.linspace(0.0, duration, 600)
            reference_xy = np.asarray(
                [
                    pid_collection_reference(t, preset).pose[:2]
                    for t in reference_times
                ],
                dtype=float,
            )

        trace = np.asarray(self._pid_pose_trace, dtype=float)
        pose = trace[-1] if trace.size else np.asarray(self._latest_pose, dtype=float)

        if self._pid_trace_times:
            elapsed = float(self._pid_trace_times[-1])
        elif self._pid_started_at > 0.0:
            elapsed = self._pid_elapsed()
        else:
            elapsed = 0.0
        elapsed = min(elapsed, duration)
        reference_pose = np.asarray(
            pid_collection_reference(elapsed, preset).pose,
            dtype=float,
        )

        x_span = max(float(np.ptp(reference_xy[:, 0])), 0.10)
        y_span = max(float(np.ptp(reference_xy[:, 1])), 0.10)
        path_scale = max(x_span, y_span)
        body_length = 0.075 * path_scale
        body_width = 0.050 * path_scale
        body_local = np.asarray(
            (
                (0.65 * body_length, 0.0),
                (-0.35 * body_length, 0.5 * body_width),
                (-0.35 * body_length, -0.5 * body_width),
            ),
            dtype=float,
        )
        heading = float(pose[2])
        rotation = np.asarray(
            (
                (math.cos(heading), -math.sin(heading)),
                (math.sin(heading), math.cos(heading)),
            ),
            dtype=float,
        )
        body = body_local @ rotation.T + pose[:2]

        if not reusable:
            for hidden_axis in axes[1:]:
                hidden_axis.clear()
                hidden_axis.set_visible(False)
            axis.set_visible(True)
            axis.set_position([0.075, 0.09, 0.90, 0.84])
            axis.clear()
            axis.set_facecolor("#17172b")
            axis.set_title("PID path collection", color="#eeeeee", fontsize=11)
            axis.set_xlabel("x (m)", color="#bbbbbb", fontsize=9)
            axis.set_ylabel("y (m)", color="#bbbbbb", fontsize=9)
            axis.tick_params(colors="#aaaaaa", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#555566")
            axis.grid(True, color="#35354a", linestyle=":", linewidth=0.7)
            axis.axhline(0.0, color="#555566", linewidth=0.6)
            axis.axvline(0.0, color="#555566", linewidth=0.6)
            axis.plot(
                reference_xy[:, 0],
                reference_xy[:, 1],
                color="#999999",
                linestyle="--",
                linewidth=1.4,
                label="reference",
                zorder=2,
            )
            measured, = axis.plot(
                [], [], color="#56B4E9", linewidth=1.8,
                label="measured", zorder=4,
            )
            reference_marker = axis.scatter(
                [reference_pose[0]], [reference_pose[1]], marker="x", s=55,
                linewidths=1.6, color="#F0E442", label="current reference",
                zorder=5,
            )
            body_patch = axis.fill(
                body[:, 0], body[:, 1], color="#E69F00",
                edgecolor="#fff3c4", linewidth=1.0, label="car", zorder=7,
            )[0]
            center_marker = axis.scatter(
                [pose[0]], [pose[1]], s=12, color="#fff3c4", zorder=8
            )
            info_text = axis.text(
                0.015, 0.985, "", transform=axis.transAxes, va="top", ha="left",
                color="#dddddd", fontsize=8,
                bbox={
                    "facecolor": "#111122",
                    "edgecolor": "#555566",
                    "alpha": 0.88,
                },
                zorder=10,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.legend(
                loc="upper right", fontsize=7, ncol=2, frameon=False,
                labelcolor="#cccccc",
            )
            artists = {
                "axis": axis,
                "reference_key": reference_key,
                "reference_xy": reference_xy,
                "measured": measured,
                "reference_marker": reference_marker,
                "body": body_patch,
                "center": center_marker,
                "info": info_text,
            }
            self._collection_trajectory_artists = artists

        assert artists is not None
        if trace.size:
            artists["measured"].set_data(trace[:, 0], trace[:, 1])
        else:
            artists["measured"].set_data([], [])
        artists["reference_marker"].set_offsets(reference_pose[:2][None, :])
        artists["body"].set_xy(body)
        artists["center"].set_offsets(pose[:2][None, :])

        all_x = np.concatenate((reference_xy[:, 0], trace[:, 0] if trace.size else pose[:1]))
        all_y = np.concatenate((reference_xy[:, 1], trace[:, 1] if trace.size else pose[1:2]))
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
        x_margin = max(0.12 * (x_max - x_min), 0.04)
        y_margin = max(0.12 * (y_max - y_min), 0.04)
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_ylim(y_min - y_margin, y_max + y_margin)

        position_error = float(np.linalg.norm(pose[:2] - reference_pose[:2]))
        artists["info"].set_text(
            (
                f"t = {elapsed:.1f} / {duration:.0f} s\n"
                f"x = {pose[0]:+.3f} m   y = {pose[1]:+.3f} m   "
                f"theta = {math.degrees(heading):+.1f} deg\n"
                f"position error = {position_error:.3f} m"
            ),
        )
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _draw_theory_signal_plots(self) -> None:
        state_text = (
            "实时"
            if self._operation_mode == "theory"
            else "实验结果（已保留）"
        )
        self._plot_header_var.set(
            f"{self._active_comparison_case} 实物对比 · 控制信号 · {state_text}"
        )
        self._plot_header_detail_var.set(
            "α₁、β₁、实测 v/v_c、实测 ω/ω_c、z₂、α₂、β₂、u_c、u · 滚轮查看全部"
        )
        self._restore_plot_grid()
        axes = tuple(self._plot_axes.flat)
        specs = (
            (
                "alpha1", r"$\alpha_1$ · outer-loop virtual command",
                (r"$v_{\alpha}$ (m/s)", r"$\omega_{\alpha}$ (rad/s)"),
            ),
            (
                "beta1", r"$\beta_1$ · filtered velocity reference",
                (r"$v_c$ (m/s)", r"$\omega_c$ (rad/s)"),
            ),
            (
                ("velocity", 0, "beta1", 0),
                r"$v$ · measured vs $v_c$ command",
                (r"$v$ (m/s)", r"$v_c$ (m/s)"),
            ),
            (
                ("velocity", 1, "beta1", 1),
                r"$\omega$ · measured vs $\omega_c$ command",
                (r"$\omega$ (rad/s)", r"$\omega_c$ (rad/s)"),
            ),
            (
                "z2", r"$z_2=V-\beta_1$ · velocity tracking error",
                (r"$z_{2,v}$ (m/s)", r"$z_{2,\omega}$ (rad/s)"),
            ),
            (
                "alpha2", r"$\alpha_2$ · inner-loop virtual command",
                (r"$i_{\alpha,R}$ (A)", r"$i_{\alpha,L}$ (A)"),
            ),
            (
                "beta2", r"$\beta_2$ · filtered current reference",
                (r"$i_{\beta,R}$ (A)", r"$i_{\beta,L}$ (A)"),
            ),
            (
                "uc", r"$u_c$ · unsaturated voltage command",
                (r"$u_{c,R}$ (V)", r"$u_{c,L}$ (V)"),
            ),
            (
                "u", r"$u$ · controller output voltage",
                (r"$u_R$ (V)", r"$u_L$ (V)"),
            ),
        )
        colors = ("#56B4E9", "#E69F00")
        for axis, (_key, title, _labels) in zip(axes, specs):
            axis.clear()
            axis.set_facecolor("#17172b")
            axis.set_title(title, color="#eeeeee", fontsize=9)
            axis.set_xlabel("t (s)", color="#bbbbbb", fontsize=8)
            axis.tick_params(colors="#aaaaaa", labelsize=7)
            for spine in axis.spines.values():
                spine.set_color("#555566")
            axis.grid(True, color="#35354a", linestyle=":", linewidth=0.6)
            axis.axhline(0.0, color="#777788", linewidth=0.6)

        if self._theory_rows:
            rows = list(self._theory_rows)
            times = np.asarray([float(row["t"]) for row in rows], dtype=float)
            for axis, (key, _title, labels) in zip(axes, specs):
                if isinstance(key, tuple):
                    values = np.asarray(
                        [
                            [row[key[0]][key[1]], row[key[2]][key[3]]]
                            for row in rows
                        ],
                        dtype=float,
                    ).T
                else:
                    values = np.asarray([row[key] for row in rows], dtype=float).T
                for index, label in enumerate(labels):
                    axis.plot(
                        times, values[index], color=colors[index],
                        linewidth=1.2, label=label,
                    )
                axis.legend(
                    loc="upper right", fontsize=6, ncol=2, frameon=False,
                    labelcolor="#cccccc",
                )
                if times.size > 1:
                    axis.set_xlim(times[0], times[-1])

        for extra_axis in axes[len(specs):]:
            extra_axis.clear()
            extra_axis.set_visible(False)

        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _draw_theory_plots(self) -> None:
        self._plot_header_var.set(
            f"{self._active_comparison_case} 实物对比 · 二维实时轨迹"
        )
        self._plot_header_detail_var.set(
            "目标轨迹、运动参考轨迹、实测轨迹、小车位置与朝向"
        )
        axes = tuple(self._plot_axes.flat)
        axis = axes[0]
        for hidden_axis in axes[1:]:
            hidden_axis.clear()
            hidden_axis.set_visible(False)

        axis.set_visible(True)
        axis.set_position([0.075, 0.09, 0.90, 0.84])
        axis.clear()
        axis.set_facecolor("#17172b")
        axis.set_title("THEORY trajectory validation", color="#eeeeee", fontsize=11)
        axis.set_xlabel("x (m)", color="#bbbbbb", fontsize=9)
        axis.set_ylabel("y (m)", color="#bbbbbb", fontsize=9)
        axis.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#555566")
        axis.grid(True, color="#35354a", linestyle=":", linewidth=0.7)
        axis.axhline(0.0, color="#555566", linewidth=0.6)
        axis.axvline(0.0, color="#555566", linewidth=0.6)

        if self._theory_rows:
            rows = list(self._theory_rows)
            pose_all = np.asarray([row["pose"] for row in rows], dtype=float)
            reference_all = np.asarray([row["reference"] for row in rows], dtype=float)
            motion_reference_all = np.asarray(
                [row["motion_reference"] for row in rows], dtype=float
            )
            axis.plot(
                reference_all[:, 0], reference_all[:, 1],
                color="#999999", linestyle="--", linewidth=1.4,
                label="target reference",
            )
            axis.plot(
                motion_reference_all[:, 0], motion_reference_all[:, 1],
                color="#009E73", linestyle="-.", linewidth=1.6,
                label="motion reference",
            )
            axis.plot(
                pose_all[:, 0], pose_all[:, 1],
                color="#56B4E9", linewidth=1.8, label="measured",
            )
            current_pose = pose_all[-1]
            current_reference = reference_all[-1]
            current_motion_reference = motion_reference_all[-1]
            axis.scatter(
                [current_reference[0]], [current_reference[1]],
                marker="x", s=55, linewidths=1.6, color="#F0E442",
                label="current reference", zorder=5,
            )
            axis.scatter(
                [current_motion_reference[0]], [current_motion_reference[1]],
                marker="D", s=28, linewidths=0.8, color="#009E73",
                edgecolors="#c8f7e8", label="current motion ref", zorder=6,
            )
            all_x = np.concatenate(
                (reference_all[:, 0], motion_reference_all[:, 0], pose_all[:, 0])
            )
            all_y = np.concatenate(
                (reference_all[:, 1], motion_reference_all[:, 1], pose_all[:, 1])
            )
            x_span = max(float(np.ptp(all_x)), 0.10)
            y_span = max(float(np.ptp(all_y)), 0.10)
            path_scale = max(x_span, y_span)
            body_length = 0.075 * path_scale
            body_width = 0.050 * path_scale
            body_local = np.asarray(
                (
                    (0.65 * body_length, 0.0),
                    (-0.35 * body_length, 0.5 * body_width),
                    (-0.35 * body_length, -0.5 * body_width),
                ),
                dtype=float,
            )
            heading = float(current_pose[2])
            rotation = np.asarray(
                (
                    (math.cos(heading), -math.sin(heading)),
                    (math.sin(heading), math.cos(heading)),
                ),
                dtype=float,
            )
            body = body_local @ rotation.T + current_pose[:2]
            axis.fill(
                body[:, 0], body[:, 1], color="#E69F00", edgecolor="#fff3c4",
                linewidth=1.0, label="car", zorder=7,
            )
            axis.scatter(
                [current_pose[0]], [current_pose[1]], s=12,
                color="#fff3c4", zorder=8,
            )

            x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
            y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
            x_margin = max(0.12 * (x_max - x_min), 0.04)
            y_margin = max(0.12 * (y_max - y_min), 0.04)
            axis.set_xlim(x_min - x_margin, x_max + x_margin)
            axis.set_ylim(y_min - y_margin, y_max + y_margin)
            axis.set_aspect("equal", adjustable="box")

            elapsed = float(rows[-1]["t"])
            position_error = float(
                np.linalg.norm(current_pose[:2] - current_reference[:2])
            )
            motion_position_error = float(
                np.linalg.norm(current_pose[:2] - current_motion_reference[:2])
            )
            heading_error = float(rows[-1]["pose_error"][2])
            axis.text(
                0.015, 0.985,
                (
                    f"t = {elapsed:.1f} s\n"
                    f"x = {current_pose[0]:+.3f} m   y = {current_pose[1]:+.3f} m   "
                    f"theta = {math.degrees(heading):+.1f} deg\n"
                    f"target error = {position_error:.3f} m   "
                    f"motion-ref error = {motion_position_error:.3f} m\n"
                    f"guidance heading error = "
                    f"{math.degrees(heading_error):+.1f} deg"
                ),
                transform=axis.transAxes, va="top", ha="left", color="#dddddd",
                fontsize=8,
                bbox={"facecolor": "#111122", "edgecolor": "#555566", "alpha": 0.88},
                zorder=10,
            )
            axis.legend(
                loc="upper right", fontsize=7, ncol=2, frameon=False,
                labelcolor="#cccccc",
            )

        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    # ==================================================================
    # UI refresh
    # ==================================================================

    def _discard_active_collection_segment(self, reason: str) -> int:
        """Quarantine the current PID segment after any safety-critical abort."""
        if self._active_collection_segment is None:
            return 0
        segment = self._active_collection_segment
        self._active_collection_segment = None
        removed = self._dataset.discard_segment(segment)
        self._synthesis = None
        self._qualification = None
        self._disturbance_diagnostics_var.set(
            "扰动诊断：采集段已变化，等待重新综合"
        )
        self._snapshot_cache_dirty = True
        self._sample_var.set(
            f"积分快照：{len(self._dataset)} · {self._dataset.segment_count} 段"
            f" · 已丢弃故障段 {segment}"
        )
        self._log(
            f"Discarded unsafe PID segment {segment}: {removed} samples · {reason}"
        )
        return removed

    def _handle_runtime_fault(self, fault: str) -> None:
        """Latch the PC UI in a safe idle state and report a fault once."""
        if not fault or fault == "none" or fault == self._reported_fault:
            return
        retain_theory_plot = bool(self._theory_rows) and (
            self._operation_mode == "theory" or self._plot_view == "theory"
        )
        self._reported_fault = fault
        self._fw_fault = fault
        removed = self._discard_active_collection_segment(fault)
        self._flush_snapshot_cache(force=True)
        self._stop_control_loop()
        self._pressed.clear()
        for direction in ("up", "down", "left", "right"):
            self._set_dpad_active(direction, False)
        self._last_ur = 0.0
        self._last_ul = 0.0
        self._operation_mode = "idle"
        self._plot_view = "theory" if retain_theory_plot else "synthesis"
        self._plot_dirty = True
        self._volt_var.set("u_R =  0.00 V    u_L =  0.00 V")
        fault_labels = {
            "imu_failure": "IMU 失效",
            "right_encoder_failure": "右轮编码器无响应",
            "left_encoder_failure": "左轮编码器无响应",
            "sensor_failure": "INA3221/传感器失效",
            "tracking_divergence": "轨迹/状态越界",
        }
        fault_text = fault_labels.get(fault, fault)
        discard_text = f" · 已丢弃本段 {removed} 条数据" if removed else ""
        self._collection_status_var.set(
            f"故障停车：{fault_text}{discard_text}"
        )
        diagnostic = ""
        if fault == "control_overrun":
            diagnostic = (
                f" · loop={self._latest_loop_us} μs"
                f" · dropped={self._latest_dropped_frames}"
            )
        self._theory_status_var.set(
            f"故障停车：{fault_text}{diagnostic} · 检查后点“停止 / 返回人工”复位"
        )
        self._status_var.set(
            f"⚠ 故障停车：{fault_text}{diagnostic}{discard_text}"
        )
        self._log(
            f"⚠ Fault latched: {fault}{diagnostic} · manual output stopped"
        )
        if retain_theory_plot:
            if self._theory_safety_reason is None:
                self._theory_safety_reason = fault_text
            had_export = self._theory_exported_path is not None
            exported = self._export_theory_trace(f"fault_{fault}")
            if exported is not None and not had_export:
                self._log(f"THEORY fault trace exported → {exported.name}")
        self._update_pid_start_state()
        self._update_theory_start_state()

    def _refresh_ui(self) -> None:
        if hasattr(self, "_apply_deadzone_button"):
            deadzone_enabled = (
                self.link.connected
                and self._connecting_stage >= 4
                and self._motor_deadzone_supported
                and self._workflow_kind is None
                and self._operation_mode == "manual"
            )
            self._apply_deadzone_button.configure(
                state=tk.NORMAL if deadzone_enabled else tk.DISABLED,
                fg="#ffffff" if deadzone_enabled else "#777777",
                bg="#0f6b4f" if deadzone_enabled else "#2a2a4a",
            )
        if not self.link.connected and self._connecting_stage >= 4:
            reason = self.link.status_text
            self._log(f"Connection lost: {reason}")
            self.link.disconnect()
            self._on_disconnected(reason)
        elif self.link.connected and self._connecting_stage >= 4:
            msgs = self.link.drain()
            for m in msgs:
                t = m.get("type")
                if t in ("ack", "error") and self._handle_workflow_message(m):
                    continue
                if (
                    t == "ack"
                    and self._operation_mode == "theory"
                    and self._theory_stop_sequence is not None
                    and m.get("seq") == self._theory_stop_sequence
                    and m.get("message") == "stop_requested"
                ):
                    self._mark_theory_complete()
                    continue
                if t == "telemetry":
                    self._latest_loop_us = int(m.get("loop_us", 0) or 0)
                    self._latest_dropped_frames = int(m.get("dropped", 0) or 0)
                    self._accept_live_state(m)
                    fault = m.get("fault", "none")
                    if fault != "none":
                        self._handle_runtime_fault(str(fault))
                    else:
                        self._accept_theory_telemetry(m)
                        self._accept_snapshot(m)
                elif t == "state":
                    self._latest_loop_us = int(m.get("loop_us", 0) or 0)
                    self._latest_dropped_frames = int(m.get("dropped", 0) or 0)
                    self._accept_live_state(m)
                    fault = m.get("fault", "none")
                    if fault != "none":
                        self._handle_runtime_fault(str(fault))
                    else:
                        self._accept_theory_telemetry(m)
                elif t == "snapshot":
                    self._accept_snapshot(m)
                elif t == "diag":
                    text = m.get("text", "")
                    if "[FAULT]" in text:
                        self._status_var.set(f"⚠ {text}")
                        self._log(f"⚠ Diag: {text}")
                elif t == "error":
                    self._status_var.set(f"⚠ Firmware error: {m.get('message', '?')}")
                    self._log(f"⚠ Error: {m.get('message', '?')}")
                elif t == "hello":
                    self._fw_mode = str(m.get("mode", self._fw_mode))
                    if self._operation_mode == "theory" and self._fw_mode == "idle":
                        try:
                            self.link.send({"cmd": "stop"})
                        except Exception:
                            pass
                        self._mark_theory_complete()

            if (
                self._workflow_current is not None
                and time.monotonic() > self._workflow_deadline
            ):
                self._abort_workflow(
                    f"{self._workflow_current['name']}等待固件确认超时"
                )

            now = time.monotonic()
            current = self._workflow_current
            if (
                current is not None
                and current["payload"].get("cmd") == "stop"
                and now - self._workflow_last_stop_retry
                >= STOP_RETRY_INTERVAL_S
            ):
                retry = dict(current["payload"])
                retry["seq"] = current["seq"]
                self.link.send(retry)
                self._workflow_last_stop_retry = now

            if (
                self._operation_mode == "theory"
                and self._theory_stop_deadline > 0.0
                and now >= self._theory_stop_deadline
                and now - self._theory_last_stop_send
                >= STOP_RETRY_INTERVAL_S
            ):
                stop_payload = {"cmd": "stop"}
                if self._theory_stop_sequence is not None:
                    stop_payload["seq"] = self._theory_stop_sequence
                self._theory_stop_sequence = self.link.send(stop_payload)
                self._theory_last_stop_send = now
                self._status_var.set(
                    "THEORY 已到运行时限 · 正在重复发送停车请求"
                )

        self.after(UI_REFRESH_MS, self._refresh_ui)

    # ==================================================================
    # Shutdown
    # ==================================================================

    def _on_close(self) -> None:
        self._flush_snapshot_cache(force=True)
        self._connecting_stage = 0
        self._reset_workflow()
        self._operation_mode = "idle"
        self._theory_stop_deadline = 0.0
        self._theory_last_stop_send = 0.0
        self._theory_stop_sequence = None
        self._stop_connect_tick()
        self._stop_control_loop()
        self._stop_keepalive()
        if self.link.connected:
            try:
                self.link.send({"cmd": "stop"})
            except Exception:
                pass
        self.link.disconnect()
        if self._plot_job:
            self.after_cancel(self._plot_job)
            self._plot_job = None
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    app = JoystickApp()

    app.update_idletasks()
    w = app.winfo_width()
    h = app.winfo_height()
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    app.geometry(f"+{x}+{y}")

    app.mainloop()


if __name__ == "__main__":
    main()
