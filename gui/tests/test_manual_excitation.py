from __future__ import annotations

import math

import numpy as np
import pytest

from joystick_car import (
    GUIDANCE_DEFAULTS,
    MOTION_REFERENCE_DEFAULTS,
    PID_COLLECTION_PRESETS,
    TRAJECTORY_PRESETS,
    PidPathController,
    compute_manual_voltages,
    pid_collection_reference,
    should_collect_snapshot,
    stacked_matrix_quality,
    theory_config_for_preset,
    validate_gyro_deadband,
    validate_motor_deadzone,
    validate_theory_safety_limits,
    without_zero_voltage_samples,
)
from mimo_car_studio.synthesis import SnapshotDataset


def test_manual_command_is_pure_driver_voltage() -> None:
    right, left = compute_manual_voltages(
        {"up", "left"},
        u_max=4.0,
        voltage_level=3.0,
    )

    assert right == pytest.approx(3.0)
    assert left == pytest.approx(0.0)


def test_manual_voltage_level_cannot_exceed_firmware_limit() -> None:
    assert compute_manual_voltages(
        {"up"},
        u_max=3.5,
        voltage_level=4.0,
    ) == pytest.approx((3.5, 3.5))


def test_deadman_stop_outputs_zero_voltage() -> None:
    assert compute_manual_voltages(
        set(),
        u_max=4.0,
        voltage_level=4.0,
    ) == (0.0, 0.0)


def test_stacked_matrix_quality_detects_full_rank_and_deficiency() -> None:
    identity = np.eye(10)
    rank, sigma_min, condition = stacked_matrix_quality(identity[:2], identity[2:])
    assert rank == 10
    assert sigma_min == pytest.approx(1.0)
    assert condition == pytest.approx(1.0)

    deficient = identity.copy()
    deficient[-1] = deficient[-2]
    rank, _sigma_min, condition = stacked_matrix_quality(
        deficient[:2], deficient[2:]
    )
    assert rank == 9
    assert np.isinf(condition)


@pytest.mark.parametrize("preset_name", tuple(TRAJECTORY_PRESETS))
def test_theory_trajectory_presets_respect_firmware_limits(
    preset_name: str,
) -> None:
    config = theory_config_for_preset(preset_name)

    assert 0.1 <= config["run_duration"] <= 600.0
    assert config["ref_a"] > 0.0
    assert config["ref_b"] >= 0.0
    assert config["ref_nu"] > 0.0
    assert config["derivative_tau"] > 0.0
    assert config["gyro_tau"] > 0.0
    assert config["velocity_tau"] > 0.0
    assert config["current_tau"] == pytest.approx(0.040)
    assert config["tau1"] > 0.0
    assert config["tau2"] > 0.0
    assert config["motor_deadzone_v"] == pytest.approx(0.5)
    assert config["theory_max_position_error"] == pytest.approx(0.20)
    assert config["theory_max_heading_error"] == pytest.approx(math.radians(60.0))
    assert config["theory_max_z2"] == pytest.approx(2.0)
    assert config["theory_max_z3"] == pytest.approx(2.0)
    for key, expected in MOTION_REFERENCE_DEFAULTS.items():
        assert config[key] == pytest.approx(expected)
    assert config["outer_max_v"] >= config["motion_max_v"]
    assert config["outer_max_omega"] >= config["motion_max_omega"]


def test_theory_defaults_enable_adaptation_and_expand_outer_envelope() -> None:
    config = theory_config_for_preset(next(iter(TRAJECTORY_PRESETS)))
    assert config["r2"] == pytest.approx(0.5)
    assert config["r3"] == pytest.approx(0.5)
    assert config["outer_max_v"] == pytest.approx(0.30)
    assert config["outer_max_omega"] == pytest.approx(3.00)


def test_unknown_theory_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown trajectory preset"):
        theory_config_for_preset("not-a-preset")


@pytest.mark.parametrize("preset_name", tuple(TRAJECTORY_PRESETS))
def test_theory_config_uses_cartesian_guidance_parameters(
    preset_name: str,
) -> None:
    config = theory_config_for_preset(preset_name)
    for key, expected in GUIDANCE_DEFAULTS.items():
        assert config[key] == pytest.approx(expected)
    assert "kx" not in config
    assert "ky" not in config
    assert "kth" not in config
    assert not any(key.startswith("frenet_") for key in config)


def test_cartesian_guidance_parameters_are_customizable() -> None:
    config = theory_config_for_preset(
        next(iter(TRAJECTORY_PRESETS)),
        outer_kp=0.9,
        outer_vbar=0.15,
        outer_ktheta=1.7,
        outer_preview_horizon=0.8,
        outer_capture_radius=0.02,
        outer_blend_radius=0.05,
    )
    assert config["outer_kp"] == pytest.approx(0.9)
    assert config["outer_vbar"] == pytest.approx(0.15)
    assert config["outer_ktheta"] == pytest.approx(1.7)
    assert config["outer_preview_horizon"] == pytest.approx(0.8)
    assert config["outer_capture_radius"] == pytest.approx(0.02)
    assert config["outer_blend_radius"] == pytest.approx(0.05)


def test_theory_filter_time_constants_are_customizable() -> None:
    preset_name = next(iter(TRAJECTORY_PRESETS))
    config = theory_config_for_preset(preset_name, tau1=0.055, tau2=0.025)
    assert config["tau1"] == pytest.approx(0.055)
    assert config["tau2"] == pytest.approx(0.025)


def test_motor_deadzone_is_customizable_and_transmitted_with_theory() -> None:
    preset_name = next(iter(TRAJECTORY_PRESETS))
    config = theory_config_for_preset(preset_name, motor_deadzone_v=0.75)
    assert config["motor_deadzone_v"] == pytest.approx(0.75)
    assert validate_motor_deadzone(0.0) == pytest.approx(0.0)


def test_theory_safety_limits_are_customizable_and_transmitted() -> None:
    preset_name = next(iter(TRAJECTORY_PRESETS))
    config = theory_config_for_preset(
        preset_name,
        theory_max_position_error=0.25,
        theory_max_heading_error=math.radians(75.0),
        theory_max_z2=2.5,
        theory_max_z3=3.0,
    )
    assert config["theory_max_position_error"] == pytest.approx(0.25)
    assert config["theory_max_heading_error"] == pytest.approx(math.radians(75.0))
    assert config["theory_max_z2"] == pytest.approx(2.5)
    assert config["theory_max_z3"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "limits",
    [
        (0.0, math.radians(60.0), 2.0, 2.0),
        (1.01, math.radians(60.0), 2.0, 2.0),
        (0.2, 0.01, 2.0, 2.0),
        (0.2, math.pi, 2.0, 2.0),
        (0.2, math.radians(60.0), 0.0, 2.0),
        (0.2, math.radians(60.0), 2.0, float("nan")),
    ],
)
def test_theory_safety_limits_reject_invalid_values(
    limits: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="上限|阈值"):
        validate_theory_safety_limits(*limits)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan")])
def test_gyro_deadband_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="陀螺角速度阈值"):
        validate_gyro_deadband(value)


@pytest.mark.parametrize("value", [-0.01, 4.0, float("inf"), float("nan")])
def test_motor_deadzone_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="电压死区"):
        validate_motor_deadzone(value)


@pytest.mark.parametrize("tau1,tau2", [(0.0, 0.03), (0.04, 0.0), (1.1, 0.03)])
def test_theory_filter_time_constants_reject_unsafe_values(
    tau1: float,
    tau2: float,
) -> None:
    with pytest.raises(ValueError, match="0.005–1.0"):
        theory_config_for_preset(
            next(iter(TRAJECTORY_PRESETS)),
            tau1=tau1,
            tau2=tau2,
        )


def test_snapshot_collection_is_exclusive_to_ready_pid_path_mode() -> None:
    assert not should_collect_snapshot("manual")
    assert should_collect_snapshot("pid_collect")
    assert not should_collect_snapshot("pid_collect", controller_ready=False)


@pytest.mark.parametrize("preset", tuple(PID_COLLECTION_PRESETS.values()))
def test_pid_collection_reference_starts_at_zero_with_aligned_heading(
    preset: dict[str, float],
) -> None:
    initial = pid_collection_reference(0.0, preset)
    moving = pid_collection_reference(4.0, preset)

    assert initial.pose == pytest.approx((0.0, 0.0, 0.0))
    assert initial.velocity == pytest.approx((0.0, 0.0))
    assert moving.velocity[0] > 0.0
    assert np.all(np.isfinite((*moving.pose, *moving.velocity)))


def test_pid_path_controller_outputs_bounded_differential_voltage() -> None:
    controller = PidPathController()
    preset = next(iter(PID_COLLECTION_PRESETS.values()))
    reference = pid_collection_reference(6.0, preset)
    right, left, metrics = controller.step(
        pose=np.zeros(3),
        velocity=np.zeros(2),
        reference=reference,
        dt=0.04,
        u_max=7.2,
    )

    assert abs(right) <= 7.2 * 0.60
    assert abs(left) <= 7.2 * 0.60
    assert right != pytest.approx(left)
    assert set(metrics) == {"e_x", "e_y", "e_theta", "v_command", "omega_command"}


@pytest.mark.parametrize(
    ("right_gain", "left_gain", "motor_tau", "deadzone"),
    (
        (0.0250, 0.0215, 0.18, 0.55),
        (0.0210, 0.0260, 0.22, 0.65),
        (0.0280, 0.0240, 0.12, 0.45),
    ),
)
def test_pid_collection_tracks_asymmetric_mock_path(
    right_gain: float,
    left_gain: float,
    motor_tau: float,
    deadzone: float,
) -> None:
    controller = PidPathController()
    preset = next(iter(PID_COLLECTION_PRESETS.values()))
    pose = np.zeros(3)
    velocity = np.zeros(2)
    wheel_velocity = np.zeros(2)
    errors: list[float] = []
    voltages: list[tuple[float, float]] = []
    peak_voltage = 0.0
    dt = 0.04
    for index in range(int(preset["duration"] / dt)):
        elapsed = index * dt
        reference = pid_collection_reference(elapsed, preset)
        right, left, _metrics = controller.step(
            pose, velocity, reference, dt, 7.2, 0.2035
        )
        voltages.append((right, left))
        peak_voltage = max(peak_voltage, abs(right), abs(left))
        motor_voltage = np.asarray((right, left), dtype=float)
        effective_voltage = np.sign(motor_voltage) * np.maximum(
            np.abs(motor_voltage) - deadzone, 0.0
        )
        wheel_target = effective_voltage * np.asarray((right_gain, left_gain))
        wheel_velocity += (dt / motor_tau) * (wheel_target - wheel_velocity)
        velocity[:] = (
            0.5 * (wheel_velocity[0] + wheel_velocity[1]),
            (wheel_velocity[0] - wheel_velocity[1]) / 0.2035,
        )
        pose[2] = (pose[2] + velocity[1] * dt + math.pi) % (2.0 * math.pi) - math.pi
        pose[0] += velocity[0] * math.cos(pose[2]) * dt
        pose[1] += velocity[0] * math.sin(pose[2]) * dt
        errors.append(math.hypot(reference.pose[0] - pose[0], reference.pose[1] - pose[1]))

    assert np.sqrt(np.mean(np.square(errors[100:]))) < 0.01
    assert peak_voltage < 5.5
    voltage_array = np.asarray(voltages[100:])
    assert np.ptp(voltage_array[:, 0]) > 1.0
    assert np.ptp(voltage_array[:, 1]) > 1.0


def test_theory_and_collection_use_validated_derivative_observer_tau() -> None:
    for preset_name in TRAJECTORY_PRESETS:
        assert theory_config_for_preset(preset_name)["derivative_tau"] == pytest.approx(0.10)


def test_zero_voltage_samples_are_excluded_from_saved_and_synthesis_data() -> None:
    dataset = SnapshotDataset()
    for voltage in ([0.0, 0.0], [1.0, 0.0], [0.0, -1.0]):
        dataset.append(
            {
                "zdot2": [0.0, 0.0],
                "y2": [0.0] * 8,
                "x3": [0.0, 0.0],
                "zdot3": [0.0, 0.0],
                "y3": [0.0] * 8,
                "x4": voltage,
            }
        )

    filtered, removed = without_zero_voltage_samples(dataset)

    assert removed == 1
    assert len(filtered) == 2
    assert np.array_equal(
        filtered.matrices()["x4"],
        np.asarray([[1.0, 0.0], [0.0, -1.0]]),
    )
