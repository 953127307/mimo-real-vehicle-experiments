from __future__ import annotations

import math

import numpy as np
import pytest

from run_pid_baseline_headless import (
    CIRCLE_NU_RAD_S,
    CIRCLE_RADIUS_M,
    CIRCLE_SPEED_MPS,
    RUN_DURATION_S,
    baseline_config,
    body_frame_error,
    center_start_circle_reference,
)


def test_center_start_circle_reference_matches_campaign_shape3():
    pose0, velocity0 = center_start_circle_reference(0.0)
    assert pose0[0] == pytest.approx(CIRCLE_RADIUS_M)
    assert pose0[1] == pytest.approx(0.0)
    assert pose0[2] == pytest.approx(math.pi / 2.0)
    assert velocity0[0] == pytest.approx(CIRCLE_SPEED_MPS)
    assert velocity0[1] == pytest.approx(CIRCLE_NU_RAD_S)

    for t in (0.7, 3.0, 12.5, RUN_DURATION_S):
        pose, velocity = center_start_circle_reference(t)
        phase = CIRCLE_NU_RAD_S * t
        assert pose[0] == pytest.approx(CIRCLE_RADIUS_M * math.cos(phase))
        assert pose[1] == pytest.approx(CIRCLE_RADIUS_M * math.sin(phase))
        assert pose[2] == pytest.approx(math.pi / 2.0 + phase)
        assert velocity[0] == pytest.approx(CIRCLE_SPEED_MPS)
        assert velocity[1] == pytest.approx(CIRCLE_NU_RAD_S)
        # The reference point stays on the campaign circle at all times.
        assert math.hypot(pose[0], pose[1]) == pytest.approx(CIRCLE_RADIUS_M)


def test_run_duration_matches_campaign():
    assert RUN_DURATION_S == pytest.approx(2.0 * math.pi / 0.30 + 3.0)


def test_baseline_config_declares_campaign_task():
    config = baseline_config()
    assert config["run_duration"] == RUN_DURATION_S
    assert config["hold_duration"] == 0.0
    assert config["ref_shape"] == 3.0
    assert config["outer_preview_horizon"] == 1.3
    assert config["u_max"] == 4.0
    # User decision 2026-09-17: the E4 ceiling matches the common 4 V bound.
    assert config["pid_base_limit_frac"] == 1.0


def test_body_frame_error_geometry():
    pose = np.asarray([0.1, -0.2, 0.5])
    reference = np.asarray([0.4, 0.1, 0.9])
    error = body_frame_error(reference, pose)
    delta = reference[:2] - pose[:2]
    cosine, sine = math.cos(0.5), math.sin(0.5)
    assert error[0] == pytest.approx(cosine * delta[0] + sine * delta[1])
    assert error[1] == pytest.approx(-sine * delta[0] + cosine * delta[1])
    assert abs(error[2]) == pytest.approx(0.4)
