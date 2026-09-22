from __future__ import annotations

from pathlib import Path

import pytest

from run_comparison_headless import without_initial_nonlinear_compensation
from joystick_car import (
    COMPARISON_CASES,
    PID_BENCHMARK_DEFAULTS,
    TRAJECTORY_PRESETS,
    comparison_case_runtime,
    theory_config_for_preset,
)


def test_adaptive_only_ablation_preserves_feedback_columns() -> None:
    payload = {
        "w2": [float(value) for value in range(16)],
        "w3": [float(value + 20) for value in range(16)],
        "route": "route_ii",
    }

    modified = without_initial_nonlinear_compensation(payload)

    assert modified["w2"] == [0.0, 1.0, *([0.0] * 6), 8.0, 9.0, *([0.0] * 6)]
    assert modified["w3"] == [20.0, 21.0, *([0.0] * 6), 28.0, 29.0, *([0.0] * 6)]
    assert payload["w2"] == [float(value) for value in range(16)]
    assert modified["route"] == "route_ii"


def test_comparison_cases_have_stable_ids_and_weight_requirements() -> None:
    cases = list(COMPARISON_CASES)
    e1 = comparison_case_runtime(cases[0], 0.5, 0.6)
    e2 = comparison_case_runtime(cases[1], 0.5, 0.6)
    e3 = comparison_case_runtime(cases[2], 0.5, 0.6)

    assert (e1["case_id"], e1["requires_weights"], e1["r2"], e1["r3"]) == (
        1,
        True,
        0.5,
        0.6,
    )
    assert (e2["case_id"], e2["requires_weights"], e2["r2"], e2["r3"]) == (
        2,
        True,
        0.0,
        0.0,
    )
    assert (e3["case_id"], e3["requires_weights"], e3["r2"], e3["r3"]) == (
        3,
        False,
        0.0,
        0.0,
    )


def test_e3_config_transmits_all_pid_parameters() -> None:
    config = theory_config_for_preset(
        next(iter(TRAJECTORY_PRESETS)),
        comparison_case=3,
        pid_velocity_kp=0.31,
        pid_current_kp=11.0,
        pid_current_ref_max=0.28,
    )

    assert config["comparison_case"] == 3.0
    assert config["pid_velocity_kp"] == pytest.approx(0.31)
    assert config["pid_current_kp"] == pytest.approx(11.0)
    assert config["pid_current_ref_max"] == pytest.approx(0.28)
    assert set(PID_BENCHMARK_DEFAULTS) <= set(config)


@pytest.mark.parametrize("case", [0, 1.5, 4, float("nan")])
def test_invalid_comparison_case_is_rejected(case: float) -> None:
    with pytest.raises(ValueError, match="E1、E2 或 E3"):
        theory_config_for_preset(
            next(iter(TRAJECTORY_PRESETS)),
            comparison_case=case,
        )


@pytest.mark.parametrize("current_limit", [0.0, 1.01, float("nan")])
def test_invalid_pid_current_reference_limit_is_rejected(
    current_limit: float,
) -> None:
    with pytest.raises(ValueError, match="电流参考限幅|有限数值"):
        theory_config_for_preset(
            next(iter(TRAJECTORY_PRESETS)),
            comparison_case=3,
            pid_current_ref_max=current_limit,
        )


def test_pc_trace_export_records_comparison_identity_and_configuration() -> None:
    pc_source = (
        Path(__file__).resolve().parents[1] / "joystick_car.py"
    ).read_text(encoding="utf-8")

    assert '"comparison_case": np.asarray([self._active_comparison_case])' in pc_source
    assert '"comparison_name": np.asarray([self._active_comparison_name])' in pc_source
    assert '"runtime_config_json": np.asarray(' in pc_source
    assert 'f"theory_trace_{case_slug}_' in pc_source
