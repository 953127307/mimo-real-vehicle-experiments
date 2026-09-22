import importlib.util
from pathlib import Path
import sys

import numpy as np

from joystick_car import GUIDANCE_DEFAULTS, theory_config_for_preset
from mimo_car_studio.synthesis import synthesize_block


def test_four_leaf_clover_preset_is_uploaded_as_shape_two() -> None:
    name = "四叶草（R 0.35 m，约 0.07–0.14 m/s，1 圈）"
    config = theory_config_for_preset(name)
    assert config["ref_shape"] == 2.0
    assert config["ref_a"] == 0.35
    assert config["ref_nu"] == 0.20
    assert config["run_duration"] == 33.5


def load_current_theory():
    path = (Path(__file__).resolve().parents[3] / "numerical_simulation"
            / "current_theory_simulation.py")
    spec = importlib.util.spec_from_file_location("current_theory_for_p4_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pc_synthesis_is_identical_to_current_theory() -> None:
    theory = load_current_theory()
    snapshots = theory.collect_snapshots(K=120)
    cases = ((2, -8.0, 0.004, 0.8), (3, -30.0, 0.008, 3.0))
    for block, pole, dbar, epsilon in cases:
        expected = theory.synthesize_block(
            block, snapshots[block], pole * np.eye(2), dbar, epsilon
        )
        actual = synthesize_block(
            snapshots[block]["Z"],
            snapshots[block]["Y"],
            snapshots[block]["X"],
            expected["route_ii_kappa"],
            dbar,
            epsilon,
        )
        assert np.allclose(actual.weights, expected["route_ii_W"], atol=1.0e-12)
        assert actual.kappa_min == expected["route_ii_kappa_lower"]
        assert actual.xi_max == expected["route_ii_certificate_max"]
        assert actual.spectral_margin == expected["route_ii_spectral_margin"]


def test_bidirectional_outer_loop_keeps_heading_feedback_dissipative() -> None:
    theory = load_current_theory()
    pose = np.array([0.0, 0.0, 0.3])
    reference_pose = np.zeros(3)
    forward, _ = theory.outer_command_from_reference(
        pose, reference_pose, np.array([0.4, 0.0])
    )
    reverse, _ = theory.outer_command_from_reference(
        pose, reference_pose, np.array([-0.4, 0.0])
    )
    assert forward[1] == reverse[1]
    assert forward[1] < 0.0


def test_guidance_defaults_match_the_manuscript_campaign() -> None:
    assert GUIDANCE_DEFAULTS == {
        "outer_kp": 0.80,
        "outer_vbar": 0.50,
        "outer_ktheta": 0.70,
        "outer_preview_horizon": 1.40,
        "outer_capture_radius": 0.010,
        "outer_blend_radius": 0.030,
    }
    # 固件源码不随本目录发布；其默认值一致性检查留在 P4实车系统/pc_app/tests。
