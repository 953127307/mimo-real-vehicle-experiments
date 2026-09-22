import json

import numpy as np
import pytest

from joystick_car import (
    load_validated_weights_cache,
    save_validated_weights_cache,
)
from mimo_car_studio.synthesis import (
    BlockQualification,
    BlockSynthesis,
    QualificationReport,
    SynthesisResult,
)


def make_valid_result() -> SynthesisResult:
    block = BlockSynthesis(
        weights=np.arange(16, dtype=float).reshape(2, 8) / 10.0,
        p_solution=np.zeros((10, 2)),
        q_solution=np.zeros((10, 6)),
        rank=10,
        rows=10,
        condition=2.0,
        match_residual=1.0e-10,
        nominal_residual=2.0e-10,
        xi_max=-0.5,
        kappa=2.0,
        kappa_min=0.5,
        sigma_min=2.0,
        chi=0.1,
        spectral_margin=1.9,
    )
    return SynthesisResult(
        block2=block,
        block3=block,
        samples=240,
        kappa2=2.0,
        kappa3=2.0,
        epsilon2=0.8,
        epsilon3=1.5,
        dbar2=0.004,
        dbar3=0.008,
    )


def make_valid_qualification() -> QualificationReport:
    return QualificationReport(
        block2=BlockQualification(
            design_dbar=0.004,
            residual_rms=0.002,
            residual_max=0.0033,
            observed_residual_covered=True,
            kappa=2.0,
            kappa_min=0.5,
            spectral_margin=1.9,
            certificate_max=-0.5,
            route_ii_feasible=True,
            split_weight_drift=0.05,
            cross_eigen_max=-0.4,
            physical_direction_ok=True,
            controller_direction_ok=True,
        ),
        block3=BlockQualification(
            design_dbar=0.008,
            residual_rms=0.004,
            residual_max=0.0066,
            observed_residual_covered=True,
            kappa=2.0,
            kappa_min=0.5,
            spectral_margin=1.9,
            certificate_max=-0.5,
            route_ii_feasible=True,
            split_weight_drift=0.06,
            cross_eigen_max=-0.5,
            physical_direction_ok=True,
            controller_direction_ok=True,
        ),
    )


def test_validated_weights_cache_roundtrip(tmp_path) -> None:
    path = tmp_path / "last_validated_weights.json"
    result = make_valid_result()
    saved = save_validated_weights_cache(
        path, result, "snapshots.npz", make_valid_qualification()
    )
    payload, loaded = load_validated_weights_cache(path)

    assert payload == result.upload_payload()
    assert loaded["sha256"] == saved["sha256"]
    assert loaded["samples"] == 240
    assert loaded["source_dataset"] == "snapshots.npz"
    assert loaded["schema"] == 8
    assert loaded["operating_config"]["gyro_deadband"] == pytest.approx(0.0)
    assert (
        loaded["qualification"]["method"]
        == "firmware_integral_v3_sgn_gyro_gate_route_ii_v1"
    )


def test_tampered_validated_weights_cache_is_rejected(tmp_path) -> None:
    path = tmp_path / "last_validated_weights.json"
    save_validated_weights_cache(
        path, make_valid_result(), qualification=make_valid_qualification()
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["w2"][0] += 1.0
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="完整性校验失败"):
        load_validated_weights_cache(path)


def test_route_i_schema5_cache_is_isolated(tmp_path) -> None:
    path = tmp_path / "last_validated_weights.json"
    save_validated_weights_cache(
        path, make_valid_result(), qualification=make_valid_qualification()
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema"] = 5
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="版本不受支持"):
        load_validated_weights_cache(path)


def test_failed_synthesis_cannot_be_cached(tmp_path) -> None:
    valid = make_valid_result()
    invalid_block = BlockSynthesis(
        **{**valid.block2.__dict__, "rank": 9}
    )
    invalid = SynthesisResult(
        **{**valid.__dict__, "block2": invalid_block}
    )

    with pytest.raises(ValueError, match="只能保存"):
        save_validated_weights_cache(
            tmp_path / "last_validated_weights.json",
            invalid,
            qualification=make_valid_qualification(),
        )


def test_paper_pass_does_not_require_optional_engineering_diagnostics(tmp_path) -> None:
    path = tmp_path / "last_validated_weights.json"
    result = make_valid_result()
    save_validated_weights_cache(path, result)
    payload, document = load_validated_weights_cache(path)
    assert payload == result.upload_payload()
    assert document["qualification"] is None


def test_residual_coverage_diagnostic_does_not_gate_paper_pass(tmp_path) -> None:
    qualification = make_valid_qualification()
    invalid_block = BlockQualification(
        **{
            **qualification.block2.__dict__,
            "observed_residual_covered": False,
        }
    )
    invalid = QualificationReport(
        block2=invalid_block,
        block3=qualification.block3,
    )
    path = tmp_path / "last_validated_weights.json"
    result = make_valid_result()
    save_validated_weights_cache(path, result, qualification=invalid)
    payload, document = load_validated_weights_cache(path)
    assert payload == result.upload_payload()
    assert document["qualification"]["block2"]["observed_residual_covered"] is False


def test_nonpaper_diagnostics_do_not_override_valid_route_ii_result(tmp_path) -> None:
    qualification = make_valid_qualification()
    invalid_block = BlockQualification(
        **{
            **qualification.block3.__dict__,
            "route_ii_feasible": False,
        }
    )
    invalid = QualificationReport(
        block2=qualification.block2,
        block3=invalid_block,
    )
    path = tmp_path / "last_validated_weights.json"
    result = make_valid_result()
    save_validated_weights_cache(path, result, qualification=invalid)
    payload, _ = load_validated_weights_cache(path)
    assert payload == result.upload_payload()


def test_positive_route_ii_certificate_cannot_be_cached(tmp_path) -> None:
    valid = make_valid_result()
    invalid_block = BlockSynthesis(
        **{**valid.block2.__dict__, "xi_max": 1.0e-6}
    )
    invalid = SynthesisResult(
        **{**valid.__dict__, "block2": invalid_block}
    )
    with pytest.raises(ValueError, match="Route II"):
        save_validated_weights_cache(
            tmp_path / "last_validated_weights.json",
            invalid,
            qualification=make_valid_qualification(),
        )
