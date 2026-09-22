import numpy as np
import pytest

from mimo_car_studio.synthesis import (
    IntegralSnapshotDataset,
    RawTimeSeriesDataset,
    SnapshotDataset,
    estimate_validation_residual,
    route_ii_threshold,
    smooth_sign_basis,
    synthesize_block,
)


def make_dataset(samples: int = 120, seed: int = 8) -> SnapshotDataset:
    rng = np.random.default_rng(seed)
    dataset = SnapshotDataset()
    for _ in range(samples):
        dataset.append(
            {
                "zdot2": rng.normal(size=2),
                "y2": rng.normal(size=8),
                "x3": rng.normal(size=2),
                "zdot3": rng.normal(size=2),
                "y3": rng.normal(size=8),
                "x4": rng.normal(size=2),
            }
        )
    return dataset


def test_synthesis_matches_route_ii_null_space_formula() -> None:
    dataset = make_dataset()
    matrices = dataset.matrices()
    z = matrices["zdot2"]
    y = matrices["y2"]
    threshold = route_ii_threshold(z, y, 0.004, 0.8)
    kappa = threshold.kappa_min + 0.5
    result = synthesize_block(z, y, matrices["x3"], kappa, 0.004, 0.8)
    c = np.vstack([np.eye(2), np.zeros((6, 2))])
    y_dagger = y.T @ np.linalg.solve(y @ y.T, np.eye(8))
    projector = np.eye(y.shape[1]) - y_dagger @ y
    projected = z @ projector
    v_selector = projector @ projected.T @ np.linalg.solve(
        projected @ projected.T, np.eye(2)
    )
    p_expected = y_dagger @ c - kappa * v_selector
    assert result.rank == 10
    assert np.allclose(result.p_solution, p_expected, atol=1.0e-12)
    assert result.match_residual < 1.0e-10
    assert result.kappa_min == pytest.approx(threshold.kappa_min)
    assert result.spectral_margin > 0.0
    assert result.xi_max <= 0.0


def test_sign_basis_matches_deployed_sgn() -> None:
    values = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    assert np.array_equal(smooth_sign_basis(values), np.sign(values))
    assert smooth_sign_basis(np.array([0.0]))[0] == pytest.approx(0.0)


def test_dataset_roundtrip(tmp_path) -> None:
    dataset = make_dataset(samples=20)
    path = tmp_path / "snapshots.npz"
    dataset.save(path)
    loaded = SnapshotDataset.load(path)
    assert len(loaded) == len(dataset)
    for key in SnapshotDataset.REQUIRED:
        assert np.array_equal(loaded.matrices()[key], dataset.matrices()[key])


def test_rank_deficiency_is_not_accepted() -> None:
    z = np.zeros((2, 20))
    y = np.zeros((8, 20))
    x = np.zeros((2, 20))
    with pytest.raises(ValueError, match="rank deficient"):
        synthesize_block(z, y, x, 1.0, 0.004, 0.8)


def test_zero_design_disturbance_bound_is_rejected_by_paper_assumption() -> None:
    dataset = make_dataset()
    matrices = dataset.matrices()
    with pytest.raises(ValueError, match="positive"):
        synthesize_block(
            matrices["zdot2"], matrices["y2"], matrices["x3"],
            1.0, 0.0, 0.8,
        )


def test_integral_snapshot_roundtrip_preserves_windows_and_raw_plot_sources(
    tmp_path,
) -> None:
    dataset = IntegralSnapshotDataset()
    segment = dataset.start_segment(
        {"source": "firmware_integral_v3_tanh_gyro_gate", "gyro_deadband": 0.040}
    )
    for index in range(24):
        row = {
            key: np.full(length, index + offset, dtype=float)
            for offset, (key, length) in enumerate(
                IntegralSnapshotDataset.EXPECTED.items()
            )
        }
        row["velocity_raw"] = np.array([index, -index], dtype=float)
        row["current_raw"] = np.array([0.1 * index, -0.1 * index])
        dataset.append_integral(
            row,
            t_us=1_000_000 + 200_000 * index,
            window_s=0.20,
            segment=segment,
        )
    path = tmp_path / "integral_v3.npz"
    dataset.save(path)
    loaded = IntegralSnapshotDataset.load(path)
    assert len(loaded) == 24
    assert (
        loaded.segment_metadata[segment]["source"]
        == "firmware_integral_v3_tanh_gyro_gate"
    )
    assert loaded.diagnostics().median_dt_s == pytest.approx(0.20)
    for key in (*IntegralSnapshotDataset.REQUIRED, *IntegralSnapshotDataset.OPTIONAL):
        assert np.array_equal(loaded.matrices()[key], dataset.matrices()[key])


def test_integral_loader_rejects_old_raw_time_series_cache(tmp_path) -> None:
    raw = RawTimeSeriesDataset()
    segment = raw.start_segment()
    raw.append_raw(
        t_us=1_000_000,
        velocity=(0.0, 0.0),
        current=(0.0, 0.0),
        voltage=(0.0, 0.0),
        segment=segment,
    )
    path = tmp_path / "raw_v2.npz"
    raw.save(path)
    with pytest.raises(ValueError, match="必须用积分快照固件重新采集"):
        IntegralSnapshotDataset.load(path)


def test_validation_residual_uses_independent_interleaved_windows() -> None:
    rng = np.random.default_rng(12)
    samples = 40
    y = rng.normal(size=(8, samples))
    x = rng.normal(size=(2, samples))
    model_a = rng.normal(size=(2, 8))
    model_b = rng.normal(size=(2, 2))
    zdot = model_a @ y + model_b @ x
    validation = np.arange(3, samples, 4)
    zdot[:, validation] += np.array([[0.3], [0.4]])
    rms, maximum = estimate_validation_residual(zdot, y, x)
    assert maximum == pytest.approx(0.5, abs=1.0e-12)
    assert rms == pytest.approx(0.5, abs=1.0e-12)


def test_route_ii_rejects_kappa_at_or_below_theorem_minimum() -> None:
    dataset = make_dataset(samples=120, seed=18)
    matrices = dataset.matrices()
    threshold = route_ii_threshold(
        matrices["zdot2"], matrices["y2"], 0.004, 0.8
    )
    with pytest.raises(ValueError, match="kappa>"):
        synthesize_block(
            matrices["zdot2"], matrices["y2"], matrices["x3"],
            threshold.kappa_min, 0.004, 0.8,
        )


def test_raw_time_series_roundtrip_and_consistent_offline_derivative(tmp_path) -> None:
    dataset = RawTimeSeriesDataset()
    segment = dataset.start_segment({"firmware": "test", "source": "pid_path_raw_v2"})
    t = np.arange(0.0, 2.0, 0.02)
    for value in t:
        dataset.append_raw(
            t_us=1_000_000 + int(value * 1.0e6),
            velocity=(0.08 * np.sin(0.7 * value), 0.4 * np.cos(0.5 * value)),
            current=(0.12 * np.sin(1.1 * value), 0.10 * np.cos(0.9 * value)),
            voltage=(2.0 * np.sin(0.8 * value), 1.8 * np.cos(0.6 * value)),
            segment=segment,
        )
    path = tmp_path / "raw_v2.npz"
    dataset.save(path)
    loaded = RawTimeSeriesDataset.load(path)
    matrices = loaded.matrices()
    recovered_vdot = matrices["zdot2"] + matrices["y2"][6:8]
    expected_vdot = 0.08 * 0.7 * np.cos(0.7 * t)

    assert len(loaded) == len(dataset)
    assert loaded.segment_count == 1
    assert loaded.segment_metadata[0]["firmware"] == "test"
    assert np.max(np.abs(recovered_vdot[0, 5:-5] - expected_vdot[5:-5])) < 2.0e-3
    # The yaw derivative is supplied by its tracking observer, not by a raw
    # finite difference.  It must remain bounded and share the yaw signal's
    # sign over this smooth reference motion.
    assert np.max(np.abs(recovered_vdot[1])) < 0.25
    assert np.mean(recovered_vdot[1, 10:] * (-np.sin(0.5 * t[10:]))) > 0.0
    assert loaded.diagnostics().median_dt_s == pytest.approx(0.02, abs=1.0e-9)


def test_raw_loader_rejects_legacy_snapshot_schema(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    make_dataset(samples=20).save(path)
    with pytest.raises(ValueError, match="旧快照格式"):
        RawTimeSeriesDataset.load(path)


def test_raw_time_series_drops_adjacent_duplicate_timestamp() -> None:
    dataset = RawTimeSeriesDataset()
    segment = dataset.start_segment()
    sample = {
        "t_us": 1_000_000,
        "velocity": (0.01, 0.02),
        "current": (0.03, 0.04),
        "voltage": (1.0, -1.0),
        "segment": segment,
    }
    dataset.append_raw(**sample)
    dataset.append_raw(**sample)
    assert len(dataset) == 1


def test_invalid_segment_can_be_discarded_without_touching_other_data() -> None:
    dataset = RawTimeSeriesDataset()
    keep = dataset.start_segment({"name": "keep"})
    discard = dataset.start_segment({"name": "discard"})
    for index in range(8):
        for segment in (keep, discard):
            dataset.append_raw(
                t_us=1_000_000 + index * 20_000,
                velocity=(0.05, 0.1),
                current=(0.02, 0.03),
                voltage=(1.0, 1.2),
                segment=segment,
            )
    assert dataset.discard_segment(discard) == 8
    assert len(dataset) == 8
    assert set(dataset.segment_metadata) == {keep}


def test_one_wheel_odometry_segment_is_rejected() -> None:
    dataset = RawTimeSeriesDataset()
    segment = dataset.start_segment()
    track_width = 0.2035
    t = np.arange(0.0, 2.0, 0.02)
    omega = 0.8 * np.sin(1.3 * t)
    forward = -0.5 * track_width * omega
    for index, value in enumerate(t):
        dataset.append_raw(
            t_us=1_000_000 + int(value * 1.0e6),
            velocity=(forward[index], omega[index]),
            current=(0.04, 0.05),
            voltage=(3.0, 3.2),
            segment=segment,
        )
    with pytest.raises(ValueError, match="右轮速度近似为零"):
        dataset.matrices()
