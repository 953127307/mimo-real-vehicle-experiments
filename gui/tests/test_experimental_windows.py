"""Offline checks for publication metrics; no vehicle connection."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# 自包含目录：生成器就在本文件夹的上一级（real_vehicle_experiments/）。
SCRIPT = Path(__file__).resolve().parents[2] / "generate_experimental_results.py"
spec = importlib.util.spec_from_file_location("experimental_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_final_rmse_does_not_cancel_opposite_errors():
    t = np.array([0., 8., 9., 9.5, 10.])
    pose = np.zeros((5, 3))
    pose[:, 0] = [0., 0., 0., 1., -1.]
    pose[:, 2] = [0., 0., 0., .2, -.2]
    result = metrics.run_metrics(
        {"t": t, "pose": pose, "reference": np.zeros((5, 3)),
         "z2": np.zeros((5, 2)), "z3": np.zeros((5, 2))}, 9.)
    assert result["final_position_rmse_m"] == pytest.approx(1.)
    assert result["final_heading_rmse_rad"] == pytest.approx(.2)


def test_terminal_boundary_must_be_inside_record():
    rows = {"t": np.array([0., 8., 10.]), "pose": np.zeros((3, 3)), "reference": np.zeros((3, 3))}
    for boundary in [8., 10., 11.]:
        with pytest.raises(ValueError, match="inside"):
            metrics.run_metrics(rows, boundary)


def test_frozen_selection_ignores_newer_unlisted_files(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(metrics, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(metrics, "DATASET_PATH", tmp_path / "data.npz")
    monkeypatch.setattr(metrics, "WEIGHTS_PATH", tmp_path / "weights.json")
    monkeypatch.setattr(metrics, "TRIALS_PER_CASE", 1)
    monkeypatch.setattr(metrics, "sha256", lambda path: "expected")

    ablated_payload = {"kappa2": 20.0, "kappa3": 30.0, "w2": [1, 0] * 8, "w3": [1, 0] * 8}
    ablated_row = [1, 0] + [0] * 6
    ablated_e3 = dict(ablated_payload,
                      w2=ablated_row + ablated_row, w3=ablated_row + ablated_row)

    def fake_peek(path):
        name = path.name
        if "e3" in name:
            return {"reason": "complete", "weights": ablated_e3,
                    "adaptive_only": True}
        if "e4" in name:
            return {"reason": "complete",
                    "weights": {"used_by_controller": False},
                    "adaptive_only": False}
        return {"reason": "complete", "weights": ablated_payload,
                "adaptive_only": False}

    monkeypatch.setattr(metrics, "peek_trace", fake_peek)
    manifest = {"training_dataset": {"name": "data.npz", "sha256": "expected"},
                "validated_weights": {"name": "weights.json", "sha256": "expected"},
                "completed": {
                    "E1": [{"name": "frozen_e1.npz", "sha256": "expected"}],
                    "E2": [{"name": "frozen_e2.npz", "sha256": "expected"}],
                    "E3": [{"name": "frozen_e3.npz", "sha256": "expected"}],
                    "E4": [{"name": "frozen_e4.npz", "sha256": "expected"}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "theory_trace_new_complete.npz").touch()
    assert metrics.select_official_traces(ablated_payload) == {
        "E1": [tmp_path / "frozen_e1.npz"],
        "E2": [tmp_path / "frozen_e2.npz"],
        "E3": [tmp_path / "frozen_e3.npz"],
        "E4": [tmp_path / "frozen_e4.npz"]}
    monkeypatch.setattr(metrics, "sha256", lambda path: "changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        metrics.select_official_traces(ablated_payload)
