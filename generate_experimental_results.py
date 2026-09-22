#!/usr/bin/env python3
"""Generate the real-vehicle figures and metrics reported in main.tex.

Self-contained: the frozen campaign data lives in ./data.  Following the same
convention as ../numerical_simulation, every invocation first creates a
timestamped run folder ./runs/<YYYYmmdd_HHMMSS>_experimental/, writes the
results JSON and figures inside it, refreshes the ./latest_experiments.txt
pointer, and copies the paper figure into ../submission/figures/.  Pass
--run <folder> to write into an existing run folder instead.  ./results
preserves the archived official output and is no longer written.

The official closed-loop evidence is the kappa=20/30 campaign on the
center-start uniform circle: five completed runs per case (E1, E2, E3) with
preview horizon T_p=1.3 s and firmware 1.7.4, plus the external sampling-PID
baseline E4 on the same task.  Candidate traces are fixed by the accompanying
campaign manifest and checked against the frozen synthesized payload.  All
trajectory errors are reconstructed from the on-board encoder/current
telemetry; they are not external-position ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TRACE_DIR = DATA_DIR / "traces"
RESULTS_DIR = HERE / "results"
SUBMISSION_FIGURES = HERE.parent / "submission" / "figures"
RUNS_DIR = HERE / "runs"
LATEST_POINTER = HERE / "latest_experiments.txt"

DATASET_PATH = DATA_DIR / "integral_pid_K504_20260922_202801.npz"
WEIGHTS_PATH = DATA_DIR / "validated_weights_integral_pid_K504_k20_k30_20260922.json"
MANIFEST_PATH = DATA_DIR / "experimental_campaign_manifest_20260922_202801.json"
TRIALS_PER_CASE = 5
EXPECTED_PREVIEW_S = 1.3
EXPECTED_FIRMWARE = "1.7.4"
EXPECTED_REF_SHAPE = 3.0
EXPECTED_RUN_DURATION_S = 2.0 * math.pi / 0.30 + 3.0
EXPECTED_HOLD_DURATION_S = 0.0


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def time_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    """Return a sampling-density-independent RMS over the recorded interval."""
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if values.shape[0] != time_s.size or time_s.size < 2:
        raise ValueError("time RMS requires at least two aligned samples")
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError("trace time must be nondecreasing")
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0.0:
        raise ValueError("trace duration must be positive")
    squared = np.square(values)
    if squared.ndim > 1:
        squared = np.sum(squared, axis=1)
    return float(np.sqrt(np.trapezoid(squared, time_s) / duration))


def body_frame_tracking_error(rows: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct the target-pose error independently of controller internals."""
    reference = np.asarray(rows["reference"], dtype=float)
    pose = np.asarray(rows["pose"], dtype=float)
    dx = reference[:, 0] - pose[:, 0]
    dy = reference[:, 1] - pose[:, 1]
    cosine = np.cos(pose[:, 2])
    sine = np.sin(pose[:, 2])
    heading_error = (reference[:, 2] - pose[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return np.column_stack(
        (cosine * dx + sine * dy, -sine * dx + cosine * dy, heading_error)
    )


STEADY_WINDOW_START_S = 8.0
# Cruise window ends at T_f = one revolution (2*pi/0.30 s), matching the
# manuscript's SS-RMSE definition and the plotted range.
STEADY_WINDOW_END_S = 2.0 * math.pi / 0.30
FINAL_SETTLE_WINDOW_S = 0.5


def run_metrics(rows: dict[str, np.ndarray], terminal_start_s: float) -> dict[str, float]:
    pose_error = body_frame_tracking_error(rows)
    time_s = np.asarray(rows["t"], dtype=float)
    t_end = float(time_s[-1])
    if not STEADY_WINDOW_START_S < terminal_start_s < t_end:
        raise ValueError("terminal window must be inside the record")
    cruise = (time_s >= STEADY_WINDOW_START_S) & (time_s <= STEADY_WINDOW_END_S)
    terminal = time_s >= terminal_start_s
    tail = time_s >= t_end - FINAL_SETTLE_WINDOW_S
    if int(cruise.sum()) < 2 or int(terminal.sum()) < 2 or int(tail.sum()) < 2:
        raise ValueError("evaluation window requires at least two samples")
    return {
        "position_rmse_m": time_rms(pose_error[:, :2], time_s),
        "heading_rmse_rad": time_rms(pose_error[:, 2], time_s),
        "z2_rms": time_rms(rows["z2"], time_s),
        "z3_rms": time_rms(rows["z3"], time_s),
        "z12_rms": time_rms(
            np.linalg.norm(rows["z2"], axis=1)
            + np.linalg.norm(rows["z3"], axis=1),
            time_s,
        ),
        "cruise_position_rmse_m": time_rms(
            pose_error[cruise, :2], time_s[cruise]
        ),
        "cruise_heading_rmse_rad": time_rms(
            pose_error[cruise, 2], time_s[cruise]
        ),
        "terminal_position_rmse_m": time_rms(
            pose_error[terminal, :2], time_s[terminal]
        ),
        "terminal_heading_rmse_rad": time_rms(
            pose_error[terminal, 2], time_s[terminal]
        ),
        "final_position_rmse_m": time_rms(pose_error[tail, :2], time_s[tail]),
        "final_heading_rmse_rad": time_rms(pose_error[tail, 2], time_s[tail]),
        "duration_s": t_end,
        "samples": int(time_s.size),
    }


def aggregate_metrics(per_run: list[dict[str, float]]) -> dict[str, object]:
    summary: dict[str, object] = {"completed_runs": len(per_run)}
    for key in (
        "position_rmse_m",
        "heading_rmse_rad",
        "z2_rms",
        "z3_rms",
        "z12_rms",
        "cruise_position_rmse_m",
        "cruise_heading_rmse_rad",
        "terminal_position_rmse_m",
        "terminal_heading_rmse_rad",
        "final_position_rmse_m",
        "final_heading_rmse_rad",
    ):
        values = np.asarray([row[key] for row in per_run], dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "sample_sd": (
                float(np.std(values, ddof=1)) if values.size > 1 else None
            ),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def normalized_case(token: object) -> str:
    text = str(token)
    return {"E1": "E1", "1.0": "E1", "1": "E1",
            "E2": "E2", "2.0": "E2", "2": "E2",
            "E3": "E3", "3.0": "E3", "3": "E3",
            "E4": "E4", "4.0": "E4", "4": "E4"}.get(text, text)


def first_record(archive, key: str) -> object | None:
    if key not in archive.files:
        return None
    return np.asarray(archive[key]).reshape(-1)[0]


def peek_trace(path: Path) -> dict[str, object] | None:
    """Read the identification fields of a trace without loading full arrays."""
    with np.load(path, allow_pickle=False) as archive:
        for key in ("reason", "weights_json", "runtime_config_json",
                    "comparison_case", "initial_nonlinear_compensation_enabled"):
            if key not in archive.files:
                return None
        return {
            "case": normalized_case(first_record(archive, "comparison_case")),
            "reason": str(first_record(archive, "reason")),
            "weights": json.loads(
                str(first_record(archive, "weights_json"))
            ),
            "config": json.loads(
                str(first_record(archive, "runtime_config_json"))
            ),
            "adaptive_only": not bool(
                first_record(archive, "initial_nonlinear_compensation_enabled")
            ),
        }


def select_official_traces(payload: dict[str, object]) -> dict[str, list[Path]]:
    """Use the frozen campaign, never directory order or modification time."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for key, path in (("training_dataset", DATASET_PATH), ("validated_weights", WEIGHTS_PATH)):
        if path.name != manifest[key]["name"] or sha256(path) != manifest[key]["sha256"]:
            raise ValueError(f"{key}: frozen campaign identity mismatch")
    names = {
        case: [entry["name"] for entry in entries]
        for case, entries in manifest["completed"].items()
    }
    if set(names) != {"E1", "E2", "E3", "E4"}:
        raise ValueError(f"manifest must freeze exactly E1-E4, got {sorted(names)}")
    selected = {}
    for case, case_names in names.items():
        if len(case_names) != TRIALS_PER_CASE:
            raise ValueError(f"{case}: expected five frozen records")
        selected[case] = []
        for name, entry in zip(case_names, manifest["completed"][case]):
            path = TRACE_DIR / name
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"{path.name}: frozen trace hash mismatch")
            fields = peek_trace(path)
            if fields is None or fields["reason"] != "complete":
                raise ValueError(f"{path.name}: not a complete record")
            if case == "E3":
                validate_adaptive_only_weights(fields["weights"], payload)
                if not fields["adaptive_only"]:
                    raise ValueError("E3 must disable initial nonlinear compensation")
            elif case == "E4":
                if fields["weights"].get("used_by_controller") is not False:
                    raise ValueError(
                        f"{path.name}: E4 must record an unused controller payload"
                    )
            elif fields["weights"] != payload or fields["adaptive_only"]:
                raise ValueError(f"{case}: unexpected deployed weights")
            selected[case].append(path)
    return selected


def representative_run(
    runs_by_case: dict[str, list[dict[str, object]]], case: str
) -> dict[str, np.ndarray]:
    """Return the median run (third by full-record position RMSE) of a case."""
    return runs_by_case[case][2]["rows"]


def steady_error_values(
    rows: dict[str, np.ndarray], column: int, window_end_s: float
) -> np.ndarray:
    error = body_frame_tracking_error(rows)
    window = (rows["t"] >= STEADY_WINDOW_START_S) & (rows["t"] <= window_end_s)
    if column == 0:
        return 1.0e3 * np.linalg.norm(error[window, :2], axis=1)
    return 1.0e3 * error[window, 2]


def plot_experimental_components(
    runs_by_case: dict[str, list[dict[str, object]]],
    terminal_start_s: float,
) -> None:
    """Compose the single 1x4 spanning experimental figure (Fig. 2 layout).

    Four square panels in one row: (a) platform photo, (b) trajectories,
    (c) position error, (d) heading error.  The two error panels keep the
    cruise-window zoom insets and the corresponding zoom-region rectangles.
    """
    cases = ("E1", "E2", "E3", "E4")
    styles = {
        "E1": dict(color=BLUE, linestyle="-"),
        "E2": dict(color=ORANGE, linestyle="--"),
        # The dash-dot period must fit the legend sample (7.2 pt at
        # handlelength 1.2 and fontsize 6); the default "-." period is
        # 7.7 pt, so its sample collapses to a solid bar. Curves and
        # legend proxies both draw from this dict.
        "E3": dict(color="#6A3D9A", dashes=(3.2, 1.15, 0.7, 1.15)),
        "E4": dict(color=GREEN, linestyle=":"),
    }
    rows_by_case = {
        case: representative_run(runs_by_case, case) for case in cases
    }
    one_rev_s = 2.0 * math.pi / 0.30
    plot_end_s = min(terminal_start_s, one_rev_s)

    def steady_limits(column: int) -> tuple[float, float]:
        # Zoom window covers Cases 1-3 only; the Case 4 tail would stretch
        # the insets far beyond the proposed-method band.
        values = [
            steady_error_values(rows_by_case[case], column, plot_end_s)
            for case in cases[:3]
        ]
        joined = np.concatenate(values)
        span = float(np.max(joined) - np.min(joined))
        pad = (
            0.10 * span
            if span > 0.0
            else max(1.0e-3, 0.1 * max(1.0, abs(float(np.max(joined)))))
        )
        return float(np.min(joined) - pad), float(np.max(joined) + pad)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.16, 2.15))
    side_x, side_y = 1.25 / 7.16, 1.25 / 2.15
    bottom = 0.205
    lefts = (0.046, 0.290, 0.535, 0.779)
    axes = [fig.add_axes([left, bottom, side_x, side_y]) for left in lefts]

    # (a) experimental platform photo, square-cropped to fill the panel.
    photo_ax = axes[0]
    img = plt.imread(DATA_DIR / "UGV.png")
    n = min(img.shape[0], img.shape[1])
    r0 = (img.shape[0] - n) // 2
    c0 = (img.shape[1] - n) // 2
    photo_ax.imshow(img[r0:r0 + n, c0:c0 + n])
    photo_ax.set_axis_off()
    photo_ax.set_title("(a) Experimental platform", fontsize=8.0, pad=6)

    # (b) representative trajectories of the three cases.
    traj_ax = axes[1]
    reference_rows = rows_by_case["E1"]
    reference_mask = reference_rows["t"] <= one_rev_s
    reference = reference_rows["reference"][reference_mask]
    traj_ax.plot(
        reference[:, 0], reference[:, 1], color=GRAY, linestyle="--",
        linewidth=0.9, label="reference", zorder=2,
    )
    endpoint = (reference[0, 0], 0.0)
    traj_ax.scatter(
        endpoint[0], endpoint[1], marker="s", facecolors="none",
        edgecolors="red", linewidths=0.9, s=12,
        label="terminal", zorder=5,
    )
    for case in cases:
        rows = rows_by_case[case]
        pose = rows["pose"][rows["t"] <= one_rev_s]
        traj_ax.plot(
            pose[:, 0], pose[:, 1], linewidth=1.0,
            label=f"Case {case[1:]}", zorder=3, **styles[case],
        )
    pose0 = rows_by_case["E1"]["pose"][0]
    traj_ax.scatter(
        pose0[0], pose0[1], marker="D", facecolors="none",
        edgecolors=GREEN, linewidths=0.9, s=9,
        label="start", zorder=6,
    )
    traj_ax.set_aspect("equal", adjustable="box")
    traj_ax.set_xlim(-0.70, 0.70)
    # Keep x/y spans equal (1.40) so the equal-aspect box stays square.
    traj_ax.set_ylim(-0.55, 0.85)
    axis_ticks = [-0.50, -0.25, 0.00, 0.25, 0.50]
    traj_ax.set_xticks(axis_ticks)
    traj_ax.set_yticks([-0.50, -0.25, 0.00, 0.25, 0.50, 0.75])
    traj_ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    traj_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    traj_ax.tick_params(labelsize=7.0)
    traj_ax.set_xlabel(r"$x$ (m)", fontsize=8.0)
    traj_ax.set_ylabel(r"$y$ (m)", fontsize=8.0, labelpad=1.0)
    traj_ax.set_title("(b) Trajectories", fontsize=8.0, pad=6)
    traj_ax.grid(True, linestyle=":", linewidth=0.45)
    marker_names = ("reference", "start", "terminal")
    handles, labels = traj_ax.get_legend_handles_labels()
    marker_handles = [handles[labels.index(name)] for name in marker_names]
    # Case samples are proxies carrying the same styles dict as the plotted
    # lines, so each legend sample renders the exact curve linestyle.
    case_names = [f"Case {case[1:]}" for case in cases]
    case_handles = [
        Line2D([], [], linewidth=1.0, **styles[case])
        for case in cases
    ]
    # Blank fourth row in the left column: legends fill column-major, so the
    # pad yields three visible entries left and the four cases right.
    blank = Line2D([], [], linestyle="none")
    traj_ax.legend(
        marker_handles + [blank] + case_handles,
        list(marker_names) + [""] + case_names,
        frameon=False, loc="upper right", fontsize=6.0, ncol=2,
        handlelength=1.2, columnspacing=0.7, borderpad=0.2,
        labelspacing=0.18,
    )

    # (c,d) position- and heading-error panels with cruise-window insets.
    for column, ax, title, ylabel in (
        (0, axes[2], "(c) Position error", r"$\Vert\boldsymbol{e}_r\Vert$ (mm)"),
        (1, axes[3], "(d) Heading error", r"$e_\theta$ (mrad)"),
    ):
        for case in cases:
            rows = rows_by_case[case]
            error = body_frame_tracking_error(rows)
            mask = rows["t"] <= one_rev_s
            values = (
                np.linalg.norm(error[:, :2], axis=1)
                if column == 0 else error[:, 2]
            )
            ax.plot(
                rows["t"][mask], 1.0e3 * values[mask], linewidth=1.0,
                label=f"Case {case[1:]}", zorder=3, **styles[case],
            )
        ax.set_xlim(0.0, one_rev_s)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=7.0)
        ax.set_xlabel("Time (s)", fontsize=8.0)
        ax.set_ylabel(ylabel, fontsize=8.0, labelpad=1.0)
        ax.set_title(title, fontsize=8.0, pad=6)
        ax.grid(True, linestyle=":", linewidth=0.45)
        ax.legend(
            case_handles, case_names,
            frameon=False, fontsize=6.0,
            loc="upper right", ncol=2, handlelength=1.2,
            borderpad=0.2, labelspacing=0.18, columnspacing=0.7,
        )
        lo, hi = steady_limits(column)
        # Raised above the decaying transients and nudged toward the
        # lower-left; Case 4 is excluded from the zoom window.
        zoom = ax.inset_axes([0.39, 0.39, 0.52, 0.26])
        zoom.set_xlim(STEADY_WINDOW_START_S, plot_end_s)
        zoom.set_ylim(lo, hi)
        for case in cases[:3]:
            rows = rows_by_case[case]
            error = body_frame_tracking_error(rows)
            mask = rows["t"] <= one_rev_s
            values = (
                np.linalg.norm(error[:, :2], axis=1)
                if column == 0 else error[:, 2]
            )
            zoom.plot(
                rows["t"][mask], 1.0e3 * values[mask],
                linewidth=0.8, **styles[case],
            )
        zoom.grid(True, linestyle=":", linewidth=0.35)
        zoom.tick_params(labelsize=7.0, pad=0.8)
        zoom.xaxis.set_major_locator(plt.MaxNLocator(3))
        zoom.yaxis.set_major_locator(plt.MaxNLocator(3))
        mark_inset(
            ax, zoom, loc1=2, loc2=4, fc="none", ec="0.7", linewidth=0.5,
        )
        ax.add_patch(
            Rectangle(
                (STEADY_WINDOW_START_S, lo),
                plot_end_s - STEADY_WINDOW_START_S,
                hi - lo,
                fill=False, edgecolor="0.35", linewidth=0.7, zorder=4,
            )
        )

    for suffix in ("pdf", "png"):
        # dpi sets the raster resolution of the embedded platform photo;
        # panel (a) is ~1.25 in wide, so ~1800 dpi embeds it at the source
        # photo's native 2246 px instead of a downsampled raster. The
        # curve panels stay vector in the PDF regardless.  Omitting the
        # date metadata keeps reruns byte-identical for the fixed archive.
        fig.savefig(
            RESULTS_DIR / f"experimental_validation.{suffix}",
            dpi=1800 if suffix == "pdf" else 300,
            metadata={"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else None,
        )
    plt.close(fig)



def validate_adaptive_only_weights(
    recorded: dict[str, object], payload: dict[str, object]
) -> None:
    """Verify the E3 ablation without confusing it with a zero-gain controller."""
    if set(recorded) != set(payload):
        raise ValueError("E3 weight metadata differs from the synthesized payload")
    for key in set(payload) - {"w2", "w3"}:
        if recorded[key] != payload[key]:
            raise ValueError(f"E3 weight metadata differs at {key}")
    for block in (2, 3):
        full = np.asarray(payload[f"w{block}"], dtype=float).reshape(2, 8)
        ablated = np.asarray(recorded[f"w{block}"], dtype=float).reshape(2, 8)
        if not np.allclose(ablated[:, :2], full[:, :2], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"E3 block {block} changed the data-derived error gains")
        if not np.allclose(ablated[:, 2:], 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"E3 block {block} retained initial nonlinear weights")


def main(run_dir: Path | None = None) -> None:
    global RESULTS_DIR
    if run_dir is None:
        run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_experimental"
    RESULTS_DIR = run_dir / "results"
    print(f"run folder: {run_dir}")
    set_style()
    weights_document = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    payload = weights_document["payload"]
    dataset = load_npz(DATASET_PATH)
    if int(weights_document["samples"]) != int(dataset["zdot2"].shape[1]):
        raise ValueError("validated weights and integral dataset have different K")
    if int(np.asarray(dataset["schema"]).reshape(-1)[0]) != 5:
        raise ValueError("unexpected integral-dataset schema")

    sample_count = int(dataset["zdot2"].shape[1])
    blocks: dict[str, dict[str, object]] = {}
    for block in (2, 3):
        g_matrix = np.vstack((dataset[f"zdot{block}"], dataset[f"y{block}"]))
        validation = weights_document["validation"][f"block{block}"]
        dbar = float(payload[f"dbar{block}"])
        blocks[str(block)] = {
            "rank": int(np.linalg.matrix_rank(g_matrix, tol=1.0e-9)),
            "condition_GGt": float(np.linalg.cond(g_matrix @ g_matrix.T)),
            "matching_residual": float(validation["match_residual"]),
            "dbar": dbar,
            "chi": float(np.sqrt(sample_count) * dbar),
            "kappa": float(payload[f"kappa{block}"]),
            "kappa_min": float(payload[f"kappa_min{block}"]),
            "spectral_margin": float(payload[f"sigma_margin{block}"]),
            "route_certificate": float(payload[f"certificate_max{block}"]),
            "weight_frobenius_norm": float(
                np.linalg.norm(np.asarray(payload[f"w{block}"], dtype=float))
            ),
        }
    training_report = {
        "sample_count": sample_count,
        "nominal_window_s": 0.10,
        "median_window_s": float(np.median(dataset["window_s"])),
        "record_span_s": float((dataset["t_us"][-1] - dataset["t_us"][0]) * 1.0e-6),
        "max_abs_applied_voltage_v": float(np.max(np.abs(dataset["x4"]))),
        "rms_applied_voltage_v": rms(dataset["x4"]),
        "blocks": blocks,
    }

    trace_paths = select_official_traces(payload)
    runs_by_case: dict[str, list[dict[str, object]]] = {}
    configs: dict[str, list[dict[str, object]]] = {}
    firmware: dict[str, set[str]] = {}
    terminal_start: dict[str, float] = {}
    for case, paths in trace_paths.items():
        runs_by_case[case] = []
        configs[case] = []
        firmware[case] = set()
        for path in paths:
            rows = load_npz(path)
            fields = peek_trace(path)
            if fields["case"] != case:
                raise ValueError(f"{path.name}: unexpected comparison case")
            for key in ("t", "reference", "pose", "velocity", "beta1"):
                if key not in rows:
                    raise ValueError(f"{path.name}: missing {key}")
            config = fields["config"]
            if str(np.asarray(rows["firmware"]).reshape(-1)[0]) != EXPECTED_FIRMWARE:
                raise ValueError(f"{path.name}: expected firmware {EXPECTED_FIRMWARE}")
            if abs(float(config.get("hold_duration", 0.0)) - EXPECTED_HOLD_DURATION_S) > 1.0e-6:
                raise ValueError(f"{path.name}: unexpected terminal hold")
            if abs(float(config["run_duration"]) - EXPECTED_RUN_DURATION_S) > 1.0e-4:
                raise ValueError(f"{path.name}: unexpected reference duration")
            expected_record_duration = float(config["run_duration"]) + float(
                config.get("hold_duration", 0.0)
            )
            if abs(float(rows["t"][-1]) - expected_record_duration) > 0.1:
                raise ValueError(f"{path.name}: incomplete recording")
            if float(config["ref_shape"]) != EXPECTED_REF_SHAPE:
                raise ValueError(f"{path.name}: unexpected reference shape")
            if abs(float(config["outer_preview_horizon"]) - EXPECTED_PREVIEW_S) > 1.0e-6:
                raise ValueError(f"{path.name}: preview horizon is not {EXPECTED_PREVIEW_S} s")
            terminal_start[case] = (
                float(config["run_duration"]) - float(config["outer_preview_horizon"])
            )
            firmware[case].add(str(np.asarray(rows["firmware"]).reshape(-1)[0]))
            configs[case].append(config)
            runs_by_case[case].append(
                {
                    "path": path,
                    "rows": rows,
                    "metrics": run_metrics(rows, terminal_start[case]),
                }
            )
        runs_by_case[case].sort(key=lambda run: run["metrics"]["position_rmse_m"])

    allowed_config_differences = {
        "comparison_case",
        "r2",
        "r3",
        "motion_max_target_error",
    }
    common_e1 = {
        key: value
        for key, value in configs["E1"][0].items()
        if key not in allowed_config_differences
    }
    for case in ("E1", "E2", "E3"):
        for config in configs[case]:
            common_case = {
                key: value
                for key, value in config.items()
                if key not in allowed_config_differences
            }
            if common_case != common_e1:
                raise ValueError(
                    f"{case}: common controller configuration differs within E1"
                )
        if len(firmware[case]) != 1:
            raise ValueError(f"{case}: inconsistent firmware versions")

    # E4 is an external baseline with its own configuration schema: validate
    # internal consistency instead of equality with the E1 controller config.
    common_e4 = {
        key: value
        for key, value in configs["E4"][0].items()
        if key not in allowed_config_differences
    }
    for config in configs["E4"]:
        common_case = {
            key: value
            for key, value in config.items()
            if key not in allowed_config_differences
        }
        if common_case != common_e4:
            raise ValueError("E4: baseline configuration differs within E4")
    if len(firmware["E4"]) != 1:
        raise ValueError("E4: inconsistent firmware versions")
    if configs["E4"][0].get("baseline") != "sampling_pid":
        raise ValueError("E4: unexpected baseline kind")

    if len({terminal_start[case] for case in terminal_start}) != 1:
        raise ValueError("inconsistent terminal capture start times")
    terminal_start_s = float(next(iter(terminal_start.values())))

    plot_experimental_components(runs_by_case, terminal_start_s)
    figure_dst = SUBMISSION_FIGURES / "experimental_validation.pdf"
    if figure_dst.exists():
        rerun_match = (
            sha256(RESULTS_DIR / "experimental_validation.pdf") == sha256(figure_dst)
        )
        print(
            "deterministic rerun: new figure "
            + ("matches" if rerun_match else "DIFFERS from")
            + " the previous submission copy"
        )
    shutil.copyfile(RESULTS_DIR / "experimental_validation.pdf", figure_dst)
    print(f"figure copied to {figure_dst}")

    case_descriptions = {
        "E1": "proposed controller with synthesized nonlinear compensation and adaptation",
        "E2": "same synthesized weights with adaptation disabled",
        "E3": (
            "adaptive ablation retaining the synthesized error-feedback columns "
            "while zeroing the initial nonlinear-compensation columns"
        ),
        "E4": (
            "external sampling-PID baseline: the data-collection excitation "
            "controller running PC-side at 25 Hz, without the guidance "
            "outer loop or the current inner loop"
        ),
    }
    case_results: dict[str, dict[str, object]] = {}
    for case in ("E1", "E2", "E3", "E4"):
        runs = runs_by_case[case]
        case_results[case] = {
            "description": case_descriptions[case],
            "firmware": sorted(firmware[case])[0],
            "runtime_config": configs[case][0],
            "runs": [
                {
                    "trace": str(run["path"]),
                    "trace_sha256": sha256(run["path"]),
                    "metrics": run["metrics"],
                }
                for run in runs
            ],
            "representative_trace": str(runs[2]["path"]),
            "metrics": aggregate_metrics([run["metrics"] for run in runs]),
        }

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report = {
        "campaign_manifest": MANIFEST_PATH.name,
        "attempt_records": [],
        "metric_definition": "Time-weighted RMSE in every window, including the final 0.5 s; window limits use the first and last available samples.",
        "data_scope": {
            "training_dataset": str(DATASET_PATH),
            "training_dataset_sha256": sha256(DATASET_PATH),
            "validated_weights": str(WEIGHTS_PATH),
            "validated_weights_sha256": sha256(WEIGHTS_PATH),
            "closed_loop_traces": {
                case: [str(run["path"]) for run in runs_by_case[case]]
                for case in ("E1", "E2", "E3", "E4")
            },
            "statistical_scope": (
                "five completed runs per case on the prescribed uniform circle "
                "with T_p=1.3 s and kappa=(20,30); "
                "mean and sample standard deviation reported over the full record, "
                "over the cruise window t>=8 s up to the terminal capture "
                "(T_f-T_p), over the terminal capture segment "
                "(T_f-T_p to T_f), and over the final 0.5-s settle window"
            ),
            "limitation": (
                "Cartesian tracking errors are reconstructed from on-board encoder "
                "odometry and are not external-position ground truth."
            ),
        },
        "training": training_report,
        "reference": {
            "shape": "uniform circle with center start (firmware ref_shape=3)",
            "equation": "q_r=R_e*[cos(nu_e*t), sin(nu_e*t)]^T until the commanded stop",
            "radius_m": 0.40,
            "nu_rad_s": 0.30,
            "reference_speed_m_s": 0.12,
            "startup_ramp_s": 0.0,
            "startup_note": "the vehicle starts at the circle center while the reference starts at (R_e,0) and moves uniformly; it continues for 3 s beyond one revolution before stopping",
            "run_duration_s": EXPECTED_RUN_DURATION_S,
            "hold_duration_s": EXPECTED_HOLD_DURATION_S,
            "hold_note": "no terminal hold is used; the reference stops at the current circular pose",
        },
        "cases": case_results,
        "figure_sources": {
            "experimental_validation": {
                "figure": "fig:expval",
                "component": (
                    "experimental_validation.pdf in this run's results folder, "
                    "copied to ../../../submission/figures/"
                ),
                "panels": (
                    "(a) UGV.png platform photo, (b) trajectories, "
                    "(c) position error, (d) heading error"
                ),
                "trace": (
                    str(runs_by_case["E1"][2]["path"]),
                    str(runs_by_case["E2"][2]["path"]),
                    str(runs_by_case["E3"][2]["path"]),
                    str(runs_by_case["E4"][2]["path"]),
                ),
                "selection": "median by full-record position RMSE per case",
            },
        },
        "pending": [
            "external absolute-position ground truth (overhead camera + ArUco) "
            "for absolute pose-error claims"
        ],
    }
    output = RESULTS_DIR / "experimental_results.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LATEST_POINTER.write_text(run_dir.name + "\n", encoding="utf-8")
    print(json.dumps(report["training"], indent=2, ensure_ascii=False))
    print(json.dumps({case: case_results[case]["metrics"] for case in case_results}, indent=2))
    print(f"wrote {output}")
    print(f"latest run pointer -> {LATEST_POINTER.name}: {run_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the experimental results inside a timestamped run "
            "folder under runs/; --run targets an existing run folder."
        )
    )
    parser.add_argument(
        "--run",
        default=None,
        help=(
            "run folder to write into: a folder name under runs/ or a path; "
            "omit to create a new <YYYYmmdd_HHMMSS>_experimental folder"
        ),
    )
    args = parser.parse_args()
    target = None
    if args.run is not None:
        target = Path(args.run)
        if not target.is_absolute():
            target = RUNS_DIR / args.run
    main(target)
