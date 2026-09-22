from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# The yaw-rate channel comes directly from the MPU6050.  Differentiating it
# with the short local polynomial alone amplifies chassis vibration.  This
# observer is deliberately slower than the 25-Hz telemetry interval while
# remaining fast enough for the small double-lemniscate collection path.
OMEGA_DERIVATIVE_OBSERVER_TAU_S = 0.30
# Smooth Coulomb-friction basis shared with firmware and simulation.
REGRESSOR_BASIS = "sgn_gyro_gate"


def smooth_sign_basis(value: np.ndarray) -> np.ndarray:
    """Return the paper's Coulomb-friction sign basis sgn(value), sgn(0)=0."""
    return np.sign(np.asarray(value, dtype=float))


@dataclass(frozen=True)
class BlockSynthesis:
    weights: np.ndarray
    p_solution: np.ndarray
    q_solution: np.ndarray
    rank: int
    rows: int
    condition: float
    match_residual: float
    nominal_residual: float
    xi_max: float
    kappa: float
    kappa_min: float
    sigma_min: float
    chi: float
    spectral_margin: float
    kappa_v: float | None = None
    kappa_w: float | None = None

    @property
    def weight_norm(self) -> float:
        return float(np.linalg.norm(self.weights, ord="fro"))

    @property
    def kappa_diagonal(self) -> bool:
        return self.kappa_v is not None and self.kappa_w is not None


@dataclass(frozen=True)
class RouteIIThreshold:
    kappa_min: float
    sigma_min: float
    chi: float
    spectral_margin: float


@dataclass(frozen=True)
class SynthesisResult:
    block2: BlockSynthesis
    block3: BlockSynthesis
    samples: int
    kappa2: float
    kappa3: float
    epsilon2: float
    epsilon3: float
    dbar2: float
    dbar3: float
    kappa2_v: float | None = None
    kappa2_w: float | None = None
    kappa3_v: float | None = None
    kappa3_w: float | None = None

    @property
    def valid(self) -> bool:
        return (
            np.isfinite(self.kappa2)
            and np.isfinite(self.kappa3)
            and np.isclose(self.kappa2, self.block2.kappa)
            and np.isclose(self.kappa3, self.block3.kappa)
            and np.isfinite(self.block2.kappa_min)
            and np.isfinite(self.block3.kappa_min)
            and self.kappa2 > self.block2.kappa_min
            and self.kappa3 > self.block3.kappa_min
            and np.isfinite(self.epsilon2)
            and np.isfinite(self.epsilon3)
            and self.epsilon2 > 0.0
            and self.epsilon3 > 0.0
            and np.isfinite(self.dbar2)
            and np.isfinite(self.dbar3)
            and self.dbar2 > 0.0
            and self.dbar3 > 0.0
            and self.block2.rank == self.block2.rows == 10
            and self.block3.rank == self.block3.rows == 10
            and self.block2.match_residual <= 1.0e-6
            and self.block3.match_residual <= 1.0e-6
            and self.block2.spectral_margin > 0.0
            and self.block3.spectral_margin > 0.0
            and self.block2.xi_max <= 0.0
            and self.block3.xi_max <= 0.0
        )

    def upload_payload(self) -> dict[str, Any]:
        payload = {
            "regressor_basis": REGRESSOR_BASIS,
            "w2": self.block2.weights.reshape(-1).tolist(),
            "w3": self.block3.weights.reshape(-1).tolist(),
            "route": "route_ii",
            "kappa2": self.kappa2,
            "kappa3": self.kappa3,
            "kappa_min2": self.block2.kappa_min,
            "kappa_min3": self.block3.kappa_min,
            "sigma_margin2": self.block2.spectral_margin,
            "sigma_margin3": self.block3.spectral_margin,
            "certificate_max2": self.block2.xi_max,
            "certificate_max3": self.block3.xi_max,
            # The deployed firmware validates these legacy metadata fields but
            # does not use them in the online control law.  Encoding -kappa
            # preserves wire compatibility while W2/W3 come exclusively from
            # the Route-II null-space construction.
            "lambda2": -self.kappa2,
            "lambda3": -self.kappa3,
            "epsilon2": self.epsilon2,
            "epsilon3": self.epsilon3,
            "dbar2": self.dbar2,
            "dbar3": self.dbar3,
            "xi_max2": self.block2.xi_max,
            "xi_max3": self.block3.xi_max,
            "rank2": self.block2.rank,
            "rank3": self.block3.rank,
        }
        if self.kappa2_v is not None:
            payload["kappa2_v"] = self.kappa2_v
            payload["kappa2_w"] = self.kappa2_w
        if self.kappa3_v is not None:
            payload["kappa3_v"] = self.kappa3_v
            payload["kappa3_w"] = self.kappa3_w
        return payload


class SnapshotDataset:
    REQUIRED = ("zdot2", "y2", "x3", "zdot3", "y3", "x4")

    def __init__(self) -> None:
        self._rows: list[dict[str, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    def append(self, snapshot: dict[str, Any]) -> None:
        expected = {"zdot2": 2, "y2": 8, "x3": 2, "zdot3": 2, "y3": 8, "x4": 2}
        row: dict[str, np.ndarray] = {}
        for key, length in expected.items():
            value = np.asarray(snapshot.get(key), dtype=float)
            if value.shape != (length,) or not np.all(np.isfinite(value)):
                raise ValueError(f"snapshot field {key} has invalid shape or value")
            row[key] = value
        self._rows.append(row)

    def matrices(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("dataset is empty")
        return {key: np.asarray([row[key] for row in self._rows], dtype=float).T for key in self.REQUIRED}

    def save(self, path: str | Path) -> None:
        matrices = self.matrices()
        np.savez_compressed(Path(path), schema=np.array([1]), **matrices)

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotDataset":
        loaded = np.load(Path(path), allow_pickle=False)
        dataset = cls()
        columns = int(loaded["zdot2"].shape[1])
        for index in range(columns):
            dataset.append({key: loaded[key][:, index] for key in cls.REQUIRED})
        return dataset


@dataclass(frozen=True)
class OfflineSnapshotDiagnostics:
    raw_samples: int
    processed_samples: int
    segments: int
    median_dt_s: float
    max_dt_s: float
    smoothing_window: int


class IntegralSnapshotDataset:
    """Non-overlapping integral snapshots produced by the 200 Hz firmware loop.

    Each stored column represents one complete window.  ``zdot2`` and
    ``zdot3`` are endpoint differences divided by the actual window duration;
    the remaining matrices are time averages over that exact same window.
    No numerical differentiation is performed on the PC.
    """

    SCHEMA = 5
    METHOD = "firmware_integral_v3_sgn_gyro_gate"
    REQUIRED = SnapshotDataset.REQUIRED
    OPTIONAL = ("velocity_raw", "current_raw")
    EXPECTED = {"zdot2": 2, "y2": 8, "x3": 2, "zdot3": 2, "y3": 8, "x4": 2}

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._segments: dict[int, dict[str, Any]] = {}
        self._next_segment = 0

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def segment_metadata(self) -> dict[int, dict[str, Any]]:
        return {key: dict(value) for key, value in self._segments.items()}

    def clear(self) -> None:
        self._rows.clear()
        self._segments.clear()
        self._next_segment = 0

    def start_segment(self, metadata: dict[str, Any] | None = None) -> int:
        segment = self._next_segment
        self._next_segment += 1
        self._segments[segment] = dict(metadata or {})
        return segment

    def discard_segment(self, segment: int) -> int:
        segment_id = int(segment)
        before = len(self._rows)
        self._rows = [
            row for row in self._rows if int(row["segment"]) != segment_id
        ]
        self._segments.pop(segment_id, None)
        return before - len(self._rows)

    def append_integral(
        self,
        snapshot: dict[str, Any],
        *,
        t_us: int,
        window_s: float,
        segment: int,
    ) -> None:
        if segment not in self._segments:
            raise ValueError("integral snapshot references an unknown segment")
        timestamp = int(t_us)
        duration = float(window_s)
        if timestamp <= 0:
            raise ValueError("integral snapshot timestamp must be positive")
        if not np.isfinite(duration) or duration < 0.05 or duration > 1.0:
            raise ValueError("integral snapshot window must be within [0.05, 1.0] s")
        row: dict[str, Any] = {
            "t_us": timestamp,
            "window_s": duration,
            "segment": int(segment),
        }
        for key, length in self.EXPECTED.items():
            value = np.asarray(snapshot.get(key), dtype=float)
            if value.shape != (length,) or not np.all(np.isfinite(value)):
                raise ValueError(f"integral snapshot field {key} is invalid")
            row[key] = value.copy()
        for key in self.OPTIONAL:
            if key not in snapshot:
                continue
            value = np.asarray(snapshot[key], dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"integral snapshot field {key} is invalid")
            row[key] = value.copy()
        if (
            self._rows
            and int(self._rows[-1]["segment"]) == int(segment)
            and int(self._rows[-1]["t_us"]) == timestamp
        ):
            return
        self._rows.append(row)

    def matrices(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("integral snapshot dataset is empty")
        matrices = {
            key: np.asarray([row[key] for row in self._rows], dtype=float).T
            for key in self.REQUIRED
        }
        for key in self.OPTIONAL:
            if all(key in row for row in self._rows):
                matrices[key] = np.asarray(
                    [row[key] for row in self._rows], dtype=float
                ).T
        return matrices

    @property
    def window_seconds(self) -> np.ndarray:
        """Actual per-window duration used by the firmware for each snapshot."""
        return np.asarray(
            [row["window_s"] for row in self._rows], dtype=float
        )

    def diagnostics(self) -> OfflineSnapshotDiagnostics:
        if not self._rows:
            raise ValueError("integral snapshot dataset is empty")
        windows = np.asarray([row["window_s"] for row in self._rows], dtype=float)
        return OfflineSnapshotDiagnostics(
            raw_samples=len(self),
            processed_samples=len(self),
            segments=self.segment_count,
            median_dt_s=float(np.median(windows)),
            max_dt_s=float(np.max(windows)),
            smoothing_window=0,
        )

    def save(self, path: str | Path) -> None:
        matrices = self.matrices()
        metadata = json.dumps(
            {str(key): value for key, value in self._segments.items()},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        np.savez_compressed(
            Path(path),
            schema=np.asarray([self.SCHEMA], dtype=np.int64),
            method=np.asarray([self.METHOD]),
            metadata=np.asarray([metadata]),
            t_us=np.asarray([row["t_us"] for row in self._rows], dtype=np.int64),
            window_s=np.asarray([row["window_s"] for row in self._rows], dtype=float),
            segment=np.asarray([row["segment"] for row in self._rows], dtype=np.int64),
            **matrices,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IntegralSnapshotDataset":
        loaded = np.load(Path(path), allow_pickle=False)
        schema = int(np.asarray(loaded["schema"]).reshape(-1)[0])
        method = (
            str(np.asarray(loaded["method"]).reshape(-1)[0])
            if "method" in loaded.files
            else "missing"
        )
        if schema != cls.SCHEMA or method != cls.METHOD:
            raise ValueError(
                f"旧数据格式 schema={schema}, method={method!r} 已隔离；"
                "必须用积分快照固件重新采集"
            )
        dataset = cls()
        metadata = json.loads(str(np.asarray(loaded["metadata"]).reshape(-1)[0]))
        if not isinstance(metadata, dict):
            raise ValueError("integral dataset segment metadata is invalid")
        for key in sorted(metadata, key=lambda value: int(value)):
            segment = int(key)
            dataset._segments[segment] = dict(metadata[key])
        dataset._next_segment = max(dataset._segments, default=-1) + 1
        t_us = np.asarray(loaded["t_us"], dtype=np.int64)
        window_s = np.asarray(loaded["window_s"], dtype=float)
        segments = np.asarray(loaded["segment"], dtype=np.int64)
        matrices = {key: np.asarray(loaded[key], dtype=float) for key in cls.REQUIRED}
        optional = {
            key: np.asarray(loaded[key], dtype=float)
            for key in cls.OPTIONAL
            if key in loaded.files
        }
        columns = int(t_us.size)
        if window_s.shape != (columns,) or segments.shape != (columns,):
            raise ValueError("integral dataset metadata shape is invalid")
        if any(value.shape != (cls.EXPECTED[key], columns) for key, value in matrices.items()):
            raise ValueError("integral dataset matrix shape is invalid")
        if any(value.shape != (2, columns) for value in optional.values()):
            raise ValueError("integral dataset optional source shape is invalid")
        for index in range(columns):
            dataset.append_integral(
                {
                    **{key: value[:, index] for key, value in matrices.items()},
                    **{key: value[:, index] for key, value in optional.items()},
                },
                t_us=int(t_us[index]),
                window_s=float(window_s[index]),
                segment=int(segments[index]),
            )
        return dataset


class RawTimeSeriesDataset:
    """Timestamped physical states used to rebuild mathematically consistent snapshots.

    Firmware-provided live derivative fields are deliberately not stored here.  The
    velocity/current states and their derivatives are reconstructed together from
    the same locally fitted polynomial, so Z and Zdot refer to the same signal.
    """

    SCHEMA = 2
    REQUIRED = ("t_us", "velocity", "current", "voltage", "segment")

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._segments: dict[int, dict[str, Any]] = {}
        self._next_segment = 0
        self._cached_matrices: dict[str, np.ndarray] | None = None
        self._cached_diagnostics: OfflineSnapshotDiagnostics | None = None

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def segment_metadata(self) -> dict[int, dict[str, Any]]:
        return {key: dict(value) for key, value in self._segments.items()}

    def clear(self) -> None:
        self._rows.clear()
        self._segments.clear()
        self._next_segment = 0
        self._invalidate()

    def start_segment(self, metadata: dict[str, Any] | None = None) -> int:
        segment = self._next_segment
        self._next_segment += 1
        self._segments[segment] = dict(metadata or {})
        self._invalidate()
        return segment

    def discard_segment(self, segment: int) -> int:
        """Remove one invalid collection segment and return its sample count."""
        segment_id = int(segment)
        before = len(self._rows)
        self._rows = [
            row for row in self._rows if int(row["segment"]) != segment_id
        ]
        self._segments.pop(segment_id, None)
        removed = before - len(self._rows)
        self._invalidate()
        return removed

    def append_raw(
        self,
        *,
        t_us: int,
        velocity: Iterable[float],
        current: Iterable[float],
        voltage: Iterable[float],
        segment: int,
    ) -> None:
        if segment not in self._segments:
            raise ValueError("raw sample references an unknown collection segment")
        timestamp = int(t_us)
        if timestamp <= 0:
            raise ValueError("raw sample timestamp must be positive")
        row: dict[str, Any] = {"t_us": timestamp, "segment": int(segment)}
        for key, value in (
            ("velocity", velocity),
            ("current", current),
            ("voltage", voltage),
        ):
            array = np.asarray(value, dtype=float)
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise ValueError(f"raw {key} must contain two finite values")
            row[key] = array.copy()
        if (
            self._rows
            and self._rows[-1]["segment"] == row["segment"]
            and self._rows[-1]["t_us"] == row["t_us"]
        ):
            # WiFi telemetry can repeat the just-sent snapshot.  It is
            # byte-identical at this timestamp and adds no physical evidence.
            return
        self._rows.append(row)
        self._invalidate()

    def _invalidate(self) -> None:
        self._cached_matrices = None
        self._cached_diagnostics = None

    def matrices(self) -> dict[str, np.ndarray]:
        if self._cached_matrices is None:
            matrices, diagnostics = rebuild_consistent_snapshots(self)
            self._cached_matrices = matrices
            self._cached_diagnostics = diagnostics
        return {key: value.copy() for key, value in self._cached_matrices.items()}

    def diagnostics(self) -> OfflineSnapshotDiagnostics:
        if self._cached_diagnostics is None:
            self.matrices()
        assert self._cached_diagnostics is not None
        return self._cached_diagnostics

    def raw_matrices(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("raw dataset is empty")
        return {
            "t_us": np.asarray([row["t_us"] for row in self._rows], dtype=np.int64),
            "velocity": np.asarray([row["velocity"] for row in self._rows], dtype=float).T,
            "current": np.asarray([row["current"] for row in self._rows], dtype=float).T,
            "voltage": np.asarray([row["voltage"] for row in self._rows], dtype=float).T,
            "segment": np.asarray([row["segment"] for row in self._rows], dtype=np.int64),
        }

    def save(self, path: str | Path) -> None:
        matrices = self.raw_matrices()
        metadata = json.dumps(
            {str(key): value for key, value in self._segments.items()},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        np.savez_compressed(
            Path(path),
            schema=np.asarray([self.SCHEMA], dtype=np.int64),
            metadata=np.asarray([metadata]),
            **matrices,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RawTimeSeriesDataset":
        loaded = np.load(Path(path), allow_pickle=False)
        schema = int(np.asarray(loaded["schema"]).reshape(-1)[0])
        if schema != cls.SCHEMA:
            raise ValueError(
                f"旧快照格式 schema={schema} 已隔离；必须重新采集原始时序数据"
            )
        dataset = cls()
        metadata_text = str(np.asarray(loaded["metadata"]).reshape(-1)[0])
        metadata = json.loads(metadata_text)
        if not isinstance(metadata, dict):
            raise ValueError("raw dataset segment metadata is invalid")
        for key in sorted(metadata, key=lambda value: int(value)):
            segment = int(key)
            dataset._segments[segment] = dict(metadata[key])
        dataset._next_segment = max(dataset._segments, default=-1) + 1
        t_us = np.asarray(loaded["t_us"], dtype=np.int64)
        velocity = np.asarray(loaded["velocity"], dtype=float)
        current = np.asarray(loaded["current"], dtype=float)
        voltage = np.asarray(loaded["voltage"], dtype=float)
        segments = np.asarray(loaded["segment"], dtype=np.int64)
        if velocity.shape != (2, t_us.size) or current.shape != (2, t_us.size):
            raise ValueError("raw dataset state matrix shape is invalid")
        if voltage.shape != (2, t_us.size) or segments.shape != (t_us.size,):
            raise ValueError("raw dataset input/segment matrix shape is invalid")
        for index in range(t_us.size):
            dataset.append_raw(
                t_us=int(t_us[index]),
                velocity=velocity[:, index],
                current=current[:, index],
                voltage=voltage[:, index],
                segment=int(segments[index]),
            )
        return dataset


def _local_polynomial_state_and_derivative(
    t: np.ndarray,
    values: np.ndarray,
    window: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase local quadratic fit for a state and its derivative."""
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if t.ndim != 1 or values.ndim != 2 or values.shape[1] != t.size:
        raise ValueError("local polynomial inputs have incompatible shapes")
    if t.size < 7 or np.any(np.diff(t) <= 0.0):
        raise ValueError("each collection segment needs >=7 strictly ordered samples")
    width = min(int(window), int(t.size))
    if width % 2 == 0:
        width -= 1
    width = max(width, 7)
    half = width // 2
    smoothed = np.empty_like(values)
    derivative = np.empty_like(values)
    for index in range(t.size):
        start = max(0, min(index - half, t.size - width))
        stop = start + width
        centered = t[start:stop] - t[index]
        design = np.column_stack((np.ones(width), centered, centered**2))
        coefficients = np.linalg.lstsq(design, values[:, start:stop].T, rcond=None)[0]
        smoothed[:, index] = coefficients[0]
        derivative[:, index] = coefficients[1]
    return smoothed, derivative


def _tracking_observer_state_and_derivative(
    t: np.ndarray,
    values: np.ndarray,
    tau_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order tracking observer with a consistent filtered derivative.

    ``state_dot=(measurement-state)/tau`` is evaluated on each measured time
    interval, then integrated with the same interval.  Unlike finite
    differencing, it does not turn MPU6050 vibration into an unbounded yaw
    acceleration estimate.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if t.ndim != 1 or values.ndim != 2 or values.shape[1] != t.size:
        raise ValueError("tracking observer inputs have incompatible shapes")
    if t.size < 2 or np.any(np.diff(t) <= 0.0) or not np.isfinite(tau_s) or tau_s <= 0.0:
        raise ValueError("tracking observer requires ordered samples and positive tau")
    state = np.empty_like(values)
    derivative = np.empty_like(values)
    state[:, 0] = values[:, 0]
    derivative[:, 0] = 0.0
    for index in range(1, t.size):
        dt = float(np.clip(t[index] - t[index - 1], 1.0e-3, 0.2))
        previous = state[:, index - 1]
        derivative[:, index] = (values[:, index] - previous) / (tau_s + dt)
        state[:, index] = previous + dt * derivative[:, index]
    return state, derivative


def _synthetic_filter(t: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    if block == 2:
        beta = np.vstack(
            (
                0.18 * np.sin(0.47 * t) + 0.06 * np.sin(1.31 * t),
                0.42 * np.sin(0.59 * t + 0.4) + 0.10 * np.sin(1.73 * t),
            )
        )
        beta_dot = np.vstack(
            (
                0.18 * 0.47 * np.cos(0.47 * t)
                + 0.06 * 1.31 * np.cos(1.31 * t),
                0.42 * 0.59 * np.cos(0.59 * t + 0.4)
                + 0.10 * 1.73 * np.cos(1.73 * t),
            )
        )
    elif block == 3:
        beta = np.vstack(
            (
                0.25 * np.sin(0.83 * t + 0.2) + 0.08 * np.sin(2.11 * t),
                0.23 * np.sin(0.71 * t - 0.5) + 0.09 * np.sin(1.89 * t + 0.7),
            )
        )
        beta_dot = np.vstack(
            (
                0.25 * 0.83 * np.cos(0.83 * t + 0.2)
                + 0.08 * 2.11 * np.cos(2.11 * t),
                0.23 * 0.71 * np.cos(0.71 * t - 0.5)
                + 0.09 * 1.89 * np.cos(1.89 * t + 0.7),
            )
        )
    else:
        raise ValueError("synthetic filter block must be 2 or 3")
    return beta, beta_dot


def rebuild_consistent_snapshots(
    dataset: RawTimeSeriesDataset,
    *,
    track_width_m: float = 0.2035,
    smoothing_window: int = 11,
    omega_observer_tau_s: float = OMEGA_DERIVATIVE_OBSERVER_TAU_S,
) -> tuple[dict[str, np.ndarray], OfflineSnapshotDiagnostics]:
    raw = dataset.raw_matrices()
    output = {key: [] for key in SnapshotDataset.REQUIRED}
    all_dt: list[np.ndarray] = []
    used_segments = 0
    for segment in np.unique(raw["segment"]):
        indices = np.flatnonzero(raw["segment"] == segment)
        if indices.size < 7:
            continue
        t_us = raw["t_us"][indices]
        order = np.argsort(t_us, kind="stable")
        indices = indices[order]
        t_us = t_us[order]
        keep = np.concatenate(([True], np.diff(t_us) > 0))
        indices = indices[keep]
        t_us = t_us[keep]
        if indices.size < 7:
            continue
        t = (t_us - t_us[0]).astype(float) * 1.0e-6
        all_dt.append(np.diff(t))
        raw_velocity = raw["velocity"][:, indices]
        wheel_proxy = np.vstack(
            (
                raw_velocity[0] + 0.5 * track_width_m * raw_velocity[1],
                raw_velocity[0] - 0.5 * track_width_m * raw_velocity[1],
            )
        )
        wheel_rms = np.sqrt(np.mean(wheel_proxy**2, axis=1))
        active_wheel_rms = float(np.max(wheel_rms))
        inactive_wheel_rms = float(np.min(wheel_rms))
        if (
            active_wheel_rms >= 0.02
            and inactive_wheel_rms <= 0.05 * active_wheel_rms
        ):
            failed_side = "右" if wheel_rms[0] < wheel_rms[1] else "左"
            raise ValueError(
                f"采集段 {int(segment)} 的{failed_side}轮速度近似为零，"
                "疑似 IMU 回退或编码器失效；该段禁止用于权重综合"
            )
        velocity, velocity_dot = _local_polynomial_state_and_derivative(
            t, raw_velocity, smoothing_window
        )
        # Keep encoder-derived forward speed on the zero-phase polynomial
        # path.  For yaw rate, use a dedicated tracking observer instead of a
        # second numerical derivative of the vibration-sensitive gyro signal.
        omega, omega_dot = _tracking_observer_state_and_derivative(
            t, raw["velocity"][1:2, indices], omega_observer_tau_s
        )
        velocity[1:2] = omega
        velocity_dot[1:2] = omega_dot
        current, current_dot = _local_polynomial_state_and_derivative(
            t, raw["current"][:, indices], smoothing_window
        )
        voltage = raw["voltage"][:, indices]
        beta1, beta1_dot = _synthetic_filter(t, 2)
        beta2, beta2_dot = _synthetic_filter(t, 3)
        z2 = velocity - beta1
        z3 = current - beta2
        wheel = np.vstack(
            (
                velocity[0] + 0.5 * track_width_m * velocity[1],
                velocity[0] - 0.5 * track_width_m * velocity[1],
            )
        )
        sigma2 = np.vstack((velocity, smooth_sign_basis(velocity), beta1_dot))
        sigma3 = np.vstack((current, wheel, beta2_dot))
        output["zdot2"].append(velocity_dot - beta1_dot)
        output["y2"].append(np.vstack((z2, sigma2)))
        output["x3"].append(current)
        output["zdot3"].append(current_dot - beta2_dot)
        output["y3"].append(np.vstack((z3, sigma3)))
        output["x4"].append(voltage)
        used_segments += 1
    if not output["zdot2"]:
        raise ValueError("没有包含至少 7 个有序样本的有效采集段")
    matrices = {key: np.hstack(value) for key, value in output.items()}
    dt_values = np.concatenate(all_dt)
    diagnostics = OfflineSnapshotDiagnostics(
        raw_samples=len(dataset),
        processed_samples=int(matrices["zdot2"].shape[1]),
        segments=used_segments,
        median_dt_s=float(np.median(dt_values)),
        max_dt_s=float(np.max(dt_values)),
        smoothing_window=int(smoothing_window),
    )
    return matrices, diagnostics


@dataclass(frozen=True)
class BlockQualification:
    design_dbar: float
    residual_rms: float
    residual_max: float
    observed_residual_covered: bool
    kappa: float
    kappa_min: float
    spectral_margin: float
    certificate_max: float
    route_ii_feasible: bool
    split_weight_drift: float
    cross_eigen_max: float
    physical_direction_ok: bool
    controller_direction_ok: bool

    @property
    def passed(self) -> bool:
        """Return only the paper-derived Route-II verdict.

        Held-out residuals, split drift, cross-segment eigenvalues, and
        direction checks are engineering diagnostics.  They are deliberately
        excluded from PASS because none is an assumption or synthesis
        condition in the manuscript.
        """
        return self.route_ii_feasible


@dataclass(frozen=True)
class QualificationReport:
    block2: BlockQualification
    block3: BlockQualification
    method: str = "firmware_integral_v3_sgn_gyro_gate_route_ii_v1"

    @property
    def passed(self) -> bool:
        return self.block2.passed and self.block3.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "passed": self.passed,
            "block2": asdict(self.block2),
            "block3": asdict(self.block3),
        }


def _fit_affine_snapshot_model(
    zdot: np.ndarray,
    y: np.ndarray,
    x_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.vstack((y, x_next)).T
    coefficients = np.linalg.lstsq(design, zdot.T, rcond=None)[0].T
    prediction = coefficients @ np.vstack((y, x_next))
    return coefficients[:, : y.shape[0]], coefficients[:, y.shape[0] :], zdot - prediction


def estimate_validation_residual(
    zdot: np.ndarray,
    y: np.ndarray,
    x_next: np.ndarray,
    *,
    validation_stride: int = 4,
) -> tuple[float, float]:
    """Return held-out residual diagnostics, not a disturbance bound.

    The residual contains model mismatch, sensor/filter errors, integration
    errors, and the window-averaged physical disturbance.  A finite-sample
    residual maximum therefore cannot certify the pointwise disturbance bound
    required by the robust synthesis theorem.
    """
    sample_count = int(np.asarray(zdot).shape[1])
    stride = int(validation_stride)
    if sample_count < 20:
        raise ValueError("independent disturbance validation requires at least 20 windows")
    if stride < 3:
        raise ValueError("validation_stride must be at least 3")
    validation = np.arange(stride - 1, sample_count, stride)
    training_mask = np.ones(sample_count, dtype=bool)
    training_mask[validation] = False
    training = np.flatnonzero(training_mask)
    if training.size < 10 or validation.size < 5:
        raise ValueError("not enough independent training/validation windows")
    model_a, model_b, _ = _fit_affine_snapshot_model(
        zdot[:, training], y[:, training], x_next[:, training]
    )
    prediction = model_a @ y[:, validation] + model_b @ x_next[:, validation]
    residual = zdot[:, validation] - prediction
    norms = np.linalg.norm(residual, axis=0)
    residual_rms = float(np.sqrt(np.mean(norms**2)))
    residual_max = float(np.max(norms))
    return residual_rms, residual_max


def _route_ii_components(
    z: np.ndarray,
    y: np.ndarray,
    dbar: float,
    epsilon: float,
) -> tuple[RouteIIThreshold, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact Route-II threshold and null-space construction data."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.ndim != 2 or y.ndim != 2:
        raise ValueError("snapshot matrices must be two-dimensional")
    if z.shape[0] != 2 or y.shape[0] != 8 or z.shape[1] != y.shape[1]:
        raise ValueError("expected Z:2xK and Y:8xK with equal sample counts")
    if z.shape[1] < 10:
        raise ValueError("at least 10 snapshots are required")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
        raise ValueError("snapshot matrices contain NaN or infinity")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(dbar) or dbar <= 0.0:
        raise ValueError("design dbar must be finite and positive")

    n, sample_count = z.shape
    c_matrix = np.vstack([np.eye(n), np.zeros((y.shape[0] - n, n))])
    yy = y @ y.T
    if np.linalg.matrix_rank(y, tol=1.0e-9) != y.shape[0]:
        raise ValueError("Y snapshot matrix is rank deficient")
    try:
        y_dagger = y.T @ np.linalg.solve(yy, np.eye(y.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Y snapshot matrix is rank deficient") from exc
    projector = np.eye(sample_count) - y_dagger @ y
    p_zero = y_dagger @ c_matrix
    projected = z @ projector
    sigma_min = float(np.linalg.svd(projected, compute_uv=False)[-1])
    chi = float(np.sqrt(sample_count) * dbar)
    spectral_margin = sigma_min - chi
    if spectral_margin <= 0.0:
        threshold = RouteIIThreshold(
            kappa_min=float("inf"),
            sigma_min=sigma_min,
            chi=chi,
            spectral_margin=spectral_margin,
        )
        return threshold, p_zero, projector, projected

    he_zp0 = z @ p_zero + (z @ p_zero).T
    numerator = (
        float(np.linalg.eigvalsh(0.5 * (he_zp0 + he_zp0.T)).max())
        + 2.0 * chi * float(np.linalg.norm(p_zero, 2))
        + epsilon
    )
    kappa_min = max(
        0.0,
        float(numerator / (2.0 * (1.0 - chi / sigma_min))),
    )
    threshold = RouteIIThreshold(
        kappa_min=kappa_min,
        sigma_min=sigma_min,
        chi=chi,
        spectral_margin=spectral_margin,
    )
    return threshold, p_zero, projector, projected


def route_ii_threshold(
    z: np.ndarray,
    y: np.ndarray,
    dbar: float,
    epsilon: float,
) -> RouteIIThreshold:
    """Compute the theorem's strict lower bound for the user-selected kappa."""
    threshold, _p_zero, _projector, _projected = _route_ii_components(
        z, y, dbar, epsilon
    )
    return threshold


def _qualify_block(
    zdot: np.ndarray,
    y: np.ndarray,
    x_next: np.ndarray,
    result: BlockSynthesis,
    *,
    kappa: float | tuple[float, float] | list[float],
    epsilon: float,
    dbar: float,
    block: int,
) -> BlockQualification:
    sample_count = zdot.shape[1]
    midpoint = sample_count // 2
    if midpoint < 10 or sample_count - midpoint < 10:
        raise ValueError("qualification requires at least 20 processed samples")
    halves = (np.arange(midpoint), np.arange(midpoint, sample_count))
    half_weights: list[np.ndarray] = []
    half_models: list[tuple[np.ndarray, np.ndarray]] = []
    for indices in halves:
        half_result = synthesize_block(
            zdot[:, indices], y[:, indices], x_next[:, indices],
            kappa, dbar, epsilon,
        )
        half_weights.append(half_result.weights)
        model_a, model_b, _ = _fit_affine_snapshot_model(
            zdot[:, indices], y[:, indices], x_next[:, indices]
        )
        half_models.append((model_a, model_b))
    denominator = max(float(np.linalg.norm(result.weights)), 1.0e-12)
    split_weight_drift = max(
        float(np.linalg.norm(weight - result.weights) / denominator)
        for weight in half_weights
    )
    cross_eigen_max = -np.inf
    for weight_index, weight in enumerate(half_weights):
        model_a, model_b = half_models[1 - weight_index]
        acl = model_a[:, :2] + model_b @ weight[:, :2]
        cross_eigen_max = max(
            cross_eigen_max,
            float(np.max(np.real(np.linalg.eigvals(acl)))),
        )
    residual_rms, residual_max = estimate_validation_residual(zdot, y, x_next)
    full_a, full_b, _ = _fit_affine_snapshot_model(zdot, y, x_next)
    del full_a
    if block == 2:
        physical_direction_ok = bool(
            0.5 * (full_b[0, 0] + full_b[0, 1]) > 0.0
            and 0.5 * (full_b[1, 0] - full_b[1, 1]) > 0.0
            and np.linalg.cond(full_b) < 100.0
        )
        controller_direction_ok = bool(
            result.weights[0, 0] < 0.0
            and result.weights[1, 0] < 0.0
            and result.weights[0, 1] < 0.0
            and result.weights[1, 1] > 0.0
        )
    else:
        physical_direction_ok = bool(
            full_b[0, 0] > 0.0
            and full_b[1, 1] > 0.0
            and abs(full_b[0, 0]) > abs(full_b[0, 1])
            and abs(full_b[1, 1]) > abs(full_b[1, 0])
            and np.linalg.cond(full_b) < 100.0
        )
        controller_direction_ok = bool(
            result.weights[0, 0] < 0.0 and result.weights[1, 1] < 0.0
        )
    kappa_rep, _kappa_v, _kappa_w = _resolve_kappa(kappa)
    return BlockQualification(
        design_dbar=float(dbar),
        residual_rms=residual_rms,
        residual_max=residual_max,
        observed_residual_covered=bool(dbar >= residual_max),
        kappa=kappa_rep,
        kappa_min=float(result.kappa_min),
        spectral_margin=float(result.spectral_margin),
        certificate_max=float(result.xi_max),
        route_ii_feasible=bool(
            result.spectral_margin > 0.0
            and kappa_rep > result.kappa_min
            and result.xi_max <= 0.0
        ),
        split_weight_drift=split_weight_drift,
        cross_eigen_max=float(cross_eigen_max),
        physical_direction_ok=physical_direction_ok,
        controller_direction_ok=controller_direction_ok,
    )


def qualify_synthesis(
    dataset: SnapshotDataset | RawTimeSeriesDataset | IntegralSnapshotDataset,
    result: SynthesisResult,
) -> QualificationReport:
    matrices = dataset.matrices()
    block2_kappa = (
        (result.kappa2_v, result.kappa2_w)
        if result.kappa2_v is not None
        else result.kappa2
    )
    block3_kappa = (
        (result.kappa3_v, result.kappa3_w)
        if result.kappa3_v is not None
        else result.kappa3
    )
    return QualificationReport(
        block2=_qualify_block(
            matrices["zdot2"], matrices["y2"], matrices["x3"], result.block2,
            kappa=block2_kappa, epsilon=result.epsilon2,
            dbar=result.dbar2, block=2,
        ),
        block3=_qualify_block(
            matrices["zdot3"], matrices["y3"], matrices["x4"], result.block3,
            kappa=block3_kappa, epsilon=result.epsilon3,
            dbar=result.dbar3, block=3,
        ),
    )


def _resolve_kappa(kappa: float | tuple[float, float] | list[float]) -> tuple[float, float | None, float | None]:
    """Resolve kappa into (representative, kappa_v, kappa_w).

    A scalar keeps the paper's ``P = P0 - kappa V`` construction; a 2-tuple
    applies a diagonal ``P = P0 - V diag(kappa_v, kappa_w)`` with one design
    gain per channel (v/omega).  The representative value is max(kappa_v,
    kappa_w) and keeps the payload's scalar ``kappa``/``lambda`` fields and
    the ``kappa > kappa_min`` gate consistent; the certificate is verified
    numerically on the actual solution either way.
    """
    if isinstance(kappa, (tuple, list, np.ndarray)):
        vector = np.asarray(kappa, dtype=float)
        if vector.shape != (2,):
            raise ValueError("kappa 向量必须为两个元素（v 通道、ω 通道）")
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
            raise ValueError("kappa 向量必须为两个正有限值")
        return float(np.max(vector)), float(vector[0]), float(vector[1])
    scalar = float(kappa)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("kappa must be finite and positive")
    return scalar, None, None


def synthesize_block(
    z: np.ndarray,
    y: np.ndarray,
    x_next: np.ndarray,
    kappa: float | tuple[float, float] | list[float],
    dbar: float,
    epsilon: float,
) -> BlockSynthesis:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    x_next = np.asarray(x_next, dtype=float)
    if z.ndim != 2 or y.ndim != 2 or x_next.ndim != 2:
        raise ValueError("snapshot matrices must be two-dimensional")
    if z.shape[0] != 2 or y.shape[0] != 8 or x_next.shape[0] != 2:
        raise ValueError("expected Z:2xK, Y:8xK, X:2xK")
    if not (z.shape[1] == y.shape[1] == x_next.shape[1]):
        raise ValueError("snapshot matrices have different sample counts")
    if z.shape[1] < 10:
        raise ValueError("at least 10 snapshots are required")
    if not all(np.all(np.isfinite(value)) for value in (z, y, x_next)):
        raise ValueError("snapshot matrices contain NaN or infinity")
    kappa_rep, kappa_v, kappa_w = _resolve_kappa(kappa)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(dbar) or dbar <= 0.0:
        raise ValueError("design dbar must be finite and positive")

    n, sample_count = z.shape
    regressor_rows = y.shape[0]
    c_matrix = np.vstack([np.eye(n), np.zeros((regressor_rows - n, n))])
    r_selector = np.vstack([np.zeros((n, regressor_rows - n)), np.eye(regressor_rows - n)])
    s_matrix = np.vstack([np.zeros((n, regressor_rows - n)), r_selector])
    g_matrix = np.vstack([z, y])
    rank = int(np.linalg.matrix_rank(g_matrix, tol=1.0e-9))
    if rank != g_matrix.shape[0]:
        raise ValueError("stacked snapshot matrix is rank deficient")
    gram = g_matrix @ g_matrix.T
    condition = float(np.linalg.cond(gram))
    threshold, p_zero, projector, projected = _route_ii_components(
        z, y, dbar, epsilon
    )
    if threshold.spectral_margin <= 0.0:
        raise ValueError(
            "Route II 数据条件不满足："
            f"sigma_min(N)={threshold.sigma_min:.6g} <= "
            f"chi={threshold.chi:.6g}"
        )
    if kappa_rep <= threshold.kappa_min:
        raise ValueError(
            f"Route II 要求 kappa>{threshold.kappa_min:.9g}，"
            f"当前代表值为 {kappa_rep:.9g}"
        )
    try:
        v_selector = projector @ projected.T @ np.linalg.solve(
            projected @ projected.T, np.eye(n)
        )
        if kappa_v is None:
            p_solution = p_zero - kappa_rep * v_selector
        else:
            # 对角 kappa：v 通道与 ω 通道独立设计增益
            p_solution = p_zero - v_selector @ np.diag([kappa_v, kappa_w])
        q_solution = g_matrix.T @ np.linalg.solve(gram, s_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Route II null-space construction is singular") from exc
    weights = x_next @ np.hstack([p_solution, q_solution])
    certificate = (
        z @ p_solution
        + (z @ p_solution).T
        + (2.0 * threshold.chi * np.linalg.norm(p_solution, 2) + epsilon)
        * np.eye(n)
    )
    p_match = y @ p_solution - c_matrix
    q_match = g_matrix @ q_solution - s_matrix
    match_residual = float(
        np.sqrt(
            np.linalg.norm(p_match, ord="fro") ** 2
            + np.linalg.norm(q_match, ord="fro") ** 2
        )
    )
    return BlockSynthesis(
        weights=weights,
        p_solution=p_solution,
        q_solution=q_solution,
        rank=rank,
        rows=int(g_matrix.shape[0]),
        condition=condition,
        match_residual=match_residual,
        nominal_residual=float(np.linalg.norm(z @ v_selector - np.eye(n), ord="fro")),
        xi_max=float(
            np.linalg.eigvalsh(0.5 * (certificate + certificate.T)).max()
        ),
        kappa=kappa_rep,
        kappa_v=kappa_v,
        kappa_w=kappa_w,
        kappa_min=threshold.kappa_min,
        sigma_min=threshold.sigma_min,
        chi=threshold.chi,
        spectral_margin=threshold.spectral_margin,
    )


def synthesize_dataset(
    dataset: SnapshotDataset | RawTimeSeriesDataset | IntegralSnapshotDataset,
    *,
    kappa2: float | tuple[float, float] | list[float] = 1.0,
    kappa3: float | tuple[float, float] | list[float] = 2.0,
    epsilon2: float = 0.8,
    epsilon3: float = 3.0,
    dbar2: float = 0.004,
    dbar3: float = 0.008,
) -> SynthesisResult:
    matrices = dataset.matrices()
    block2 = synthesize_block(matrices["zdot2"], matrices["y2"], matrices["x3"], kappa2, dbar2, epsilon2)
    block3 = synthesize_block(matrices["zdot3"], matrices["y3"], matrices["x4"], kappa3, dbar3, epsilon3)
    return SynthesisResult(
        block2=block2,
        block3=block3,
        samples=len(dataset),
        kappa2=block2.kappa,
        kappa3=block3.kappa,
        kappa2_v=block2.kappa_v,
        kappa2_w=block2.kappa_w,
        kappa3_v=block3.kappa_v,
        kappa3_w=block3.kappa_w,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
        dbar2=dbar2,
        dbar3=dbar3,
    )
