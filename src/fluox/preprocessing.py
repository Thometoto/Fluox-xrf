from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

from .config import DEFAULT_LINES, SpectrumConfig
from .simulator import energy_grid


def transform_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts[None, :]
    counts = np.maximum(counts, 0.0)
    transformed = np.log1p(counts)
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    # float64 avoids numerical instability in some BLAS backends while fitting
    # linear classifiers.
    return transformed / np.maximum(norms, 1e-12)


def estimate_background(counts: np.ndarray, config: SpectrumConfig) -> np.ndarray:
    """Estimate a nonlinear continuum with a compact SNIP-style algorithm."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    log_counts = np.log1p(np.maximum(values, 0.0))
    points_per_kev = config.channels / (config.energy_max_kev - config.energy_min_kev)
    minimum_size = max(9, int(round(0.16 * points_per_kev)) | 1)
    smooth_sigma = max(2.0, 0.055 * points_per_kev)
    baseline = minimum_filter1d(log_counts, size=minimum_size, axis=1, mode="nearest")
    max_shift = max(4, int(round(0.30 * points_per_kev)))
    shifts = np.unique(np.geomspace(2, max_shift, 10).astype(int))[::-1]
    for shift in shifts:
        center = baseline[:, shift:-shift]
        clipped = 0.5 * (baseline[:, :-2 * shift] + baseline[:, 2 * shift:])
        center[:] = np.minimum(center, clipped)
    return gaussian_filter1d(baseline, sigma=smooth_sigma, axis=1, mode="nearest")


def energy_reliability(config: SpectrumConfig) -> np.ndarray:
    """Reliability prior for a Mo tube; no channel is discarded."""
    energy = energy_grid(config)
    weight = np.ones_like(energy, dtype=np.float64)
    transition = np.clip((energy - 14.5) / 2.0, 0.0, 1.0)
    weight *= 1.0 - 0.55 * transition
    for center, width, depth in ((16.74, 0.38, 0.70), (17.48, 0.24, 0.82),
                                 (19.61, 0.30, 0.76)):
        weight *= 1.0 - depth * np.exp(-0.5 * ((energy - center) / width) ** 2)
    return np.clip(weight, 0.08, 1.0)


def extract_features(counts: np.ndarray, config: SpectrumConfig) -> np.ndarray:
    """Physics-informed features from every channel of the calibrated spectrum."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    values = np.maximum(values, 0.0)
    log_counts = np.log1p(values)
    background = estimate_background(values, config)
    residual = np.maximum(log_counts - background, 0.0)
    reliability = energy_reliability(config)[None, :]

    # Pool four neighboring channels for the global branches. All 4096 input
    # channels still contribute, while the local line features below retain
    # the instrument's original sampling.
    usable = (config.channels // 4) * 4
    global_log = log_counts[:, :usable].reshape(len(values), -1, 4).mean(axis=2)
    global_residual = (residual[:, :usable] * reliability[:, :usable]).reshape(
        len(values), -1, 4
    ).mean(axis=2)
    global_log /= np.maximum(np.linalg.norm(global_log, axis=1, keepdims=True), 1e-12)
    global_residual /= np.maximum(
        np.linalg.norm(global_residual, axis=1, keepdims=True), 1e-12
    )

    energy = energy_grid(config)
    local = []
    for element in config.elements:
        element_features = []
        for line_energy, _ in DEFAULT_LINES[element][:2]:
            if not (config.energy_min_kev <= line_energy <= config.energy_max_kev):
                zeros = np.zeros(len(values), dtype=np.float64)
                element_features.extend((zeros, zeros, zeros))
                continue
            signal = np.abs(energy - line_energy) <= 0.10
            side = ((np.abs(energy - line_energy) >= 0.16)
                    & (np.abs(energy - line_energy) <= 0.32))
            peak = residual[:, signal]
            raw_signal = log_counts[:, signal]
            side_values = log_counts[:, side]
            local_floor = np.median(side_values, axis=1) if side.any() else 0.0
            element_features.extend((
                np.max(peak, axis=1),
                np.mean(peak, axis=1),
                np.maximum(np.max(raw_signal, axis=1) - local_floor, 0.0),
            ))
        local.extend(element_features)
    local_features = np.column_stack(local)
    return np.hstack((global_log, global_residual, local_features))


def load_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read and return every valid point from a two-column spectrum file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    raw = np.genfromtxt(file_path, delimiter=None, comments="#", invalid_raise=False)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.shape[0] < 3:
        # A textual header becomes NaN and is removed below, while a numeric
        # first row is preserved.
        raw = np.genfromtxt(file_path, delimiter=",", comments="#", invalid_raise=False)
    raw = raw[np.all(np.isfinite(raw[:, :2]), axis=1)]
    if raw.shape[0] < 3:
        raise ValueError("The file must contain at least 3 valid Energy, Counts rows")
    order = np.argsort(raw[:, 0])
    energy, counts = raw[order, 0], np.maximum(raw[order, 1], 0.0)
    return energy.astype(np.float32), counts.astype(np.float32)


def read_spectrum(path: str, config: SpectrumConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a spectrum onto the fixed grid expected by the model."""
    energy, counts = load_spectrum(path)
    target = energy_grid(config)
    if energy.min() > target.min() or energy.max() < target.max():
        raise ValueError(
            f"Insufficient range: {energy.min():.3f}-{energy.max():.3f} keV; "
            f"expected {target.min():.3f}-{target.max():.3f} keV"
        )
    return target.astype(np.float32), np.interp(target, energy, counts).astype(np.float32)
