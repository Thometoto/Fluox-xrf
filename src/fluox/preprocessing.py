from pathlib import Path
from typing import Tuple

import numpy as np

from .config import SpectrumConfig
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


def read_spectrum(path: str, config: SpectrumConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Read two-column CSV/TSV/text and interpolate onto the model grid."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    raw = np.genfromtxt(file_path, delimiter=None, comments="#", invalid_raise=False)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.shape[0] < 3:
        # Deuxieme essai pour les CSV a virgules avec en-tete.
        raw = np.genfromtxt(file_path, delimiter=",", comments="#", skip_header=1)
    raw = raw[np.all(np.isfinite(raw[:, :2]), axis=1)]
    if raw.shape[0] < 3:
        raise ValueError("The file must contain at least 3 valid Energy, Counts rows")
    order = np.argsort(raw[:, 0])
    energy, counts = raw[order, 0], np.maximum(raw[order, 1], 0.0)
    target = energy_grid(config)
    if energy.min() > target.min() or energy.max() < target.max():
        raise ValueError(
            f"Insufficient range: {energy.min():.3f}-{energy.max():.3f} keV; "
            f"expected {target.min():.3f}-{target.max():.3f} keV"
        )
    return target.astype(np.float32), np.interp(target, energy, counts).astype(np.float32)
