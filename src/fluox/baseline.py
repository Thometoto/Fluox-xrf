from typing import Dict

import numpy as np

from .config import DEFAULT_LINES, SpectrumConfig
from .preprocessing import estimate_background


def peak_scores(energy: np.ndarray, counts: np.ndarray, config: SpectrumConfig) -> Dict[str, float]:
    """Return an interpretable signal/background score around the main line."""
    scores = {}
    counts = np.asarray(counts, dtype=float)
    for element in config.elements:
        center = DEFAULT_LINES[element][0][0]
        signal = (np.abs(energy - center) <= 0.10)
        side = (np.abs(energy - center) >= 0.18) & (np.abs(energy - center) <= 0.35)
        if not signal.any() or not side.any():
            scores[element] = 0.0
            continue
        background = np.median(counts[side])
        excess = max(float(np.max(counts[signal]) - background), 0.0)
        scores[element] = excess / np.sqrt(max(background, 1.0))
    return scores


def multi_line_evidence(counts: np.ndarray, config: SpectrumConfig) -> Dict[str, dict]:
    """Return background-corrected evidence for every accessible line."""
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1:
        raise ValueError("multi_line_evidence expects one spectrum")
    energy = np.linspace(config.energy_min_kev, config.energy_max_kev, config.channels)
    background = np.expm1(estimate_background(values, config)[0])
    evidence = {}
    for element in config.elements:
        line_scores = []
        for line_energy, _ in DEFAULT_LINES[element]:
            if not (energy[0] <= line_energy <= energy[-1]):
                continue
            signal = np.abs(energy - line_energy) <= 0.10
            if not signal.any():
                continue
            excess = np.maximum(values[signal] - background[signal], 0.0)
            noise = np.sqrt(np.maximum(background[signal].sum(), 1.0))
            line_scores.append(float(excess.sum() / noise))
        supported = sum(score >= 3.0 for score in line_scores)
        evidence[element] = {
            "line_snrs": line_scores,
            "max_snr": max(line_scores, default=0.0),
            "supported_lines": supported,
        }
    return evidence
