from typing import Dict

import numpy as np

from .config import DEFAULT_LINES, SpectrumConfig


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
