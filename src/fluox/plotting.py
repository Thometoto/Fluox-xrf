from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_LINES


def plot_spectrum(
    energy: np.ndarray,
    counts: np.ndarray,
    output: str,
    elements: Iterable[str] = (),
    title: str = "XRF spectrum",
    probabilities: Optional[dict] = None,
) -> None:
    """Save a spectrum annotated with the expected emission lines."""
    selected = list(elements)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(energy, counts, color="#155eef", linewidth=0.9, label="Comptages")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
    ymax = max(float(np.max(counts)), 1.0)
    for color, element in zip(colors, selected):
        for line_index, (line_energy, _) in enumerate(DEFAULT_LINES[element]):
            if energy.min() <= line_energy <= energy.max():
                ax.axvline(line_energy, color=color, alpha=0.55, linewidth=1.0, linestyle="--")
                if line_index == 0:
                    suffix = ""
                    if probabilities and element in probabilities:
                        suffix = f" ({probabilities[element]:.2f})"
                    ax.text(line_energy, ymax * 0.92, element + suffix, rotation=90,
                            va="top", ha="right", color=color, fontsize=9)
    ax.set(title=title, xlabel="Energy (keV)", ylabel="Counts")
    ax.set_xlim(float(energy.min()), float(energy.max()))
    ax.grid(alpha=0.18)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
