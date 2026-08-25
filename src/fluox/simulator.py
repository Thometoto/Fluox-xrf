from typing import Optional, Tuple

import numpy as np

from .config import ANODE_LINES, DEFAULT_LINES, SpectrumConfig


def energy_grid(config: SpectrumConfig) -> np.ndarray:
    return np.linspace(config.energy_min_kev, config.energy_max_kev, config.channels)


def _gaussian(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _sigma_kev(energy_kev: float, config: SpectrumConfig) -> float:
    # SDD approximation: FWHM scales with sqrt(E).
    fwhm = config.resolution_fwhm_ev_at_5_9kev / 1000.0
    return (fwhm * np.sqrt(max(energy_kev, 0.1) / 5.9)) / 2.35482


def simulate_spectrum(
    config: SpectrumConfig,
    rng: np.random.Generator,
    selected: Optional[np.ndarray] = None,
    background_template: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (energy, counts, binary labels)."""
    elements = config.elements
    if selected is None:
        count = int(rng.integers(config.min_elements, config.max_elements + 1))
        selected = rng.choice(len(elements), size=count, replace=False)

    labels = np.zeros(len(elements), dtype=np.uint8)
    labels[selected] = 1
    nominal_energy = energy_grid(config)
    shift = rng.uniform(-config.calibration_shift_ev, config.calibration_shift_ev) / 1000.0
    gain = rng.uniform(-config.calibration_gain_fraction, config.calibration_gain_fraction)
    observed_energy = nominal_energy * (1.0 + gain) + shift

    if background_template is None:
        # Generic continuum when no experimental reference is supplied.
        decay = rng.uniform(2.5, 8.0)
        spectrum = rng.uniform(0.03, 0.12) * np.exp(-(observed_energy - config.energy_min_kev) / decay)
        spectrum += rng.uniform(0.002, 0.02)
        spectrum += rng.uniform(0.0, 0.006) * (observed_energy - config.energy_min_kev)
    else:
        template = np.asarray(background_template, dtype=float)
        if template.shape != nominal_energy.shape:
            raise ValueError("Background template does not match the model grid")
        template = np.maximum(template, 0.0)
        template /= max(float(template.max()), 1e-12)
        spectrum = template * rng.uniform(0.035, 0.16) + rng.uniform(0.002, 0.015)
        spectrum *= rng.uniform(0.8, 1.2) + rng.uniform(-0.015, 0.015) * (
            observed_energy - observed_energy.mean()
        )

    # Element contributions with log-uniform concentrations.
    for idx in selected:
        element = elements[int(idx)]
        # Include trace-like peaks as well as major components.
        abundance = 10.0 ** rng.uniform(-2.1, 0.5)
        matrix_attenuation = rng.uniform(0.55, 1.15)
        for line_energy, relative in DEFAULT_LINES[element]:
            if config.energy_min_kev <= line_energy <= config.energy_max_kev:
                width = _sigma_kev(line_energy, config) * rng.uniform(0.9, 1.15)
                spectrum += abundance * relative * matrix_attenuation * _gaussian(
                    observed_energy, line_energy, width
                )

    # Scattering of source-anode lines varies independently of that element in
    # the sample, intentionally making the anode material non-identifiable.
    scatter_scale = 10.0 ** rng.uniform(-1.5 if background_template is not None else -1.2,
                                       -0.15 if background_template is not None else 0.2)
    for line_energy, relative in ANODE_LINES[config.anode_material]:
        if config.energy_min_kev <= line_energy <= config.energy_max_kev:
            width = _sigma_kev(line_energy, config) * rng.uniform(1.2, 2.3)
            center = line_energy + rng.uniform(-0.12, 0.12)
            spectrum += scatter_scale * relative * _gaussian(observed_energy, center, width)

        # Broad Compton component. Its energy follows the photon-scattering
        # relation for a randomized effective angle and is lower than the
        # elastic line. This is essential around the Mo K lines, where the
        # Compton structure overlaps Y/Zr emission regions.
        angle_rad = np.deg2rad(rng.uniform(55.0, 145.0))
        compton_center = line_energy / (
            1.0 + (line_energy / 511.0) * (1.0 - np.cos(angle_rad))
        )
        compton_center += rng.uniform(-0.12, 0.12)
        compton_width = _sigma_kev(line_energy, config) * rng.uniform(2.0, 4.5)
        compton_scale = scatter_scale * relative * rng.uniform(0.25, 1.1)
        # Keep the low-energy tail when a Compton structure lies just above
        # the analysis boundary (notably for a Mo tube).
        if compton_center - 5.0 * compton_width <= config.energy_max_kev:
            spectrum += compton_scale * _gaussian(
                observed_energy, compton_center, compton_width
            )

    spectrum = np.maximum(spectrum, 0.0)
    total_counts = int(10.0 ** rng.uniform(np.log10(config.total_counts_min),
                                          np.log10(config.total_counts_max)))
    expected = spectrum / spectrum.sum() * total_counts
    counts = rng.poisson(expected).astype(np.float32)
    return nominal_energy.astype(np.float32), counts, labels


def generate_dataset(
    config: SpectrumConfig,
    samples: int,
    seed: Optional[int] = None,
    background_templates: Optional[np.ndarray] = None,
):
    rng = np.random.default_rng(config.random_seed if seed is None else seed)
    x = np.empty((samples, config.channels), dtype=np.float32)
    y = np.empty((samples, len(config.elements)), dtype=np.uint8)
    energy = energy_grid(config).astype(np.float32)
    for i in range(samples):
        template = None
        if background_templates is not None and len(background_templates):
            template = background_templates[int(rng.integers(len(background_templates)))]
        _, x[i], y[i] = simulate_spectrum(config, rng, background_template=template)
    return energy, x, y
