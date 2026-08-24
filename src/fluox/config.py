from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


# Approximate energies (keV) and relative intensities. These values support the
# prototype; a future version will load them from a versioned atomic database.
DEFAULT_LINES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "P": ((2.014, 1.0), (2.139, 0.10)),
    "Ar": ((2.958, 1.0), (3.190, 0.11)),
    "Ca": ((3.692, 1.0), (4.013, 0.13)),
    "Ti": ((4.511, 1.0), (4.932, 0.12)),
    "V": ((4.952, 1.0), (5.427, 0.13)),
    "Cr": ((5.415, 1.0), (5.947, 0.14)),
    "Mn": ((5.899, 1.0), (6.490, 0.14)),
    "Fe": ((6.404, 1.0), (7.058, 0.15)),
    "Co": ((6.930, 1.0), (7.649, 0.15)),
    "Ni": ((7.478, 1.0), (8.265, 0.16)),
    "Cu": ((8.048, 1.0), (8.905, 0.16)),
    "Zn": ((8.638, 1.0), (9.572, 0.17)),
    "Ga": ((9.251, 1.0), (10.264, 0.17)),
    "As": ((10.544, 1.0), (11.726, 0.18)),
    "Se": ((11.222, 1.0), (12.496, 0.18)),
    "Rb": ((13.395, 1.0), (14.961, 0.19)),
    "Sr": ((14.165, 1.0), (15.835, 0.19)),
    "Y": ((14.958, 1.0), (16.738, 0.20)),
    "Zr": ((15.775, 1.0), (17.668, 0.20)),
    # Pb L lines fall within the usual measurement range.
    "Pb": ((10.552, 0.45), (12.614, 1.0), (14.765, 0.28)),
    "Mo": ((17.480, 1.0), (19.608, 0.52)),
    "Ag": ((22.163, 1.0), (24.942, 0.48)),
}

ANODE_LINES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "Cu": ((8.048, 1.0), (8.905, 0.16)),
    "Mo": ((17.479, 1.0), (19.608, 0.52)),
    "Ag": ((22.163, 1.0), (24.942, 0.48)),
}


@dataclass(frozen=True)
class SpectrumConfig:
    energy_min_kev: float = 1.0
    energy_max_kev: float = 20.0
    channels: int = 1024
    resolution_fwhm_ev_at_5_9kev: float = 160.0
    min_elements: int = 1
    max_elements: int = 6
    total_counts_min: int = 20_000
    total_counts_max: int = 250_000
    calibration_shift_ev: float = 25.0
    calibration_gain_fraction: float = 0.0015
    random_seed: int = 42
    anode_material: str = "Mo"

    @property
    def elements(self) -> Tuple[str, ...]:
        if self.anode_material not in ANODE_LINES:
            raise ValueError(f"Unsupported anode material: {self.anode_material}")
        # The anode element is excluded because its sample lines overlap with
        # scattering from the source.
        return tuple(
            element
            for element, lines in DEFAULT_LINES.items()
            if element != self.anode_material
            and any(self.energy_min_kev <= energy <= self.energy_max_kev for energy, _ in lines)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SpectrumConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in values.items() if k in allowed})
