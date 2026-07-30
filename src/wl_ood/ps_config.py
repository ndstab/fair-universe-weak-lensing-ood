"""Fixed configuration for the Phase-2 power-spectrum baseline."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerSpectrumConfig:
    algorithm_version: str = "ps-baseline-v1"
    seed: int = 20260730
    noise_namespace: int = 1
    galaxy_density: float = 30.0
    pixel_size_arcmin: float = 2.0
    ell_min: float = 100.0
    ell_max: float = 10_000.0
    n_bins: int = 10
    feature_dtype: str = "float64"

    @property
    def pixel_size_radian(self) -> float:
        return self.pixel_size_arcmin / 60.0 / 180.0 * np.pi

    @property
    def ell_edges(self) -> np.ndarray:
        return np.logspace(
            np.log10(self.ell_min),
            np.log10(self.ell_max),
            self.n_bins + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ell_edges"] = self.ell_edges.tolist()
        payload["pixel_size_radian"] = self.pixel_size_radian
        return payload
