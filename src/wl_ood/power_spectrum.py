"""One-map angular power-spectrum summaries."""

import numpy as np

from wl_ood.ps_config import PowerSpectrumConfig


class PowerSpectrumExtractor:
    """Precompute Fourier-mode bins, then summarize individual maps."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        config: PowerSpectrumConfig,
    ) -> None:
        self.map_shape = map_shape
        self.config = config

        frequency_y = np.fft.fftfreq(
            map_shape[0],
            d=config.pixel_size_radian,
        )[:, None]
        frequency_x = np.fft.rfftfreq(
            map_shape[1],
            d=config.pixel_size_radian,
        )[None, :]
        ell = 2.0 * np.pi * np.sqrt(frequency_y**2 + frequency_x**2)
        bins = np.searchsorted(config.ell_edges, ell) - 1

        weights = np.full(ell.shape, 2.0, dtype=np.float64)
        weights[:, 0] = 1.0
        if map_shape[1] % 2 == 0:
            weights[:, -1] = 1.0

        flat_bins = bins.ravel()
        flat_weights = weights.ravel()
        self._indices = tuple(
            np.flatnonzero(flat_bins == index) for index in range(config.n_bins)
        )
        self._weights = tuple(
            flat_weights[indices] for indices in self._indices
        )
        self._mode_weights = np.asarray(
            [weights_for_bin.sum() for weights_for_bin in self._weights],
            dtype=np.float64,
        )
        if np.any(self._mode_weights == 0.0):
            raise ValueError("every power-spectrum bin must contain a mode")

    def extract_power(self, full_map: np.ndarray) -> np.ndarray:
        field = np.asarray(full_map, dtype=np.float64)
        if field.shape != self.map_shape:
            raise ValueError(
                f"map shape: expected {self.map_shape}, found {field.shape}"
            )

        squared = (np.abs(np.fft.rfft2(field)) ** 2).ravel()
        power = np.asarray(
            [
                np.sum(squared[indices] * weights) / mode_weight
                for indices, weights, mode_weight in zip(
                    self._indices,
                    self._weights,
                    self._mode_weights,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        power *= self.config.pixel_size_radian**2 / field.size
        return power

    def extract_log_power(self, full_map: np.ndarray) -> np.ndarray:
        power = self.extract_power(full_map)
        if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
            raise ValueError(
                "power-spectrum bins must be finite and strictly positive"
            )
        return np.log10(power)
