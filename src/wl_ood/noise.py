"""Organizer-compatible shape noise for noiseless training maps."""

import numpy as np

from wl_ood.maps import reconstruct_map
from wl_ood.ps_config import PowerSpectrumConfig


def noise_standard_deviation(config: PowerSpectrumConfig) -> float:
    return 0.4 / np.sqrt(
        2.0 * config.galaxy_density * config.pixel_size_arcmin**2
    )


def training_noise(
    size: int,
    config: PowerSpectrumConfig,
    *,
    cosmology: int,
    realization: int,
) -> np.ndarray:
    seed = np.random.SeedSequence(
        [config.seed, config.noise_namespace, cosmology, realization]
    )
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, noise_standard_deviation(config), size=size)


def noisy_training_map(
    values: np.ndarray,
    mask: np.ndarray,
    config: PowerSpectrumConfig,
    *,
    cosmology: int,
    realization: int,
) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    full = reconstruct_map(values, mask_array, fill_value=0.0).astype(
        np.float64,
        copy=False,
    )
    full[mask_array] += training_noise(
        int(np.count_nonzero(mask_array)),
        config,
        cosmology=cosmology,
        realization=realization,
    )
    return full
