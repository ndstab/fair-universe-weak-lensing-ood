import numpy as np

from wl_ood.noise import (
    noise_standard_deviation,
    noisy_training_map,
    training_noise,
)
from wl_ood.ps_config import PowerSpectrumConfig


def test_official_noise_scale() -> None:
    config = PowerSpectrumConfig()
    expected = 0.4 / np.sqrt(2 * 30 * 2.0**2)
    assert noise_standard_deviation(config) == expected


def test_map_noise_is_order_independent() -> None:
    config = PowerSpectrumConfig(seed=17)
    first = training_noise(8, config, cosmology=3, realization=5)
    _ = training_noise(8, config, cosmology=8, realization=2)
    repeated = training_noise(8, config, cosmology=3, realization=5)
    np.testing.assert_array_equal(first, repeated)


def test_different_map_indices_receive_different_noise() -> None:
    config = PowerSpectrumConfig(seed=17)
    first = training_noise(8, config, cosmology=3, realization=5)
    second = training_noise(8, config, cosmology=3, realization=6)
    assert not np.array_equal(first, second)


def test_noisy_training_map_keeps_masked_pixels_zero() -> None:
    mask = np.array([[True, False], [True, False]])
    full = noisy_training_map(
        np.array([0.1, 0.2]),
        mask,
        PowerSpectrumConfig(seed=17),
        cosmology=0,
        realization=0,
    )
    assert full.shape == mask.shape
    assert np.all(full[~mask] == 0.0)
    assert np.all(np.isfinite(full[mask]))
