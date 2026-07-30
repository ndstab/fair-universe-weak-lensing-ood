import numpy as np
import pytest

from wl_ood.power_spectrum import PowerSpectrumExtractor
from wl_ood.ps_config import PowerSpectrumConfig


def independent_reference(
    field: np.ndarray,
    pixel_size: float,
    edges: np.ndarray,
) -> np.ndarray:
    transform = np.fft.rfft2(field)
    result = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        total = 0.0
        weight = 0
        for i, fy in enumerate(np.fft.fftfreq(field.shape[0], d=pixel_size)):
            for j, fx in enumerate(np.fft.rfftfreq(field.shape[1], d=pixel_size)):
                ell = 2.0 * np.pi * np.sqrt(fy**2 + fx**2)
                if low < ell <= high:
                    symmetry = (
                        1
                        if j == 0
                        or (
                            field.shape[1] % 2 == 0
                            and j == field.shape[1] // 2
                        )
                        else 2
                    )
                    total += symmetry * float(abs(transform[i, j]) ** 2)
                    weight += symmetry
        result.append(total / weight * pixel_size**2 / field.size)
    return np.asarray(result)


def tiny_config() -> PowerSpectrumConfig:
    return PowerSpectrumConfig(
        pixel_size_arcmin=60.0 * 180.0 / np.pi,
        ell_min=0.5,
        ell_max=4.0,
        n_bins=3,
    )


def test_power_matches_independent_small_reference() -> None:
    config = tiny_config()
    field = np.arange(64, dtype=np.float64).reshape(8, 8)
    extractor = PowerSpectrumExtractor(field.shape, config)
    expected = independent_reference(
        field,
        config.pixel_size_radian,
        config.ell_edges,
    )
    np.testing.assert_allclose(extractor.extract_power(field), expected)


def test_log_power_rejects_non_positive_bins() -> None:
    extractor = PowerSpectrumExtractor((8, 8), tiny_config())
    with pytest.raises(ValueError, match="strictly positive"):
        extractor.extract_log_power(np.zeros((8, 8)))


def test_extractor_rejects_wrong_map_shape() -> None:
    extractor = PowerSpectrumExtractor((8, 8), tiny_config())
    with pytest.raises(ValueError, match="map shape"):
        extractor.extract_power(np.zeros((4, 4)))
