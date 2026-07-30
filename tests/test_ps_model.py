from pathlib import Path

import numpy as np
import pytest

from wl_ood.ps_model import PowerSpectrumModel, fit_moments


COSMOLOGIES = np.array(
    [[0.2, 0.7], [0.2, 0.9], [0.4, 0.7], [0.4, 0.9]],
    dtype=np.float64,
)


def gaussian_model() -> PowerSpectrumModel:
    means = np.column_stack([COSMOLOGIES[:, 0], COSMOLOGIES[:, 1]])
    covariances = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    return PowerSpectrumModel(COSMOLOGIES, means, covariances)


def test_fit_moments_uses_organizer_denominator() -> None:
    features = np.arange(4 * 8 * 2, dtype=np.float64).reshape(4, 8, 2)
    _, covariances = fit_moments(features, np.arange(8))
    delta = features[0] - features[0].mean(axis=0)
    expected = delta.T @ delta / (8 - 2 - 2)
    np.testing.assert_allclose(covariances[0], expected)


def test_likelihood_and_chi2_match_known_gaussian() -> None:
    model = gaussian_model()
    feature = np.array([0.3, 0.8])
    assert model.chi_squared(np.array([0.3, 0.8]), feature) == pytest.approx(0.0)
    assert model.negative_log_likelihood(
        np.array([0.2, 0.7]),
        feature,
    ) == pytest.approx(0.01)


def test_interpolation_outside_hull_is_rejected() -> None:
    model = gaussian_model()
    assert np.isinf(
        model.negative_log_likelihood(
            np.array([0.1, 0.6]),
            np.array([0.3, 0.8]),
        )
    )


def test_score_one_returns_finite_best_fit() -> None:
    result = gaussian_model().score_one(np.array([0.31, 0.79]))
    np.testing.assert_allclose(result.parameters, [0.31, 0.79], atol=1e-4)
    assert result.chi_squared == pytest.approx(0.0, abs=1e-8)
    assert np.isfinite(result.objective)
    assert result.evaluations > 0


def test_model_round_trip(tmp_path: Path) -> None:
    original = gaussian_model()
    loaded = PowerSpectrumModel.load(original.save(tmp_path / "model.npz"))
    np.testing.assert_array_equal(loaded.cosmologies, original.cosmologies)
    np.testing.assert_array_equal(loaded.means, original.means)
    np.testing.assert_array_equal(loaded.covariances, original.covariances)
