from __future__ import annotations

import numpy as np

from scripts.score_physics_field_detector import (
    empirical_cdf_score,
    fit_embedding_typicality,
    stratified_calibration_rows,
)


def test_empirical_cdf_score_uses_only_frozen_calibration_values() -> None:
    """A test-batch rank would violate the single-instance submission protocol."""

    calibration = np.asarray([0.0, 1.0, 2.0, 3.0])
    values = np.asarray([-1.0, 0.5, 4.0])

    scores = empirical_cdf_score(calibration, values)

    np.testing.assert_allclose(scores, np.asarray([0.1, 0.3, 0.9]))
    one_at_a_time = np.asarray(
        [empirical_cdf_score(calibration, value[None])[0] for value in values]
    )
    np.testing.assert_allclose(scores, one_at_a_time)


def test_embedding_typicality_assigns_larger_distance_to_far_field() -> None:
    """The standalone typicality head must respond to multivariate field shifts."""

    calibration = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    scorer = fit_embedding_typicality(calibration)

    distances = scorer(np.asarray([[0.0, 0.0], [8.0, 8.0]]))

    assert distances.shape == (2,)
    assert distances[1] > 20.0 * distances[0]


def test_calibration_rows_cover_cosmologies_before_reusing_one() -> None:
    """Low-FPR calibration must not be dominated by the first cosmologies on disk."""

    rows = stratified_calibration_rows(
        cosmologies=3, realizations=8, held_fold=3, count=6
    )

    np.testing.assert_array_equal(rows, np.asarray([3, 11, 19, 7, 15, 23]))
