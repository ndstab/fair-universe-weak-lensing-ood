import numpy as np
import pytest
from sklearn.metrics import roc_curve

from wl_ood.metric import compute_low_fpr_metric


def official_reference(labels: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(labels, scores)
    target_fpr = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    target_tpr = np.interp(target_fpr, fpr, tpr)
    return target_fpr, target_tpr, float(np.mean(target_tpr))


def test_matches_official_reference_formula() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0.05, 0.90, 0.30, 0.60, 0.10, 0.80, 0.20, 0.70])

    expected_fpr, expected_tpr, expected_score = official_reference(labels, scores)
    result = compute_low_fpr_metric(labels, scores)

    np.testing.assert_allclose(result.fpr, expected_fpr)
    np.testing.assert_allclose(result.tpr, expected_tpr)
    assert result.score == pytest.approx(expected_score)


def test_perfect_ranking_scores_one() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = compute_low_fpr_metric(labels, scores)

    assert result.score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        (np.array([[0, 1]]), np.array([0.1, 0.9]), "one-dimensional"),
        (np.array([0, 1]), np.array([[0.1, 0.9]]), "one-dimensional"),
        (np.array([0, 1]), np.array([0.1]), "same length"),
        (np.array([0, 2]), np.array([0.1, 0.9]), "binary"),
        (np.array([0, 0]), np.array([0.1, 0.9]), "both classes"),
        (np.array([0, 1]), np.array([0.1, np.nan]), "finite"),
    ],
)
def test_rejects_invalid_inputs(labels, scores, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_low_fpr_metric(labels, scores)


def test_rejects_invalid_fpr_configuration() -> None:
    with pytest.raises(ValueError, match="FPR"):
        compute_low_fpr_metric(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            min_fpr=0.1,
            max_fpr=0.05,
        )
