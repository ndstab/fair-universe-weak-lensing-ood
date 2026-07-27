"""Exact local implementation of the FAIR Universe Phase-2 metric."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class LowFprMetricResult:
    """Interpolated ROC values and their mean over the challenge FPR grid."""

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    score: float


def compute_low_fpr_metric(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    min_fpr: float = 0.001,
    max_fpr: float = 0.05,
    n_points: int = 100,
) -> LowFprMetricResult:
    """Compute the organizer's mean TPR over a logarithmic low-FPR grid."""
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=np.float64)

    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if labels_array.size == 0 or labels_array.size != scores_array.size:
        raise ValueError("labels and scores must have the same length")
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must contain only finite values")

    unique_labels = np.unique(labels_array)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError("labels must be binary values 0 or 1")
    if unique_labels.size != 2:
        raise ValueError("labels must contain both classes")

    if not (0.0 < min_fpr < max_fpr <= 1.0) or n_points < 2:
        raise ValueError("FPR configuration must satisfy 0 < min < max <= 1")

    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    target_fpr = np.logspace(
        np.log10(min_fpr),
        np.log10(max_fpr),
        n_points,
    )
    target_tpr = np.interp(target_fpr, fpr, tpr)

    return LowFprMetricResult(
        fpr=target_fpr,
        tpr=target_tpr,
        score=float(np.mean(target_tpr)),
    )
