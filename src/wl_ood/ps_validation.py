"""Training-only split and InD-tail diagnostics."""

from dataclasses import dataclass

import numpy as np


DEFAULT_FPR_LEVELS = (0.05, 0.01, 0.005, 0.001)


@dataclass(frozen=True)
class ValidationSplit:
    fit: np.ndarray
    calibration: np.ndarray
    validation: np.ndarray


def split_realizations(count: int) -> ValidationSplit:
    if count < 1:
        raise ValueError("realization count must be positive")
    indices = np.arange(count)
    return ValidationSplit(
        fit=indices[np.isin(indices % 5, [2, 3, 4])],
        calibration=indices[indices % 5 == 1],
        validation=indices[indices % 5 == 0],
    )


def calibration_thresholds(
    scores: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_FPR_LEVELS,
) -> dict[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("calibration scores must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    if any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("false-positive levels must lie between zero and one")
    return {
        float(level): float(
            np.quantile(values, 1.0 - level, method="higher")
        )
        for level in levels
    }


def evaluate_ind_tail(
    calibration_scores: np.ndarray,
    validation_scores: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_FPR_LEVELS,
) -> dict:
    thresholds = calibration_thresholds(calibration_scores, levels)
    validation = np.asarray(validation_scores, dtype=np.float64)
    if validation.ndim != 1 or validation.size == 0:
        raise ValueError("validation scores must be a non-empty vector")
    if not np.all(np.isfinite(validation)):
        raise ValueError("validation scores must be finite")
    rates = {
        level: float(np.mean(validation > threshold))
        for level, threshold in thresholds.items()
    }
    return {
        "thresholds": thresholds,
        "false_positive_rates": rates,
        "calibration_count": int(np.size(calibration_scores)),
        "validation_count": int(validation.size),
    }
