import numpy as np

from wl_ood.ps_validation import (
    DEFAULT_FPR_LEVELS,
    calibration_thresholds,
    evaluate_ind_tail,
    split_realizations,
)


def test_official_split_membership_and_counts() -> None:
    split = split_realizations(256)
    assert split.fit.size == 153
    assert split.calibration.size == 51
    assert split.validation.size == 52
    assert np.all(np.isin(split.fit % 5, [2, 3, 4]))
    assert np.all(split.calibration % 5 == 1)
    assert np.all(split.validation % 5 == 0)


def test_thresholds_use_higher_calibration_quantile_only() -> None:
    calibration = np.arange(100, dtype=np.float64)
    thresholds = calibration_thresholds(calibration, (0.05,))
    assert thresholds[0.05] == 95.0


def test_validation_values_do_not_change_thresholds() -> None:
    calibration = np.arange(100, dtype=np.float64)
    low = evaluate_ind_tail(calibration, np.zeros(20))
    high = evaluate_ind_tail(calibration, np.full(20, 1000.0))
    assert low["thresholds"] == high["thresholds"]
    assert set(low["false_positive_rates"]) == set(DEFAULT_FPR_LEVELS)
    assert all(value == 0.0 for value in low["false_positive_rates"].values())
    assert all(value == 1.0 for value in high["false_positive_rates"].values())
