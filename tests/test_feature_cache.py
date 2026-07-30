from pathlib import Path

import numpy as np
import pytest

from wl_ood.feature_cache import FeatureCache, require_run_config


def test_run_config_rejects_incompatible_resume(tmp_path: Path) -> None:
    require_run_config(tmp_path, {"algorithm_version": "v1", "seed": 1})
    with pytest.raises(ValueError, match="incompatible"):
        require_run_config(tmp_path, {"algorithm_version": "v1", "seed": 2})


def test_written_rows_survive_reopen_and_are_not_pending(tmp_path: Path) -> None:
    cache = FeatureCache.open(tmp_path, "train", (3, 2))
    cache.write((1,), np.array([4.0, 5.0]))

    reopened = FeatureCache.open(tmp_path, "train", (3, 2))

    np.testing.assert_array_equal(reopened.values[1], [4.0, 5.0])
    assert reopened.complete.tolist() == [False, True, False]
    assert list(reopened.pending_indices()) == [(0,), (2,)]


def test_cache_rejects_shape_mismatch(tmp_path: Path) -> None:
    FeatureCache.open(tmp_path, "test", (3, 2))
    with pytest.raises(ValueError, match="shape"):
        FeatureCache.open(tmp_path, "test", (4, 2))


def test_cache_rejects_non_finite_feature_row(tmp_path: Path) -> None:
    cache = FeatureCache.open(tmp_path, "test", (3, 2))
    with pytest.raises(ValueError, match="finite"):
        cache.write((0,), np.array([1.0, np.nan]))
    assert not bool(cache.complete[0])
