import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from wl_ood.data import CompetitionDataset, DatasetPaths
from wl_ood.feature_cache import FeatureCache
from wl_ood.power_spectrum import PowerSpectrumExtractor
from wl_ood.ps_config import PowerSpectrumConfig
from wl_ood.ps_pipeline import (
    benchmark_features,
    extract_test_features,
    extract_training_features,
    run_full_baseline,
    run_validation,
)


@pytest.fixture
def tiny_ps_dataset(tmp_path: Path) -> CompetitionDataset:
    rng = np.random.default_rng(2026)
    mask = np.ones((8, 8), dtype=bool)
    training = rng.normal(size=(4, 15, 64)).astype(np.float64)
    test = rng.normal(size=(3, 64)).astype(np.float64)
    labels = np.zeros((4, 15, 5), dtype=np.float64)
    cosmologies = np.array(
        [[0.2, 0.7], [0.2, 0.9], [0.4, 0.7], [0.4, 0.9]],
    )
    labels[:, :, :2] = cosmologies[:, None, :]
    return CompetitionDataset(
        paths=DatasetPaths.from_dir(tmp_path),
        training=training,
        test=test,
        mask=mask,
        labels=labels,
    )


def tiny_ps_config(seed: int = 17) -> PowerSpectrumConfig:
    return PowerSpectrumConfig(
        seed=seed,
        pixel_size_arcmin=60.0 * 180.0 / np.pi,
        ell_min=0.5,
        ell_max=4.0,
        n_bins=3,
    )


def test_training_extraction_resumes_completed_rows(
    tiny_ps_dataset: CompetitionDataset,
    tmp_path: Path,
) -> None:
    config = tiny_ps_config()
    cache = FeatureCache.open(tmp_path, "train", (4, 15, config.n_bins))
    extract_training_features(tiny_ps_dataset, cache, config, limit=2)
    assert int(cache.complete.sum()) == 2

    extract_training_features(tiny_ps_dataset, cache, config)

    assert bool(np.all(cache.complete))


def test_test_extraction_does_not_add_training_noise(
    tiny_ps_dataset: CompetitionDataset,
    tmp_path: Path,
) -> None:
    config = tiny_ps_config()
    cache = FeatureCache.open(tmp_path, "test", (3, config.n_bins))
    extract_test_features(tiny_ps_dataset, cache, config)
    extractor = PowerSpectrumExtractor((8, 8), config)
    expected = extractor.extract_log_power(tiny_ps_dataset.test[0].reshape(8, 8))
    np.testing.assert_allclose(cache.values[0], expected)


def test_benchmark_is_deterministic(
    tiny_ps_dataset: CompetitionDataset,
) -> None:
    report = benchmark_features(tiny_ps_dataset, tiny_ps_config(), count=2)
    assert report["repeat_bitwise_equal"] is True
    assert report["maps_per_second"] > 0.0
    assert report["maps_benchmarked"] == 4


@pytest.fixture
def tiny_ps_run(
    tiny_ps_dataset: CompetitionDataset,
    tmp_path: Path,
) -> SimpleNamespace:
    config = tiny_ps_config()
    run_dir = tmp_path / "run"
    train_cache = FeatureCache.open(
        run_dir,
        "train",
        (4, 15, config.n_bins),
    )
    test_cache = FeatureCache.open(
        run_dir,
        "test",
        (3, config.n_bins),
    )
    extract_training_features(tiny_ps_dataset, train_cache, config)
    extract_test_features(tiny_ps_dataset, test_cache, config)
    return SimpleNamespace(
        dataset=tiny_ps_dataset,
        config=config,
        run_dir=run_dir,
        train_cache=train_cache,
        test_cache=test_cache,
    )


def test_validation_uses_fixed_training_only_partitions(
    tiny_ps_run: SimpleNamespace,
) -> None:
    report_path = run_validation(
        tiny_ps_run.dataset,
        tiny_ps_run.train_cache,
        tiny_ps_run.run_dir,
        tiny_ps_run.config,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"]["fit_residues"] == [2, 3, 4]
    assert report["split"]["calibration_residue"] == 1
    assert report["split"]["validation_residue"] == 0
    assert report["optimizer_failures"] == 0
    assert report["counts"] == {
        "fit": 36,
        "calibration": 12,
        "validation": 12,
    }


def test_full_run_creates_ordered_finite_submission(
    tiny_ps_run: SimpleNamespace,
) -> None:
    output = run_full_baseline(
        tiny_ps_run.dataset,
        tiny_ps_run.train_cache,
        tiny_ps_run.test_cache,
        tiny_ps_run.run_dir,
        tiny_ps_run.config,
        submission_path=tiny_ps_run.run_dir / "submission.zip",
        expected_test_count=3,
    )
    with zipfile.ZipFile(output) as archive:
        scores = np.asarray(
            json.loads(archive.read("result.json"))["ood_scores"],
        )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.unique(scores).size > 1

    manifest = json.loads(
        (tiny_ps_run.run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["score_count"] == 3
    assert manifest["scores_all_finite"] is True
    assert manifest["scores_non_constant"] is True
    assert "score_mean" not in manifest
