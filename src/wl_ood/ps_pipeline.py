"""Memory-bounded orchestration for the power-spectrum baseline."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from wl_ood.audit import sha256_file
from wl_ood.data import CompetitionDataset
from wl_ood.feature_cache import FeatureCache
from wl_ood.maps import reconstruct_map
from wl_ood.noise import noisy_training_map
from wl_ood.power_spectrum import PowerSpectrumExtractor
from wl_ood.ps_config import PowerSpectrumConfig
from wl_ood.ps_model import PowerSpectrumModel, fit_moments
from wl_ood.ps_validation import evaluate_ind_tail, split_realizations
from wl_ood.submission import create_submission_zip


def extract_training_features(
    dataset: CompetitionDataset,
    cache: FeatureCache,
    config: PowerSpectrumConfig,
    *,
    limit: int | None = None,
) -> int:
    extractor = PowerSpectrumExtractor(dataset.mask.shape, config)
    written = 0
    for cosmology, realization in cache.pending_indices():
        try:
            full = noisy_training_map(
                dataset.training[cosmology, realization],
                dataset.mask,
                config,
                cosmology=cosmology,
                realization=realization,
            )
            features = extractor.extract_log_power(full)
        except (ValueError, FloatingPointError) as error:
            raise ValueError(
                "failed to extract training map "
                f"({cosmology}, {realization}): {error}"
            ) from error
        cache.write((cosmology, realization), features)
        written += 1
        if limit is not None and written >= limit:
            break
    return written


def extract_test_features(
    dataset: CompetitionDataset,
    cache: FeatureCache,
    config: PowerSpectrumConfig,
    *,
    limit: int | None = None,
) -> int:
    extractor = PowerSpectrumExtractor(dataset.mask.shape, config)
    written = 0
    for (index,) in cache.pending_indices():
        try:
            full = reconstruct_map(
                dataset.test[index],
                dataset.mask,
                fill_value=0.0,
            )
            features = extractor.extract_log_power(full)
        except (ValueError, FloatingPointError) as error:
            raise ValueError(
                f"failed to extract test map {index}: {error}"
            ) from error
        cache.write((index,), features)
        written += 1
        if limit is not None and written >= limit:
            break
    return written


def _benchmark_pass(
    dataset: CompetitionDataset,
    config: PowerSpectrumConfig,
    count: int,
) -> np.ndarray:
    extractor = PowerSpectrumExtractor(dataset.mask.shape, config)
    output = np.empty((count * 2, config.n_bins), dtype=np.float64)
    for realization in range(count):
        training_map = noisy_training_map(
            dataset.training[0, realization],
            dataset.mask,
            config,
            cosmology=0,
            realization=realization,
        )
        output[realization] = extractor.extract_log_power(training_map)
    for index in range(count):
        test_map = reconstruct_map(
            dataset.test[index],
            dataset.mask,
            fill_value=0.0,
        )
        output[count + index] = extractor.extract_log_power(test_map)
    return output


def benchmark_features(
    dataset: CompetitionDataset,
    config: PowerSpectrumConfig,
    *,
    count: int = 32,
) -> dict:
    if count < 1:
        raise ValueError("benchmark count must be positive")
    if dataset.training.shape[1] < count or dataset.test.shape[0] < count:
        raise ValueError("dataset does not contain enough benchmark maps")

    started = time.perf_counter()
    first = _benchmark_pass(dataset, config, count)
    repeated = _benchmark_pass(dataset, config, count)
    elapsed = time.perf_counter() - started
    distinct_maps = count * 2
    processed_maps = distinct_maps * 2
    maps_per_second = processed_maps / elapsed
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = (
        int(maximum_rss)
        if sys.platform == "darwin"
        else int(maximum_rss) * 1024
    )
    full_map_count = (
        int(np.prod(dataset.training.shape[:2])) + dataset.test.shape[0]
    )
    return {
        "maps_benchmarked": distinct_maps,
        "repeat_bitwise_equal": bool(np.array_equal(first, repeated)),
        "maps_per_second": float(maps_per_second),
        "elapsed_seconds": float(elapsed),
        "peak_rss_bytes": peak_rss_bytes,
        "projected_full_extraction_seconds": float(
            full_map_count / maps_per_second
        ),
    }


def _require_complete(cache: FeatureCache, role: str) -> None:
    if not bool(np.all(cache.complete)):
        missing = int(np.count_nonzero(~cache.complete))
        raise ValueError(f"{role} feature cache is incomplete: {missing} rows missing")


def _score_rows(
    model: PowerSpectrumModel,
    rows: np.ndarray,
) -> tuple[np.ndarray, int]:
    features = np.asarray(rows, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("feature rows must be two-dimensional")
    scores = np.empty(features.shape[0], dtype=np.float64)
    evaluations = 0
    for index, feature_vector in enumerate(features):
        try:
            result = model.score_one(feature_vector)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"failed to score feature row {index}: {error}") from error
        scores[index] = result.chi_squared
        evaluations += result.evaluations
    return scores, evaluations


def _covariance_diagnostics(covariances: np.ndarray) -> dict:
    cholesky = np.linalg.cholesky(covariances)
    diagonal = np.diagonal(cholesky, axis1=1, axis2=2)
    conditions = np.asarray(
        [np.linalg.cond(covariance) for covariance in covariances],
        dtype=np.float64,
    )
    return {
        "minimum_cholesky_diagonal": float(np.min(diagonal)),
        "maximum_condition_number": float(np.max(conditions)),
    }


def _write_json(payload: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def run_validation(
    dataset: CompetitionDataset,
    train_cache: FeatureCache,
    run_dir: str | Path,
    config: PowerSpectrumConfig,
) -> Path:
    """Fit only the fixed training partition and evaluate held-out InD tails."""
    _require_complete(train_cache, "training")
    started = time.perf_counter()
    features = np.asarray(train_cache.values)
    split = split_realizations(features.shape[1])
    cosmologies = np.asarray(dataset.labels[:, 0, :2], dtype=np.float64)
    means, covariances = fit_moments(features, split.fit)
    model = PowerSpectrumModel(cosmologies, means, covariances)

    feature_count = features.shape[-1]
    calibration_rows = np.asarray(
        features[:, split.calibration, :],
        dtype=np.float64,
    ).reshape(-1, feature_count)
    validation_rows = np.asarray(
        features[:, split.validation, :],
        dtype=np.float64,
    ).reshape(-1, feature_count)
    calibration_scores, calibration_evaluations = _score_rows(
        model,
        calibration_rows,
    )
    validation_scores, validation_evaluations = _score_rows(
        model,
        validation_rows,
    )
    diagnostics = evaluate_ind_tail(calibration_scores, validation_scores)
    cosmology_count = features.shape[0]
    report = {
        "schema_version": 1,
        "config": config.to_dict(),
        "split": {
            "fit_residues": [2, 3, 4],
            "calibration_residue": 1,
            "validation_residue": 0,
            "modulus": 5,
        },
        "counts": {
            "fit": int(cosmology_count * split.fit.size),
            "calibration": int(cosmology_count * split.calibration.size),
            "validation": int(cosmology_count * split.validation.size),
        },
        "thresholds": diagnostics["thresholds"],
        "false_positive_rates": diagnostics["false_positive_rates"],
        "score_ranges": {
            "calibration": [
                float(np.min(calibration_scores)),
                float(np.max(calibration_scores)),
            ],
            "validation": [
                float(np.min(validation_scores)),
                float(np.max(validation_scores)),
            ],
        },
        "finite_score_count": int(
            np.count_nonzero(np.isfinite(calibration_scores))
            + np.count_nonzero(np.isfinite(validation_scores))
        ),
        "optimizer_failures": 0,
        "function_evaluations": int(
            calibration_evaluations + validation_evaluations
        ),
        "covariance": _covariance_diagnostics(covariances),
        "runtime_seconds": float(time.perf_counter() - started),
        "uses_public_test_statistics": False,
    }
    return _write_json(
        report,
        Path(run_dir) / "validation-report.json",
    )


def _git_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def run_full_baseline(
    dataset: CompetitionDataset,
    train_cache: FeatureCache,
    test_cache: FeatureCache,
    run_dir: str | Path,
    config: PowerSpectrumConfig,
    *,
    submission_path: str | Path,
    expected_test_count: int = 10_000,
) -> Path:
    """Fit all training simulations and score test maps independently."""
    _require_complete(train_cache, "training")
    _require_complete(test_cache, "test")
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)

    training_features = np.asarray(train_cache.values)
    all_realizations = np.arange(training_features.shape[1])
    means, covariances = fit_moments(training_features, all_realizations)
    cosmologies = np.asarray(dataset.labels[:, 0, :2], dtype=np.float64)
    model = PowerSpectrumModel(cosmologies, means, covariances)
    model_path = model.save(root / "full-model.npz")

    scores, evaluations = _score_rows(model, np.asarray(test_cache.values))
    scores_all_finite = bool(np.all(np.isfinite(scores)))
    scores_non_negative = bool(np.all(scores >= 0.0))
    scores_non_constant = not bool(np.all(scores == scores[0]))
    if not scores_all_finite:
        raise ValueError("test scores must all be finite")
    if not scores_non_negative:
        raise ValueError("chi-squared scores must all be non-negative")
    if not scores_non_constant:
        raise ValueError("test scores must not all be identical")
    if scores.size != expected_test_count:
        raise ValueError(
            f"expected {expected_test_count} test scores, found {scores.size}"
        )

    score_path = root / "test-chi2.npy"
    np.save(score_path, scores)
    submission = create_submission_zip(
        scores,
        submission_path,
        expected_count=expected_test_count,
    )
    revision, dirty = _git_state()
    completed_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "algorithm": config.algorithm_version,
        "config": config.to_dict(),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "runtime_seconds": float(time.perf_counter() - started),
        "git_revision": revision,
        "git_dirty": dirty,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "package": version("fair-universe-wl-ood"),
        },
        "training_feature_count": int(np.prod(training_features.shape[:2])),
        "score_count": int(scores.size),
        "scores_all_finite": scores_all_finite,
        "scores_non_negative": scores_non_negative,
        "scores_non_constant": scores_non_constant,
        "optimizer_failures": 0,
        "function_evaluations": int(evaluations),
        "covariance": _covariance_diagnostics(covariances),
        "artifacts": {
            "model": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            },
            "scores": {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
            },
        },
        "submission": {
            "path": str(submission),
            "size_bytes": submission.stat().st_size,
            "sha256": sha256_file(submission),
        },
        "uses_public_test_statistics": False,
        "single_instance_scoring": True,
    }
    _write_json(manifest, root / "run-manifest.json")
    return submission
