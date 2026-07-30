"""Memory-bounded orchestration for the power-spectrum baseline."""

import resource
import sys
import time

import numpy as np

from wl_ood.data import CompetitionDataset
from wl_ood.feature_cache import FeatureCache
from wl_ood.maps import reconstruct_map
from wl_ood.noise import noisy_training_map
from wl_ood.power_spectrum import PowerSpectrumExtractor
from wl_ood.ps_config import PowerSpectrumConfig


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
