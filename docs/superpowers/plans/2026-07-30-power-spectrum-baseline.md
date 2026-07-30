# Power-Spectrum Baseline and First Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the organizer's CPU power-spectrum/chi-squared baseline, validate its InD tail without leakage, and create the first reproducible 10,000-score Codabench submission.

**Architecture:** Focused modules own deterministic training noise, one-map FFT features, resumable memory-mapped caches, cosmology-conditioned Gaussian fitting, and workflow orchestration. The CLI composes them into benchmark, validation, and full-submission commands while all public-test inference remains independent per map.

**Tech Stack:** Python 3.12.8, NumPy 2.3.3, SciPy 1.18.0, scikit-learn 1.8.0, pytest 8.4.2, standard-library JSON/hash/time/resource utilities.

## Global Constraints

- This milestone is CPU-only; do not configure or use GCloud or GPUs.
- Notify the user before any later GPU or cloud-credential activity.
- Raw competition arrays remain read-only memory maps.
- Use the exact organizer noise scale `0.4 / sqrt(2 * 30 * 2^2)`.
- Use map-specific `SeedSequence([base_seed, 1, cosmology, realization])`.
- Use base seed `20260730` for the canonical run.
- Use reconstructed shape `(1424, 176)` without numerical transposition.
- Use `ell_edges = numpy.logspace(2, 4, 11)` and `float64` features.
- Use cache algorithm version `ps-baseline-v1`.
- Fit/calibration/validation realization residues are `{2,3,4}`, `{1}`, and `{0}` modulo 5.
- Fit public-test scores only from training artifacts and one test map at a time.
- Never calculate public-test population statistics, ranks, clusters, or fitted transforms.
- Generated scientific artifacts remain in ignored `artifacts/` and `submissions/`.
- Introduce every production behavior through a test that fails for the expected reason.

---

## File Structure

- Modify `pyproject.toml`: pin SciPy as a direct dependency.
- Create `src/wl_ood/ps_config.py`: immutable baseline configuration and JSON payload.
- Create `src/wl_ood/noise.py`: official noise scale and deterministic map noise.
- Create `src/wl_ood/power_spectrum.py`: precomputed FFT binning and one-map features.
- Create `src/wl_ood/feature_cache.py`: compatible resumable `.npy` feature caches.
- Create `src/wl_ood/ps_model.py`: moments, interpolation, likelihood, MAP, and chi-squared.
- Create `src/wl_ood/ps_validation.py`: fixed split, calibration thresholds, and InD-tail report.
- Create `src/wl_ood/ps_pipeline.py`: benchmark, extraction, validation, full fit, manifest, submission.
- Modify `src/wl_ood/cli.py`: expose the three baseline commands.
- Modify `README.md` and `experiments/README.md`: document the exact baseline workflow.
- Create or modify corresponding test modules under `tests/`.

### Task 1: Configuration, Direct SciPy Dependency, and Official Training Noise

**Files:**
- Modify: `pyproject.toml`
- Create: `src/wl_ood/ps_config.py`
- Create: `src/wl_ood/noise.py`
- Create: `tests/test_noise.py`
- Modify: `tests/test_project_environment.py`

**Interfaces:**
- Produces: `PowerSpectrumConfig`.
- Produces: `noise_standard_deviation(config) -> float`.
- Produces: `training_noise(size, config, cosmology, realization) -> ndarray`.
- Produces: `noisy_training_map(values, mask, config, cosmology, realization) -> ndarray`.

- [ ] **Step 1: Write failing configuration and noise tests**

Create `tests/test_noise.py`:

```python
import numpy as np

from wl_ood.noise import (
    noise_standard_deviation,
    noisy_training_map,
    training_noise,
)
from wl_ood.ps_config import PowerSpectrumConfig


def test_official_noise_scale() -> None:
    config = PowerSpectrumConfig()
    expected = 0.4 / np.sqrt(2 * 30 * 2.0**2)
    assert noise_standard_deviation(config) == expected


def test_map_noise_is_order_independent() -> None:
    config = PowerSpectrumConfig(seed=17)
    first = training_noise(8, config, cosmology=3, realization=5)
    _ = training_noise(8, config, cosmology=8, realization=2)
    repeated = training_noise(8, config, cosmology=3, realization=5)
    np.testing.assert_array_equal(first, repeated)


def test_different_map_indices_receive_different_noise() -> None:
    config = PowerSpectrumConfig(seed=17)
    first = training_noise(8, config, cosmology=3, realization=5)
    second = training_noise(8, config, cosmology=3, realization=6)
    assert not np.array_equal(first, second)


def test_noisy_training_map_keeps_masked_pixels_zero() -> None:
    mask = np.array([[True, False], [True, False]])
    full = noisy_training_map(
        np.array([0.1, 0.2]),
        mask,
        PowerSpectrumConfig(seed=17),
        cosmology=0,
        realization=0,
    )
    assert full.shape == mask.shape
    assert np.all(full[~mask] == 0.0)
    assert np.all(np.isfinite(full[mask]))
```

Append to `tests/test_project_environment.py`:

```python
def test_scipy_runtime_version_is_pinned() -> None:
    assert version("scipy") == "1.18.0"


def test_scipy_is_a_direct_runtime_dependency() -> None:
    project_requirements = requires("fair-universe-wl-ood") or []
    assert "scipy==1.18.0" in project_requirements
```

Add `requires` to the existing `importlib.metadata` import.

- [ ] **Step 2: Run the focused tests and verify the missing-module failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_noise.py tests/test_project_environment.py -q
```

Expected: collection fails because `wl_ood.noise` and `wl_ood.ps_config` do not exist.

- [ ] **Step 3: Add the direct dependency and configuration**

Add to `pyproject.toml` dependencies:

```toml
  "scipy==1.18.0",
```

Create `src/wl_ood/ps_config.py`:

```python
"""Fixed configuration for the Phase-2 power-spectrum baseline."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerSpectrumConfig:
    algorithm_version: str = "ps-baseline-v1"
    seed: int = 20260730
    noise_namespace: int = 1
    galaxy_density: float = 30.0
    pixel_size_arcmin: float = 2.0
    ell_min: float = 100.0
    ell_max: float = 10_000.0
    n_bins: int = 10
    feature_dtype: str = "float64"

    @property
    def pixel_size_radian(self) -> float:
        return self.pixel_size_arcmin / 60.0 / 180.0 * np.pi

    @property
    def ell_edges(self) -> np.ndarray:
        return np.logspace(
            np.log10(self.ell_min),
            np.log10(self.ell_max),
            self.n_bins + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ell_edges"] = self.ell_edges.tolist()
        payload["pixel_size_radian"] = self.pixel_size_radian
        return payload
```

- [ ] **Step 4: Implement deterministic noise**

Create `src/wl_ood/noise.py`:

```python
"""Organizer-compatible shape noise for noiseless training maps."""

import numpy as np

from wl_ood.maps import reconstruct_map
from wl_ood.ps_config import PowerSpectrumConfig


def noise_standard_deviation(config: PowerSpectrumConfig) -> float:
    return 0.4 / np.sqrt(
        2.0 * config.galaxy_density * config.pixel_size_arcmin**2
    )


def training_noise(
    size: int,
    config: PowerSpectrumConfig,
    *,
    cosmology: int,
    realization: int,
) -> np.ndarray:
    seed = np.random.SeedSequence(
        [config.seed, config.noise_namespace, cosmology, realization]
    )
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, noise_standard_deviation(config), size=size)


def noisy_training_map(
    values: np.ndarray,
    mask: np.ndarray,
    config: PowerSpectrumConfig,
    *,
    cosmology: int,
    realization: int,
) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    full = reconstruct_map(values, mask_array, fill_value=0.0).astype(
        np.float64,
        copy=False,
    )
    full[mask_array] += training_noise(
        int(np.count_nonzero(mask_array)),
        config,
        cosmology=cosmology,
        realization=realization,
    )
    return full
```

- [ ] **Step 5: Reinstall, verify, and commit**

Run:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_noise.py tests/test_project_environment.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

Commit:

```bash
git add pyproject.toml src/wl_ood/ps_config.py src/wl_ood/noise.py tests/test_noise.py tests/test_project_environment.py
git commit -m "feat: add deterministic organizer noise"
```

### Task 2: Organizer-Compatible One-Map Power Spectrum

**Files:**
- Create: `src/wl_ood/power_spectrum.py`
- Create: `tests/test_power_spectrum.py`

**Interfaces:**
- Consumes: `PowerSpectrumConfig`.
- Produces: `PowerSpectrumExtractor(mask_shape, config)`.
- Produces: `extract_power(full_map) -> ndarray`.
- Produces: `extract_log_power(full_map) -> ndarray`.

- [ ] **Step 1: Write failing FFT feature tests**

Create `tests/test_power_spectrum.py`:

```python
import numpy as np
import pytest

from wl_ood.power_spectrum import PowerSpectrumExtractor
from wl_ood.ps_config import PowerSpectrumConfig


def independent_reference(
    field: np.ndarray,
    pixel_size: float,
    edges: np.ndarray,
) -> np.ndarray:
    transform = np.fft.rfft2(field)
    result = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        total = 0.0
        weight = 0
        for i, fy in enumerate(np.fft.fftfreq(field.shape[0], d=pixel_size)):
            for j, fx in enumerate(np.fft.rfftfreq(field.shape[1], d=pixel_size)):
                ell = 2.0 * np.pi * np.sqrt(fy**2 + fx**2)
                if low < ell <= high:
                    symmetry = 1 if j == 0 or (
                        field.shape[1] % 2 == 0
                        and j == field.shape[1] // 2
                    ) else 2
                    total += symmetry * float(abs(transform[i, j]) ** 2)
                    weight += symmetry
        result.append(total / weight * pixel_size**2 / field.size)
    return np.asarray(result)


def test_power_matches_independent_small_reference() -> None:
    config = PowerSpectrumConfig(
        pixel_size_arcmin=60.0 * 180.0 / np.pi,
        ell_min=0.5,
        ell_max=4.0,
        n_bins=3,
    )
    field = np.arange(64, dtype=np.float64).reshape(8, 8)
    extractor = PowerSpectrumExtractor(field.shape, config)
    expected = independent_reference(field, config.pixel_size_radian, config.ell_edges)
    np.testing.assert_allclose(extractor.extract_power(field), expected)


def test_log_power_rejects_non_positive_bins() -> None:
    config = PowerSpectrumConfig(
        pixel_size_arcmin=60.0 * 180.0 / np.pi,
        ell_min=0.5,
        ell_max=4.0,
        n_bins=3,
    )
    extractor = PowerSpectrumExtractor((8, 8), config)
    with pytest.raises(ValueError, match="strictly positive"):
        extractor.extract_log_power(np.zeros((8, 8)))


def test_extractor_rejects_wrong_map_shape() -> None:
    config = PowerSpectrumConfig(
        pixel_size_arcmin=60.0 * 180.0 / np.pi,
        ell_min=0.5,
        ell_max=4.0,
        n_bins=3,
    )
    extractor = PowerSpectrumExtractor((8, 8), config)
    with pytest.raises(ValueError, match="map shape"):
        extractor.extract_power(np.zeros((4, 4)))
```

- [ ] **Step 2: Verify the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_power_spectrum.py -q
```

Expected: collection fails because `wl_ood.power_spectrum` does not exist.

- [ ] **Step 3: Implement the precomputed extractor**

Create `src/wl_ood/power_spectrum.py`:

```python
"""One-map angular power-spectrum summaries."""

import numpy as np

from wl_ood.ps_config import PowerSpectrumConfig


class PowerSpectrumExtractor:
    def __init__(
        self,
        map_shape: tuple[int, int],
        config: PowerSpectrumConfig,
    ) -> None:
        self.map_shape = map_shape
        self.config = config
        fy = np.fft.fftfreq(map_shape[0], d=config.pixel_size_radian)[:, None]
        fx = np.fft.rfftfreq(map_shape[1], d=config.pixel_size_radian)[None, :]
        ell = 2.0 * np.pi * np.sqrt(fy**2 + fx**2)
        self._bin = np.searchsorted(config.ell_edges, ell) - 1
        weights = np.full(ell.shape, 2.0)
        weights[:, 0] = 1.0
        if map_shape[1] % 2 == 0:
            weights[:, -1] = 1.0
        self._weights = weights
        self._mode_weight = np.asarray(
            [
                weights[self._bin == index].sum()
                for index in range(config.n_bins)
            ],
            dtype=np.float64,
        )
        if np.any(self._mode_weight == 0):
            raise ValueError("every power-spectrum bin must contain a mode")

    def extract_power(self, full_map: np.ndarray) -> np.ndarray:
        field = np.asarray(full_map, dtype=np.float64)
        if field.shape != self.map_shape:
            raise ValueError(
                f"map shape: expected {self.map_shape}, found {field.shape}"
            )
        squared = np.abs(np.fft.rfft2(field)) ** 2
        power = np.asarray(
            [
                np.sum(squared[self._bin == index] * self._weights[self._bin == index])
                / self._mode_weight[index]
                for index in range(self.config.n_bins)
            ],
            dtype=np.float64,
        )
        power *= self.config.pixel_size_radian**2 / field.size
        return power

    def extract_log_power(self, full_map: np.ndarray) -> np.ndarray:
        power = self.extract_power(full_map)
        if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
            raise ValueError("power-spectrum bins must be finite and strictly positive")
        return np.log10(power)
```

- [ ] **Step 4: Run focused and full tests, then commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_power_spectrum.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/power_spectrum.py tests/test_power_spectrum.py
git commit -m "feat: extract organizer power-spectrum features"
```

### Task 3: Compatible Resumable Feature Cache

**Files:**
- Create: `src/wl_ood/feature_cache.py`
- Create: `tests/test_feature_cache.py`

**Interfaces:**
- Produces: `require_run_config(run_dir, payload) -> Path`.
- Produces: `FeatureCache.open(run_dir, role, shape) -> FeatureCache`.
- Produces: `pending_indices()`.
- Produces: `write(index, values)`.

- [ ] **Step 1: Write failing cache behavior tests**

Create `tests/test_feature_cache.py`:

```python
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
```

- [ ] **Step 2: Verify the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_feature_cache.py -q
```

- [ ] **Step 3: Implement configuration and row caches**

Create `src/wl_ood/feature_cache.py`:

```python
"""Crash-resumable feature arrays with explicit compatibility metadata."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib.format import open_memmap


def require_run_config(run_dir: str | Path, payload: dict) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    normalized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != normalized:
        raise ValueError("run directory contains an incompatible configuration")
    if not path.exists():
        path.write_text(normalized, encoding="utf-8")
    return path


@dataclass
class FeatureCache:
    values: np.memmap
    complete: np.memmap

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        role: str,
        shape: tuple[int, ...],
    ) -> "FeatureCache":
        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        values_path = root / f"{role}-log-power.npy"
        complete_path = root / f"{role}-complete.npy"
        leading_shape = shape[:-1]
        if values_path.exists():
            values = np.load(values_path, mmap_mode="r+")
            complete = np.load(complete_path, mmap_mode="r+")
            if values.shape != shape or complete.shape != leading_shape:
                raise ValueError("existing feature cache shape is incompatible")
        else:
            values = open_memmap(values_path, mode="w+", dtype="float64", shape=shape)
            complete = open_memmap(
                complete_path,
                mode="w+",
                dtype="bool",
                shape=leading_shape,
            )
            complete[...] = False
            values.flush()
            complete.flush()
        return cls(values=values, complete=complete)

    def pending_indices(self) -> Iterator[tuple[int, ...]]:
        for index in np.argwhere(~self.complete):
            yield tuple(int(value) for value in index)

    def write(self, index: tuple[int, ...], features: np.ndarray) -> None:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (self.values.shape[-1],):
            raise ValueError("feature row has the wrong shape")
        if not np.all(np.isfinite(vector)):
            raise ValueError("feature row must be finite")
        self.values[index] = vector
        self.values.flush()
        self.complete[index] = True
        self.complete.flush()
```

- [ ] **Step 4: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_feature_cache.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/feature_cache.py tests/test_feature_cache.py
git commit -m "feat: add resumable power-spectrum cache"
```

### Task 4: Cosmology-Conditioned Gaussian Model and Independent Scoring

**Files:**
- Create: `src/wl_ood/ps_model.py`
- Create: `tests/test_ps_model.py`

**Interfaces:**
- Produces: `fit_moments(features, cosmologies, realization_indices)`.
- Produces: `PowerSpectrumModel`.
- Produces: `score_one(features) -> FitResult`.
- Produces: `save(path)` and `load(path)`.

- [ ] **Step 1: Write failing model tests**

Create `tests/test_ps_model.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from wl_ood.ps_model import PowerSpectrumModel, fit_moments


COSMOLOGIES = np.array(
    [[0.2, 0.7], [0.2, 0.9], [0.4, 0.7], [0.4, 0.9]],
    dtype=np.float64,
)


def test_fit_moments_uses_organizer_denominator() -> None:
    features = np.arange(4 * 8 * 2, dtype=np.float64).reshape(4, 8, 2)
    _, covariances = fit_moments(features, np.arange(8))
    delta = features[0] - features[0].mean(axis=0)
    expected = delta.T @ delta / (8 - 2 - 2)
    np.testing.assert_allclose(covariances[0], expected)


def test_likelihood_and_chi2_match_known_gaussian() -> None:
    means = np.column_stack([COSMOLOGIES[:, 0], COSMOLOGIES[:, 1]])
    covariances = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    model = PowerSpectrumModel(COSMOLOGIES, means, covariances)
    feature = np.array([0.3, 0.8])
    assert model.chi_squared(np.array([0.3, 0.8]), feature) == pytest.approx(0.0)
    assert model.negative_log_likelihood(
        np.array([0.2, 0.7]),
        feature,
    ) == pytest.approx(0.01)


def test_interpolation_outside_hull_is_rejected() -> None:
    means = np.column_stack([COSMOLOGIES[:, 0], COSMOLOGIES[:, 1]])
    covariances = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    model = PowerSpectrumModel(COSMOLOGIES, means, covariances)
    assert np.isinf(
        model.negative_log_likelihood(np.array([0.1, 0.6]), np.array([0.3, 0.8]))
    )


def test_model_round_trip(tmp_path: Path) -> None:
    means = np.column_stack([COSMOLOGIES[:, 0], COSMOLOGIES[:, 1]])
    covariances = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    original = PowerSpectrumModel(COSMOLOGIES, means, covariances)
    loaded = PowerSpectrumModel.load(original.save(tmp_path / "model.npz"))
    np.testing.assert_array_equal(loaded.cosmologies, COSMOLOGIES)
    np.testing.assert_array_equal(loaded.means, means)
```

- [ ] **Step 2: Verify the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_model.py -q
```

- [ ] **Step 3: Implement moments and model**

Create `src/wl_ood/ps_model.py` with:

```python
"""Cosmology-conditioned Gaussian model for log-power features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize


def fit_moments(
    features: np.ndarray,
    realization_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(features[:, realization_indices], dtype=np.float64)
    means = selected.mean(axis=1)
    n_realizations = selected.shape[1]
    n_features = selected.shape[2]
    denominator = n_realizations - n_features - 2
    if denominator <= 0:
        raise ValueError("not enough realizations for covariance correction")
    delta = selected - means[:, None, :]
    covariances = np.einsum("crf,crg->cfg", delta, delta) / denominator
    return means, covariances


@dataclass(frozen=True)
class FitResult:
    parameters: np.ndarray
    chi_squared: float
    objective: float
    evaluations: int


class PowerSpectrumModel:
    def __init__(self, cosmologies, means, covariances) -> None:
        self.cosmologies = np.asarray(cosmologies, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        for index, covariance in enumerate(self.covariances):
            if not np.allclose(covariance, covariance.T):
                raise ValueError(f"covariance {index} is not symmetric")
            np.linalg.cholesky(covariance)
        self._mean_interp = LinearNDInterpolator(
            self.cosmologies,
            self.means,
            fill_value=np.nan,
        )
        self._cov_interp = LinearNDInterpolator(
            self.cosmologies,
            self.covariances,
            fill_value=np.nan,
        )

    def _moments(self, parameters):
        mean = np.asarray(self._mean_interp(parameters), dtype=np.float64).reshape(-1)
        covariance = np.asarray(
            self._cov_interp(parameters),
            dtype=np.float64,
        ).reshape(self.means.shape[1], self.means.shape[1])
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            return None
        return mean, covariance

    def chi_squared(self, parameters, features) -> float:
        moments = self._moments(parameters)
        if moments is None:
            return float("inf")
        mean, covariance = moments
        delta = np.asarray(features, dtype=np.float64) - mean
        return float(delta @ np.linalg.solve(covariance, delta))

    def negative_log_likelihood(self, parameters, features) -> float:
        moments = self._moments(parameters)
        if moments is None:
            return float("inf")
        _, covariance = moments
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return float("inf")
        return 0.5 * logdet + 0.5 * self.chi_squared(parameters, features)

    def score_one(self, features: np.ndarray) -> FitResult:
        result = minimize(
            lambda parameters: self.negative_log_likelihood(parameters, features),
            x0=np.array([0.3, 0.8]),
            method="Nelder-Mead",
        )
        if not result.success:
            raise RuntimeError(f"MAP optimization failed: {result.message}")
        objective = self.negative_log_likelihood(result.x, features)
        chi2 = self.chi_squared(result.x, features)
        if not np.isfinite(objective) or not np.isfinite(chi2) or chi2 < 0:
            raise RuntimeError("MAP optimization did not produce a valid score")
        return FitResult(
            parameters=np.asarray(result.x),
            chi_squared=chi2,
            objective=objective,
            evaluations=int(result.nfev),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            cosmologies=self.cosmologies,
            means=self.means,
            covariances=self.covariances,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PowerSpectrumModel":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                payload["cosmologies"],
                payload["means"],
                payload["covariances"],
            )
```

Add a focused `score_one` test with the same four-corner Gaussian and assert
that it returns finite parameters and non-negative chi-squared.

- [ ] **Step 4: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_model.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/ps_model.py tests/test_ps_model.py
git commit -m "feat: fit and score cosmology Gaussian model"
```

### Task 5: Fixed Leakage-Safe Validation

**Files:**
- Create: `src/wl_ood/ps_validation.py`
- Create: `tests/test_ps_validation.py`

**Interfaces:**
- Produces: `ValidationSplit`.
- Produces: `split_realizations(count)`.
- Produces: `calibration_thresholds(scores, levels)`.
- Produces: `evaluate_ind_tail(calibration_scores, validation_scores)`.

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_ps_validation.py`:

```python
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
    thresholds = calibration_thresholds(calibration, np.array([0.05]))
    expected = np.quantile(calibration, 0.95, method="higher")
    assert thresholds[0.05] == expected


def test_validation_values_do_not_change_thresholds() -> None:
    calibration = np.arange(100, dtype=np.float64)
    low = evaluate_ind_tail(calibration, np.zeros(20))
    high = evaluate_ind_tail(calibration, np.full(20, 1000.0))
    assert low["thresholds"] == high["thresholds"]
    assert set(low["false_positive_rates"]) == set(DEFAULT_FPR_LEVELS)
```

- [ ] **Step 2: Verify the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_validation.py -q
```

- [ ] **Step 3: Implement the fixed split and thresholds**

Create `src/wl_ood/ps_validation.py`:

```python
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
    indices = np.arange(count)
    return ValidationSplit(
        fit=indices[np.isin(indices % 5, [2, 3, 4])],
        calibration=indices[indices % 5 == 1],
        validation=indices[indices % 5 == 0],
    )


def calibration_thresholds(
    scores: np.ndarray,
    levels: np.ndarray | tuple[float, ...] = DEFAULT_FPR_LEVELS,
) -> dict[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        float(level): float(np.quantile(values, 1.0 - level, method="higher"))
        for level in levels
    }


def evaluate_ind_tail(
    calibration_scores: np.ndarray,
    validation_scores: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_FPR_LEVELS,
) -> dict:
    thresholds = calibration_thresholds(calibration_scores, levels)
    validation = np.asarray(validation_scores, dtype=np.float64)
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
```

- [ ] **Step 4: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_validation.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/ps_validation.py tests/test_ps_validation.py
git commit -m "feat: add leakage-safe InD validation split"
```

### Task 6: Memory-Bounded Extraction and CPU Benchmark

**Files:**
- Create: `src/wl_ood/ps_pipeline.py`
- Create: `tests/test_ps_pipeline.py`

**Interfaces:**
- Consumes: dataset, config, extractor, and cache.
- Produces: `extract_training_features(...)`.
- Produces: `extract_test_features(...)`.
- Produces: `benchmark_features(...) -> dict`.

- [ ] **Step 1: Write failing extraction and benchmark tests**

Create a synthetic dataset helper with an `8 x 8` all-true mask, four
cosmologies, 15 training realizations, and three test maps. Test:

```python
def test_training_extraction_resumes_completed_rows(tiny_ps_dataset, tmp_path):
    config = tiny_ps_config(seed=17)
    cache = FeatureCache.open(tmp_path, "train", (4, 15, config.n_bins))
    extract_training_features(tiny_ps_dataset, cache, config, limit=2)
    assert int(cache.complete.sum()) == 2
    extract_training_features(tiny_ps_dataset, cache, config)
    assert bool(np.all(cache.complete))


def test_test_extraction_does_not_add_training_noise(tiny_ps_dataset, tmp_path):
    config = tiny_ps_config(seed=17)
    cache = FeatureCache.open(tmp_path, "test", (3, config.n_bins))
    extract_test_features(tiny_ps_dataset, cache, config)
    extractor = PowerSpectrumExtractor((8, 8), config)
    expected = extractor.extract_log_power(tiny_ps_dataset.test[0].reshape(8, 8))
    np.testing.assert_allclose(cache.values[0], expected)


def test_benchmark_is_deterministic(tiny_ps_dataset):
    report = benchmark_features(tiny_ps_dataset, tiny_ps_config(seed=17), count=2)
    assert report["repeat_bitwise_equal"] is True
    assert report["maps_per_second"] > 0
```

- [ ] **Step 2: Verify failures because the pipeline does not exist**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_pipeline.py -q
```

- [ ] **Step 3: Implement bounded extraction**

Implement in `src/wl_ood/ps_pipeline.py`:

```python
def extract_training_features(dataset, cache, config, *, limit=None) -> int:
    extractor = PowerSpectrumExtractor(dataset.mask.shape, config)
    written = 0
    for cosmology, realization in cache.pending_indices():
        full = noisy_training_map(
            dataset.training[cosmology, realization],
            dataset.mask,
            config,
            cosmology=cosmology,
            realization=realization,
        )
        cache.write((cosmology, realization), extractor.extract_log_power(full))
        written += 1
        if limit is not None and written >= limit:
            break
    return written


def extract_test_features(dataset, cache, config, *, limit=None) -> int:
    extractor = PowerSpectrumExtractor(dataset.mask.shape, config)
    written = 0
    for (index,) in cache.pending_indices():
        full = reconstruct_map(
            dataset.test[index],
            dataset.mask,
            fill_value=0.0,
        )
        cache.write((index,), extractor.extract_log_power(full))
        written += 1
        if limit is not None and written >= limit:
            break
    return written
```

Implement `benchmark_features` by extracting fixed first indices twice without
writing a cache, asserting bitwise equality, measuring `time.perf_counter()`,
and reporting `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`,
maps-per-second, and projected full extraction seconds.

- [ ] **Step 4: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_pipeline.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/ps_pipeline.py tests/test_ps_pipeline.py
git commit -m "feat: extract and benchmark baseline features"
```

### Task 7: Validation Run, Full Fit, Manifest, and Submission

**Files:**
- Modify: `src/wl_ood/ps_pipeline.py`
- Modify: `tests/test_ps_pipeline.py`

**Interfaces:**
- Produces: `run_validation(...) -> Path`.
- Produces: `run_full_baseline(...) -> Path`.
- Produces: `run-manifest.json`.
- Produces: deterministic submission ZIP.

- [ ] **Step 1: Write failing end-to-end synthetic tests**

Add tests that:

```python
def test_validation_uses_fixed_training_only_partitions(tiny_ps_run):
    report_path = run_validation(
        tiny_ps_run.dataset,
        tiny_ps_run.train_cache,
        tiny_ps_run.run_dir,
        tiny_ps_run.config,
    )
    report = json.loads(report_path.read_text())
    assert report["split"]["fit_residues"] == [2, 3, 4]
    assert report["split"]["calibration_residue"] == 1
    assert report["split"]["validation_residue"] == 0
    assert report["optimizer_failures"] == 0


def test_full_run_creates_ordered_finite_submission(tiny_ps_run):
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
        scores = np.asarray(json.loads(archive.read("result.json"))["ood_scores"])
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
    assert np.unique(scores).size > 1
```

The synthetic fixture must provide enough fitting realizations for
`N - n_features - 2 > 0` and non-singular covariances.

- [ ] **Step 2: Run and verify expected missing-function failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_pipeline.py -q
```

- [ ] **Step 3: Implement validation workflow**

`run_validation` must:

1. require complete training features;
2. create the fixed modulo-five split;
3. fit moments only on `split.fit`;
4. score flattened calibration and validation feature rows independently;
5. call `evaluate_ind_tail`;
6. record counts, FPRs, thresholds, score ranges, evaluations, and runtime;
7. write deterministic `validation-report.json`.

Use a private `_score_rows(model, rows)` loop that calls `score_one` for each
row and returns chi-squared plus total function evaluations. It must not accept
or inspect a test collection when called for validation.

- [ ] **Step 4: Implement full model and submission workflow**

`run_full_baseline` must:

1. require both caches complete;
2. fit moments with all realization indices;
3. save `full-model.npz`;
4. score test rows in existing index order;
5. save `test-chi2.npy`;
6. reject non-finite, negative, or constant scores;
7. call `create_submission_zip(..., expected_count=expected_test_count)`;
8. fingerprint the model, score array, and ZIP with `sha256_file`;
9. record environment versions, Git revision/dirty flag, runtime, configuration,
   score count, cache counts, and the no-test-aggregation declaration in
   `run-manifest.json`.

The manifest may record score count and finiteness only. It must not record
test-score mean, extrema, quantiles, histogram, or ranks.

- [ ] **Step 5: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_ps_pipeline.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/ps_pipeline.py tests/test_ps_pipeline.py
git commit -m "feat: run validated baseline submission pipeline"
```

### Task 8: CLI and Research Documentation

**Files:**
- Modify: `src/wl_ood/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `experiments/README.md`
- Modify: `tests/test_documentation.py`

**Interfaces:**
- Produces: `wl-ood benchmark-ps-baseline`.
- Produces: `wl-ood validate-ps-baseline`.
- Produces: `wl-ood run-ps-baseline`.

- [ ] **Step 1: Write failing CLI tests**

Add parser-level and synthetic execution tests for:

```text
wl-ood benchmark-ps-baseline --data-dir PATH --run-dir PATH --seed 17
wl-ood validate-ps-baseline --data-dir PATH --run-dir PATH --seed 17
wl-ood run-ps-baseline --data-dir PATH --run-dir PATH --seed 17 --submission PATH
```

Extend `main` with test injection parameters only for `PowerSpectrumConfig` and
expected test count. Assert the benchmark writes `benchmark.json`, validation
writes `validation-report.json`, and full run creates the ZIP.

- [ ] **Step 2: Verify focused failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_documentation.py -q
```

- [ ] **Step 3: Wire commands**

Each command must:

1. load the memory-mapped dataset;
2. derive a configuration from the explicit seed;
3. validate/write `config.json`, including streamed raw-file SHA-256 hashes;
4. open the correctly shaped cache;
5. invoke only its named pipeline stage;
6. print the output path and concise runtime status.

`validate-ps-baseline` completes training extraction before validation.
`run-ps-baseline` requires the validation report, completes train/test
extraction, then creates the full model and ZIP. Compatible existing caches are
resumed.

- [ ] **Step 4: Document commands and experiment record**

README must state:

- the baseline is CPU-only;
- the exact three commands;
- the canonical seed/run directory/submission path;
- test maps are independently scored;
- generated artifacts are ignored;
- no leaderboard result should be interpreted before the manifest is saved.

Update `experiments/README.md` so the first baseline run references
`run-manifest.json` and `validation-report.json`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_documentation.py -q
.venv/bin/python -m pytest -q
```

Commit:

```bash
git add src/wl_ood/cli.py tests/test_cli.py README.md experiments/README.md tests/test_documentation.py
git commit -m "feat: expose power-spectrum baseline workflow"
```

### Task 9: Real CPU Benchmark, Validation, and First Submission Artifact

**Files:**
- Generated, ignored: `artifacts/power-spectrum/seed-20260730/*`
- Generated, ignored: `submissions/ps-baseline-seed-20260730.zip`
- Create after scoring: `experiments/20260730-ps-baseline/record.md`

**Interfaces:**
- Consumes: the audited real Phase-2 arrays.
- Produces: a benchmark, validation report, full model, 10,000 scores, manifest,
  and Codabench-ready ZIP.

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: every test passes.

- [ ] **Step 2: Run the fixed CPU benchmark**

Run:

```bash
.venv/bin/wl-ood benchmark-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730
```

Inspect `benchmark.json`. Require deterministic repeat features, finite values,
positive throughput, and bounded peak memory. Record the projected runtime.

- [ ] **Step 3: Extract training features and validate**

Run:

```bash
.venv/bin/wl-ood validate-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730
```

This may take minutes. If interrupted, rerun the exact command and verify that
completed cache rows are skipped.

Inspect only training-derived validation fields:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; r=json.loads(Path("artifacts/power-spectrum/seed-20260730/validation-report.json").read_text()); print(r["counts"]); print(r["thresholds"]); print(r["false_positive_rates"]); print(r["optimizer_failures"])'
```

Require zero numerical/optimizer failures before public-test inference.

- [ ] **Step 4: Run full independent test inference**

Run:

```bash
.venv/bin/wl-ood run-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730 \
  --submission submissions/ps-baseline-seed-20260730.zip
```

Do not inspect public-test score distributions. Inspect only:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; r=json.loads(Path("artifacts/power-spectrum/seed-20260730/run-manifest.json").read_text()); print(r["score_count"], r["scores_all_finite"], r["scores_non_constant"]); print(r["submission"])'
```

Expected: count 10,000 and all boolean checks true.

- [ ] **Step 5: Verify archive and reproducibility**

Run:

```bash
.venv/bin/python -c 'import json,zipfile; from pathlib import Path; p=Path("submissions/ps-baseline-seed-20260730.zip"); z=zipfile.ZipFile(p); assert z.namelist()==["result.json"]; s=json.loads(z.read("result.json"))["ood_scores"]; assert len(s)==10000; print(p.stat().st_size)'
.venv/bin/python -m pytest -q
git status --short --branch
git check-ignore artifacts/power-spectrum/seed-20260730/run-manifest.json
git check-ignore submissions/ps-baseline-seed-20260730.zip
```

Expected: archive schema passes, tests pass, repository is clean, and artifacts
are ignored.

- [ ] **Step 6: Submit and record the leaderboard reference**

Open the user's authenticated Codabench competition page, upload
`submissions/ps-baseline-seed-20260730.zip`, wait for terminal scoring status,
and record:

- submission timestamp;
- public leaderboard score;
- the 100 logged TPR values;
- submission archive SHA-256;
- source Git revision; and
- conclusion that this is the organizer-compatible reference baseline.

Write these fields to
`artifacts/power-spectrum/seed-20260730/leaderboard-result.json`, then create
`experiments/20260730-ps-baseline/record.md` following the repository's
experiment template. The committed record may reference ignored artifact paths
and checksums but must not embed the 10,000 scores.

Do not modify the method in response to individual logged thresholds during
this task. Any improvement becomes a separately specified experiment.

## Plan Self-Review

- **Spec coverage:** Noise, exact FFT bins, covariance correction, interpolation,
  MAP/chi-squared, deterministic caching, leakage-safe validation, benchmark,
  full independent inference, manifest, ZIP, and Codabench submission are
  covered.
- **Scope:** This is one end-to-end classical baseline. Pseudo-OoD generation,
  higher-order summaries, ensembles, neural models, GCloud, and GPUs are
  explicitly deferred.
- **Type consistency:** Configuration, cache, extractor, model, validation, and
  pipeline names are consistent across producing and consuming tasks.
- **No placeholder work:** Every task has concrete files, interfaces, tests,
  commands, expected behavior, and a commit boundary.
