# Power-Spectrum Baseline and First Submission Design

## Objective

Build a faithful, reproducible implementation of the organizer's Phase-2
power-spectrum baseline and use it to produce the project's first scientific
Codabench submission.

This milestone is CPU-only. It must stop at a validated submission archive; it
does not configure cloud credentials, provision accelerators, or train a neural
network.

## Official Reference

The implementation follows the public organizer materials:

- [Phase-2 power-spectrum starting kit](https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/Phase_2_Startingkit_WL_PSAnalysis.ipynb)
- [FAIR Universe challenge white paper](https://arxiv.org/abs/2604.14451)

The reference method:

1. adds HSC-like shape noise to noiseless training convergence maps;
2. extracts a 10-bin angular power spectrum;
3. models the mean and covariance of log power at each cosmology;
4. linearly interpolates those quantities across
   `(\Omega_m, S_8)`;
5. finds the maximum-likelihood cosmology for a map;
6. evaluates its chi-squared residual at that cosmology; and
7. uses the chi-squared statistic, or a monotonic empirical-p-value transform,
   as the OoD score.

## Fixed Competition and Safety Constraints

- Each test map must be scoreable from that map plus artifacts fitted only from
  training data.
- Test maps must remain in organizer-provided order.
- No preprocessing, normalization, fitting, calibration, ranking, threshold,
  or ensemble weight may be derived from the public-test population.
- Test features may be cached individually for resumability, but the pipeline
  must not compute test-wide means, quantiles, histograms, ranks, clusters, or
  other distribution summaries.
- Raw arrays remain read-only memory maps under ignored `public_data/`.
- Generated features, models, manifests, reports, and submissions remain under
  ignored `artifacts/` and `submissions/`.
- All randomness must be reproducible and independent of processing order.
- The implementation must remain memory-bounded; it may reconstruct only a
  bounded number of full maps at once.
- The first submission uses no GPU. Before any later GPU or GCloud activity,
  the user must be notified so credentials and access can be coordinated.

## Physics and Feature Definition

### Survey geometry

Each stored map contains the 132,019 values at `True` positions in the
`(1424, 176)` boolean survey mask. Power-spectrum extraction reconstructs one
full map with zeros outside the valid mask.

The FFT orientation follows the organizer notebook exactly: the reconstructed
array is passed to `numpy.fft.rfft2` with shape `(1424, 176)`. Display-only
transposition is never applied to numerical features.

### Training noise

Training maps are noiseless. Each training map receives independent Gaussian
shape noise at valid mask positions:

```text
sigma_noise = 0.4 / sqrt(2 * n_g * pixel_size_arcmin^2)
n_g = 30 galaxies / arcmin^2
pixel_size_arcmin = 2
```

This gives `sigma_noise = 0.4 / sqrt(240)`.

Noise outside the survey mask is zero. Public-test maps are already noisy and
must not receive additional noise.

Noise for training map `(cosmology, realization)` is produced from a
map-specific NumPy `SeedSequence` containing:

```text
(base_seed, 1, cosmology, realization)
```

The fixed namespace value `1` identifies canonical training-noise generation
and makes a map's noise independent of loop order, chunk size, interruption,
or future parallel execution.

The first canonical run uses `base_seed = 20260730`.

### Angular power spectrum

The pixel size in radians is:

```text
pixel_size_radian = 2 / 60 / 180 * pi
```

The 10 multipole bins use 11 logarithmically spaced edges:

```text
ell_edges = logspace(2, 4, 11)
```

For one reconstructed map:

1. calculate `rfft2`;
2. square the complex magnitude;
3. construct the two-dimensional angular-frequency grid;
4. bin modes using the organizer's real-FFT symmetry correction;
5. average the power in each populated bin;
6. normalize by pixel area divided by map area; and
7. return `log10(power)` for the 10 bins.

All 10 powers must be finite and strictly positive before taking `log10`.
Violations stop the run with the map index in the error message.

## Feature Cache and Resumability

The canonical run directory is:

```text
artifacts/power-spectrum/seed-20260730/
```

It contains:

```text
config.json
train-log-power.npy
train-complete.npy
test-log-power.npy
test-complete.npy
validation-report.json
full-model.npz
test-chi2.npy
run-manifest.json
```

The training feature shape is `(101, 256, 10)`. The test feature shape is
`(10000, 10)`. Both feature arrays use `float64`. Feature arrays and completion
arrays are created with NumPy's memory-mapped `.npy` support. The algorithm
version stored in cache metadata is `ps-baseline-v1`.

For each map:

1. calculate and validate the 10 features;
2. write the feature row;
3. flush the feature memory map;
4. mark the corresponding completion flag;
5. flush the completion memory map.

On restart, completed rows are skipped. A cache is reusable only when
`config.json` exactly matches:

- algorithm version;
- dataset SHA-256 fingerprints;
- mask shape and valid-pixel count;
- base seed and noise convention;
- pixel size and galaxy density;
- multipole edges; and
- feature dtype.

A mismatch produces an explicit error instead of silently mixing artifacts.

Test caching is a batch-efficiency mechanism only. `score_one_map()` must accept
one map and a fitted training artifact without requiring any other test map.

## Baseline Model

### Cosmology-conditioned moments

Let `d` be a 10-element log-power vector. At each of the 101 cosmologies, fit:

```text
mean(theta) = mean_j d_j
cov(theta) =
    sum_j [(d_j - mean(theta)) (d_j - mean(theta))^T]
    / (N - n_d - 2)
```

where `n_d = 10` and `N` is the number of selected realizations. The denominator
matches the organizer notebook.

Every covariance must be finite, symmetric, and positive definite. Validation
reports its minimum Cholesky diagonal and condition number. A failed covariance
stops the run; regularization is not silently introduced in the faithful
baseline.

### Emulator

Use SciPy `LinearNDInterpolator` over the 101 provided
`(\Omega_m, S_8)` points for:

- the 10-element mean;
- the `10 x 10` covariance; and
- a zero log-prior inside the convex hull with negative infinity outside.

SciPy `1.18.0` is a direct, pinned project dependency rather than an accidental
transitive dependency.

### Best fit and score

For a map with feature vector `d`, minimize the negative Gaussian log
likelihood with SciPy Nelder-Mead starting at `(0.3, 0.8)`, matching the
organizer notebook. Set both `maxfev` and `maxiter` to `4,000`; the organizer's
default budget can stop before convergence for maps whose optimum lies near the
interpolation boundary.

The objective includes:

```text
0.5 * log(det(cov(theta)))
+ 0.5 * (d - mean(theta))^T cov(theta)^-1 (d - mean(theta))
```

Linear solves are used instead of explicitly forming a matrix inverse. The
optimizer result must:

- lie inside the interpolation hull;
- have finite interpolated moments;
- have a finite objective;
- produce a finite, non-negative chi-squared statistic.

Any failure records the map index and stops rather than silently substituting a
different method.

The submitted OoD score is raw chi-squared. The organizer explicitly permits
this, and it avoids ties introduced by an empirical p-value transform while
preserving the required ordering: larger means less compatible with the
training simulation family.

## Leakage-Safe Local Validation

The local validation split is fixed by realization index and applied at every
cosmology:

```text
realization % 5 in {2, 3, 4}: model fitting
realization % 5 == 1: calibration
realization % 5 == 0: validation
```

This yields, per cosmology:

- 153 fitting realizations;
- 51 calibration realizations; and
- 52 validation realizations.

The split holds out nuisance-parameter realizations as well as independent
simulation seeds while retaining all 101 cosmological locations in each
partition.

The model is fitted only on the fitting partition. Calibration chi-squared
scores define training-only thresholds at false-positive levels:

```text
0.05, 0.01, 0.005, 0.001
```

For false-positive level `alpha`, the threshold is the
`(1 - alpha)` empirical quantile of calibration scores using NumPy's
`method="higher"`. This rule is fixed before validation scores are inspected.

The validation report measures the fraction of validation InD scores exceeding
each threshold. It also records:

- counts in each partition;
- optimizer failure count;
- finite-score count;
- score minimum and maximum for calibration and validation;
- empirical false-positive rates at the four thresholds;
- feature and scoring runtimes; and
- the complete configuration and Git revision.

These are training-data diagnostics. No public-test value participates.

Because this milestone has no labeled OoD examples, validation cannot estimate
OoD detection power. Its purpose is to catch leakage, numerical failure, and a
badly miscalibrated InD tail before spending a leaderboard submission.
Pseudo-OoD stress tests are a separate later milestone.

## Full Fit and Test Inference

After validation succeeds:

1. fit cosmology-conditioned means and covariances using all 256 training
   realizations;
2. build the full interpolator;
3. extract or resume all 10,000 test feature rows;
4. score every test map independently;
5. verify exactly 10,000 finite, non-negative, non-constant scores;
6. preserve test ordering;
7. write `test-chi2.npy`;
8. create a ZIP containing only root-level `result.json`; and
9. calculate a SHA-256 checksum of the ZIP.

The canonical archive is:

```text
submissions/ps-baseline-seed-20260730.zip
```

## Benchmark Gate

Before the full feature extraction, run a deterministic CPU benchmark on fixed
indices:

```text
training: (0, 0) through (0, 31)
test: 0 through 31
```

The benchmark verifies feature equality across repeated runs, reports peak
resident memory and maps per second, and estimates the full extraction time.
It does not calculate test-population statistics.

The full run proceeds automatically if:

- all benchmark features are finite and positive before log transform;
- repeat extraction is bitwise identical;
- peak memory remains bounded well below available local RAM; and
- the projected runtime is recorded.

No GPU is requested based only on FFT runtime. GPU coordination is reserved for
later learned models unless the CPU baseline becomes genuinely impractical.

## Software Boundaries

### `wl_ood.noise`

Owns the official noise scale and order-independent per-map random generation.
It has no knowledge of files, FFTs, or model fitting.

### `wl_ood.power_spectrum`

Owns FFT-grid preparation and one-map log-power extraction. A reusable
`PowerSpectrumExtractor` precomputes mode bins and counts for the fixed map
shape.

### `wl_ood.feature_cache`

Owns cache metadata validation, memory-mapped output arrays, completion flags,
flushing, and resumption. It does not interpret feature values statistically.

### `wl_ood.ps_model`

Owns cosmology-conditioned means/covariances, interpolation, likelihood
evaluation, MAP fitting, and independent chi-squared scoring.

### `wl_ood.ps_pipeline`

Owns the benchmark, fixed validation split, full-fit workflow, manifests,
runtime measurements, and submission creation by composing the focused units
above.

### CLI

Expose:

```text
wl-ood benchmark-ps-baseline
wl-ood validate-ps-baseline
wl-ood run-ps-baseline
```

All commands require explicit `--data-dir` and `--run-dir`. The seed defaults to
`20260730`. `run-ps-baseline` requires an explicit submission output path and
resumes compatible caches.

## Testing Strategy

Tests use small synthetic maps and datasets; private competition arrays are not
required for the automated suite.

Coverage includes:

- exact official noise standard deviation;
- deterministic, order-independent map noise;
- no added noise outside the mask;
- FFT frequency grid and real-FFT symmetry weighting;
- a power-spectrum result checked against a small independent reference;
- positive finite log-power enforcement;
- feature-cache creation, flush ordering, resumption, and metadata mismatch;
- exact fixed validation membership and counts;
- organizer covariance denominator;
- covariance validation;
- interpolation inside and outside the cosmology hull;
- known Gaussian likelihood and chi-squared examples;
- one-map scoring independence;
- optimizer failure reporting;
- validation thresholds fitted only from calibration scores;
- full score length, order, finiteness, and non-constancy;
- CLI behavior and resumability;
- submission ZIP structure; and
- manifest contents and checksums.

The real-data integration run then benchmarks, validates, fits, extracts, and
scores the actual arrays.

## Failure Handling

- Missing or changed raw files: stop before using caches.
- Incompatible cache metadata: stop and require a new run directory.
- Non-finite or non-positive power: stop with split and map index.
- Invalid covariance: stop with cosmology and diagnostic values.
- Interpolation or optimizer failure: stop with map index and parameters.
- Interrupted extraction: resume only rows whose completion flags were flushed.
- Wrong number of test scores: refuse to create a submission.
- Any attempt to derive a fit or threshold from test features: prohibited by
  module interfaces and tested workflow separation.

## Reproducibility Record

`run-manifest.json` records:

- UTC start and completion timestamps;
- Git revision and dirty-state flag;
- Python, NumPy, SciPy, and package versions;
- raw dataset filenames, sizes, and SHA-256 hashes;
- complete baseline configuration;
- validation split definition and report path;
- cache paths and completion counts;
- benchmark, extraction, fitting, and scoring runtimes;
- number of optimizer failures;
- score count and finite check;
- submission path, size, and SHA-256; and
- confirmation that inference used no public-test aggregate statistics.

## Success Criteria

- The full automated suite passes.
- The CPU benchmark is deterministic and memory-bounded.
- Leakage-safe validation completes with zero numerical or optimizer failures.
- Validation reports the InD false-positive tail at all four fixed levels.
- The full model uses all 25,856 training maps.
- All 10,000 public-test maps receive finite, non-negative, non-constant scores
  in original order.
- The submission ZIP contains only `result.json` at its root.
- The submission and all inputs are fingerprinted in the run manifest.
- The repository remains clean because generated artifacts are ignored.
- Codabench accepts the archive, and its public score and 100 TPR values are
  recorded as the reference experiment for later improvements.
