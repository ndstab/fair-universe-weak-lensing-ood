# Matched Phase-Surrogate OoD Sprint

**Date:** 2026-08-02

**Status:** Approved design; implementation requires written-spec review

**Competition:** FAIR Universe Weak-Lensing OoD Challenge, Phase 2

## Objective

Build a fast, independently scoreable weak-lensing OoD component that tests
whether the spatial organization of a map is compatible with the InD
simulation family after controlling for its ordinary two-point power and
one-point pixel distribution.

The sprint is successful only if the new component improves the frozen
incumbent on held proxy families without reintroducing cosmology leakage or
low-FPR instability. A successful offline result may produce submission
archives, but no archive may be uploaded without explicit user approval.

## Why This Is a New Experiment

The teammate repository already contains three related lines of work:

- raw high-order, bicoherence, WPH, scattering, and topology summaries;
- synthetic phase-mixing and phase-coupling proxy families; and
- a KappaTNG hydro-minus-DMO, power-matched, phase/topology classifier.

The existing KappaTNG phase-only classifier is not the proposed method. It
learns the direction of one external baryonic response. Its strongest
configuration reached an external curve of `0.7675363133` but only
`0.0753986588` on the broad synthetic proxy set. Every tested blend with the
offline incumbent reduced the aggregate proxy curve, so it was correctly not
promoted. Its ready ZIP is not evidence for submitting it.

The new method instead constructs a null distribution separately for each map.
Its controls preserve that map's own power spectrum and pixel histogram while
destroying higher-order phase organization. It therefore asks a narrower
question: does the observed morphology differ from what its own lower-order
statistics can explain?

## Scientific Basis

Weak-lensing convergence fields become non-Gaussian through nonlinear
large-scale-structure evolution and astrophysical effects. Their power
spectrum does not retain all of the resulting information. Wavelet phase
harmonics, scattering coefficients, peak counts, and Minkowski functionals are
established summaries of the missing higher-order morphology.

Phase-randomized surrogate tests preserve Fourier amplitudes while removing
phase correlations. Iterative amplitude-adjusted Fourier transform (IAAFT)
surrogates additionally preserve the one-point amplitude distribution. The
observed-versus-surrogate contrast can therefore suppress two major nuisance
channels before any learned conditional calibration is applied.

Primary references:

- DES Y3 simulation validation of wavelet phase harmonics and scattering:
  <https://arxiv.org/abs/2310.17557>
- Weak-lensing scattering transform and non-Gaussian information:
  <https://academic.oup.com/mnras/article/507/1/1012/6325549>
- Fourier-phase surrogate tests with preserved power and morphological
  comparisons:
  <https://academic.oup.com/mnras/article/428/1/551/1055634>

These references motivate the statistic; they do not establish that it will
transfer to the hidden competition OoD mixture. Transfer must be demonstrated
by the preregistered offline gates below.

## Constraints and Non-Goals

- Score every test map independently. A score may use only that map and
  artifacts fitted on training data.
- Do not rank, normalize, calibrate, fit, or choose weights using a collection
  of test scores.
- Do not inspect or print test-score distributions during research selection.
- Keep the first offline pilot below three hours wall time and below thirty
  minutes of GPU compute.
- Use four patches and four surrogates for the first pilot. More patches or
  surrogates require evidence from the pilot or a measured runtime need.
- Reuse frozen maps, masks, splits, proxy rows, incumbent scores, feature
  conventions, metric code, and deterministic ZIP packaging.
- Do not retrain the OTFM or any large neural network.
- Do not use the public leaderboard to tune feature subsets, conditioning,
  aggregation, or blend weights.
- Do not submit the pre-existing KappaTNG phase-only ZIP as part of this sprint.

## Compared Approaches

### A. Matched per-map IAAFT surrogates — selected

Generate controls from each patch, compare compact phase/morphology summaries,
and condition the contrasts on train-fitted lower-order anchors.

Advantages:

- cancels much of the power-spectrum and one-point-distribution variation
  within each map;
- uses a direct scientific null rather than an external-family classifier;
- operates on small patches and can be cached quickly;
- can detect phase organization not represented by the current incumbent.

Risks:

- the footprint mask makes exact constrained surrogates impossible without a
  fixed fill convention;
- residual phase organization still varies with cosmology;
- four surrogates estimate the null center more reliably than its variance.

The design addresses these risks with deterministic inpainting, explicit
spectral-fidelity audits, train-fitted conditional location and scale, and a
cache of raw contrasts rather than prematurely collapsed scores.

### B. Direct WPH or bicoherence residuals — rejected for this sprint

The literature support is strong, but the repository already contains WPH,
high-order, bicoherence, and conditional-profile experiments. Repeating them
without a new null construction would add little information and would retain
their existing cosmology dependence.

### C. Existing KappaTNG phase-only moonshot — rejected

It can be packaged in seconds and has high upside on its matched external
family, but it failed broad proxy transfer and worsened every incumbent blend
tested offline. Uploading it would spend a submission on an already rejected
hypothesis.

## Frozen Pilot Population

The pilot reuses the Flow Mismatch evaluation population so comparisons are
paired rather than anecdotal:

- `384` held-InD rows covering all `101` cosmologies;
- two independent shape-noise realizations for cross-noise stability;
- `1,664` proxy rows, exactly `128` from each of `13` physics families;
- the same four realization-held folds;
- no competition test maps.

The proxy set must retain the existing family labels. Aggregate improvement
alone is insufficient because a detector can win by recognizing noise or
resolution artifacts.

## Patch Contract

The map has shape `1424 × 176`, and the survey mask contains no completely
observed `64 × 64` patch. The pilot uses fixed `64 × 64` patches on a stride-32
grid.

Candidate patches require at least `90%` mask coverage. Divide the map into
four equal row-coordinate quarters and choose the highest-coverage candidate
whose center lies in each quarter; break ties by row and then column. For the
current frozen mask this gives:

| Quarter | Row | Column | Coverage |
|---:|---:|---:|---:|
| 0 | 128 | 0 | 0.966064453125 |
| 1 | 640 | 32 | 0.966308593750 |
| 2 | 736 | 0 | 0.969726562500 |
| 3 | 1344 | 0 | 0.965332031250 |

The implementation must recompute these coordinates from the mask, assert the
exact frozen result, and record the mask checksum. Patch selection cannot use
map values, labels, proxies, or test data.

## Deterministic Mask Fill

IAAFT operates on a dense rectangular field. For each patch:

1. retain all observed pixel values unchanged;
2. fill missing pixels using a fixed normalized `5 × 5` local mean of observed
   neighbors, applied repeatedly until every missing location has support;
3. use the observed-pixel global mean only as the final fallback;
4. record the original Boolean patch mask and evaluate morphology only on its
   valid support.

The same fill algorithm is used for train, held, proxy, and any later test map.
It has no fitted parameters and consumes no other map.

## Matched-Surrogate Construction

For each sample, patch, and surrogate ID:

1. derive a BLAKE2 seed from the frozen global seed `20260802`, sample
   namespace, sample ID, patch coordinate, and surrogate ID;
2. initialize the dense patch with a deterministic random permutation of its
   filled values;
3. replace its Fourier magnitudes with those of the filled target patch while
   retaining the current phases;
4. inverse transform and rank-remap the result to the sorted filled target
   values;
5. repeat steps 3–4 exactly `16` times;
6. retain the rank-remapped iteration as the surrogate and reapply the original
   mask for feature measurement.

The pilot uses `4` surrogates per patch. The implementation must be independent
of batch order and batch size. Changing a sample's neighbors in a batch must
not change any of its surrogates or features.

Each row-patch record stores:

- target and surrogate radial log-power summaries;
- RMS log-power mismatch for each surrogate;
- exact sorted-value mismatch over the dense filled patch;
- the deterministic contract hash.

Promotion requires a median surrogate radial log-power RMS mismatch no greater
than `0.01`, a 99th percentile no greater than `0.03`, and exact histogram rank
matching within float32 tolerance. Failure means the null is invalid; it is not
fixed by loosening the score gates.

## Phase-Contrast Features

Compute target and surrogate features with the same fixed patch mask,
apodization, filters, and normalization. The pilot feature contract contains:

- phase-only squeezed bicoherence across the six existing disjoint Fourier
  bands;
- adjacent-scale wavelet coupling terms already defined by the generic feature
  extractor;
- standardized peak and void densities;
- Minkowski perimeter and Euler characteristic at fixed standardized
  thresholds;
- patch skewness and excess kurtosis only as audited controls, not automatic
  production features.

For every feature, cache:

`contrast = target_feature - median(surrogate_features)`

and

`spread = 1.4826 × median(abs(surrogate_feature - surrogate_median))`.

With only four surrogates, the per-map spread is diagnostic. It is not used as
the sole denominator. Feature scaling is learned from training contrasts.

Two preregistered views are evaluated:

1. `phase_core`: bicoherence and wavelet-coupling contrasts;
2. `phase_morphology`: `phase_core` plus peak, void, perimeter, and Euler
   contrasts.

The skewness and kurtosis controls are reported separately. If they dominate a
selected score, the candidate fails the intended phase-specific hypothesis.

## Conditional Location-and-Scale Model

Self-matching reduces but does not guarantee removal of cosmology dependence.
The production-compatible nuisance inputs are train-fitted, per-map lower-order
anchors:

- the frozen radial log-power features;
- global mean and standard deviation;
- the existing analytic noise and mask constants.

Ground-truth cosmology labels are used only for leakage audits, never as an
input required at test inference.

Within each realization-held fold:

1. standardize anchors on fit rows;
2. apply a fixed quadratic expansion;
3. standardize each contrast feature on fit rows and fit ridge conditional
   means with `alpha=10.0`;
4. fit ridge log variances with `alpha=10.0` to
   `log(mean_residual² + 1e-4)` and clamp the predicted conditional standard
   deviation to `[0.25, 4.0]` in standardized-feature units;
5. studentize the held contrasts;
6. fit OAS covariance whitening on fit-row studentized residuals;
7. score held and proxy rows with that frozen transform.

The fixed ridge strengths, floor, and clipping limits cannot be selected by
proxy-family performance. The same fold models later score a test map from
that map's anchors and contrast features only.

## Patch and Map Scores

For each view, compute a non-negative whitened residual energy per patch. Tail
calibration uses fit/held InD scores only.

Define two fixed map aggregations:

- `coherent`: the mean of the four calibrated patch scores;
- `localized`: the maximum of the four calibrated patch scores.

Define the primary specialist as the equal-weight average of the independently
InD-calibrated coherent and localized scores. This balances a field-wide phase
shift against a localized morphology defect without selecting an aggregation
from proxy results.

The pilot reports both views and both aggregations, but only the predefined
primary specialist enters the promotion blend tests.

## Validation and Promotion Gates

All gates are evaluated independently for `phase_core` and
`phase_morphology`. A view is eligible only if every scientific-integrity gate
passes:

1. **Surrogate fidelity:** power and histogram requirements above pass.
2. **Held-InD calibration:** at target FPRs
   `(0.001, 0.003, 0.01, 0.03, 0.05)`, threshold calibration scores with the
   corresponding higher empirical quantile and require the held empirical FPR
   to lie inside the exact two-sided `95%` binomial interval for every target,
   matching the Flow Mismatch pilot.
3. **Cosmology leakage:** maximum absolute Spearman correlation with any of the
   five labels is below `0.15` after conditional studentization.
4. **Noise stability:** cross-noise Spearman correlation is at least `0.60`.
5. **Proxy breadth:** standalone improvement over the aligned incumbent is
   positive in at least three of four folds.
6. **No shortcut family:** noise or resolution/easy families contribute no more
   than `50%` of total improvement.
7. **Hypothesis integrity:** on the proxy population, the squared studentized
   residual energy of the skewness/kurtosis controls divided by control plus
   selected-view energy is no greater than `0.50`.
8. **Runtime:** the offline pilot finishes below three hours wall time and
   thirty minutes GPU time.

For each eligible specialist, evaluate only these frozen combinations with the
aligned incumbent:

- `standalone_phase`;
- `blend_phase_10 = 0.90 × incumbent + 0.10 × phase` after independent InD
  calibration of both components;
- `tail_union_phase`, the maximum of the incumbent and phase component after
  subtracting each component's train-fitted 99th-percentile InD threshold.

A combination is promotable only if its mean aggregate proxy delta is at least
`+0.003`, its mean low-FPR delta is at least `+0.002`, at least three of four
fold deltas are positive, and the worst fold delta is no worse than `-0.002`.
The standalone candidate may be packaged as a diagnostic only when it passes
all scientific-integrity gates; it need not beat the incumbent blend gate.

If no candidate passes, stop without reading competition test maps or creating
a submission archive.

## Cache and Fast-Submission Architecture

The expensive unit is a deterministic row-patch contrast, not a model score.
Cache it once and make scoring and packaging cheap.

Each cache is contract-keyed by:

- source map checksum and namespace;
- mask checksum and selected patch coordinates;
- fill, IAAFT, seed, iteration, and surrogate count configuration;
- feature names and feature implementation version;
- dtype and software versions.

The cache stores float32 target features, surrogate medians, surrogate MADs,
contrasts, fidelity diagnostics, committed-row markers, and per-record BLAKE2
integrity tags. Interrupted or corrupt records reset to missing and can be
recomputed without invalidating sound rows.

Once train, held, proxy, and—only after promotion—test contrasts exist, fold
fitting, score generation, blend evaluation, audit manifests, and deterministic
ZIP packaging must be CPU-only and complete in under five minutes. This is the
mechanism for faster meaningful submissions: one expensive immutable cache can
support the three preregistered candidates without repeated map inference.

## Production and Submission Contract

Production is permitted only after an offline promotion decision is recorded.
Before test scoring, freeze:

- the selected view and primary aggregation;
- all fold models and calibration arrays;
- the exact incumbent score artifact and checksum;
- candidate formulas and names;
- source/test/mask checksums;
- deterministic sample order.

The current production incumbent source is
`submissions/multitask_inception_topology_v1/scores.npz`, checksum
`c93b7b765a1be7200024ee693aee0f791824ff0e0fefa43f0c88b14eac06363c`,
as referenced by the scale-conditioned manifest. If the team has since
submitted and accepted a newer incumbent, its identity must be recorded before
test scoring; it cannot be chosen after inspecting new test scores.

Every production score vector must:

- contain exactly `10,000` finite float scores in original order;
- be reproducible across batch sizes;
- score a map identically when run alone;
- avoid any test-batch statistic;
- be packaged as a ZIP containing only root-level `result.json` with the
  `ood_scores` key;
- include a manifest with hashes, runtime, configuration, fold replay checks,
  cache-integrity counts, and the offline promotion evidence.

Archive creation is not upload authorization. Present all eligible candidates,
formulas, hashes, offline evidence, and risks to the user and wait for explicit
approval before any Codabench upload.

## Error Handling

- Reject non-finite input, features, contrasts, model outputs, and final scores.
- Reject a changed mask, patch contract, source ordering, or feature-name
  contract rather than silently adapting.
- Reject cache rows whose integrity tag or committed marker fails.
- Resume only missing or invalid row-patch records.
- Stop if surrogate fidelity fails, even if a downstream proxy score looks
  strong.
- Treat a batch-invariance failure as a correctness failure.
- Treat the inherited unrelated full-suite submission-count test failure as
  separate from this sprint; do not modify it without explicit scope.

## Test Strategy

Implementation follows test-driven development. At minimum, automated tests
must cover:

- deterministic seeds and independence from batch composition/order;
- exact frozen patch selection and mask-fill behavior;
- IAAFT histogram matching and spectral-fidelity diagnostics;
- phase destruction on a constructed coupled-phase field;
- near-zero average contrast for Gaussian random fields;
- sensitivity to a power-matched phase-coupled synthetic field;
- cache resume, corruption recovery, and contract invalidation;
- conditional mean and log-variance fitting without label inputs at inference;
- fold isolation and single-map inference invariance;
- all scientific and blend promotion gates;
- exact score count, ordering, JSON schema, ZIP contents, and deterministic
  packaging;
- a CLI guard proving that validation accepts no competition test input.

## Execution Checkpoints

1. **CPU synthetic unit tests:** validate the null on small Gaussian and
   phase-coupled fields.
2. **A100 microbenchmark:** score at most `32` held rows and `32` proxy rows;
   measure surrogate fidelity, batch invariance, memory, and projected runtime.
3. **Offline pilot:** score the frozen held/proxy population, then run all gates.
4. **Decision:** record promote or reject before touching competition test maps.
5. **Production cache:** only for promoted candidates; resume safely and do not
   print distribution summaries.
6. **Packaging:** create only eligible archives, verify them, and stop for user
   approval before upload.

The pilot may terminate early at any failed checkpoint. A fast scientific
rejection is a valid result and preserves both GPU time and submission budget.
