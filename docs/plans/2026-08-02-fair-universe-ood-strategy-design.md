# FAIR Universe Phase-2 OoD Strategy Design

**Date:** 2026-08-02

**Status:** Approved for implementation planning
**Competition objective:** Maximize the mean TPR over logarithmically spaced FPRs from 0.001 to 0.05 on the unseen final holdout while preserving single-instance eligibility.

## 1. Current position

- Best public score: `0.562092` from `drc19_hgb_balanced_tailgate_w0p150.zip`.
- Clean holdout-safe anchor: `0.551032` from `multitask_inception_topology_v1`.
- Public leaders at the audit time: `0.7336` and `0.9392`.
- The public-feedback lineage gains mostly at high FPR and already shows an approximately `0.011` gap over the clean anchor.
- Incremental feature/blend tuning cannot plausibly close the leaderboard gap.

## 2. Constraints

- Five submissions are available per day and 100 submissions in total.
- Target three meaningful submissions on an active submission day, with a fourth only when it answers a distinct question.
- Keep one daily slot unused for recovery or an unusually informative follow-up.
- Normal compute jobs should finish within three hours; no uninterrupted 10–12 hour jobs.
- Longer production work must be resumable in two- to three-hour checkpoints.
- Every submitted score must be computed from one map at a time. No test-batch ranks, mixture fits, index partitions, or statistics estimated from the 10,000-map test collection.
- No public score may be used as a direct label for fitting a test-set transformation.

## 3. What the repository audit changes

The following ideas already exist and are not first-sprint candidates by themselves:

- Global WPH profile and LOCO discriminative WPH.
- Density-split band-energy summaries.
- Marked power and cross spectra.
- Nearest-cosmology conditional manifolds.
- Five-label inverse/forward residual models.
- Cross-scale conditional null scores.
- DenseNet two-block spatial dispersion.
- Raw, typicality, conditional, and feature-space flow likelihoods.

Observed limitations include:

- Global WPH transferred poorly to the external physics suite (`0.1265` challenge-style score).
- Density-split summaries were essentially random on matched CAMELS hydro directions (`0.0180`).
- Marked spectra added about `0.0014` in one fold and hurt the low-FPR component.
- DenseNet spatial summaries helped some folds by `0.002–0.007` but were inconsistent.
- Existing conditional OTFM proxy variants remained weak and highly selective for phase/resolution corruptions.
- No conditional feature-flow candidate passed its promotion gate.

Therefore, the first sprint must not rebuild these components under new names.

## 4. Leaderboard interpretation

The filename sequence `twoblock_probe -> partition -> ranking_combo -> lb -> exp27 -> exp28` is more consistent with leaderboard probing or test partitioning than with a recognizable astrophysical pipeline. This is an inference, not proof.

Consequences:

1. The `0.9392` public score is not assumed to estimate final-holdout performance.
2. We will not copy batch-based probing because it is unlikely to generalize and may violate final eligibility.
3. Public submissions will decide among pre-specified architectures, not tune a large family of test-set weights.

## 5. Primary new strategy: flow mismatching

The frozen OTFM model already learns a velocity field from Gaussian noise to normal weak-lensing patches. Raw likelihood requires numerical divergence integration and is confounded by cosmology and nuisance parameters.

Flow Mismatching instead evaluates the learned velocity along direct paths from deterministic Gaussian noise to the observed patch:

\[
x_t=(1-t)x_0+ty, \qquad g=y-x_0,
\]

and measures

\[
Z_t=\lVert v_\theta(x_t,t)-g\rVert^2.
\]

For normal patches, the learned normal velocity should agree with the geometric target direction. For anomalous patches, the velocity discrepancy contains a Fisher-divergence term between the anomalous and normal path distributions. The method uses forward passes only and avoids Hutchinson divergence estimation. The source method is [Chen et al., 2026](https://arxiv.org/html/2605.23070).

### 5.1 Patch score

- Use the checkpoint's exact fixed-mask patch layout and data scaling.
- Add the challenge shape noise deterministically for training calibration rows.
- Use interior times `(0.2, 0.4, 0.6, 0.8)` initially.
- Weight time discrepancies by `t^2`.
- Use four deterministic Gaussian paths for the pilot.
- Preserve the complete pixel heatmap until aggregation.
- Evaluate both the 25th path percentile and path mean. The low percentile is expected to suppress false positives.

### 5.2 Map aggregations

Derive independent per-map candidates from the same patch matrix:

- `coherent_mean`: robust mean of patch-average mismatch, targeting field-wide physics shifts.
- `coherent_trimmed`: 10% trimmed mean, limiting isolated numerical failures.
- `localized_topk`: mean of the largest 10% patch scores, targeting spatially localized changes.
- `heatmap_tail`: maximum plus top-1%-pixel mean within each patch, then robustly aggregate across patches.
- `dual_union`: empirically calibrated maximum of coherent and localized evidence.

All tail calibration references are frozen held-out InD training rows.

### 5.3 Five-parameter deconfounding

For every map retain a signature over patch, time, and aggregation channels. Fit the expected signature from the five training labels

`(Omega_m, S8, T_AGN, f0, delta_z)`

using cross-fitted ridge/quadratic response models. On test maps, estimate labels only from existing train-fitted inverse models. Subtract the expected signature and score the residual with shrinkage covariance and robust coordinate tails.

Promotion requires absolute Spearman correlation below `0.15` with every known label on held InD rows. This is stricter than the current raw OTFM behavior.

## 6. Secondary physics strategy

If flow mismatching passes the first gate, the next independent addition is not global WPH or ordinary density split. It is a conditional phase-response statistic:

- Define the environment from a coarse, mask-normalized field.
- Measure cross-scale phase-harmonic coupling separately in void, intermediate, and halo environments.
- Fit the five-parameter response surface.
- Project the residual orthogonally to the local five-parameter tangent.
- Calibrate halo, phase, growth, and numerical blocks separately.
- Fuse specialist p-values by an empirical min-p/max-surprisal calibration fitted only on held InD rows.

This targets the mismatch between the training HMcode global transfer `T(k,z)` and environment-dependent hydrodynamical rearrangement. It will be implemented only after the flow-mismatch sprint because the existing global WPH and density-split results do not independently justify submissions.

## 7. Validation protocol

### 7.1 Type-I calibration

- Split by realization slot and cosmology, never by individual noisy view.
- Use disjoint noise seeds for fit, calibration, and evaluation.
- At target FPRs `0.001, 0.003, 0.01, 0.03, 0.05`, require empirical held-InD FPR within the fold's binomial 95% interval.
- Require finite outputs, deterministic replay, and identical single-map scores whether evaluated alone or in a batch.

### 7.2 Power tests

- Use the 13-family physical proxy bank only for evaluation.
- Report the official low-FPR metric overall and by family.
- Keep CAMELS, KappaTNG, and FLAMINGO as external transfer audits after geometry, noise, mask, and marginal power matching.
- External performance cannot be the sole reason to submit.

### 7.3 Promotion gates

A new standalone detector is promotable only if:

- Held-InD tail calibration passes.
- Maximum absolute known-label Spearman correlation is below `0.15` after deconfounding.
- Cross-noise rank correlation is at least `0.60`.
- Proxy improvement is present in at least three of four folds.
- No single easy family such as noise or resolution contributes more than half of the total improvement.
- Pilot wall time is under three hours.

A blend with the incumbent is promotable only if:

- Mean proxy metric improves by at least `0.003`.
- Low-FPR proxy component improves by at least `0.002`.
- At least three of four folds improve.
- Worst-fold degradation is no worse than `-0.002`.

## 8. First submission quartet

Only candidates passing the gates are submitted:

1. Deconfounded coherent flow-mismatch score.
2. Deconfounded localized flow-mismatch score.
3. Empirically calibrated coherent/localized union.
4. Conservative union/incumbent fusion.

If fewer than three candidates pass, unused daily slots remain unused. Submission count is not a substitute for independent evidence.

## 9. Compute schedule

- Unit and CPU synthetic tests: under five minutes.
- Frozen-model pilot on a small balanced validation set: under 30 minutes.
- Four-fold proxy pilot with `K=4`, `T=4`, and four patches: under two hours.
- Full 38-patch validation: resumable, with a three-hour checkpoint boundary.
- Test scoring: resumable patch cache with integrity tags; package candidates without rescoring.
- No retraining of OTFM during the first sprint.

## 10. Experiment ledger

Every run and submission records:

- Hypothesis and expected direction.
- Exact command and configuration.
- Source and checkpoint hashes.
- Fit/calibration/evaluation splits and seeds.
- Runtime and peak GPU memory.
- Overall, low-, mid-, and high-FPR metrics.
- Per-family proxy results.
- Output score and archive hashes.
- Codabench filename, timestamp, public score, and conclusion.

The ledger is append-only. A public result may reject or support a hypothesis, but it may not silently alter the historical record.
