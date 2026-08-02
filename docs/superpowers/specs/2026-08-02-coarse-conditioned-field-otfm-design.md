# Coarse-Conditioned Field-Level OTFM Sprint

**Date:** 2026-08-02

**Status:** Approved design; implementation requires written-spec review

**Competition:** FAIR Universe Weak-Lensing OoD Challenge, Phase 2

## Objective

Test whether a field-level continuous-time flow can detect hidden simulation
mismatch from the conditional relationship between a weak-lensing map's
large-scale structure and its fine-scale structure. The first cycle must reach
a valid standalone submission in at most five hours of wall time, while also
producing two preregistered blends that protect the verified `0.562092` public
anchor.

The sprint is successful only if it establishes a calibrated, single-map OoD
score with credible low-FPR validation. It is not successful merely because a
large model trains, an average AUROC rises, or a public score changes.

## Decision Context

The most recent CNN blend scored `0.5414`, below the verified `0.562092`
anchor. This was not a small miss. Relative to the older `0.560062` ordering,
the CNN candidate retained:

- none of the prior top 10;
- `15%` of the prior top 100;
- `40.6%` of the prior top 500; and
- `84.5%` of the prior top 1,000.

Its median absolute rank displacement was `298`, its 90th percentile was
`882`, and its maximum was `2,953`. The official metric concentrates on the
extreme low-FPR tail, so a broad synthetic-proxy improvement could not rescue
this tail destruction. The root failure was research selection against an
insufficient proxy objective.

Consequently, this sprint freezes three rules:

1. Tail retention and low-FPR calibration are first-class validation outputs.
2. The new representation receives a standalone submission; blending cannot
   hide whether the scientific idea works.
3. The public anchor is never retrained as part of this sprint. It appears only
   as a frozen score channel in two risk-controlled submission candidates.

## Scientific Hypothesis

Let a masked convergence patch be decomposed by a one-level orthonormal Haar
transform into a coarse low-pass component `c` and fine detail coefficients
`d`. Train a continuous-time flow on InD maps to model

\[
p_\theta(d \mid c, m),
\]

where `m` is the transformed mask/support information.

Large-scale amplitude, cosmology, and much of the ordinary nuisance variation
are represented in `c`. A changed baryonic prescription, gravity model,
numerical solver, or sub-grid response can change the conditional fine-scale
organization even when the map's broad power and amplitude remain plausible.
The score therefore asks whether the observed fine structure is typical for
that map's own coarse structure, rather than whether the complete map has high
raw likelihood under a cosmology-mixed density.

This hypothesis is motivated by:

- the challenge definition and its hidden change-of-physical-assumptions
  setting: <https://arxiv.org/abs/2604.14451>;
- field-level continuous-time flow results for weak-lensing model mismatch:
  <https://arxiv.org/abs/2505.00632>;
- nuisance hardening by removing sensitivity to known parameter directions:
  <https://arxiv.org/abs/1903.01473>; and
- field-level and scale-dependent baryonic responses:
  <https://arxiv.org/abs/2401.15891> and
  <https://arxiv.org/abs/2510.09761>.

The papers motivate the direction but do not predict the competition score.
Only the preregistered validation and leaderboard experiments may do that.

## Why This Is Not the Discarded Flow Strategy

The existing production OTFM is unconditional. Its raw field likelihood is
strongly affected by cosmology, pixel energy, and nuisance variation. The
unconditional four-patch pilot produced weak proxy performance, and a
post-hoc regression of raw likelihood could not fully remove the confounding.
The existing conditional *feature* flow compresses the map before density
modelling and also transferred poorly to physical proxies.

This sprint instead trains the already-prepared conditional *field* branch.
It conditions every fine-scale coefficient prediction directly on the same
map's coarse field and mask. No trained conditional field checkpoint currently
exists in the audited repository, so this is a new experiment rather than a
rerun or reweighting of the failed score.

## Compared Approaches

### A. Coarse-conditioned field OTFM — selected

Advantages:

- retains field-level information discarded by feature models;
- conditions out large-scale realization and amplitude map by map;
- requires only InD training examples;
- is deployable without hidden cosmology labels; and
- has prepared repository infrastructure, making a same-day result possible.

Risks:

- conditional likelihood can still prefer over-smoothed or low-complexity
  inputs;
- exact likelihood integration may be expensive;
- mask boundaries can dominate fine coefficients; and
- a four-patch pilot can miss spatially localized anomalies.

These risks are controlled by two-sided typicality, explicit mask
conditioning, independent-noise audits, and a staged four-to-sixteen-patch
schedule.

### B. Nuisance-tangent-hardened summary likelihood — deferred

Project high-order summaries away from response directions induced by the five
known training parameters, then fit a regularized conditional density. This is
interpretable and fast, but the repository already contains many related
summary, residual, ratio, and feature-flow models. Their evidence suggests
incremental upside and information loss relative to a field model.

### C. External hydrodynamic likelihood ratio — deferred

Train directly on FLAMINGO, CAMELS, or KappaTNG hydro-versus-DMO effects. This
has strong physical semantics but has transferred weakly after challenge
geometry, mask, noise, and marginal-spectrum matching. It remains an
evaluation audit, not the primary training distribution.

## Data and Split Contract

- Train only on official InD training maps and their masks.
- Split by realization slot and cosmology group, never by individual noisy
  view.
- Keep fit, calibration, and evaluation noise seeds disjoint.
- Reuse the repository's frozen held rows, four-fold assignments, map scaling,
  patch coordinates, and physical proxy bank where compatible.
- Do not inspect, summarize, rank, normalize, or fit against the collection of
  competition test scores.
- A map's output must be identical whether scored alone or inside any batch.
- The five ground-truth simulation labels may enter the oracle diagnostic
  branch, but never the deployable coarse branch or test-time inference.

## Model Branches

Train three branches using matched architecture, data rows, optimizer budget,
and seeds so the comparison is interpretable.

### 1. Unconditional control

Model the Haar fine coefficients without a coarse condition. This control
measures whether the new training run merely reproduces the known raw
likelihood confounding.

### 2. Coarse-conditioned deployable model

Condition the vector field on:

- the low-pass Haar component of the same patch;
- coarse valid-support/mask channels;
- the continuous flow time; and
- fixed analytic noise/mask constants already permitted by the challenge.

The fine coefficients remain the flow state and likelihood target. Ground
truth cosmology and nuisance labels are excluded.

### 3. Oracle-conditioned diagnostic

Condition on the same inputs plus the five known labels
`(Omega_m, S8, T_AGN, f0, delta_z)`. This branch is never deployable. It
estimates whether the conditional likelihood idea has a useful ceiling and
whether the coarse field recovers enough of the nuisance information.

## Frozen Stage-0 Training Budget

- Use the prepared `train_conditional_otfm.py` and conditional-field modules.
- Use four fixed high-coverage `64 x 64` patches per map.
- Use the repository's existing scaling and mask conventions exactly.
- Start with base width `32`, `20` epochs, at most `400` optimizer steps per
  epoch, and one fixed model seed.
- Train unconditional, coarse, and oracle branches under the same cap.
- Write resumable checkpoints and a manifest containing data, code, config,
  and checkpoint hashes.
- Stop the stage if cumulative wall time reaches two hours before all three
  branches finish; diagnose the bottleneck instead of silently reducing data.

Hyperparameter search is outside stage 0. A second seed is authorized only
after the coarse branch passes the scientific and runtime gates.

## Score Construction

### Patch evidence

For each branch and patch, compute conditional log likelihood with the same
integration settings. Raw negative log likelihood is retained for audit but is
not the primary anomaly score.

Fit the expected InD location and scale of log likelihood from held calibration
rows, conditional on permitted coarse diagnostics if required. Define signed
standardized typicality

\[
z = \frac{\ell - \widehat{E}[\ell \mid c,m]}
         {\widehat{\sigma}[\ell \mid c,m] + \epsilon}.
\]

Use two-sided evidence `abs(z)` because both implausibly low and implausibly
high likelihood can indicate model mismatch. All location, scale, clipping,
and empirical-tail parameters are fitted on training/held InD rows only.

### Map aggregators

Preregister two map-level views from the same patch evidence:

1. `coherent`: a robust mean of patch evidence for field-wide mismatch;
2. `sparse`: a fixed calibrated union of the largest patch evidence values,
   targeting localized mismatch.

Calibrate their union on held InD rows with no family-specific or leaderboard
tuning. Stage 0 uses four patches. Sixteen patches are scored only if the
four-patch coarse model passes the promotion gates.

### Anchor compatibility

The public `0.562092` prediction vector is immutable. It is not a model input,
training target, pseudo-label, or validation label.

Before blending, transform both anchor and new evidence through calibration
maps fitted from corresponding held training scores. If a reproducible held
version of the anchor score cannot be generated, no numerical blend is
authorized; submit the standalone new score and reconstruct the anchor
calibration lineage first. Test-set ranks or test-distribution moments are
forbidden.

The two frozen blend weights are:

\[
s_{30} = 0.70s_{anchor} + 0.30s_{field},
\]

\[
s_{10} = 0.90s_{anchor} + 0.10s_{field}.
\]

No alternate weights may be created in response to the first public result.

## Validation Ladder

### Gate 0: contract and determinism

- finite score for every evaluated map;
- deterministic replay from identical artifact hashes;
- score invariant to batch order and batch companions;
- transformed mask aligned with every Haar channel; and
- exact separation between fit, calibration, and evaluation rows/seeds.

### Gate 1: nuisance and noise control

On held InD rows report:

- false-positive calibration at FPR `0.001`, `0.003`, `0.01`, `0.03`, and
  `0.05`, with binomial intervals;
- absolute Spearman correlation with each of the five known labels;
- cross-noise rank correlation for repeated views of the same signal; and
- score distributions by cosmology group and mask coverage.

The coarse branch must improve maximum label correlation over the
unconditional control and reach cross-noise rank correlation at least `0.60`.
Any severe low-FPR inflation blocks submission regardless of proxy power.

### Gate 2: conditional-model test

On the frozen physical proxy evaluation, compare unconditional, coarse, and
oracle branches using the official 100-point low-FPR metric, per fold and per
family.

Let `G_coarse` and `G_oracle` be their gains over the unconditional control.
Continue only if:

- the oracle has positive gain in at least three of four folds;
- the coarse branch has positive gain in at least three of four folds;
- the coarse branch recovers at least `60%` of the oracle's positive aggregate
  gain; and
- no easy noise/resolution family supplies more than half the gain.

If the oracle fails, the conditional-likelihood hypothesis lacks evidence and
the sprint stops. If the oracle passes but coarse fails, improve the
conditioning representation before scaling the flow.

### Gate 3: anchor complementarity

For standalone and preregistered blends, report:

- the official aggregate metric;
- its low-, middle-, and high-FPR thirds;
- four-fold mean and worst-fold change;
- top-10, top-20, top-50, top-100, and top-500 retention relative to the
  incumbent held ordering; and
- family-level wins and losses.

A protective blend is promotable only if it improves at least three of four
folds, does not reduce the low-FPR third by more than `0.002`, and retains at
least `70%` of the incumbent top 100 on the paired held evaluation.

## First Submission Cycle

If the standalone coarse model passes Gates 0–2, prepare exactly these
archives before observing any new public result:

1. `coarse_conditional_field_otfm_pure_v1`: pure new calibrated field score;
2. `anchor70_coarse30_v1`: 70% frozen anchor, 30% field score; and
3. `anchor90_coarse10_v1`: 90% frozen anchor, 10% field score.

Upload order is pure, 70/30, then 90/10, because the pure score gives the most
scientific information. Keep at least one daily slot unused. Do not generate a
post-result weight sweep.

Public outcomes are interpreted as follows:

- Pure strong, blends weaker: the anchor suppresses a better representation;
  move toward the pure lineage.
- Pure weak, 70/30 or 90/10 strong: the signal is complementary but not
  calibrated or comprehensive enough to stand alone.
- All weak: reject the coarse-conditioned field score rather than making a
  blind weight search.
- Pure and blends close to anchor: acquire the per-threshold TPR curve before
  deciding whether the score adds a narrow tail specialist.

No archive may be uploaded without explicit user approval after local
integrity checks and archive hashes are shown.

## Runtime and Stop Rules

Target wall-clock budget to a submittable JSON/ZIP:

- data and contract smoke test: `15–30` minutes;
- three stage-0 training branches: `45–90` minutes total;
- four-patch held/proxy likelihood scoring: `60–120` minutes;
- gates, calibration, test scoring, and packaging: `60–120` minutes.

The target is `3–5` hours. No individual job may be launched with an expected
runtime above three hours. Sixteen-patch scoring must be resumable and is
authorized only after the four-patch gate. If the five-hour target is missed,
preserve checkpoints/caches and report the exact bottleneck before continuing.

## Deliverables and Ledger

Every run records:

- hypothesis and branch role;
- exact command, configuration, seed, and environment;
- source-data, split, code, checkpoint, and mask hashes;
- fit/calibration/evaluation row identities;
- runtime and peak GPU memory;
- label correlations and cross-noise stability;
- overall, per-fold, per-family, and FPR-third metrics;
- incumbent tail-retention table;
- score-vector and archive hashes; and
- Codabench filename, timestamp, public score, and conclusion after upload.

The ledger is append-only. A public score can reject or support the experiment,
but cannot retroactively alter its gates, calibration, weights, or history.

## Explicit Non-Goals

- Do not retrain or fine-tune the `0.562092` anchor.
- Do not train the new flow to imitate the anchor scores.
- Do not use the competition test collection for calibration or ranking.
- Do not submit the protected-tail candidate as a claimed new strategy.
- Do not revive the rejected phase surrogate, unconditional raw likelihood,
  external-domain classifier, or CNN weight sweep inside this sprint.
- Do not claim first-place performance from proxy metrics or papers.

## Review Decision

Implementation planning may begin only after the user reviews this written
specification. GPU execution remains separately gated by the resulting plan
and the submission upload remains separately gated by archive review.
