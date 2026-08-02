# Paired Conditional Likelihood-Ratio Submission Design

Date: 2026-08-02

## Status

Approved in direction by the user after the raw coarse-conditioned OTFM failed
its four-patch promotion gate. This sprint must reuse the completed matched
checkpoints and likelihood caches. It must not retrain the three OTFM branches.

## Goal

Build a single-instance eligible weak-lensing OoD score from the paired
likelihood difference

\[
R(x) = \operatorname{NLL}_{U}(x) - \operatorname{NLL}_{C}(x),
\]

where `U` is the matched unconditional OTFM and `C` is the coarse-conditioned
OTFM. Promote the method only if the ratio passes fixed held-only physics,
stability, leakage, fold, and low-FPR gates. If it passes, score the 10,000 test
maps once per branch and produce verified submission ZIPs in the same sprint.

The implementation can guarantee that every ZIP satisfies the frozen gates and
packaging contract. It cannot guarantee that the observed ratio will pass; the
thresholds must not be lowered after results are seen.

## Why this is materially different

The rejected method calibrated `NLL_C` as a standalone typicality score. Its
performance worsened as patches were added, its cross-noise Spearman correlation
was only 0.5275, and even the oracle-label branch failed to beat unconditional.
Those observations indicate that marginal high-dimensional field typicality and
model error dominate the raw likelihood.

The paired difference uses the same noisy map, patch, Haar representation, ODE
solver, Hutchinson probes, sample ID, and architecture in both terms. Common
field difficulty, shape-noise realization, and much of the likelihood-estimation
error should cancel. The remaining quantity estimates how much knowing the
map's own coarse field improves the fine-field likelihood. OoD maps can be
anomalous in either direction: coarse context may help unusually little or
unusually much.

## Alternatives considered

1. **Paired analytic likelihood ratio — selected.** It directly attacks the
   diagnosed typicality failure, reuses all expensive validation caches, needs no
   training, and can reach a production decision in minutes.
2. **Learned contrastive coarse/fine ratio — fallback.** Train an InD-only
   classifier on true versus deliberately mismatched coarse/fine pairs. This is
   higher upside but requires a new model, new negative-pair controls, and a GPU
   training sprint. It starts only if the analytic paired ratio fails.
3. **Package raw `NLL_C` despite rejection — declined.** It is technically
   possible but does not address the observed physics, stability, or label-
   leakage failures.

## Isolation and provenance

- Preserve the completed branch `codex/coarse-conditional-otfm` and its worktree.
- Create branch `codex/paired-conditional-likelihood-ratio` from commit
  `674a007` in a new worktree.
- Store new artifacts under
  `/home/neelshah/neurips_wl_runs/paired_conditional_ratio_20260802`.
- Treat `/home/neelshah/neurips_wl` as read-only except for reading official data,
  frozen feature/anchor artifacts, and the completed run caches.
- Reuse the exact unconditional, coarse, and oracle checkpoint hashes recorded
  by the rejected sprint.
- Reuse the immutable cache contract with four patches `[6, 33, 63, 48]`, 32
  Heun steps, two Hutchinson probes, probe seed 20260901, primary noise seed
  20260929, and repeat noise seed 20260930.

## Score construction

For each map and patch `j`, construct paired improvements

\[
R^C_j = \operatorname{NLL}_{U,j} - \operatorname{NLL}_{C,j},
\qquad
R^O_j = \operatorname{NLL}_{U,j} - \operatorname{NLL}_{O,j}.
\]

`R^C` is deployable. `R^O` is an oracle-label diagnostic and must never be used
for test scoring, calibration selection, or packaging.

Fit the following transformation separately for `R^C` and `R^O` using only the
512 calibration InD rows:

1. Per patch, compute the median and the scale
   `max(1.4826 * MAD, 1e-6)`.
2. Standardize each paired improvement into a signed robust z-score.
3. Form a coherent view `abs(median(z_j))` and a sparse view
   `max(abs(z_j))`.
4. Fit separate continuous empirical-surprisal calibrations for coherent and
   sparse views on the calibration rows.
5. Define the fixed map score as the maximum of the two calibrated views.

The primary candidate is the four-patch deployable union. One- and two-patch
versions are diagnostics only and cannot replace the primary after results are
seen. Raw paired differences, z-scores, coherent scores, and sparse scores are
retained for audit.

## Stage A: zero-new-GPU promotion gate

Stage A reads the completed calibration, evaluation, repeat-noise, and twelve
physical-proxy caches for `U`, `C`, and `O`. It must reject missing, non-finite,
misaligned, or contract-incompatible arrays before calculating a metric.

The four-patch deployable ratio passes only if every gate below is true:

1. **Paired lineage:** `U` and `C` matrices have identical row IDs, patch IDs,
   likelihood settings, source fingerprints, and shape-noise seeds.
2. **Low-FPR calibration:** no empirical held-InD FPR exceeds its exact 95%
   binomial upper bound at targets 0.001, 0.003, 0.01, 0.03, or 0.05.
3. **Cross-noise stability:** Spearman correlation between primary and repeat-
   noise deployable ratio scores is at least 0.60.
4. **Known-label leakage:** maximum absolute correlation with the five known InD
   labels is below 0.15 and lower than the matched unconditional score's maximum.
5. **Fold consistency:** the ratio beats both standalone `U` and standalone `C`
   on the official metric in at least three of four realization folds.
6. **Low-FPR fold consistency:** the ratio beats both standalone branches on the
   low-FPR metric in at least three of four folds.
7. **Material pooled gain:** the ratio's pooled promotion-family metric exceeds
   the better standalone branch by at least 0.005.
8. **Clustered uncertainty:** the 95% clustered-bootstrap lower bound of the
   ratio-minus-best-standalone gain is no worse than -0.002.
9. **Family breadth:** gain is positive on at least six of eight promotion
   families and at least eight of all twelve physical families.
10. **No easy-family domination:** nuisance-overlap families contribute at most
    half of total positive family gain.
11. **Oracle sanity:** the oracle paired ratio beats standalone `U` in at least
    three folds. This verifies that the matched models can express conditional
    information under the same score construction.
12. **Patch accumulation:** the four-patch metric is no more than 0.002 below the
    two-patch ratio metric. This rejects a repeat of the monotonic degradation
    seen in the raw likelihood sprint.

All decisions, component metrics, fold values, family gains, intervals, and
failed gate names are written to a JSON validation report. A failed Stage A
report cannot be used by the production packager.

## Stage B: frozen-anchor complementarity

Run Stage B only after Stage A passes. The completed run already contains
coarse scores for the exact first 128 rows of the anchor's thirteen generic
proxy families. Score only the missing unconditional likelihoods, then form the
same paired ratio without changing calibration or weights.

Fit field and frozen-anchor empirical-surprisal calibrations independently on
held InD rows. Evaluate exactly two preregistered blends:

- `anchor70_ratio30 = 0.70 * anchor + 0.30 * ratio`
- `anchor90_ratio10 = 0.90 * anchor + 0.10 * ratio`

A blend is production-eligible only if it improves the anchor in at least three
of four folds, its worst low-FPR fold delta is at least -0.002, and it retains at
least 70% of the anchor top-100. No alternative weights are evaluated.

The pure ratio remains eligible when Stage A passes even if one or both blends
fail Stage B. Only blends that pass are packaged.

## Production scoring

Production begins only after Stage A passes and at least the pure candidate is
eligible.

- Score each official test map with both `U` and `C` using identical patches,
  sample IDs, solver, steps, probes, and probe seed.
- Test maps are already noisy; never add shape noise.
- Score each map independently. Scores must be invariant to batch companions
  and operational batch size.
- Do not fit or read any test mean, variance, quantile, rank, histogram, mixture,
  or blend weight.
- Fill NaN-initialized memmaps only for missing cells and flush after each patch
  so the run is resumable.
- Compute the test ratio with the frozen held-only robust and empirical-
  surprisal calibrations.

Expected compute after Stage A is approximately 5–10 GPU minutes for the missing
generic unconditional cache and 60–75 GPU minutes for the two 10,000-map test
likelihood passes. Abort rather than launch if the current throughput estimate
exceeds two hours.

## Submission artifacts

Create a deterministic root-only ZIP for each eligible candidate:

- `paired_conditional_ratio_pure_v1.zip`
- `anchor70_paired_ratio30_v1.zip`, only if its Stage B gate passes
- `anchor90_paired_ratio10_v1.zip`, only if its Stage B gate passes

Each archive contains only `result.json` with 10,000 finite continuous scores in
original test order. Require at least 9,500 unique values and deterministic
byte-for-byte replay. The manifest records formulas, held calibrations, source
and checkpoint fingerprints, cache/report/code hashes, gate results, runtimes,
archive hashes, and the single-instance eligibility statement.

No archive is uploaded automatically. The user receives absolute paths and
SHA-256 hashes for manual submission.

## Implementation boundaries

Add a focused paired-ratio module and tests rather than embedding new logic into
the already large validator. Extend production scoring to load and score both
matched branches from one immutable cache contract. Do not modify the frozen
anchor artifact, the rejected run report, or unrelated legacy packagers.

Required tests include:

- paired subtraction uses exact row and patch alignment;
- common additive likelihood error cancels;
- two-sided coherent and sparse anomalies score highly;
- calibration uses held rows only;
- batch-companion invariance for both branches and their difference;
- cache-contract mismatch rejection;
- failed Stage A reports cannot package;
- only Stage B-passing blends are emitted;
- deterministic root-only ZIP replay with 10,000 scores.

## Stop conditions

- Stop immediately on cache/checkpoint lineage mismatch or any non-finite score.
- Stop without GPU use if Stage A fails. Preserve the report and diagnostics.
- Do not change a gate, candidate view, patch count, or blend weight after seeing
  Stage A or Stage B results.
- If the analytic ratio fails, record the result and begin a separately approved
  learned-contrastive-ratio design; do not package the failed raw coarse score as
  a substitute.
