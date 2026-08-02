# Matched Phase-Surrogate Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and—only after offline promotion—package a deterministic matched phase-surrogate weak-lensing OoD candidate as a root-only Codabench ZIP.

**Architecture:** A GPU-native primitive creates four deterministic IAAFT controls for each of four fixed high-coverage patches, and a compact extractor caches target-minus-control phase/morphology contrasts. A CPU-only conditional location/scale model removes lower-order nuisance dependence, validates the frozen specialist and two incumbent combinations, then reuses the immutable cache to package eligible 10,000-row production scores.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch, scikit-learn, pytest, existing weak-lensing metric and submission utilities.

## Global Constraints

- Work only in a new remote worktree and branch; never modify `/home/neelshah/neurips_wl`.
- Keep the live teammate repository read-only and use it only for large source data and frozen artifacts.
- Score every test map independently; never derive ranks, moments, quantiles, thresholds, calibration, model weights, or feature selection from the test batch.
- Freeze global seed `20260802`, four patches, four surrogates, and sixteen IAAFT iterations.
- The pilot population is 384 held-InD rows, two independent shape-noise views, and 1,664 proxy rows: 128 from each of 13 families.
- The pilot must stay below three hours wall time and thirty minutes GPU time.
- Validation must not accept a competition-test input.
- Do not read competition test maps unless an offline candidate is promoted.
- A submittable artifact is a deterministic ZIP containing only root-level `result.json` with exactly 10,000 finite scores in original order.
- Never upload a ZIP without explicit user approval.
- Follow test-driven development: establish a focused failing test before each implementation.
- Preserve the inherited unrelated full-suite failure in `tests/test_build_otfm_cosmology_conditional_submissions.py`; do not change its source or test.

## Worktree

- Source commit: `9be777d0b806ec8d0d034b23b745601e400486d2` from `/home/neelshah/neurips_wl_worktrees/codex_flow_mismatch_20260802`.
- Branch: `codex/phase-surrogate-sprint`.
- Worktree: `/home/neelshah/neurips_wl_worktrees/codex_phase_surrogate_20260802`.
- Large inputs remain at `/home/neelshah/neurips_wl/data`, `/home/neelshah/neurips_wl/features`, `/home/neelshah/neurips_wl/models`, and `/home/neelshah/neurips_wl/submissions` and are passed by explicit absolute CLI paths.

## File Map

- `scripts/wl_phase_surrogate.py`: patch selection, mask filling, deterministic IAAFT controls, and fidelity diagnostics.
- `scripts/wl_phase_contrast.py`: compact bicoherence, wavelet-coupling, topology, and control features plus target-minus-surrogate contrasts.
- `scripts/cache_phase_surrogate.py`: contract-keyed resumable row/patch cache with integrity tags.
- `scripts/phase_surrogate_scores.py`: quadratic anchor conditioning, conditional scale, OAS whitening, and map aggregation.
- `scripts/validate_phase_surrogate.py`: fidelity, leakage, stability, proxy, shortcut, runtime, and blend promotion gates.
- `scripts/run_phase_surrogate_pilot.py`: frozen microbenchmark and held/proxy pilot orchestration.
- `scripts/build_phase_surrogate_submissions.py`: promoted-model replay, production scoring, audit manifest, and deterministic ZIP packaging.
- `tests/test_wl_phase_surrogate.py`: primitive and synthetic-null tests.
- `tests/test_wl_phase_contrast.py`: feature and contrast tests.
- `tests/test_cache_phase_surrogate.py`: resume, corruption, and contract tests.
- `tests/test_phase_surrogate_scores.py`: conditional inference and aggregation tests.
- `tests/test_validate_phase_surrogate.py`: exact gate tests and no-test-input CLI guard.
- `tests/test_run_phase_surrogate_pilot.py`: frozen population and stop-decision tests.
- `tests/test_build_phase_surrogate_submissions.py`: independent inference and root-only ZIP tests.
- `reports/phase_surrogate_s4_i16_p4_pilot_v1.md`: committed scientific decision after the real pilot.

---

### Task 1: Isolate the sprint

**Files:**
- Create worktree: `/home/neelshah/neurips_wl_worktrees/codex_phase_surrogate_20260802`

**Interfaces:**
- Consumes: source commit `9be777d`.
- Produces: clean branch `codex/phase-surrogate-sprint` for all following tasks.

- [ ] **Step 1: Verify the source and destination**

Run:

```bash
git -C /home/neelshah/neurips_wl_worktrees/codex_flow_mismatch_20260802 rev-parse HEAD
test ! -e /home/neelshah/neurips_wl_worktrees/codex_phase_surrogate_20260802
```

Expected: HEAD is `9be777d0b806ec8d0d034b23b745601e400486d2` and the destination check exits zero.

- [ ] **Step 2: Create the branch worktree**

Run from the version-controlled repository:

```bash
git worktree add -b codex/phase-surrogate-sprint /home/neelshah/neurips_wl_worktrees/codex_phase_surrogate_20260802 9be777d0b806ec8d0d034b23b745601e400486d2
```

- [ ] **Step 3: Verify isolation and baseline tests**

Run:

```bash
git -C /home/neelshah/neurips_wl_worktrees/codex_phase_surrogate_20260802 status --short
cd /home/neelshah/neurips_wl_worktrees/codex_phase_surrogate_20260802
.venv/bin/python -m pytest -q tests/test_wl_flow_mismatch.py tests/test_cache_flow_mismatch.py tests/test_flow_mismatch_scores.py tests/test_validate_flow_mismatch.py tests/test_run_flow_mismatch_pilot.py
```

Expected: clean status and `121 passed, 1 deselected` for the existing focused suite.

### Task 2: Deterministic matched-surrogate primitive

**Files:**
- Create: `scripts/wl_phase_surrogate.py`
- Create: `tests/test_wl_phase_surrogate.py`

**Interfaces:**
- Produces: `PhaseSurrogateConfig`, `PatchSpec`, `select_quarter_patches(mask)`, `fill_masked_patch(values, mask)`, `iaaft_surrogates(filled, sample_ids, namespaces, patch, config)`, and `surrogate_fidelity(target, surrogates)`.
- `iaaft_surrogates` returns shape `(batch, 4, 64, 64)` and is invariant to batch order and composition.

- [ ] **Step 1: Write failing patch, fill, determinism, and fidelity tests**

Create tests that include these assertions:

```python
CONFIG = PhaseSurrogateConfig(seed=20260802, surrogates=4, iterations=16)

def test_frozen_quarter_patch_contract(mask_path):
    mask = np.load(mask_path)
    patches = select_quarter_patches(mask)
    assert [(p.row, p.column) for p in patches] == [(128, 0), (640, 32), (736, 0), (1344, 0)]

def test_mask_fill_preserves_observed_values_and_is_finite():
    filled = fill_masked_patch(values, mask)
    torch.testing.assert_close(filled[mask], values[mask])
    assert torch.isfinite(filled).all()

def test_surrogates_are_independent_of_batch_order():
    together = iaaft_surrogates(batch, [7, 11], ["held", "held"], patch, CONFIG)
    reversed_rows = iaaft_surrogates(batch.flip(0), [11, 7], ["held", "held"], patch, CONFIG)
    torch.testing.assert_close(together, reversed_rows.flip(0), rtol=0, atol=0)

def test_iaaft_matches_histogram_and_power():
    controls = iaaft_surrogates(coupled_field[None], [3], ["synthetic"], patch, CONFIG)
    report = surrogate_fidelity(coupled_field[None], controls)
    assert report.sorted_value_max_error <= torch.finfo(torch.float32).eps * 8
    assert float(torch.median(report.radial_log_power_rms)) <= 0.01
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_wl_phase_surrogate.py -q
```

Expected: import failure for `wl_phase_surrogate`.

- [ ] **Step 3: Implement the primitive**

Use immutable contracts:

```python
@dataclass(frozen=True)
class PhaseSurrogateConfig:
    seed: int = 20260802
    patch_size: int = 64
    stride: int = 32
    minimum_coverage: float = 0.90
    surrogates: int = 4
    iterations: int = 16

@dataclass(frozen=True)
class PatchSpec:
    row: int
    column: int
    coverage: float

@dataclass(frozen=True)
class SurrogateFidelity:
    radial_log_power_rms: torch.Tensor
    sorted_value_max_error: float
```

Use BLAKE2 over `seed:namespace:sample_id:row:column:surrogate_id`; never seed from a batch index. Perform sixteen alternating Fourier-amplitude and sorted-value projections and retain the final sorted-value projection. Reject non-finite values, duplicate sample keys, wrong shapes, and changed frozen patch coordinates.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_wl_phase_surrogate.py -q
git add scripts/wl_phase_surrogate.py tests/test_wl_phase_surrogate.py
git commit -m "feat: add deterministic matched phase surrogates"
```

Expected: all focused tests pass.

### Task 3: Compact phase-contrast features

**Files:**
- Create: `scripts/wl_phase_contrast.py`
- Create: `tests/test_wl_phase_contrast.py`

**Interfaces:**
- Consumes: `PhaseSurrogateConfig` and the Task 2 primitive.
- Produces: `PhaseContrastBatch` and `PhaseContrastExtractor.extract(flat_maps, sample_ids, namespaces)`.
- Output arrays have shape `(maps, 4, features)` with stable names and view masks for `phase_core`, `phase_morphology`, and `controls`.

- [ ] **Step 1: Write failing synthetic feature tests**

Cover contracts and scientific direction:

```python
def test_feature_names_and_views_are_stable(extractor, gaussian_maps):
    result = extractor.extract(gaussian_maps, [1, 2], ["unit", "unit"])
    assert result.contrast.shape[:2] == (2, 4)
    assert result.target.shape == result.surrogate_median.shape == result.contrast.shape
    assert result.views["phase_core"].dtype == np.bool_
    assert result.views["phase_morphology"].sum() > result.views["phase_core"].sum()

def test_phase_coupling_changes_contrast_without_changing_power(extractor):
    gaussian, coupled = power_matched_pair(seed=41)
    score_g = np.mean(np.square(extractor.extract(gaussian[None], [1], ["g"]).contrast))
    score_c = np.mean(np.square(extractor.extract(coupled[None], [2], ["c"]).contrast))
    assert score_c > score_g
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run `.venv/bin/python -m pytest tests/test_wl_phase_contrast.py -q`.

Expected: import failure for `wl_phase_contrast`.

- [ ] **Step 3: Implement fixed feature groups**

Define:

```python
@dataclass(frozen=True)
class PhaseContrastBatch:
    contrast: np.ndarray
    target: np.ndarray
    surrogate_median: np.ndarray
    surrogate_mad: np.ndarray
    radial_log_power_rms: np.ndarray
    sorted_value_max_error: np.ndarray
    names: tuple[str, ...]
    views: dict[str, np.ndarray]
```

Reuse the six `BAND_EDGES` from `wl_high_order.py`. Extract fifteen phase-only squeezed bicoherences, twenty adjacent-band wavelet coupling terms, fixed peak/void densities, Minkowski perimeter/Euler terms, and two control moments. Compute `target - median(surrogates)` and `1.4826 * MAD`; do not divide by four-surrogate MAD.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_wl_phase_surrogate.py tests/test_wl_phase_contrast.py -q
git add scripts/wl_phase_contrast.py tests/test_wl_phase_contrast.py
git commit -m "feat: add matched phase contrast features"
```

### Task 4: Resumable integrity-checked cache

**Files:**
- Create: `scripts/cache_phase_surrogate.py`
- Create: `tests/test_cache_phase_surrogate.py`

**Interfaces:**
- Consumes: `PhaseContrastBatch`.
- Produces: `create_phase_cache(root, contract)`, `open_phase_cache(root, contract)`, `write_row_patch(cache, row, patch, payload)`, `audit_phase_cache(cache)`, and a CLI that scores explicit indices only.

- [ ] **Step 1: Write failing resume and corruption tests**

Test that a committed row-patch survives reopen, an uncommitted or hash-corrupt record resets to NaN, contract changes reject reuse, and retry computes only missing records.

```python
def test_corrupt_committed_record_is_reset(tmp_path):
    cache = create_phase_cache(tmp_path, contract(rows=2, patches=4, features=7))
    write_row_patch(cache, 0, 1, payload(features=7))
    cache.contrast[0, 1, 0] += 1.0
    reopened = open_phase_cache(tmp_path, cache.contract)
    assert np.isnan(reopened.contrast[0, 1]).all()
    assert reopened.integrity[0, 1] == 0
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run `.venv/bin/python -m pytest tests/test_cache_phase_surrogate.py -q`.

- [ ] **Step 3: Implement cache and CLI**

Store separate `.npy` memmaps for target, surrogate median, surrogate MAD, contrast, power RMS, sorted-value error, committed flags, and 64-bit integrity tags. Write payload, flush, hash contract plus bytes, write tag, flush, then set committed flag and flush. The CLI accepts source maps, mask, explicit source indices, namespace, output root, batch size, and device; it does not accept implicit test-wide fitting.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_cache_phase_surrogate.py tests/test_wl_phase_contrast.py -q
git add scripts/cache_phase_surrogate.py tests/test_cache_phase_surrogate.py
git commit -m "feat: cache resumable matched phase contrasts"
```

### Task 5: Conditional location, scale, whitening, and aggregation

**Files:**
- Create: `scripts/phase_surrogate_scores.py`
- Create: `tests/test_phase_surrogate_scores.py`

**Interfaces:**
- Consumes: contrast array `(rows, 4, features)`, lower-order anchors, and a Boolean view mask.
- Produces: `QuadraticLocationScale`, `fit_phase_fold`, `score_phase_fold`, and `aggregate_phase_patches`.

- [ ] **Step 1: Write failing conditional and invariant tests**

```python
def test_location_scale_removes_anchor_mean_and_variance():
    model = QuadraticLocationScale(alpha=10.0, floor=1e-4, sigma_bounds=(0.25, 4.0))
    model.fit(fit_anchor, fit_contrast)
    residual = model.studentize(held_anchor, held_contrast)
    assert abs(spearmanr(np.abs(residual[:, 0]), held_anchor[:, 0]).statistic) < 0.15

def test_scoring_one_row_matches_scoring_in_batch(fitted_fold):
    together = score_phase_fold(fitted_fold, anchors, contrasts)
    for row in range(len(anchors)):
        np.testing.assert_allclose(together[row], score_phase_fold(fitted_fold, anchors[row:row+1], contrasts[row:row+1])[0])
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run `.venv/bin/python -m pytest tests/test_phase_surrogate_scores.py -q`.

- [ ] **Step 3: Implement fixed conditional model**

Standardize fit anchors, use `PolynomialFeatures(degree=2, include_bias=False)`, fit multi-target `Ridge(alpha=10.0)`, fit log variance using `log(residual**2 + 1e-4)`, clamp sigma to `[0.25, 4.0]`, then fit `OAS` covariance on studentized fit residuals. Patch energy is the whitened quadratic form divided by view dimension. Tail-calibrate each patch using training scores only. Return coherent mean, localized max, and their independently calibrated equal-weight specialist.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_phase_surrogate_scores.py -q
git add scripts/phase_surrogate_scores.py tests/test_phase_surrogate_scores.py
git commit -m "feat: condition and aggregate phase surrogate scores"
```

### Task 6: Exact offline promotion gates

**Files:**
- Create: `scripts/validate_phase_surrogate.py`
- Create: `tests/test_validate_phase_surrogate.py`

**Interfaces:**
- Consumes: held primary/repeat caches, proxy cache, anchors, five audit labels, proxy families, and aligned incumbent held/proxy scores.
- Produces: JSON report with per-view fidelity, calibration, label correlation, cross-noise stability, family curves, runtime, specialist decision, and three candidate decisions.

- [ ] **Step 1: Write failing exact-boundary gate tests**

Test strict label `<0.15`, stability `>=0.60`, easy-family fraction `<=0.50`, runtime `<10800`, aggregate delta `>=0.003`, low delta `>=0.002`, positive folds `>=3`, and worst fold `>=-0.002`. Also test parser rejection of `--test`, `--test-maps`, and `--test-scores`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run `.venv/bin/python -m pytest tests/test_validate_phase_surrogate.py -q`.

- [ ] **Step 3: Implement report and gates**

Reuse `FPR_GRID`, `phase2_metric`, empirical high-tail calibration, and exact `binom.interval(0.95, n, target)` semantics from `validate_flow_mismatch.py`. Evaluate `phase_core` and `phase_morphology` independently. Compute only `standalone_phase`, `0.90*incumbent + 0.10*phase`, and the frozen 99th-percentile tail union.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_validate_phase_surrogate.py tests/test_phase_surrogate_scores.py -q
git add scripts/validate_phase_surrogate.py tests/test_validate_phase_surrogate.py
git commit -m "feat: validate phase surrogate promotion gates"
```

### Task 7: Frozen pilot runner and early-stop decision

**Files:**
- Create: `scripts/run_phase_surrogate_pilot.py`
- Create: `tests/test_run_phase_surrogate_pilot.py`

**Interfaces:**
- Consumes: train maps/labels, mask, existing proxy transformation code, existing generic anchor caches, aligned incumbent artifact, and output directory.
- Produces: resumable primary/repeat/proxy caches, runtime records, report JSON, score NPZ, and promote/reject status.

- [ ] **Step 1: Write failing protocol tests**

Assert exactly 384 held rows covering 101 cosmologies, two distinct noise seeds, 128 rows per 13 proxy families, exact four patches/four controls/sixteen iterations, no competition-test argument, and no packaging callback on rejection.

- [ ] **Step 2: Run the focused test and confirm RED**

Run `.venv/bin/python -m pytest tests/test_run_phase_surrogate_pilot.py -q`.

- [ ] **Step 3: Implement orchestration**

Reuse the held-row and proxy-family construction in `run_flow_mismatch_pilot.py` without importing competition test paths. Add `--microbenchmark` that limits execution to 32 held and 32 proxy rows, verifies batch-size replay, measures A100 memory/runtime, and projects the full pilot/runtime. The full mode builds all three caches then calls `validate_phase_surrogate.py`.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_run_phase_surrogate_pilot.py tests/test_validate_phase_surrogate.py -q
git add scripts/run_phase_surrogate_pilot.py tests/test_run_phase_surrogate_pilot.py
git commit -m "feat: add matched phase surrogate pilot runner"
```

### Task 8: Production replay and deterministic submission packaging

**Files:**
- Create: `scripts/build_phase_surrogate_submissions.py`
- Create: `tests/test_build_phase_surrogate_submissions.py`

**Interfaces:**
- Consumes: report JSON whose selected candidate is promoted, frozen fold artifacts, production contrast cache, exact incumbent artifact, and explicit output directory.
- Produces: `build_submissions(*, report_path: Path, fold_models_path: Path, production_cache: Path, incumbent_scores: Path, output_dir: Path) -> tuple[SubmissionArtifact, ...]`, eligible `scores.npz`, manifest JSON, and one root-only ZIP per promoted candidate; does not upload.

- [ ] **Step 1: Write failing guard, independence, and ZIP tests**

```python
def test_rejects_unpromoted_report(tmp_path):
    with pytest.raises(ValueError, match="not promoted"):
        build_submissions(
            report_path=write_report(tmp_path, promoted=False),
            fold_models_path=write_fold_models(tmp_path),
            production_cache=write_complete_cache(tmp_path, rows=10_000),
            incumbent_scores=write_incumbent_scores(tmp_path, rows=10_000),
            output_dir=tmp_path / "output",
        )

def test_zip_has_only_root_result_json(eligible_build):
    with zipfile.ZipFile(eligible_build.archive) as zf:
        assert zf.namelist() == ["result.json"]
        payload = json.loads(zf.read("result.json"))
    assert list(payload) == ["ood_scores"]
    assert len(payload["ood_scores"]) == 10_000
    assert np.isfinite(payload["ood_scores"]).all()
```

Verify bit-identical scores when rows are replayed alone, in different batch sizes, and in permuted batches restored to original order.

- [ ] **Step 2: Run the focused test and confirm RED**

Run `.venv/bin/python -m pytest tests/test_build_phase_surrogate_submissions.py -q`.

- [ ] **Step 3: Implement production builder**

Refuse reports without an eligible candidate, checksum mismatches, incomplete cache rows, fold replay mismatch, missing incumbent source, non-finite/wrong-count scores, and any output directory containing an unexpected pre-existing archive. Reuse `atomic_write_submission_zip` and archive audit utilities from `build_otfm_submissions.py`. Manifest score summaries are forbidden; record only count, finiteness, replay error, hashes, elapsed time, and configuration.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_build_phase_surrogate_submissions.py -q
git add scripts/build_phase_surrogate_submissions.py tests/test_build_phase_surrogate_submissions.py
git commit -m "feat: package promoted phase surrogate submissions"
```

### Task 9: Microbenchmark, offline pilot, and production decision

**Files:**
- Create after real run: `reports/phase_surrogate_s4_i16_p4_pilot_v1.md`
- Create untracked: `experiments/phase_surrogate_s4_i16_p4_pilot_v1.json`
- Create untracked: `experiments/phase_surrogate_s4_i16_p4_pilot_v1.scores.npz`
- Create untracked: `cache/phase_surrogate_s4_i16_p4_pilot_v1/`

**Interfaces:**
- Consumes: Tasks 2–8.
- Produces: verified reject decision or production-ready candidate artifacts.

- [ ] **Step 1: Run all new CPU tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_wl_phase_surrogate.py \
  tests/test_wl_phase_contrast.py \
  tests/test_cache_phase_surrogate.py \
  tests/test_phase_surrogate_scores.py \
  tests/test_validate_phase_surrogate.py \
  tests/test_run_phase_surrogate_pilot.py \
  tests/test_build_phase_surrogate_submissions.py
```

- [ ] **Step 2: Run the 64-row A100 microbenchmark**

Run the pilot CLI with `--microbenchmark`, the explicit live train/label/mask paths, output under the isolated worktree, and device `cuda`. Stop if power fidelity, batch replay, projected runtime, or memory fails.

- [ ] **Step 3: Run the frozen offline pilot**

Run the same CLI without `--microbenchmark`. Do not provide or access the Phase-2 competition test file. Record hashes, row counts, integrity counts, runtime, fidelity, gates, and decision.

- [ ] **Step 4: Commit the scientific decision**

Write the Markdown report from the JSON evidence, run `git diff --check`, and commit:

```bash
git add reports/phase_surrogate_s4_i16_p4_pilot_v1.md
git commit -m "exp: record matched phase surrogate pilot"
```

- [ ] **Step 5: If and only if promoted, build the production cache**

Score 10,000 test rows independently into a contract-keyed resumable cache. Do not compute or print test-score distribution summaries. Abort if measured production GPU time reaches three hours.

- [ ] **Step 6: If and only if promoted, package eligible candidates**

Run `build_phase_surrogate_submissions.py`, then independently audit each ZIP member list, `result.json` schema, count, finiteness, source order, score replay, and SHA-256. Do not upload.

- [ ] **Step 7: Run final verification**

Run the complete new suite and the existing focused Flow Mismatch suite. Then run the full repository suite and report the known inherited failure separately if it remains byte-identical to baseline.

- [ ] **Step 8: Present artifacts for user approval**

Report offline gates, exact candidate formula, ZIP path, SHA-256, score count, runtime, and residual risk. Stop and wait for explicit user approval before any leaderboard upload.
