# Flow-Mismatch First Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, per-map Flow Mismatching detector around the frozen weak-lensing OTFM, validate it on held InD and physics proxies, and package only candidates that pass low-FPR and deconfounding gates.

**Architecture:** Reuse `MaskAwareOTFMUNet`, `FixedPatchLayout`, checkpoint scaling, and frozen validation splits. Compute velocity disagreement along deterministic affine noise-to-data paths using forward passes, cache patch signatures, remove five-label response directions with train-only models, then derive coherent, localized, and calibrated-union scores.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy, scikit-learn, pytest, existing FAIR Universe submission helpers.

## Global Constraints

- Work in `/home/neelshah/neurips_wl` on the A100 VM.
- Do not modify or overwrite existing experiments, caches, checkpoints, or submissions.
- No uninterrupted GPU command may exceed three hours; all production caches must resume safely.
- Use only training rows and frozen held-InD rows for fit/calibration.
- Score each competition test map independently; never use test-batch ranks or distributions.
- Use `K=4` paths and times `(0.2, 0.4, 0.6, 0.8)` for the first pilot.
- Do not retrain the OTFM during this sprint.
- Use `apply_patch` for source edits.
- Run focused tests before broad tests.
- The project root is not currently a Git repository. Before implementation, create a code-only version-controlled working copy or obtain explicit approval to initialize version control in place; do not initialize Git silently.

---

### Task 1: Pure Flow Mismatching primitives

**Files:**
- Create: `scripts/wl_flow_mismatch.py`
- Create: `tests/test_wl_flow_mismatch.py`

**Interfaces:**
- Consumes: a frozen model with `forward(x: Tensor, time: Tensor, conditioning: Tensor) -> Tensor`.
- Produces: `FlowMismatchConfig`, `deterministic_path_noise`, `velocity_mismatch_heatmap`, and `reduce_mismatch_heatmap`.

- [ ] **Step 1: Write deterministic-noise and exact-mismatch tests**

```python
from __future__ import annotations

import torch

from scripts.wl_flow_mismatch import (
    FlowMismatchConfig,
    deterministic_path_noise,
    reduce_mismatch_heatmap,
    velocity_mismatch_heatmap,
)


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, time, conditioning):
        return torch.zeros_like(x)


def test_deterministic_path_noise_replays_and_separates_paths() -> None:
    mask = torch.ones(2, 1, 4, 4)
    first = deterministic_path_noise([3, 7], 5, 0, mask, seed=19)
    second = deterministic_path_noise([3, 7], 5, 0, mask, seed=19)
    other = deterministic_path_noise([3, 7], 5, 1, mask, seed=19)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, other)


def test_zero_velocity_has_exact_weighted_geometric_mismatch() -> None:
    target = torch.ones(1, 1, 4, 4)
    mask = torch.ones_like(target)
    conditioning = torch.ones(1, 3, 4, 4)
    base = torch.zeros(1, 1, 4, 4)
    config = FlowMismatchConfig(times=(0.25, 0.5), paths=1)
    heatmap = velocity_mismatch_heatmap(
        ZeroVelocity(), target, conditioning, [11], 2, mask, config,
        probe_seed=23, supplied_noise=base[:, None],
    )
    torch.testing.assert_close(heatmap, torch.full_like(heatmap, 0.15625))


def test_reduction_ignores_masked_pixels() -> None:
    heatmap = torch.tensor([[[1.0, 2.0], [1000.0, 4.0]]])
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    reduced = reduce_mismatch_heatmap(heatmap, mask, top_fraction=0.34)
    torch.testing.assert_close(reduced["mean"], torch.tensor([7.0 / 3.0]))
    torch.testing.assert_close(reduced["top"], torch.tensor([4.0]))
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cd /home/neelshah/neurips_wl
.venv/bin/pytest -q tests/test_wl_flow_mismatch.py
```

Expected: collection fails because `scripts.wl_flow_mismatch` does not exist.

- [ ] **Step 3: Implement the pure scoring module**

Implement these exact public contracts:

```python
@dataclass(frozen=True)
class FlowMismatchConfig:
    times: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    paths: int = 4
    path_quantile: float = 0.25
    time_power: float = 2.0


def deterministic_path_noise(
    sample_ids: Sequence[int],
    patch_id: int,
    path_id: int,
    mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Return sample/patch/path deterministic N(0,1) noise masked to data support."""


@torch.inference_mode()
def velocity_mismatch_heatmap(
    model: torch.nn.Module,
    target: torch.Tensor,
    conditioning: torch.Tensor,
    sample_ids: Sequence[int],
    patch_id: int,
    mask: torch.Tensor,
    config: FlowMismatchConfig,
    *,
    probe_seed: int,
    supplied_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `(batch, height, width)` weighted path-quantile velocity mismatch."""


def reduce_mismatch_heatmap(
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_fraction: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return masked `mean`, `top`, `max`, and `max_plus_top` map scores."""
```

Validate constructor values, reject endpoint times, generate noise independently per sample ID, evaluate all paths/times in vectorized batches, take channel-mean squared velocity error, aggregate paths with `torch.quantile`, and average `t ** time_power` discrepancies across times.

- [ ] **Step 4: Run primitive tests**

Run:

```bash
.venv/bin/pytest -q tests/test_wl_flow_mismatch.py
```

Expected: all tests pass.

- [ ] **Step 5: Record the change**

When the isolated working copy has Git:

```bash
git add scripts/wl_flow_mismatch.py tests/test_wl_flow_mismatch.py
git commit -m "feat: add deterministic flow mismatch scoring"
```

### Task 2: Frozen-checkpoint patch scorer and resumable cache

**Files:**
- Create: `scripts/cache_flow_mismatch.py`
- Create: `tests/test_cache_flow_mismatch.py`
- Reuse: `scripts/validate_otfm.py`
- Reuse: `scripts/build_otfm_submissions.py`

**Interfaces:**
- Consumes: `load_frozen_model`, `flatten_challenge_maps`, checkpoint `validation_indices`, `score_patch_ids`, and Task 1 primitives.
- Produces: a float32 cache shaped `(n_maps, n_patches, 4)` with channels `mean`, `top`, `max`, `max_plus_top`, plus JSON contract and integrity tags.

- [ ] **Step 1: Write cache-contract tests**

Test that:

1. A newly created cache is NaN-filled.
2. Reopening with the identical contract preserves completed rows.
3. A changed checkpoint hash, time grid, path count, or patch list raises `ValueError`.
4. A corrupt completed row is reset to NaN when its integrity tag does not match.
5. Scoring a sample alone and inside a batch produces identical cached values.

Use temporary directories, an eight-pixel toy mask, and a zero-velocity model; do not load the production checkpoint in unit tests.

- [ ] **Step 2: Run cache tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_cache_flow_mismatch.py
```

Expected: collection fails because `scripts.cache_flow_mismatch` does not exist.

- [ ] **Step 3: Implement CLI and cache contract**

The CLI must accept:

```text
--checkpoint PATH --maps PATH --mask PATH --cache-dir PATH
--split {heldout,proxy,test} --patch-limit INT --batch-size INT
--times 0.2 0.4 0.6 0.8 --paths 4 --path-quantile 0.25
--probe-seed INT --shape-noise-seed INT --device cuda
```

For `heldout`, verify the source against the checkpoint data contract and use only `validation_indices`. For `test`, require exactly 10,000 maps and do not inspect aggregate test statistics. For every patch batch:

- Extract via `FixedPatchLayout`.
- Add deterministic shape noise only when the source is noiseless training data.
- Apply checkpoint `data_scale` and mask.
- Call `velocity_mismatch_heatmap`.
- Store all four reductions.
- Flush the memmap and integrity tag before reporting progress.

- [ ] **Step 4: Run cache tests**

```bash
.venv/bin/pytest -q tests/test_cache_flow_mismatch.py tests/test_wl_flow_mismatch.py
```

Expected: all tests pass.

- [ ] **Step 5: Run a CPU smoke cache**

Use the test fixture command produced by `tests/test_cache_flow_mismatch.py`; expected output is a complete two-map, two-patch cache with no NaNs and a valid contract digest.

- [ ] **Step 6: Record the change**

```bash
git add scripts/cache_flow_mismatch.py tests/test_cache_flow_mismatch.py
git commit -m "feat: cache resumable flow mismatch patch scores"
```

### Task 3: Five-label residualization and map aggregations

**Files:**
- Create: `scripts/flow_mismatch_scores.py`
- Create: `tests/test_flow_mismatch_scores.py`

**Interfaces:**
- Consumes: patch cache `(n_maps, n_patches, 4)`, five training labels, and train-fitted label estimates.
- Produces: `coherent_mean`, `coherent_trimmed`, `localized_topk`, `heatmap_tail`, and `dual_union` raw/calibrated arrays.

- [ ] **Step 1: Write aggregation and deconfounding tests**

Create synthetic data in which a score signature equals a known quadratic function of five labels plus Gaussian noise. Assert that:

- Cross-fitted residualization reduces every absolute label correlation below `0.05`.
- A sixth injected direction remains separated after residualization.
- `coherent_trimmed` discards the correct 10% tails.
- `localized_topk` uses the largest `ceil(0.10 * n_patches)` patches.
- Empirical tail calibration depends only on the supplied InD reference.
- Permuting unrelated test rows leaves every per-row score unchanged.

- [ ] **Step 2: Run score tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_flow_mismatch_scores.py
```

Expected: collection fails because `scripts.flow_mismatch_scores` does not exist.

- [ ] **Step 3: Implement exact score interfaces**

```python
def aggregate_patch_signatures(
    patch_scores: np.ndarray,
    *,
    localized_fraction: float = 0.10,
    trim_fraction: float = 0.10,
) -> dict[str, np.ndarray]:
    """Map `(rows, patches, channels)` to named one-dimensional raw scores."""


class FiveLabelResponseResidualizer:
    def fit(self, signatures: np.ndarray, labels: np.ndarray) -> "FiveLabelResponseResidualizer": ...
    def residual(self, signatures: np.ndarray, estimated_labels: np.ndarray) -> np.ndarray: ...


def empirical_union_score(
    members: Mapping[str, np.ndarray],
    calibration_members: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Calibrate members on InD, take maximum surprisal, and recalibrate the union."""
```

Use `PolynomialFeatures(degree=2, include_bias=True)`, standardized labels, ridge regression, and OAS covariance. Fit every transformer on training/calibration rows only.

- [ ] **Step 4: Run score tests**

```bash
.venv/bin/pytest -q tests/test_flow_mismatch_scores.py tests/test_ood_metric.py
```

Expected: all tests pass.

- [ ] **Step 5: Record the change**

```bash
git add scripts/flow_mismatch_scores.py tests/test_flow_mismatch_scores.py
git commit -m "feat: add deconfounded flow mismatch map scores"
```

### Task 4: Held-InD and proxy validation report

**Files:**
- Create: `scripts/validate_flow_mismatch.py`
- Create: `tests/test_validate_flow_mismatch.py`
- Reuse: `scripts/ood_metric.py`
- Reuse: `scripts/validate_feature_detectors.py`

**Interfaces:**
- Consumes: held-InD cache, proxy cache, five labels, existing incumbent OOF scores.
- Produces: `experiments/flow_mismatch_k4_t4_pilot_v1.json` and `.scores.npz`.

- [ ] **Step 1: Write validation-gate tests**

Use synthetic four-fold reports to verify:

- A candidate with calibration failure is rejected.
- A candidate with maximum label correlation `0.151` is rejected.
- A candidate with cross-noise correlation `0.59` is rejected.
- A blend with deltas `(0.004, 0.004, 0.004, -0.003)` is rejected by worst-fold degradation.
- A blend with all-delta `0.0035`, low-delta `0.0025`, three positive folds, and worst `-0.001` is promotable.
- A candidate whose improvement is more than 50% noise/resolution-driven is rejected.

- [ ] **Step 2: Run validation tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_validate_flow_mismatch.py
```

Expected: collection fails because `scripts.validate_flow_mismatch` does not exist.

- [ ] **Step 3: Implement validation CLI**

The report must contain:

- Complete configuration and source hashes.
- Runtime and patch coverage.
- Fold-level official metric components.
- Per-family proxy scores.
- Held-InD empirical FPR at `0.001, 0.003, 0.01, 0.03, 0.05`.
- Cross-noise Spearman correlation.
- Five known-label Spearman correlations.
- Correlation and top-tail overlap with the incumbent.
- Standalone and blend promotion decisions with explicit failed gates.

No test-map input is accepted by this validation script.

- [ ] **Step 4: Run validation tests**

```bash
.venv/bin/pytest -q tests/test_validate_flow_mismatch.py tests/test_flow_mismatch_scores.py tests/test_ood_metric.py
```

Expected: all tests pass.

- [ ] **Step 5: Record the change**

```bash
git add scripts/validate_flow_mismatch.py tests/test_validate_flow_mismatch.py
git commit -m "feat: validate flow mismatch promotion gates"
```

### Task 5: Short A100 pilot and decision checkpoint

**Files:**
- Create: `reports/flow_mismatch_k4_t4_pilot_v1.md`
- Generate: `experiments/flow_mismatch_k4_t4_pilot_v1.json`
- Generate: `experiments/flow_mismatch_k4_t4_pilot_v1.scores.npz`

**Interfaces:**
- Consumes: Tasks 1–4 and the frozen production OTFM checkpoint.
- Produces: a written promote/stop decision; no competition submission yet.

- [ ] **Step 1: Run the full focused test set**

```bash
.venv/bin/pytest -q \
  tests/test_wl_flow_mismatch.py \
  tests/test_cache_flow_mismatch.py \
  tests/test_flow_mismatch_scores.py \
  tests/test_validate_flow_mismatch.py \
  tests/test_wl_otfm.py \
  tests/test_validate_otfm.py \
  tests/test_ood_metric.py
```

Expected: all tests pass.

- [ ] **Step 2: Cache a balanced four-patch held-InD pilot**

Run with `K=4`, `T=4`, four highest-coverage checkpoint patches, and enough held rows to cover all 101 cosmologies. Record wall time and GPU memory. Stop if extrapolated 38-patch scoring exceeds the three-hour checkpoint policy without safe resume.

- [ ] **Step 3: Cache the matching four-patch proxy pilot**

Use the identical checkpoint, patches, scaling, paths, times, and seeds. Do not fit on proxy labels or proxy families.

- [ ] **Step 4: Generate the validation report**

Run `scripts/validate_flow_mismatch.py` and write the exact report values into `reports/flow_mismatch_k4_t4_pilot_v1.md`, including every failed gate.

- [ ] **Step 5: Apply the checkpoint decision**

- If no standalone candidate passes, stop this track without test scoring.
- If a standalone candidate passes but no blend passes, authorize full held/proxy patch coverage only.
- If a blend passes, authorize full held/proxy coverage and test-cache preparation.

- [ ] **Step 6: Record the pilot evidence**

```bash
git add reports/flow_mismatch_k4_t4_pilot_v1.md
git commit -m "exp: record flow mismatch pilot decision"
```

### Task 6: Production scoring, packaging, and ledger

**Files:**
- Create: `scripts/package_flow_mismatch_submissions.py`
- Create: `tests/test_package_flow_mismatch_submissions.py`
- Create or append: `reports/submission_ledger.jsonl`
- Generate: `submissions/flow_mismatch_k4_t4_v1/`

**Interfaces:**
- Consumes: a complete, integrity-checked test patch cache and the frozen validation artifact.
- Produces: at most four validated ZIP files and append-only ledger entries.

- [ ] **Step 1: Write packaging tests**

Verify that packaging rejects:

- Any missing or corrupt cache cell.
- A checkpoint/config hash mismatch.
- A candidate absent from the promotable list.
- Non-finite or non-10,000-length scores.
- A submission whose isolated-map audit differs from its batched score.
- An attempt to overwrite an existing ledger record or ZIP.

- [ ] **Step 2: Run packaging tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_package_flow_mismatch_submissions.py
```

Expected: collection fails because the packaging module does not exist.

- [ ] **Step 3: Implement immutable packaging**

Package only these validation-approved names:

```text
flow_mismatch_coherent_k4t4_v1.zip
flow_mismatch_localized_k4t4_v1.zip
flow_mismatch_union_k4t4_v1.zip
flow_mismatch_incumbent_fusion_k4t4_v1.zip
```

Each ledger line must contain the hypothesis, source hashes, score hash, ZIP hash, validation artifact, exact candidate name, timestamp, and a null Codabench result field. A separate append operation may add the Codabench submission ID and score without rewriting prior lines.

- [ ] **Step 4: Run packaging tests**

```bash
.venv/bin/pytest -q tests/test_package_flow_mismatch_submissions.py tests/test_build_otfm_submissions.py
```

Expected: all tests pass.

- [ ] **Step 5: Fill the resumable test cache**

Run in checkpointed sessions of at most three hours. After each session, verify cache progress and integrity tags. Packaging must not begin until progress is 100%.

- [ ] **Step 6: Package approved candidates**

Run the packaging CLI, then validate every ZIP using `generate_smoke_submission.validate_archive`. Compare isolated and batched scores for fixed test map IDs before upload.

- [ ] **Step 7: Record the production artifacts**

```bash
git add scripts/package_flow_mismatch_submissions.py \
  tests/test_package_flow_mismatch_submissions.py \
  reports/submission_ledger.jsonl
git commit -m "feat: package audited flow mismatch submissions"
```

### Task 7: Final verification before any upload

**Files:**
- Verify: all Task 1–6 files and artifacts.

**Interfaces:**
- Consumes: complete sprint artifacts.
- Produces: an upload recommendation listing exact ZIP hashes, or a stop decision.

- [ ] **Step 1: Run the complete project test suite**

```bash
.venv/bin/pytest -q tests
```

Expected: all tests pass.

- [ ] **Step 2: Verify validation gates from immutable artifacts**

Recompute the promotion decision from the saved `.npz` and compare it byte-for-byte with the JSON decision fields. Any mismatch blocks upload.

- [ ] **Step 3: Verify runtime and eligibility constraints**

Confirm that test scoring used only single-map inputs, no aggregate test calibration, the frozen checkpoint, and the exact validation-approved configuration.

- [ ] **Step 4: Present candidates for user approval**

Report each candidate's hypothesis, local metric deltas, low-FPR delta, fold consistency, correlations, SHA-256, and remaining daily/total submission budget. Do not upload automatically.
