# Coarse-Conditioned Field-Level OTFM Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and validate a coarse-conditioned field-level OTFM, then produce one standalone and two frozen-anchor submission archives without using any test-batch statistic.

**Architecture:** Reuse the repository's existing Haar conditional-flow model, training script, exact-likelihood integrator, and checkpoint contracts. Add a focused score-calibration module, extend the held-only validator to enforce the approved nuisance/oracle/tail gates, and add a resumable production scorer/packager for the deployable coarse branch. Expensive 10,000-map scoring starts only after the four-patch model passes the offline gate.

**Tech Stack:** Python 3, NumPy, SciPy, PyTorch, pytest, deterministic NPY memmaps, JSON manifests, ZIP submissions, NVIDIA A100 40 GB.

## Global Constraints

- Work only in the new VM worktree `/home/neelshah/neurips_wl_worktrees/codex_coarse_conditional_otfm_20260802` on branch `codex/coarse-conditional-otfm`. Keep `/home/neelshah/neurips_wl` read-only except for reading data and frozen artifacts.
- Store checkpoints and caches under `/home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802`.
- Train only on official InD maps. Oracle labels are diagnostic and never enter deployable test scoring.
- Use four fixed `64 x 64` patches, base width `32`, `20` epochs, at most `400` steps per epoch, and seed `20260901` for stage 0.
- Every test map must score identically alone or in any operational batch. No test means, variances, quantiles, ranks, histograms, mixture fits, or weight fitting.
- Fit all typicality and empirical-surprisal transformations from held InD rows only.
- Keep individual jobs below three expected hours. Stop the first cycle at five cumulative hours for diagnosis.
- Do not retrain, fine-tune, imitate, or use the `0.562092` anchor as a model input. Use its frozen held/test score vectors only for 70/30 and 90/10 fusions.
- Freeze submission weights before a new public result. Do not create a leaderboard-driven weight sweep.
- Do not upload any archive until the user sees validation evidence and SHA-256 hashes and explicitly approves upload.
- Update the private experiment ledger and `docs/AI_USAGE.md`.

## File Structure

- Create `scripts/conditional_otfm_scores.py`: held-only patch typicality, coherent/sparse map views, empirical-surprisal calibration, and fixed score fusion.
- Create `tests/test_conditional_otfm_scores.py`: unit coverage for those primitives.
- Modify `scripts/validate_conditional_otfm.py`: second-noise cache, known-label leakage, FPR calibration, four-fold metrics, oracle recovery, easy-family dominance, and anchor complementarity.
- Modify `tests/test_validate_conditional_otfm.py`: tests for the new diagnostics and gates.
- Create `scripts/build_conditional_otfm_submissions.py`: resumable held/test likelihood cache plus pure/70-30/90-10 packaging.
- Create `tests/test_build_conditional_otfm_submissions.py`: cache-contract, independence, blend, and ZIP tests.
- Create `reports/conditional_field_otfm_stage0_v1.md`: append-only observed run ledger.
- Modify `docs/AI_USAGE.md`: disclose the AI-assisted research and implementation work.

---

### Task 1: Create the isolated worktree and verify the inherited baseline

**Files:**
- Verify: `scripts/wl_conditional_otfm.py`
- Verify: `scripts/train_conditional_otfm.py`
- Verify: `scripts/validate_conditional_otfm.py`
- Verify: `scripts/run_conditional_field_stage0.py`
- Verify: `tests/test_wl_conditional_otfm.py`
- Verify: `tests/test_validate_conditional_otfm.py`

**Interfaces:**
- Consumes: VM Git branch `main` at `d06132c`.
- Produces: clean `codex/coarse-conditional-otfm` worktree and passing inherited tests.

- [ ] **Step 1: Invoke the worktree setup skill**

Use `superpowers:using-git-worktrees`. Verify the target path and branch do not contain unrelated work.

- [ ] **Step 2: Create the worktree**

```bash
git worktree add -b codex/coarse-conditional-otfm \
  /home/neelshah/neurips_wl_worktrees/codex_coarse_conditional_otfm_20260802 \
  main
```

Expected: new branch at `d06132c`.

- [ ] **Step 3: Verify the branch and sources**

```bash
cd /home/neelshah/neurips_wl_worktrees/codex_coarse_conditional_otfm_20260802
git status --short --branch
test -r /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy
test -r /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy
test -r /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy
test -r /home/neelshah/neurips_wl/data/label_newrealization.npy
```

Expected: clean status and four successful read checks.

- [ ] **Step 4: Run inherited focused tests**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_wl_conditional_otfm.py \
  tests/test_validate_conditional_otfm.py -q
```

Expected: all pass. Stop on a baseline failure.

- [ ] **Step 5: Record source fingerprints**

```bash
git rev-parse HEAD
sha256sum \
  /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy \
  /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  /home/neelshah/neurips_wl/data/label_newrealization.npy
```

Expected: one commit ID and four hashes captured. This task changes no tracked file.

---

### Task 2: Implement held-only score calibration

**Files:**
- Create: `scripts/conditional_otfm_scores.py`
- Create: `tests/test_conditional_otfm_scores.py`

**Interfaces:**
- Consumes: finite patch NLL matrices shaped `(n_maps, n_patches)`.
- Produces: `PatchTypicalityCalibration.fit`, `evidence`, `map_views`, `EmpiricalSurprisalCalibration.fit`, `score`, and `fixed_anchor_blends`.

- [ ] **Step 1: Write failing typicality tests**

Create `tests/test_conditional_otfm_scores.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from scripts.conditional_otfm_scores import (
    EmpiricalSurprisalCalibration,
    PatchTypicalityCalibration,
    fixed_anchor_blends,
    map_views,
)


def test_patch_typicality_is_two_sided_and_patch_specific() -> None:
    held = np.asarray([
        [9.0, 98.0],
        [10.0, 99.0],
        [11.0, 101.0],
        [12.0, 102.0],
    ])
    calibration = PatchTypicalityCalibration.fit(held)
    evidence = calibration.evidence(
        np.asarray([[10.5, 100.0], [2.0, 100.0], [20.0, 100.0]])
    )
    assert evidence.shape == (3, 2)
    assert evidence[0, 0] < evidence[1, 0] < evidence[2, 0]
    assert evidence[0, 1] == pytest.approx(0.0)


def test_patch_typicality_rejects_bad_inputs() -> None:
    calibration = PatchTypicalityCalibration.fit(np.arange(12.0).reshape(6, 2))
    with pytest.raises(ValueError, match="patch count"):
        calibration.evidence(np.ones((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        calibration.evidence(np.asarray([[1.0, np.nan]]))
```

- [ ] **Step 2: Run the tests to verify failure**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_conditional_otfm_scores.py -q
```

Expected: `ModuleNotFoundError` for `scripts.conditional_otfm_scores`.

- [ ] **Step 3: Implement per-patch typicality**

Create `scripts/conditional_otfm_scores.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return result


@dataclass(frozen=True)
class PatchTypicalityCalibration:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    count: int

    @classmethod
    def fit(cls, held_patch_nll: np.ndarray) -> "PatchTypicalityCalibration":
        values = _finite_matrix(held_patch_nll, "held patch NLL")
        if len(values) < 2:
            raise ValueError("held patch NLL requires at least two maps")
        std = values.std(axis=0, ddof=1)
        if np.any(std <= 0.0):
            raise ValueError("held patch NLL contains a zero-variance patch")
        return cls(
            tuple(values.mean(axis=0).tolist()),
            tuple(std.tolist()),
            len(values),
        )

    def evidence(self, patch_nll: np.ndarray) -> np.ndarray:
        values = _finite_matrix(patch_nll, "patch NLL")
        if values.shape[1] != len(self.mean):
            raise ValueError("patch count differs from calibration")
        return np.abs(
            (values - np.asarray(self.mean)) / np.asarray(self.std)
        )

    def to_json(self) -> dict[str, object]:
        return asdict(self)
```

- [ ] **Step 4: Run the typicality tests**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_conditional_otfm_scores.py -q
```

Expected: the two tests pass.

- [ ] **Step 5: Add failing map-view, tail-calibration, and blend tests**

Append:

```python
def test_map_views_separate_distributed_and_localized_evidence() -> None:
    evidence = np.asarray([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 4.0]])
    views = map_views(evidence)
    np.testing.assert_allclose(views["coherent"], [1.0, 0.0])
    np.testing.assert_allclose(views["sparse"], [1.0, 4.0])


def test_empirical_surprisal_uses_frozen_held_scores_only() -> None:
    calibration = EmpiricalSurprisalCalibration.fit(
        np.asarray([0.0, 1.0, 2.0, 3.0])
    )
    first = calibration.score(np.asarray([0.5, 2.5, 9.0]))
    second = calibration.score(np.asarray([0.5, 2.5, 9.0]))
    np.testing.assert_allclose(first, second)
    assert first[0] < first[1] < first[2]
    extrapolated = calibration.score(np.asarray([9.0, 10.0]))
    assert extrapolated[0] < extrapolated[1]


def test_fixed_anchor_blends_use_only_preregistered_weights() -> None:
    blends = fixed_anchor_blends(
        np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0])
    )
    np.testing.assert_allclose(blends["anchor70_coarse30"], [0.30, 0.70])
    np.testing.assert_allclose(blends["anchor90_coarse10"], [0.10, 0.90])
    assert set(blends) == {"anchor70_coarse30", "anchor90_coarse10"}
```

- [ ] **Step 6: Run to verify the new interfaces fail**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_conditional_otfm_scores.py -q
```

Expected: failures name `map_views`, `EmpiricalSurprisalCalibration`, and `fixed_anchor_blends`.

- [ ] **Step 7: Implement the three interfaces**

Append:

```python
def map_views(evidence: np.ndarray) -> dict[str, np.ndarray]:
    values = _finite_matrix(evidence, "patch evidence")
    if values.shape[1] < 1:
        raise ValueError("patch evidence requires at least one patch")
    return {
        "coherent": np.median(values, axis=1),
        "sparse": values.max(axis=1),
    }


@dataclass(frozen=True)
class EmpiricalSurprisalCalibration:
    sorted_ind: tuple[float, ...]

    @classmethod
    def fit(cls, held_scores: np.ndarray) -> "EmpiricalSurprisalCalibration":
        values = np.asarray(held_scores, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("held scores must be a finite one-dimensional array")
        return cls(tuple(np.sort(values).tolist()))

    def score(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("scores must be a finite one-dimensional array")
        held = np.asarray(self.sorted_ind)
        unique, first, counts = np.unique(
            held, return_index=True, return_counts=True
        )
        if len(unique) < 2:
            raise ValueError("held scores require at least two unique values")
        midrank = first + 0.5 * (counts - 1)
        rank = np.interp(values, unique, midrank)
        probability = (len(held) - rank) / (len(held) + 1.0)
        positive_spacing = np.diff(unique)
        scale = float(np.median(positive_spacing[positive_spacing > 0.0]))
        below = values < unique[0]
        if np.any(below):
            p0 = (len(held) - midrank[0]) / (len(held) + 1.0)
            probability[below] = 1.0 - (1.0 - p0) * np.exp(
                (values[below] - unique[0]) / scale
            )
        above = values > unique[-1]
        if np.any(above):
            p1 = (len(held) - midrank[-1]) / (len(held) + 1.0)
            probability[above] = p1 * np.exp(
                -(values[above] - unique[-1]) / scale
            )
        return -np.log10(
            np.clip(probability, np.finfo(np.float64).tiny, 1.0)
        )

    def to_json(self) -> dict[str, object]:
        return {"sorted_ind": list(self.sorted_ind), "count": len(self.sorted_ind)}


def fixed_anchor_blends(
    anchor: np.ndarray, field: np.ndarray
) -> dict[str, np.ndarray]:
    anchor_values = np.asarray(anchor, dtype=np.float64)
    field_values = np.asarray(field, dtype=np.float64)
    if anchor_values.shape != field_values.shape or anchor_values.ndim != 1:
        raise ValueError("anchor and field scores must be aligned vectors")
    if not np.isfinite(anchor_values).all() or not np.isfinite(field_values).all():
        raise ValueError("anchor and field scores must be finite")
    return {
        "anchor70_coarse30": 0.70 * anchor_values + 0.30 * field_values,
        "anchor90_coarse10": 0.90 * anchor_values + 0.10 * field_values,
    }
```

- [ ] **Step 8: Run the complete score test file**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_conditional_otfm_scores.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit the score-calibration unit**

```bash
git add scripts/conditional_otfm_scores.py tests/test_conditional_otfm_scores.py
git commit -m "feat: calibrate conditional OTFM map scores"
```

---

### Task 3: Extend validation with nuisance, oracle, and tail gates

**Files:**
- Modify: `scripts/validate_conditional_otfm.py`
- Modify: `tests/test_validate_conditional_otfm.py`

**Interfaces:**
- Consumes: three aligned branch caches, two held-noise views, five labels, physical proxy families, and frozen anchor audit scores.
- Produces: `empirical_fpr_report`, `known_label_correlations`, `promotion_decision`, four-fold metrics, cross-noise stability, and anchor complementarity.

- [ ] **Step 1: Write failing diagnostic tests**

Append to `tests/test_validate_conditional_otfm.py`:

```python
from scripts.validate_conditional_otfm import (
    empirical_fpr_report,
    known_label_correlations,
    promotion_decision,
)


def test_empirical_fpr_uses_frozen_calibration_thresholds() -> None:
    calibration = np.linspace(0.0, 100.0, 10_001)
    evaluation = np.linspace(0.0, 100.0, 10_001)
    report = empirical_fpr_report(calibration, evaluation)
    assert set(report) == {"0.001", "0.003", "0.01", "0.03", "0.05"}
    assert report["0.01"]["observed"] == pytest.approx(0.01, abs=2e-4)


def test_known_label_correlations_report_five_columns() -> None:
    score = np.arange(20.0)
    labels = np.column_stack([score, -score, np.sin(score), np.ones(20), score % 3])
    report = known_label_correlations(score, labels)
    assert set(report) == {"Omega_m", "S8", "T_AGN", "f0", "delta_z"}
    assert report["Omega_m"] == pytest.approx(1.0)
    assert report["S8"] == pytest.approx(-1.0)
    assert report["f0"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_validate_conditional_otfm.py -q
```

Expected: import failures name the new helpers.

- [ ] **Step 3: Implement FPR and label diagnostics**

Add:

```python
from scipy.stats import binom, spearmanr

TARGET_FPRS = (0.001, 0.003, 0.01, 0.03, 0.05)
LABEL_NAMES = ("Omega_m", "S8", "T_AGN", "f0", "delta_z")


def empirical_fpr_report(
    calibration_scores: np.ndarray, evaluation_scores: np.ndarray
) -> dict[str, dict[str, float]]:
    calibration = np.asarray(calibration_scores, dtype=np.float64)
    evaluation = np.asarray(evaluation_scores, dtype=np.float64)
    if calibration.ndim != 1 or evaluation.ndim != 1:
        raise ValueError("FPR inputs must be vectors")
    report = {}
    for target in TARGET_FPRS:
        threshold = float(np.quantile(calibration, 1.0 - target, method="higher"))
        observed = float(np.mean(evaluation >= threshold))
        lower_count, upper_count = binom.interval(0.95, len(evaluation), target)
        report[str(target)] = {
            "target": target,
            "threshold": threshold,
            "observed": observed,
            "lower_95": float(lower_count / len(evaluation)),
            "upper_95": float(upper_count / len(evaluation)),
        }
    return report


def known_label_correlations(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if targets.shape != (len(values), 5):
        raise ValueError("known labels must align with scores")
    result = {}
    for column, name in enumerate(LABEL_NAMES):
        if np.std(targets[:, column]) == 0.0:
            result[name] = 0.0
        else:
            result[name] = float(
                spearmanr(values, targets[:, column]).statistic
            )
    return result
```

- [ ] **Step 4: Write failing promotion tests**

Append:

```python
def test_promotion_requires_sixty_percent_oracle_recovery() -> None:
    report = promotion_decision(
        unconditional_fold=np.full(4, 0.10),
        coarse_fold=np.asarray([0.18, 0.17, 0.18, 0.17]),
        oracle_fold=np.full(4, 0.20),
        coarse_low_fold=np.full(4, 0.16),
        unconditional_low_fold=np.full(4, 0.10),
        family_gains={"physics": 0.07, "noise": 0.01},
        cross_noise=0.80,
        max_label_correlation=0.10,
        unconditional_max_label_correlation=0.20,
        severe_fpr_inflation=False,
    )
    assert report["coarse_recovery_fraction"] == pytest.approx(0.75)
    assert report["passed"]


def test_promotion_rejects_easy_family_dominance() -> None:
    report = promotion_decision(
        unconditional_fold=np.zeros(4),
        coarse_fold=np.full(4, 0.08),
        oracle_fold=np.full(4, 0.10),
        coarse_low_fold=np.full(4, 0.08),
        unconditional_low_fold=np.zeros(4),
        family_gains={"noise": 0.09, "physics": 0.01},
        cross_noise=0.80,
        max_label_correlation=0.10,
        unconditional_max_label_correlation=0.20,
        severe_fpr_inflation=False,
    )
    assert report["easy_family_gain_fraction"] == pytest.approx(0.90)
    assert not report["passed"]
```

- [ ] **Step 5: Implement the frozen gate**

Add:

```python
def promotion_decision(
    *,
    unconditional_fold: np.ndarray,
    coarse_fold: np.ndarray,
    oracle_fold: np.ndarray,
    coarse_low_fold: np.ndarray,
    unconditional_low_fold: np.ndarray,
    family_gains: dict[str, float],
    cross_noise: float,
    max_label_correlation: float,
    unconditional_max_label_correlation: float,
    severe_fpr_inflation: bool,
) -> dict[str, object]:
    u = np.asarray(unconditional_fold, dtype=np.float64)
    c = np.asarray(coarse_fold, dtype=np.float64)
    o = np.asarray(oracle_fold, dtype=np.float64)
    oracle_gain = float(o.mean() - u.mean())
    coarse_gain = float(c.mean() - u.mean())
    recovery = coarse_gain / oracle_gain if oracle_gain > 0.0 else float("-inf")
    positive = {name: max(0.0, value) for name, value in family_gains.items()}
    total = sum(positive.values())
    easy = sum(
        value for name, value in positive.items()
        if "noise" in name or "resolution" in name
    )
    easy_fraction = easy / total if total > 0.0 else 1.0
    gates = {
        "oracle_positive_three_folds": int(np.sum(o > u)) >= 3,
        "coarse_positive_three_folds": int(np.sum(c > u)) >= 3,
        "coarse_low_positive_three_folds": int(
            np.sum(
                np.asarray(coarse_low_fold)
                > np.asarray(unconditional_low_fold)
            )
        ) >= 3,
        "coarse_recovers_60pct_oracle": recovery >= 0.60,
        "easy_family_at_most_half_gain": easy_fraction <= 0.50,
        "cross_noise_at_least_0p60": cross_noise >= 0.60,
        "label_correlation_below_0p15": max_label_correlation < 0.15,
        "label_correlation_improves_unconditional": (
            max_label_correlation < unconditional_max_label_correlation
        ),
        "no_severe_fpr_inflation": not severe_fpr_inflation,
    }
    return {
        "oracle_gain": oracle_gain,
        "coarse_gain": coarse_gain,
        "coarse_recovery_fraction": recovery,
        "easy_family_gain_fraction": easy_fraction,
        "gates": gates,
        "passed": all(gates.values()),
    }
```

- [ ] **Step 6: Add the second held-noise cache**

Create `evaluation_repeat_unconditional.npy`, `evaluation_repeat_coarse.npy`,
and `evaluation_repeat_oracle.npy` using the same source rows and
`shape_noise_seed + 1`. Include both seeds in `cache_contract.json`. Score with
`score_signal_patch_matrix` and compute Spearman correlation between the two
deployable map-score vectors.

Use a separate namespace so the original argument object remains immutable:

```python
repeat_args = argparse.Namespace(**vars(args))
repeat_args.shape_noise_seed = args.shape_noise_seed + 1
repeat_cache = _open_cache(
    args.cache_dir / f"evaluation_repeat_{mode}.npy",
    (len(evaluation_indices), len(patch_ids)),
)
score_signal_patch_matrix(
    maps[evaluation_indices],
    evaluation_indices,
    labels[evaluation_indices],
    branch,
    patch_ids,
    repeat_args,
    repeat_cache,
)
```

- [ ] **Step 7: Replace summed-NLL selection with held-only map views**

Import `PatchTypicalityCalibration`, `EmpiricalSurprisalCalibration`, and
`map_views`. Fit patch typicality separately from
`calibration_unconditional.npy`, `calibration_coarse.npy`, and
`calibration_oracle.npy`. Produce coherent and sparse raw views, fit
empirical-surprisal maps from calibration views, and define the deployable
union as `np.maximum(coherent, sparse)`. Preserve summed raw NLL under an
audit-only key.

Implement the shared transformation as:

```python
def calibrated_map_scores(
    calibration_patch_nll: np.ndarray,
    target_patch_nll: np.ndarray,
    count: int,
) -> dict[str, np.ndarray]:
    patch = PatchTypicalityCalibration.fit(calibration_patch_nll[:, :count])
    calibration_views = map_views(
        patch.evidence(calibration_patch_nll[:, :count])
    )
    target_views = map_views(patch.evidence(target_patch_nll[:, :count]))
    coherent_calibration = EmpiricalSurprisalCalibration.fit(
        calibration_views["coherent"]
    )
    sparse_calibration = EmpiricalSurprisalCalibration.fit(
        calibration_views["sparse"]
    )
    coherent = coherent_calibration.score(target_views["coherent"])
    sparse = sparse_calibration.score(target_views["sparse"])
    return {
        "coherent": coherent,
        "sparse": sparse,
        "union": np.maximum(coherent, sparse),
    }
```

- [ ] **Step 8: Add four folds and the recovery report**

Assign fold `(source_id % maps_per_group) % 4`. Report official and low-FPR metrics for each branch/fold. Feed the vectors, cross-noise result, maximum absolute label correlation, family gains, and FPR status to `promotion_decision`.

Use the exact fold contract:

```python
def realization_folds(source_ids: np.ndarray, per_group: int) -> np.ndarray:
    values = np.asarray(source_ids, dtype=np.int64)
    if values.ndim != 1 or per_group <= 0:
        raise ValueError("source IDs and maps-per-group are invalid")
    return (values % per_group) % 4
```

Define severe FPR inflation as any evaluation rate above the corresponding
`upper_95` value returned by `empirical_fpr_report`.

- [ ] **Step 9: Add frozen-anchor complementarity**

Add CLI argument:

```python
parser.add_argument("--anchor-audit", type=Path, required=True)
```

Load `audit_ind[source_id]` for nominal held rows. Align proxy rows by exact `(proxy_type, proxy_source_index)` pairs and reject missing/duplicate pairs. Calibrate anchor and field independently with held InD scores, construct only 70/30 and 90/10 blends, and report fold changes plus top-10/20/50/100/500 retention. Do not read the test `candidate` vector in validation.

Build the alignment index without fuzzy matching:

```python
def align_anchor_proxy(
    proxy_type: np.ndarray,
    proxy_source_index: np.ndarray,
    requested_type: np.ndarray,
    requested_source_index: np.ndarray,
) -> np.ndarray:
    lookup: dict[tuple[str, int], int] = {}
    for position, (name, source) in enumerate(
        zip(proxy_type.astype(str), proxy_source_index.astype(np.int64))
    ):
        key = (name, int(source))
        if key in lookup:
            raise ValueError("duplicate anchor proxy pair")
        lookup[key] = position
    keys = list(zip(requested_type.astype(str), requested_source_index.astype(np.int64)))
    if any((name, int(source)) not in lookup for name, source in keys):
        raise ValueError("anchor audit is missing a requested proxy pair")
    return np.asarray(
        [lookup[(name, int(source))] for name, source in keys], dtype=np.int64
    )
```

For each protective blend, require improvement in at least three folds,
low-FPR degradation no worse than `-0.002`, and top-100 retention at least
`0.70`. Record this separately from the standalone promotion decision.

- [ ] **Step 10: Run focused tests**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_conditional_otfm_scores.py \
  tests/test_validate_conditional_otfm.py -q
```

Expected: all pass.

- [ ] **Step 11: Commit validation**

```bash
git add scripts/validate_conditional_otfm.py tests/test_validate_conditional_otfm.py
git commit -m "feat: gate conditional OTFM on tail stability"
```

---

### Task 4: Implement resumable test scoring and packaging

**Files:**
- Create: `scripts/build_conditional_otfm_submissions.py`
- Create: `tests/test_build_conditional_otfm_submissions.py`

**Interfaces:**
- Consumes: coarse checkpoint, train/test/mask/labels, anchor artifact, and likelihood settings.
- Produces: `cache` and `package` subcommands, held/test NLL caches, three ZIPs, `scores.npz`, and `manifest.json`.

- [ ] **Step 1: Write failing cache-contract tests**

Create `tests/test_build_conditional_otfm_submissions.py`:

```python
from __future__ import annotations

import json
import zipfile
import numpy as np
import pytest

from scripts.build_conditional_otfm_submissions import (
    EXPECTED_TEST_MAPS,
    enforce_production_contract,
    package_score_vectors,
    write_archives,
)


def test_contract_rejects_semantic_cache_reuse(tmp_path) -> None:
    enforce_production_contract(tmp_path, {"checkpoint": "abc", "steps": 32})
    enforce_production_contract(tmp_path, {"steps": 32, "checkpoint": "abc"})
    with pytest.raises(ValueError, match="contract differs"):
        enforce_production_contract(
            tmp_path, {"checkpoint": "abc", "steps": 64}
        )


def test_contract_rejects_legacy_arrays(tmp_path) -> None:
    np.save(tmp_path / "test_patch_nll.npy", np.zeros((10, 4)))
    with pytest.raises(ValueError, match="no contract"):
        enforce_production_contract(tmp_path, {"checkpoint": "abc"})
```

- [ ] **Step 2: Run to verify module failure**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_build_conditional_otfm_submissions.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Add CLI and immutable contract**

Create a parser with `cache` and `package` subcommands. The cache contract must include format version, checkpoint SHA-256, train/test/mask/label fingerprints, validation indices, patch IDs, solver, steps, probes, probe seed, held-noise seed, data scale, and test shape. Use atomic JSON writes and NaN-initialized float64 memmaps. Reject arrays without a matching contract.

Core contract function:

```python
def enforce_production_contract(cache_dir: Path, contract: dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cache_contract.json"
    canonical = json.loads(json.dumps(contract, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text()) != canonical:
            raise ValueError("production cache contract differs")
        return
    if any(cache_dir.glob("*.npy")):
        raise ValueError("production cache arrays have no contract")
    atomic_write_json(path, canonical)
```

- [ ] **Step 4: Add a batch-companion invariance test**

Use the tiny conditional U-Net configuration and an eight-pixel full mask. Score two realized rows together and separately with identical sample IDs, then assert `np.testing.assert_allclose` with `rtol=1e-6` and `atol=1e-6`.

The assertion must compare original row order exactly:

```python
def test_realized_scoring_is_batch_companion_invariant(
    fake_branch, fake_maps, fake_args
) -> None:
    patch_ids = np.asarray([0], dtype=np.int64)
    together = score_realized_patch_matrix(
        fake_maps, np.asarray([10, 11]), fake_branch, patch_ids, fake_args
    )
    separate = np.concatenate([
        score_realized_patch_matrix(
            fake_maps[index:index + 1],
            np.asarray([10 + index]),
            fake_branch,
            patch_ids,
            fake_args,
        )
        for index in range(2)
    ])
    np.testing.assert_allclose(together, separate, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 5: Implement held and test scoring**

Held rows use deterministic organizer-compatible shape noise keyed by source row and patch. Test maps are already noisy: divide by checkpoint `data_scale`, extract each patch, call `prepare_realized_detail` in `coarse` mode with an all-zero `(B, 5)` label tensor, then call `conditional_patch_negative_log_likelihood`. Never add test noise. Fill only missing memmap cells and flush after every patch.

The deployable test loop must have this shape:

```python
def score_realized_patch_matrix(
    maps: np.ndarray,
    sample_ids: np.ndarray,
    branch: LoadedBranch,
    patch_ids: np.ndarray,
    args: argparse.Namespace,
    output: np.ndarray | None = None,
) -> np.ndarray:
    if branch.mode != "coarse":
        raise ValueError("production scoring requires the coarse branch")
    if output is None:
        output = np.full((len(maps), len(patch_ids)), np.nan, dtype=np.float64)
    device = next(branch.model.parameters()).device
    scale = float(branch.checkpoint["data_contract"]["data_scale"])
    for column, patch_value in enumerate(patch_ids):
        patch_id = int(patch_value)
        missing = np.flatnonzero(~np.isfinite(output[:, column]))
        fixed = branch.layout.conditioning(patch_id, device=device)
        for cursor in range(0, len(missing), args.batch_size):
            positions = missing[cursor:cursor + args.batch_size]
            patches = np.stack([
                branch.layout.extract(maps[int(position)], patch_id)
                for position in positions
            ]) / scale
            data = torch.as_tensor(
                patches[:, None], device=device, dtype=torch.float32
            )
            labels = torch.zeros((len(positions), 5), device=device)
            detail, conditioning = prepare_realized_detail(
                data, fixed, labels, branch.standardization, "coarse"
            )
            nll = conditional_patch_negative_log_likelihood(
                branch.model,
                detail,
                conditioning,
                np.asarray(sample_ids)[positions].tolist(),
                patch_id,
                steps=args.steps,
                solver=args.solver,
                hutchinson_probes=args.hutchinson_probes,
                probe_seed=args.probe_seed,
            )
            output[positions, column] = nll.detach().cpu().numpy()
        if hasattr(output, "flush"):
            output.flush()
    if not np.isfinite(output).all():
        raise RuntimeError("production likelihood cache is incomplete")
    return output
```

- [ ] **Step 6: Write failing packaging tests**

Append:

```python
def test_packaging_creates_only_three_preregistered_vectors() -> None:
    vectors, calibrations = package_score_vectors(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.5, 2.5]),
        np.asarray([3.0, 2.0, 1.0, 0.0]),
        np.asarray([2.5, 0.5]),
    )
    assert set(vectors) == {
        "coarse_conditional_field_otfm_pure",
        "anchor70_coarse30",
        "anchor90_coarse10",
    }
    assert set(calibrations) == {"field", "anchor"}


def test_zip_contains_only_root_result_json(tmp_path) -> None:
    paths = write_archives(
        tmp_path, {"pure": np.linspace(0.0, 1.0, EXPECTED_TEST_MAPS)}
    )
    with zipfile.ZipFile(paths["pure"]) as archive:
        assert archive.namelist() == ["result.json"]
        payload = json.loads(archive.read("result.json"))
    assert len(payload["ood_scores"]) == EXPECTED_TEST_MAPS
```

- [ ] **Step 7: Implement frozen packaging**

Fit `EmpiricalSurprisalCalibration` separately on held field-union scores and `audit_ind` anchor scores. Transform field test scores and frozen anchor `candidate`. Call `fixed_anchor_blends` once. Write exactly:

```text
coarse_conditional_field_otfm_pure_v1.zip
anchor70_coarse30_v1.zip
anchor90_coarse10_v1.zip
scores.npz
manifest.json
```

Require 10,000 finite scores, at least 9,500 unique scores, only root `result.json` in each ZIP, deterministic replay, and SHA-256 hashes. The manifest records formulas, calibrations, sources, code/checkpoint/cache/archive hashes, runtime, and the single-instance eligibility statement.

Implement score construction independently of file I/O:

```python
def package_score_vectors(
    held_field: np.ndarray,
    test_field: np.ndarray,
    held_anchor: np.ndarray,
    test_anchor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, EmpiricalSurprisalCalibration]]:
    field_calibration = EmpiricalSurprisalCalibration.fit(held_field)
    anchor_calibration = EmpiricalSurprisalCalibration.fit(held_anchor)
    field = field_calibration.score(test_field)
    anchor = anchor_calibration.score(test_anchor)
    vectors = {"coarse_conditional_field_otfm_pure": field}
    vectors.update(fixed_anchor_blends(anchor, field))
    return vectors, {
        "field": field_calibration,
        "anchor": anchor_calibration,
    }
```

- [ ] **Step 8: Run focused production tests**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_conditional_otfm_scores.py \
  tests/test_build_conditional_otfm_submissions.py -q
```

Expected: all pass on CPU.

- [ ] **Step 9: Commit production scoring**

```bash
git add \
  scripts/build_conditional_otfm_submissions.py \
  tests/test_build_conditional_otfm_submissions.py
git commit -m "feat: package conditional OTFM submissions"
```

---

### Task 5: Verify all implementation before GPU use

**Files:**
- Verify all conditional OTFM scripts and tests.

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: clean tested commit used by every artifact manifest.

- [ ] **Step 1: Compile scripts**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m py_compile \
  scripts/wl_conditional_otfm.py \
  scripts/train_conditional_otfm.py \
  scripts/run_conditional_field_stage0.py \
  scripts/conditional_otfm_scores.py \
  scripts/validate_conditional_otfm.py \
  scripts/build_conditional_otfm_submissions.py
```

Expected: exit zero.

- [ ] **Step 2: Run conditional-flow tests**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest \
  tests/test_wl_otfm.py \
  tests/test_wl_conditional_otfm.py \
  tests/test_validate_otfm.py \
  tests/test_validate_conditional_otfm.py \
  tests/test_conditional_otfm_scores.py \
  tests/test_build_conditional_otfm_submissions.py -q
```

Expected: all pass.

- [ ] **Step 3: Run full suite**

```bash
/home/neelshah/neurips_wl/.venv/bin/python -m pytest -q
```

Expected: no failure.

- [ ] **Step 4: Verify clean branch**

```bash
git status --short --branch
git log -3 --oneline
```

Expected: clean worktree with three new commits.

---

### Task 6: Run contracts and train three matched branches

**Files:**
- Execute: `scripts/run_conditional_field_stage0.py`
- Execute: `scripts/train_conditional_otfm.py`
- Produce: `/home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/contracts/stage0_controls.json`
- Produce: three best and three last checkpoints under `checkpoints/`.

**Interfaces:**
- Consumes: verified code and official train data.
- Produces: aligned `unconditional.pt`, `coarse.pt`, and `oracle.pt`.

- [ ] **Step 1: Create run directories**

```bash
mkdir -p \
  /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/contracts \
  /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints \
  /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/logs \
  /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/cache \
  /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/reports \
  /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/submissions
```

- [ ] **Step 2: Run CPU/data controls**

```bash
cd /home/neelshah/neurips_wl_worktrees/codex_coarse_conditional_otfm_20260802
/home/neelshah/neurips_wl/.venv/bin/python scripts/run_conditional_field_stage0.py \
  --maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  --labels /home/neelshah/neurips_wl/data/label_newrealization.npy \
  --mask /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  --output /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/contracts/stage0_controls.json
```

Expected: `passed` is true. Stop on any contract failure.

- [ ] **Step 3: Record GPU state and UTC start**

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ
```

- [ ] **Step 4: Train unconditional**

```bash
/usr/bin/time -v /home/neelshah/neurips_wl/.venv/bin/python \
  scripts/train_conditional_otfm.py \
  --maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  --labels /home/neelshah/neurips_wl/data/label_newrealization.npy \
  --mask /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  --mode unconditional --base-channels 32 --epochs 20 --steps-per-epoch 400 \
  --score-patches 4 --seed 20260901 --device cuda \
  --output /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/unconditional.pt \
  > /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/logs/train_unconditional.log 2>&1
```

- [ ] **Step 5: Train coarse with the identical arguments**

```bash
/usr/bin/time -v /home/neelshah/neurips_wl/.venv/bin/python \
  scripts/train_conditional_otfm.py \
  --maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  --labels /home/neelshah/neurips_wl/data/label_newrealization.npy \
  --mask /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  --mode coarse --base-channels 32 --epochs 20 --steps-per-epoch 400 \
  --score-patches 4 --seed 20260901 --device cuda \
  --output /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/coarse.pt \
  > /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/logs/train_coarse.log 2>&1
```

- [ ] **Step 6: Train oracle with the identical arguments**

```bash
/usr/bin/time -v /home/neelshah/neurips_wl/.venv/bin/python \
  scripts/train_conditional_otfm.py \
  --maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  --labels /home/neelshah/neurips_wl/data/label_newrealization.npy \
  --mask /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  --mode oracle --base-channels 32 --epochs 20 --steps-per-epoch 400 \
  --score-patches 4 --seed 20260901 --device cuda \
  --output /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/oracle.pt \
  > /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/logs/train_oracle.log 2>&1
```

- [ ] **Step 7: Verify checkpoint contracts**

```bash
/home/neelshah/neurips_wl/.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import torch

root = Path("/home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints")
values = {
    mode: torch.load(root / f"{mode}.pt", map_location="cpu", weights_only=False)
    for mode in ("unconditional", "coarse", "oracle")
}
reference = values["unconditional"]
for mode, value in values.items():
    assert value["mode"] == mode
    assert value["model_config"] == reference["model_config"]
    assert value["layout_contract"] == reference["layout_contract"]
    for key in ("train_indices", "validation_indices", "score_patch_ids"):
        assert (value[key] == reference[key]).all()
    path = root / f"{mode}.pt"
    print(mode, value["best_validation_loss"], hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Expected: all assertions pass. Stop if cumulative training exceeds two hours.

---

### Task 7: Run the four-patch gate

**Files:**
- Execute: `scripts/validate_conditional_otfm.py`
- Produce: `cache/held_proxy_four_patch/`
- Produce: `reports/conditional_otfm_four_patch.json`

**Interfaces:**
- Consumes: three checkpoints and anchor audit artifact.
- Produces: authoritative `promotion_decision.passed`.

- [ ] **Step 1: Run resumable validation**

```bash
/usr/bin/time -v /home/neelshah/neurips_wl/.venv/bin/python \
  scripts/validate_conditional_otfm.py \
  --unconditional /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/unconditional.pt \
  --coarse /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/coarse.pt \
  --oracle /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/oracle.pt \
  --maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  --labels /home/neelshah/neurips_wl/data/label_newrealization.npy \
  --mask /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  --anchor-audit /home/neelshah/neurips_wl/submissions/drc19_hgb_tailgate_moonshot_v1/scores.npz \
  --cache-dir /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/cache/held_proxy_four_patch \
  --output /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/reports/conditional_otfm_four_patch.json \
  --patch-counts 1 2 4 --steps 32 --solver heun --hutchinson-probes 2 \
  --batch-size 32 --bootstrap-replicates 500 --device cuda \
  > /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/logs/validate_four_patch.log 2>&1
```

- [ ] **Step 2: Print only preregistered decision fields**

```bash
/home/neelshah/neurips_wl/.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("/home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/reports/conditional_otfm_four_patch.json")
report = json.loads(path.read_text())
print(json.dumps({
    "promotion_decision": report["promotion_decision"],
    "known_label_correlations": report["known_label_correlations"],
    "cross_noise_rank_correlation": report["cross_noise_rank_correlation"],
    "fpr_calibration": report["fpr_calibration"],
    "anchor_complementarity": report["anchor_complementarity"],
}, indent=2, sort_keys=True))
PY
```

- [ ] **Step 3: Apply the hard gate**

If `promotion_decision.passed` is false, stop, preserve caches, and record rejection. Do not score test maps or weaken gates. If true, continue.

- [ ] **Step 4: Audit elapsed budget**

If cumulative elapsed time exceeds four hours, report the production estimate before continuing. Do not launch a continuation expected above three hours.

---

### Task 8: Score test maps and package three archives

**Files:**
- Execute: `scripts/build_conditional_otfm_submissions.py`
- Produce: `cache/production_four_patch/`
- Produce: three ZIPs, `scores.npz`, and `manifest.json`.

**Interfaces:**
- Consumes: passing report, coarse checkpoint, official held/test data, and anchor vectors.
- Produces: three verified submittable archives.

- [ ] **Step 1: Cache held and test NLLs**

```bash
/usr/bin/time -v /home/neelshah/neurips_wl/.venv/bin/python \
  scripts/build_conditional_otfm_submissions.py cache \
  --coarse /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/coarse.pt \
  --train-maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy \
  --test-maps /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy \
  --labels /home/neelshah/neurips_wl/data/label_newrealization.npy \
  --mask /home/neelshah/neurips_wl/data/WIDE12H_bin2_2arcmin_mask.npy \
  --cache-dir /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/cache/production_four_patch \
  --patch-count 4 --steps 32 --solver heun --hutchinson-probes 2 \
  --batch-size 32 --device cuda \
  > /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/logs/cache_production.log 2>&1
```

- [ ] **Step 2: Audit completeness without test-distribution inspection**

```bash
/home/neelshah/neurips_wl/.venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path
root = Path("/home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/cache/production_four_patch")
for name, expected in (("calibration_patch_nll.npy", None), ("test_patch_nll.npy", 10_000)):
    value = np.load(root / name, mmap_mode="r", allow_pickle=False)
    assert value.ndim == 2 and value.shape[1] == 4
    if expected is not None:
        assert len(value) == expected
    assert np.isfinite(value).all()
    print(name, value.shape, value.dtype, "finite")
PY
```

Expected: shape/dtype/finiteness only; no test summary statistics.

- [ ] **Step 3: Package from the single cache**

```bash
/home/neelshah/neurips_wl/.venv/bin/python \
  scripts/build_conditional_otfm_submissions.py package \
  --coarse /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/checkpoints/coarse.pt \
  --cache-dir /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/cache/production_four_patch \
  --anchor /home/neelshah/neurips_wl/submissions/drc19_hgb_tailgate_moonshot_v1/scores.npz \
  --validation-report /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/reports/conditional_otfm_four_patch.json \
  --output-dir /home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/submissions
```

Expected: packager refuses a failed validation report.

- [ ] **Step 4: Verify archives and hashes**

```bash
/home/neelshah/neurips_wl/.venv/bin/python - <<'PY'
import hashlib
import json
import zipfile
from pathlib import Path

root = Path("/home/neelshah/neurips_wl_runs/coarse_conditional_otfm_20260802/submissions")
for name in (
    "coarse_conditional_field_otfm_pure_v1.zip",
    "anchor70_coarse30_v1.zip",
    "anchor90_coarse10_v1.zip",
):
    path = root / name
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["result.json"]
        scores = json.loads(archive.read("result.json"))["ood_scores"]
    assert len(scores) == 10_000
    assert all(isinstance(value, (int, float)) for value in scores)
    print(name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Expected: filenames, sizes, and hashes only.

- [ ] **Step 5: Stop for archive review**

Show gate evidence and archive hashes. Do not upload.

---

### Task 9: Record and verify the sprint

**Files:**
- Create: `reports/conditional_field_otfm_stage0_v1.md`
- Modify: `docs/AI_USAGE.md`

**Interfaces:**
- Consumes: commands, hashes, runtimes, gates, and archive metadata.
- Produces: reproducible private record and clean branch.

- [ ] **Step 1: Write the append-only report**

Record the approved hypothesis/spec path, VM branch and commits, data/checkpoint/cache/archive hashes, exact commands/seeds, runtime and GPU memory, branch losses, FPR/label/noise/fold/family/oracle/tail outputs, gate result, and stop-or-package decision. Leave the public-result section empty until an approved upload returns.

- [ ] **Step 2: Update AI disclosure**

Add a dated `docs/AI_USAGE.md` entry that Codex assisted with the coarse-conditioned field hypothesis, preregistered gates, implementation, testing, and orchestration, while data and outputs remained in the private VM.

- [ ] **Step 3: Verify docs and tests**

```bash
git diff --check
/home/neelshah/neurips_wl/.venv/bin/python -m pytest -q
```

Expected: no whitespace error and no test failure.

- [ ] **Step 4: Commit the record**

```bash
git add reports/conditional_field_otfm_stage0_v1.md docs/AI_USAGE.md
git commit -m "exp: record conditional field OTFM sprint"
```

- [ ] **Step 5: Invoke completion verification**

Use `superpowers:verification-before-completion`, then run:

```bash
git status --short --branch
git log -5 --oneline
```

Expected: clean branch.

- [ ] **Step 6: Request upload approval**

If the gate passed and archives exist, recommend pure first, then 70/30 and 90/10 without changing weights. Upload only after explicit user approval.
