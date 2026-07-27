# Project Foundation and Submission Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible Python project that exactly implements the public Phase-2 metric, validates OoD score arrays, and builds deterministic Codabench submission archives without requiring the full competition dataset.

**Architecture:** A small `wl_ood` package will separate metric calculation, submission validation/serialization, and command-line orchestration. Tests will compare the metric against the organizers' published NumPy/scikit-learn formula and inspect ZIP contents byte-for-byte. Project documentation will record privacy rules, experiment discipline, and AI usage from the first commit.

**Tech Stack:** Python 3.12.8, NumPy 2.3.3, scikit-learn 1.8.0, pytest 8.x, standard-library `argparse`, `json`, and `zipfile`.

## Global Constraints

- Each test score must be computed independently of all other test maps.
- A valid Phase-2 result contains exactly 10,000 finite floating-point scores in original test order.
- Larger scores must mean greater OoD confidence.
- The archive must contain exactly one root-level file named `result.json`.
- The metric must use 100 logarithmically spaced FPR values from 0.001 to 0.05 and NumPy linear interpolation over `sklearn.metrics.roc_curve`.
- Competition code, methods, data, and generated submissions remain private.
- Python runtime is pinned to 3.12.8 for local reproducibility; the Codabench `codalab/codalab-legacy:py39` image only scores the result JSON.

---

## File Structure

- `.gitignore`: exclude environments, raw data, generated artifacts, notebooks' transient state, and macOS metadata.
- `.python-version`: document Python 3.12.8.
- `pyproject.toml`: package metadata, runtime dependencies, pytest settings, and `wl-ood` console entry point.
- `README.md`: project purpose, privacy warning, setup, verification, and first commands.
- `src/wl_ood/__init__.py`: expose the package version.
- `src/wl_ood/metric.py`: implement the exact official low-FPR metric and input validation.
- `src/wl_ood/submission.py`: validate scores, serialize `result.json`, and create deterministic ZIP archives.
- `src/wl_ood/cli.py`: provide `make-dummy` and `score` commands.
- `tests/test_metric.py`: verify official-formula parity and invalid-input behavior.
- `tests/test_submission.py`: verify score validation, JSON schema, ordering, and clean ZIP structure.
- `tests/test_cli.py`: verify deterministic dummy creation and local score reporting.
- `docs/AI_USAGE.md`: disclose Codex's role and define the ongoing logging convention.
- `experiments/README.md`: define one-directory-per-experiment records.
- `submissions/.gitkeep`: preserve the generated-submission directory while ignoring its contents.

---

### Task 1: Python Project Skeleton

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `src/wl_ood/__init__.py`
- Create: `submissions/.gitkeep`

**Interfaces:**
- Produces: importable package `wl_ood` with `wl_ood.__version__ == "0.1.0"`.
- Produces: console entry point `wl-ood = wl_ood.cli:main`, implemented in Task 4.

- [ ] **Step 1: Write the package import test**

Create `tests/test_package.py`:

```python
import wl_ood


def test_package_version() -> None:
    assert wl_ood.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_package.py -v
```

Expected: failure because pytest and/or `wl_ood` is not installed yet.

- [ ] **Step 3: Add project configuration**

Create `.python-version`:

```text
3.12.8
```

Create `.gitignore`:

```gitignore
.DS_Store
__MACOSX/
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
*.egg-info/
build/
dist/

data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
artifacts/*
!artifacts/.gitkeep
submissions/*
!submissions/.gitkeep

.ipynb_checkpoints/
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75,<76"]
build-backend = "setuptools.build_meta"

[project]
name = "fair-universe-wl-ood"
version = "0.1.0"
description = "Research pipeline for the FAIR Universe Weak Lensing Phase-2 OoD challenge"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy==2.3.3",
  "scikit-learn==1.8.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3,<9",
]

[project.scripts]
wl-ood = "wl_ood.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

Create `src/wl_ood/__init__.py`:

```python
"""FAIR Universe weak-lensing OoD research package."""

__version__ = "0.1.0"
```

Create an empty `submissions/.gitkeep`.

- [ ] **Step 4: Create and install the isolated development environment**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Expected: editable package and pytest install successfully.

- [ ] **Step 5: Run the package test**

Run:

```bash
.venv/bin/python -m pytest tests/test_package.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore .python-version pyproject.toml src/wl_ood/__init__.py submissions/.gitkeep tests/test_package.py
git commit -m "build: initialize weak-lensing OoD package"
```

---

### Task 2: Exact Low-FPR Competition Metric

**Files:**
- Create: `src/wl_ood/metric.py`
- Create: `tests/test_metric.py`

**Interfaces:**
- Produces: `LowFprMetricResult(fpr: np.ndarray, tpr: np.ndarray, score: float)`.
- Produces: `compute_low_fpr_metric(labels, scores, *, min_fpr=0.001, max_fpr=0.05, n_points=100) -> LowFprMetricResult`.
- Consumes: one-dimensional binary labels and finite one-dimensional scores of equal nonzero length.

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metric.py`:

```python
import numpy as np
import pytest
from sklearn.metrics import roc_curve

from wl_ood.metric import compute_low_fpr_metric


def official_reference(labels: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(labels, scores)
    target_fpr = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    target_tpr = np.interp(target_fpr, fpr, tpr)
    return target_fpr, target_tpr, float(np.mean(target_tpr))


def test_matches_official_reference_formula() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0.05, 0.90, 0.30, 0.60, 0.10, 0.80, 0.20, 0.70])

    expected_fpr, expected_tpr, expected_score = official_reference(labels, scores)
    result = compute_low_fpr_metric(labels, scores)

    np.testing.assert_allclose(result.fpr, expected_fpr)
    np.testing.assert_allclose(result.tpr, expected_tpr)
    assert result.score == pytest.approx(expected_score)


def test_perfect_ranking_scores_one() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = compute_low_fpr_metric(labels, scores)

    assert result.score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        (np.array([[0, 1]]), np.array([0.1, 0.9]), "one-dimensional"),
        (np.array([0, 1]), np.array([[0.1, 0.9]]), "one-dimensional"),
        (np.array([0, 1]), np.array([0.1]), "same length"),
        (np.array([0, 2]), np.array([0.1, 0.9]), "binary"),
        (np.array([0, 0]), np.array([0.1, 0.9]), "both classes"),
        (np.array([0, 1]), np.array([0.1, np.nan]), "finite"),
    ],
)
def test_rejects_invalid_inputs(labels, scores, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_low_fpr_metric(labels, scores)


def test_rejects_invalid_fpr_configuration() -> None:
    with pytest.raises(ValueError, match="FPR"):
        compute_low_fpr_metric(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            min_fpr=0.1,
            max_fpr=0.05,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_metric.py -v
```

Expected: collection error because `wl_ood.metric` does not exist.

- [ ] **Step 3: Implement the metric**

Create `src/wl_ood/metric.py`:

```python
"""Exact local implementation of the FAIR Universe Phase-2 metric."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class LowFprMetricResult:
    """Interpolated ROC values and their mean over the challenge FPR grid."""

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    score: float


def compute_low_fpr_metric(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    min_fpr: float = 0.001,
    max_fpr: float = 0.05,
    n_points: int = 100,
) -> LowFprMetricResult:
    """Compute the organizer's mean TPR over a logarithmic low-FPR grid."""
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=np.float64)

    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if labels_array.size == 0 or labels_array.size != scores_array.size:
        raise ValueError("labels and scores must have the same length")
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must contain only finite values")

    unique_labels = np.unique(labels_array)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError("labels must be binary values 0 or 1")
    if unique_labels.size != 2:
        raise ValueError("labels must contain both classes")

    if not (0.0 < min_fpr < max_fpr <= 1.0) or n_points < 2:
        raise ValueError("FPR configuration must satisfy 0 < min < max <= 1")

    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    target_fpr = np.logspace(
        np.log10(min_fpr),
        np.log10(max_fpr),
        n_points,
    )
    target_tpr = np.interp(target_fpr, fpr, tpr)

    return LowFprMetricResult(
        fpr=target_fpr,
        tpr=target_tpr,
        score=float(np.mean(target_tpr)),
    )
```

- [ ] **Step 4: Run the metric tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_metric.py -v
```

Expected: all metric tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/wl_ood/metric.py tests/test_metric.py
git commit -m "feat: implement exact Phase-2 metric"
```

---

### Task 3: Submission Validation and Deterministic Archive

**Files:**
- Create: `src/wl_ood/submission.py`
- Create: `tests/test_submission.py`

**Interfaces:**
- Produces: `validate_scores(scores, *, expected_count=10_000) -> np.ndarray`.
- Produces: `write_result_json(scores, path, *, expected_count=10_000) -> pathlib.Path`.
- Produces: `create_submission_zip(scores, path, *, expected_count=10_000) -> pathlib.Path`.
- ZIP output contains only `result.json` at the archive root.

- [ ] **Step 1: Write failing submission tests**

Create `tests/test_submission.py`:

```python
import json
import zipfile

import numpy as np
import pytest

from wl_ood.submission import (
    create_submission_zip,
    validate_scores,
    write_result_json,
)


def test_validate_scores_preserves_order_and_values() -> None:
    scores = np.linspace(-1.0, 1.0, 10_000)

    validated = validate_scores(scores)

    np.testing.assert_array_equal(validated, scores)
    assert validated.dtype == np.float64


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.zeros(9_999), "exactly 10000"),
        (np.zeros((100, 100)), "one-dimensional"),
        (np.concatenate([np.zeros(9_999), [np.nan]]), "finite"),
        (np.concatenate([np.zeros(9_999), [np.inf]]), "finite"),
    ],
)
def test_validate_scores_rejects_invalid_values(scores, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_scores(scores)


def test_write_result_json_uses_required_schema(tmp_path) -> None:
    scores = np.arange(10_000, dtype=np.float64)
    output = write_result_json(scores, tmp_path / "result.json")

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert list(payload) == ["ood_scores"]
    assert payload["ood_scores"] == scores.tolist()


def test_submission_zip_contains_only_root_result_json(tmp_path) -> None:
    scores = np.linspace(0.0, 1.0, 10_000)
    archive = create_submission_zip(scores, tmp_path / "dummy.zip")

    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist() == ["result.json"]
        payload = json.loads(zipped.read("result.json"))

    assert payload["ood_scores"] == scores.tolist()


def test_submission_zip_is_deterministic(tmp_path) -> None:
    scores = np.linspace(0.0, 1.0, 10_000)
    first = create_submission_zip(scores, tmp_path / "first.zip")
    second = create_submission_zip(scores, tmp_path / "second.zip")

    assert first.read_bytes() == second.read_bytes()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_submission.py -v
```

Expected: collection error because `wl_ood.submission` does not exist.

- [ ] **Step 3: Implement submission utilities**

Create `src/wl_ood/submission.py`:

```python
"""Validation and deterministic packaging for Codabench result submissions."""

import json
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_scores(
    scores: ArrayLike,
    *,
    expected_count: int = 10_000,
) -> NDArray[np.float64]:
    """Return validated float64 scores without changing their order."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("ood_scores must be one-dimensional")
    if score_array.size != expected_count:
        raise ValueError(f"ood_scores must contain exactly {expected_count} values")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("ood_scores must contain only finite values")
    return score_array


def _result_json_bytes(scores: NDArray[np.float64]) -> bytes:
    payload = {"ood_scores": scores.tolist()}
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8")


def write_result_json(
    scores: ArrayLike,
    path: str | Path,
    *,
    expected_count: int = 10_000,
) -> Path:
    """Write the exact result JSON schema expected by Codabench."""
    score_array = validate_scores(scores, expected_count=expected_count)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_result_json_bytes(score_array))
    return output_path


def create_submission_zip(
    scores: ArrayLike,
    path: str | Path,
    *,
    expected_count: int = 10_000,
) -> Path:
    """Create a deterministic ZIP containing only root-level result.json."""
    score_array = validate_scores(scores, expected_count=expected_count)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zip_info = zipfile.ZipInfo("result.json", date_time=(2026, 1, 1, 0, 0, 0))
    zip_info.compress_type = zipfile.ZIP_DEFLATED
    zip_info.external_attr = 0o600 << 16

    with zipfile.ZipFile(output_path, mode="w") as archive:
        archive.writestr(zip_info, _result_json_bytes(score_array))

    return output_path
```

- [ ] **Step 4: Run submission tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_submission.py -v
```

Expected: all submission tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/wl_ood/submission.py tests/test_submission.py
git commit -m "feat: validate and package Codabench scores"
```

---

### Task 4: Command-Line Workflow

**Files:**
- Create: `src/wl_ood/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `wl-ood make-dummy --output PATH --seed INTEGER`.
- Produces: `wl-ood score --labels PATH --result-json PATH`.
- `make-dummy` generates deterministic independent random scores solely to test submission plumbing.
- `score` reads `.npy` binary labels and a result JSON, then prints the scalar metric.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.py`:

```python
import json
import zipfile

import numpy as np
import pytest

from wl_ood.cli import main


def test_make_dummy_is_reproducible(tmp_path, capsys) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    assert main(["make-dummy", "--output", str(first), "--seed", "20260727"]) == 0
    assert main(["make-dummy", "--output", str(second), "--seed", "20260727"]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert "10000" in capsys.readouterr().out


def test_score_command_reports_official_metric(tmp_path, capsys) -> None:
    labels = np.array([0, 0, 1, 1])
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"ood_scores": [0.1, 0.2, 0.8, 0.9]}),
        encoding="utf-8",
    )

    assert main(
        [
            "score",
            "--labels",
            str(labels_path),
            "--result-json",
            str(result_path),
        ]
    ) == 0
    assert "1.000000" in capsys.readouterr().out


def test_score_command_rejects_missing_key(tmp_path) -> None:
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, np.array([0, 1]))
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="ood_scores"):
        main(
            [
                "score",
                "--labels",
                str(labels_path),
                "--result-json",
                str(result_path),
            ]
        )


def test_make_dummy_zip_has_expected_schema(tmp_path) -> None:
    archive_path = tmp_path / "dummy.zip"
    main(["make-dummy", "--output", str(archive_path)])

    with zipfile.ZipFile(archive_path) as archive:
        payload = json.loads(archive.read("result.json"))

    assert len(payload["ood_scores"]) == 10_000
    assert np.all(np.isfinite(payload["ood_scores"]))
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: collection error because `wl_ood.cli` does not exist.

- [ ] **Step 3: Implement the CLI**

Create `src/wl_ood/cli.py`:

```python
"""Command-line entry points for metric checks and submission creation."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from wl_ood.metric import compute_low_fpr_metric
from wl_ood.submission import create_submission_zip


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wl-ood")
    commands = parser.add_subparsers(dest="command", required=True)

    dummy = commands.add_parser("make-dummy")
    dummy.add_argument("--output", type=Path, required=True)
    dummy.add_argument("--seed", type=int, default=20260727)

    score = commands.add_parser("score")
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "make-dummy":
        rng = np.random.default_rng(args.seed)
        scores = rng.random(10_000)
        output = create_submission_zip(scores, args.output)
        print(f"Wrote {scores.size} deterministic dummy scores to {output}")
        return 0

    labels = np.load(args.labels)
    with args.result_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if "ood_scores" not in payload:
        raise ValueError("result JSON must contain ood_scores")

    result = compute_low_fpr_metric(labels, payload["ood_scores"])
    print(f"Phase-2 score: {result.score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: all CLI tests pass.

- [ ] **Step 5: Generate and inspect the first dummy archive**

Run:

```bash
.venv/bin/wl-ood make-dummy \
  --output submissions/dummy-seed-20260727.zip \
  --seed 20260727
unzip -l submissions/dummy-seed-20260727.zip
```

Expected: output reports 10,000 scores; archive listing contains exactly `result.json`.

- [ ] **Step 6: Commit**

```bash
git add src/wl_ood/cli.py tests/test_cli.py
git commit -m "feat: add local metric and dummy submission CLI"
```

---

### Task 5: Research and Privacy Documentation

**Files:**
- Create: `README.md`
- Create: `docs/AI_USAGE.md`
- Create: `experiments/README.md`

**Interfaces:**
- Produces: setup and verification instructions using only commands tested in Tasks 1–4.
- Produces: append-only AI disclosure fields for future work.
- Produces: experiment-record schema required before a result may be submitted.

- [ ] **Step 1: Write the documentation smoke test**

Create `tests/test_documentation.py`:

```python
from pathlib import Path


def test_required_research_documents_exist() -> None:
    required = [
        Path("README.md"),
        Path("docs/AI_USAGE.md"),
        Path("experiments/README.md"),
    ]
    assert all(path.is_file() for path in required)


def test_readme_contains_verified_entry_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert ".venv/bin/python -m pytest" in readme
    assert ".venv/bin/wl-ood make-dummy" in readme
    assert "PRIVATE" in readme
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_documentation.py -v
```

Expected: failures because the required documents do not exist.

- [ ] **Step 3: Create the project README**

Create `README.md`:

```markdown
# FAIR Universe Weak-Lensing OoD Research

> **PRIVATE COMPETITION WORKSPACE:** Do not publish code, methodology, data,
> trained artifacts, scores, or submissions during the competition.

This repository develops independently evaluated OoD scores for Phase 2 of the
FAIR Universe Weak Lensing ML Uncertainty Challenge.

## Current milestone

The initial package reproduces the public scoring formula and creates clean,
deterministic result-submission archives. It does not yet train an OoD model.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Codabench uses `codalab/codalab-legacy:py39` to score uploaded JSON results.
Our model code runs locally; only `result.json` is submitted.

## Verify

```bash
.venv/bin/python -m pytest
```

## Create a plumbing-only dummy submission

```bash
.venv/bin/wl-ood make-dummy \
  --output submissions/dummy-seed-20260727.zip \
  --seed 20260727
```

Dummy scores test file formatting only and are not a scientific method.

## Research rules

- One test map is scored without access to other test maps.
- Test-set statistics never fit preprocessing, calibration, or ensembles.
- Every experiment records its source revision, configuration, seed, local
  evidence, artifacts, and conclusion.
- AI assistance is disclosed in `docs/AI_USAGE.md`.
```

- [ ] **Step 4: Create the AI usage record**

Create `docs/AI_USAGE.md`:

```markdown
# AI Usage Disclosure

The project uses OpenAI Codex as a collaborative AI agent.

## Roles

- Explain competition physics and rules.
- Help design the research and validation strategy.
- Draft and review source code, tests, and documentation.
- Assist with debugging and reproducibility checks.
- Summarize experiment evidence under human direction.

## Human responsibility

The participant approves research decisions, controls Codabench registration
and submissions, verifies generated work, and remains responsible for rule
compliance and the final methodology.

## Ongoing log

For material AI-assisted decisions, append:

- Date
- Human objective
- AI contribution
- Files or experiments affected
- Human verification performed
```

- [ ] **Step 5: Create the experiment convention**

Create `experiments/README.md`:

```markdown
# Experiment Records

Every scientific run receives a directory named:

`YYYYMMDD-HHMM-short-hypothesis`

Each directory must contain `record.md` with:

1. Hypothesis
2. Source commit
3. Configuration and random seed
4. Training and calibration split
5. Features and model
6. Local low-FPR results
7. Pseudo-OoD results
8. Runtime and hardware
9. Artifact and submission checksums
10. Conclusion and next decision

A Codabench submission must reference exactly one experiment directory.
```

- [ ] **Step 6: Run documentation and full test suites**

Run:

```bash
.venv/bin/python -m pytest tests/test_documentation.py -v
.venv/bin/python -m pytest
```

Expected: documentation tests pass; full suite passes.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/AI_USAGE.md experiments/README.md tests/test_documentation.py
git commit -m "docs: establish private reproducible research workflow"
```

---

### Task 6: Final Milestone Verification

**Files:**
- Modify only if verification exposes a defect in files from Tasks 1–5.

**Interfaces:**
- Consumes: complete package, tests, CLI, and documentation.
- Produces: passing test output and a clean deterministic dummy ZIP ready for manual Codabench upload.

- [ ] **Step 1: Run all automated tests from the isolated environment**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass with no warnings caused by project code.

- [ ] **Step 2: Generate the dummy archive twice**

Run:

```bash
.venv/bin/wl-ood make-dummy \
  --output submissions/dummy-a.zip \
  --seed 20260727
.venv/bin/wl-ood make-dummy \
  --output submissions/dummy-b.zip \
  --seed 20260727
```

Expected: both commands report 10,000 scores.

- [ ] **Step 3: Verify deterministic bytes and clean contents**

Run:

```bash
shasum -a 256 submissions/dummy-a.zip submissions/dummy-b.zip
unzip -l submissions/dummy-a.zip
```

Expected: SHA-256 hashes match; archive contains exactly one `result.json`.

- [ ] **Step 4: Verify the package is cleanly reinstallable**

Run:

```bash
.venv/bin/python -m pip uninstall -y fair-universe-wl-ood
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Expected: installation succeeds and all tests pass again.

- [ ] **Step 5: Record repository status**

Run:

```bash
git status --short
git log --oneline --decorate -6
```

Expected: only ignored generated submission archives are absent from status; milestone commits are visible.

- [ ] **Step 6: Close or reopen the milestone**

If Steps 1–5 all match their expected results, close the milestone without an
empty commit. If a check fails, return to the task that owns the failing file,
add a regression test there, and repeat that task's test-and-commit cycle before
rerunning all of Task 6.

---

## Self-Review Results

- **Spec coverage:** This milestone covers project structure, Python reproducibility, the exact public metric, result validation, deterministic ZIP creation, privacy, AI disclosure, experiment discipline, and dummy-submission generation. Full-data loading and scientific baselines intentionally remain in the next plan because they require the downloaded dataset.
- **Placeholder scan:** No `TBD`, `TODO`, replacement token, or unspecified implementation body remains.
- **Type consistency:** `compute_low_fpr_metric`, `validate_scores`, `write_result_json`, `create_submission_zip`, and `main` use the same names and argument conventions across source, tests, CLI, and documentation.
