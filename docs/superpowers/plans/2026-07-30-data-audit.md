# Competition Data Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a tested, memory-mapped structural audit of the four local Phase-2 NumPy files, fingerprint them, reconstruct individual maps, and generate fixed previews without using aggregate public-test statistics.

**Architecture:** `wl_ood.data` owns file resolution, schemas, and memory-mapped loading. `wl_ood.audit` validates structure, scans finiteness in bounded chunks, fingerprints files, and writes JSON. `wl_ood.maps` reconstructs flattened valid pixels and renders only fixed individual examples. Existing CLI wiring exposes `audit-data` and `preview-data`.

**Tech Stack:** Python 3.12.8, NumPy 2.3.3, scikit-learn 1.8.0, Matplotlib 3.11.1, pytest 8.4.2, standard-library `hashlib`, `json`, `pathlib`, and `dataclasses`.

## Global Constraints

- Raw competition files remain read-only and local under `public_data/`.
- `public_data/` must be ignored by Git.
- Large arrays must be opened with `numpy.load(..., mmap_mode="r", allow_pickle=False)`.
- No full training or test array may be cast, copied, or loaded into RAM.
- Test-set validation may record only structure, fingerprints, and non-finite counts.
- Do not compute or save test-wide means, standard deviations, quantiles, histograms, ranks, clusters, or fitted normalization.
- Preview indices are fixed before image inspection: training `(0, 0)`, training `(100, 255)`, test `0`, and test `9999`.
- Generated reports and images live under ignored `artifacts/`.
- Every production function is introduced through a failing test.

---

## File Structure

- Modify `.gitignore`: ignore `public_data/` and preserve ignored artifact directories.
- Modify `pyproject.toml`: add pinned Matplotlib 3.11.1.
- Create `src/wl_ood/data.py`: schemas, path resolution, and memory-mapped loading.
- Create `src/wl_ood/audit.py`: bounded finiteness scan, structural validation, SHA-256, and report writing.
- Create `src/wl_ood/maps.py`: one-map reconstruction and fixed preview rendering.
- Modify `src/wl_ood/cli.py`: add `audit-data` and `preview-data`.
- Create `tests/test_data.py`: path, schema, and memmap tests.
- Create `tests/__init__.py`: make shared synthetic-data fixtures importable
  without colliding with unrelated installed `tests` packages.
- Create `tests/test_audit.py`: structural, chunk, fingerprint, and report tests.
- Create `tests/test_maps.py`: reconstruction and PNG tests.
- Modify `tests/test_cli.py`: audit and preview command tests.
- Modify `tests/test_documentation.py`: require flattened-layout documentation.
- Modify `README.md`: document data location, storage shape, audit commands, and outputs.

---

### Task 1: Protect Raw Data and Add Preview Dependency

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `tests/test_data_privacy.py`

**Interfaces:**
- Produces: Git ignore rule `public_data/`.
- Produces: runtime dependency `matplotlib==3.11.1`.

- [ ] **Step 1: Write the failing configuration tests**

Create `tests/test_data_privacy.py`:

```python
from pathlib import Path


def test_public_data_is_ignored() -> None:
    rules = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "public_data/" in rules


def test_matplotlib_is_pinned() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"matplotlib==3.11.1"' in project
```

- [ ] **Step 2: Run the tests and confirm the expected failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_data_privacy.py -v
```

Expected: both tests fail because neither rule exists.

- [ ] **Step 3: Add the minimal configuration**

Append this line to `.gitignore` after the environment/build rules:

```gitignore
public_data/
```

Add this entry to `[project].dependencies` in `pyproject.toml`:

```toml
  "matplotlib==3.11.1",
```

- [ ] **Step 4: Reinstall and verify**

Run:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_data_privacy.py -v
git check-ignore public_data/WIDE12H_bin2_2arcmin_mask.npy
```

Expected: two tests pass and `git check-ignore` prints the mask path.

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml tests/test_data_privacy.py
git commit -m "build: protect raw data and add preview dependency"
```

---

### Task 2: Dataset Schema and Memory-Mapped Loader

**Files:**
- Create: `src/wl_ood/data.py`
- Create: `tests/__init__.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `DatasetSchema`.
- Produces: `PHASE2_SCHEMA`.
- Produces: `DatasetPaths.from_dir(data_dir)`.
- Produces: `CompetitionDataset`.
- Produces: `load_dataset(data_dir) -> CompetitionDataset`.

- [ ] **Step 1: Write failing loader tests**

Create `tests/test_data.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from wl_ood.data import DatasetPaths, load_dataset


FILENAMES = {
    "training": "WIDE12H_bin2_2arcmin_kappa_newrealization.npy",
    "test": "WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy",
    "mask": "WIDE12H_bin2_2arcmin_mask.npy",
    "labels": "label_newrealization.npy",
}


def write_tiny_dataset(root: Path) -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    training = np.arange(2 * 3 * 3, dtype=np.float16).reshape(2, 3, 3)
    test = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    labels = np.zeros((2, 3, 5), dtype=np.float64)
    labels[0, :, :2] = [0.2, 0.7]
    labels[1, :, :2] = [0.3, 0.8]
    np.save(root / FILENAMES["training"], training)
    np.save(root / FILENAMES["test"], test)
    np.save(root / FILENAMES["mask"], mask)
    np.save(root / FILENAMES["labels"], labels)


def test_paths_report_missing_files(tmp_path: Path) -> None:
    paths = DatasetPaths.from_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing competition data"):
        paths.require_all()


def test_loader_opens_every_array_as_memmap(tmp_path: Path) -> None:
    write_tiny_dataset(tmp_path)
    dataset = load_dataset(tmp_path)

    assert isinstance(dataset.training, np.memmap)
    assert isinstance(dataset.test, np.memmap)
    assert isinstance(dataset.mask, np.memmap)
    assert isinstance(dataset.labels, np.memmap)
    assert dataset.training.shape == (2, 3, 3)
    assert dataset.test.shape == (4, 3)
    assert dataset.mask.shape == (2, 3)
    assert dataset.labels.shape == (2, 3, 5)
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_data.py -v
```

Expected: collection fails because `wl_ood.data` does not exist.

- [ ] **Step 3: Implement the loader**

Create `src/wl_ood/data.py`:

```python
"""Memory-safe access to the FAIR Universe Phase-2 NumPy arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAINING_FILENAME = "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"
TEST_FILENAME = "WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy"
MASK_FILENAME = "WIDE12H_bin2_2arcmin_mask.npy"
LABELS_FILENAME = "label_newrealization.npy"


@dataclass(frozen=True)
class DatasetSchema:
    training_shape: tuple[int, int, int]
    test_shape: tuple[int, int]
    mask_shape: tuple[int, int]
    labels_shape: tuple[int, int, int]
    training_dtype: np.dtype
    test_dtype: np.dtype
    mask_dtype: np.dtype
    labels_dtype: np.dtype
    valid_pixels: int
    unique_cosmologies: int


PHASE2_SCHEMA = DatasetSchema(
    training_shape=(101, 256, 132_019),
    test_shape=(10_000, 132_019),
    mask_shape=(1424, 176),
    labels_shape=(101, 256, 5),
    training_dtype=np.dtype("float16"),
    test_dtype=np.dtype("float64"),
    mask_dtype=np.dtype("bool"),
    labels_dtype=np.dtype("float64"),
    valid_pixels=132_019,
    unique_cosmologies=101,
)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    training: Path
    test: Path
    mask: Path
    labels: Path

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "DatasetPaths":
        root = Path(data_dir)
        return cls(
            root=root,
            training=root / TRAINING_FILENAME,
            test=root / TEST_FILENAME,
            mask=root / MASK_FILENAME,
            labels=root / LABELS_FILENAME,
        )

    def require_all(self) -> None:
        missing = [
            path.name
            for path in (self.training, self.test, self.mask, self.labels)
            if not path.is_file()
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"Missing competition data: {joined}")


@dataclass(frozen=True)
class CompetitionDataset:
    paths: DatasetPaths
    training: np.memmap
    test: np.memmap
    mask: np.memmap
    labels: np.memmap


def load_dataset(data_dir: str | Path) -> CompetitionDataset:
    """Open all competition arrays read-only without loading them into RAM."""
    paths = DatasetPaths.from_dir(data_dir)
    paths.require_all()
    options = {"mmap_mode": "r", "allow_pickle": False}
    return CompetitionDataset(
        paths=paths,
        training=np.load(paths.training, **options),
        test=np.load(paths.test, **options),
        mask=np.load(paths.mask, **options),
        labels=np.load(paths.labels, **options),
    )
```

Create `tests/__init__.py`:

```python
"""Project test package."""
```

- [ ] **Step 4: Run loader and regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_data.py -v
.venv/bin/python -m pytest
```

Expected: loader tests and the full suite pass.

- [ ] **Step 5: Commit**

```bash
git add src/wl_ood/data.py tests/__init__.py tests/test_data.py
git commit -m "feat: add memory-mapped competition dataset loader"
```

---

### Task 3: Structural Audit, Bounded Scan, and Fingerprints

**Files:**
- Create: `src/wl_ood/audit.py`
- Create: `tests/test_audit.py`

**Interfaces:**
- Produces: `count_non_finite(array, chunk_rows=16) -> int`.
- Produces: `sha256_file(path, block_size=8_388_608) -> str`.
- Produces: `validate_dataset(dataset, schema) -> dict`.
- Produces: `audit_dataset(data_dir, schema=PHASE2_SCHEMA, chunk_rows=16) -> dict`.
- Produces: `write_audit_report(report, output_path) -> Path`.

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_audit.py`:

```python
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wl_ood.audit import (
    audit_dataset,
    count_non_finite,
    sha256_file,
    validate_dataset,
    write_audit_report,
)
from wl_ood.data import DatasetSchema, load_dataset
from tests.test_data import FILENAMES, write_tiny_dataset


TINY_SCHEMA = DatasetSchema(
    training_shape=(2, 3, 3),
    test_shape=(4, 3),
    mask_shape=(2, 3),
    labels_shape=(2, 3, 5),
    training_dtype=np.dtype("float16"),
    test_dtype=np.dtype("float64"),
    mask_dtype=np.dtype("bool"),
    labels_dtype=np.dtype("float64"),
    valid_pixels=3,
    unique_cosmologies=2,
)


def test_count_non_finite_finds_late_chunk_value() -> None:
    values = np.zeros((7, 3), dtype=np.float64)
    values[6, 2] = np.nan
    assert count_non_finite(values, chunk_rows=2) == 1


def test_sha256_matches_standard_library(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"weak-lensing-audit")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path, block_size=3) == expected


def test_validate_tiny_dataset(tmp_path: Path) -> None:
    write_tiny_dataset(tmp_path)
    result = validate_dataset(load_dataset(tmp_path), TINY_SCHEMA, chunk_rows=2)
    assert result["mask"]["valid_pixels"] == 3
    assert result["labels"]["unique_cosmologies"] == 2
    assert result["non_finite_counts"] == {"training": 0, "test": 0}


def test_validate_rejects_shape_mismatch(tmp_path: Path) -> None:
    write_tiny_dataset(tmp_path)
    np.save(tmp_path / FILENAMES["test"], np.zeros((5, 3)))
    with pytest.raises(ValueError, match="test shape"):
        validate_dataset(load_dataset(tmp_path), TINY_SCHEMA)


def test_audit_and_report_schema(tmp_path: Path) -> None:
    write_tiny_dataset(tmp_path)
    report = audit_dataset(tmp_path, schema=TINY_SCHEMA, chunk_rows=2)
    output = write_audit_report(report, tmp_path / "audit.json")
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert set(loaded["files"]) == {"training", "test", "mask", "labels"}
    assert all(len(item["sha256"]) == 64 for item in loaded["files"].values())
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_audit.py -v
```

Expected: collection fails because `wl_ood.audit` does not exist.

- [ ] **Step 3: Implement audit functions**

Create `src/wl_ood/audit.py`:

```python
"""Structural integrity audit for competition arrays."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from wl_ood.data import (
    PHASE2_SCHEMA,
    CompetitionDataset,
    DatasetSchema,
    load_dataset,
)


def count_non_finite(array: np.ndarray, *, chunk_rows: int = 16) -> int:
    """Count non-finite values while reading bounded leading-row chunks."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    rows = array.reshape(-1, array.shape[-1])
    total = 0
    for start in range(0, rows.shape[0], chunk_rows):
        chunk = rows[start : start + chunk_rows]
        total += int(np.count_nonzero(~np.isfinite(chunk)))
    return total


def sha256_file(path: str | Path, *, block_size: int = 8_388_608) -> str:
    """Hash a file without reading it all into memory."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _expect(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected}, found {actual}")


def validate_dataset(
    dataset: CompetitionDataset,
    schema: DatasetSchema,
    *,
    chunk_rows: int = 16,
) -> dict[str, Any]:
    """Validate structural facts without computing test distribution summaries."""
    _expect("training shape", dataset.training.shape, schema.training_shape)
    _expect("test shape", dataset.test.shape, schema.test_shape)
    _expect("mask shape", dataset.mask.shape, schema.mask_shape)
    _expect("labels shape", dataset.labels.shape, schema.labels_shape)
    _expect("training dtype", dataset.training.dtype, schema.training_dtype)
    _expect("test dtype", dataset.test.dtype, schema.test_dtype)
    _expect("mask dtype", dataset.mask.dtype, schema.mask_dtype)
    _expect("labels dtype", dataset.labels.dtype, schema.labels_dtype)

    valid_pixels = int(np.count_nonzero(dataset.mask))
    _expect("mask valid pixels", valid_pixels, schema.valid_pixels)
    _expect("training feature count", dataset.training.shape[-1], valid_pixels)
    _expect("test feature count", dataset.test.shape[-1], valid_pixels)

    if not np.all(np.isfinite(dataset.labels)):
        raise ValueError("labels contain non-finite values")
    cosmology_constant = bool(
        np.all(dataset.labels[:, :, :2] == dataset.labels[:, :1, :2])
    )
    if not cosmology_constant:
        raise ValueError("cosmology labels vary across realizations")
    unique_cosmologies = int(
        np.unique(dataset.labels[:, :, :2].reshape(-1, 2), axis=0).shape[0]
    )
    _expect(
        "unique cosmologies",
        unique_cosmologies,
        schema.unique_cosmologies,
    )

    non_finite_counts = {
        "training": count_non_finite(dataset.training, chunk_rows=chunk_rows),
        "test": count_non_finite(dataset.test, chunk_rows=chunk_rows),
    }
    if any(non_finite_counts.values()):
        raise ValueError(f"map arrays contain non-finite values: {non_finite_counts}")

    return {
        "arrays": {
            "training": {
                "shape": list(dataset.training.shape),
                "dtype": str(dataset.training.dtype),
            },
            "test": {
                "shape": list(dataset.test.shape),
                "dtype": str(dataset.test.dtype),
            },
            "mask": {
                "shape": list(dataset.mask.shape),
                "dtype": str(dataset.mask.dtype),
            },
            "labels": {
                "shape": list(dataset.labels.shape),
                "dtype": str(dataset.labels.dtype),
            },
        },
        "mask": {
            "valid_pixels": valid_pixels,
            "masked_pixels": int(dataset.mask.size - valid_pixels),
        },
        "labels": {
            "unique_cosmologies": unique_cosmologies,
            "cosmology_constant_per_group": cosmology_constant,
            "all_finite": True,
        },
        "non_finite_counts": non_finite_counts,
    }


def audit_dataset(
    data_dir: str | Path,
    *,
    schema: DatasetSchema = PHASE2_SCHEMA,
    chunk_rows: int = 16,
) -> dict[str, Any]:
    """Validate and fingerprint a dataset directory."""
    dataset = load_dataset(data_dir)
    validation = validate_dataset(dataset, schema, chunk_rows=chunk_rows)
    file_paths = {
        "training": dataset.paths.training,
        "test": dataset.paths.test,
        "mask": dataset.paths.mask,
        "labels": dataset.paths.labels,
    }
    files = {
        role: {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in file_paths.items()
    }
    return {"schema_version": 1, "files": files, **validation}


def write_audit_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write an audit report with deterministic key ordering."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 4: Run audit and regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_audit.py -v
.venv/bin/python -m pytest
```

Expected: audit tests and the full suite pass.

- [ ] **Step 5: Commit**

```bash
git add src/wl_ood/audit.py tests/test_audit.py
git commit -m "feat: add bounded structural data audit"
```

---

### Task 4: Map Reconstruction and Fixed Previews

**Files:**
- Create: `src/wl_ood/maps.py`
- Create: `tests/test_maps.py`

**Interfaces:**
- Produces: `reconstruct_map(values, mask, fill_value=np.nan) -> np.ndarray`.
- Produces: `save_map_preview(values, mask, output_path, title) -> Path`.
- Produces: `save_mask_preview(mask, output_path) -> Path`.
- Produces: `generate_fixed_previews(dataset, output_dir) -> list[Path]`.

- [ ] **Step 1: Write failing map tests**

Create `tests/test_maps.py`:

```python
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from wl_ood.data import load_dataset
from wl_ood.maps import (
    generate_fixed_previews,
    reconstruct_map,
    save_map_preview,
)
from tests.test_data import write_tiny_dataset


def test_reconstruct_map_preserves_boolean_index_order() -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    result = reconstruct_map(np.array([10.0, 20.0, 30.0]), mask)
    np.testing.assert_array_equal(result[mask], [10.0, 20.0, 30.0])
    assert np.all(np.isnan(result[~mask]))


def test_reconstruct_map_rejects_wrong_value_count() -> None:
    mask = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError, match="2 valid pixels"):
        reconstruct_map(np.array([1.0]), mask)


def test_save_map_preview_creates_png(tmp_path: Path) -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    output = save_map_preview(
        np.array([1.0, 2.0, 3.0]),
        mask,
        tmp_path / "map.png",
        title="Tiny map",
    )
    image = mpimg.imread(output)
    assert output.is_file()
    assert image.ndim == 3


def test_generate_fixed_previews_with_explicit_indices(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_tiny_dataset(data_dir)
    outputs = generate_fixed_previews(
        load_dataset(data_dir),
        tmp_path / "previews",
        training_indices=((0, 0), (1, 2)),
        test_indices=(0, 3),
    )
    assert [path.name for path in outputs] == [
        "train-c000-r000.png",
        "train-c001-r002.png",
        "test-00000.png",
        "test-00003.png",
        "mask.png",
    ]
    assert all(path.is_file() for path in outputs)
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_maps.py -v
```

Expected: collection fails because `wl_ood.maps` does not exist.

- [ ] **Step 3: Implement reconstruction and previews**

Create `src/wl_ood/maps.py`:

```python
"""Reconstruct and render individual masked convergence maps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wl_ood.data import CompetitionDataset


def reconstruct_map(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Restore one flattened valid-pixel vector to the survey mask layout."""
    vector = np.asarray(values)
    mask_array = np.asarray(mask, dtype=bool)
    expected = int(np.count_nonzero(mask_array))
    if vector.ndim != 1 or vector.size != expected:
        raise ValueError(
            f"values must be one-dimensional with {expected} valid pixels"
        )
    dtype = np.result_type(vector.dtype, np.float32)
    full = np.full(mask_array.shape, fill_value, dtype=dtype)
    full[mask_array] = vector
    return full


def save_map_preview(
    values: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Render one map using colour limits calculated from that map alone."""
    full = reconstruct_map(values, mask)
    valid = full[np.asarray(mask, dtype=bool)]
    vmin, vmax = np.percentile(valid, [1.0, 99.0])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 10))
    image = axis.imshow(full, cmap="magma", vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label="kappa")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_mask_preview(mask: np.ndarray, output_path: str | Path) -> Path:
    """Render the fixed survey mask."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 10))
    axis.imshow(mask, cmap="gray", vmin=0, vmax=1)
    axis.set_title("WIDE12H survey mask")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_fixed_previews(
    dataset: CompetitionDataset,
    output_dir: str | Path,
    *,
    training_indices: tuple[tuple[int, int], ...] = ((0, 0), (100, 255)),
    test_indices: tuple[int, ...] = (0, 9999),
) -> list[Path]:
    """Render predeclared individual maps and the survey mask."""
    root = Path(output_dir)
    outputs: list[Path] = []
    for cosmology, realization in training_indices:
        outputs.append(
            save_map_preview(
                dataset.training[cosmology, realization],
                dataset.mask,
                root / f"train-c{cosmology:03d}-r{realization:03d}.png",
                title=f"Training cosmology {cosmology}, realization {realization}",
            )
        )
    for index in test_indices:
        outputs.append(
            save_map_preview(
                dataset.test[index],
                dataset.mask,
                root / f"test-{index:05d}.png",
                title=f"Test map {index}",
            )
        )
    outputs.append(save_mask_preview(dataset.mask, root / "mask.png"))
    return outputs
```

- [ ] **Step 4: Run map and regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_maps.py -v
.venv/bin/python -m pytest
```

Expected: map tests and the full suite pass.

- [ ] **Step 5: Commit**

```bash
git add src/wl_ood/maps.py tests/test_maps.py
git commit -m "feat: reconstruct and preview masked maps"
```

---

### Task 5: Audit and Preview CLI Commands

**Files:**
- Modify: `src/wl_ood/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `tests/test_documentation.py`

**Interfaces:**
- Produces: `wl-ood audit-data --data-dir PATH --output PATH --chunk-rows N`.
- Produces: `wl-ood preview-data --data-dir PATH --output-dir PATH`.

- [ ] **Step 1: Add failing CLI and documentation tests**

Append to `tests/test_cli.py`:

```python
from tests.test_audit import TINY_SCHEMA
from tests.test_data import write_tiny_dataset


def test_audit_data_command(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_tiny_dataset(data_dir)
    output = tmp_path / "audit.json"
    assert main(
        [
            "audit-data",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--chunk-rows",
            "2",
        ],
        schema=TINY_SCHEMA,
    ) == 0
    assert output.is_file()
    assert "Audit passed" in capsys.readouterr().out


def test_preview_data_command(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_tiny_dataset(data_dir)
    output_dir = tmp_path / "previews"
    assert main(
        [
            "preview-data",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ],
        preview_training_indices=((0, 0), (1, 2)),
        preview_test_indices=(0, 3),
    ) == 0
    assert len(list(output_dir.glob("*.png"))) == 5
    assert "5 previews" in capsys.readouterr().out
```

Append to `tests/test_documentation.py`:

```python
def test_readme_documents_flattened_data_and_audit_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "132,019 valid mask pixels" in readme
    assert "wl-ood audit-data" in readme
    assert "wl-ood preview-data" in readme
```

- [ ] **Step 2: Run focused tests and confirm failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_documentation.py -v
```

Expected: failures because the commands, injection parameters, and documentation do not exist.

- [ ] **Step 3: Extend the CLI**

Update `src/wl_ood/cli.py` imports:

```python
from wl_ood.audit import audit_dataset, write_audit_report
from wl_ood.data import PHASE2_SCHEMA, DatasetSchema, load_dataset
from wl_ood.maps import generate_fixed_previews
```

Add parsers inside `_parser()` before `return parser`:

```python
    audit = commands.add_parser("audit-data")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--chunk-rows", type=int, default=16)

    preview = commands.add_parser("preview-data")
    preview.add_argument("--data-dir", type=Path, required=True)
    preview.add_argument("--output-dir", type=Path, required=True)
```

Change `main` to this signature:

```python
def main(
    argv: Sequence[str] | None = None,
    *,
    schema: DatasetSchema = PHASE2_SCHEMA,
    preview_training_indices: tuple[tuple[int, int], ...] = ((0, 0), (100, 255)),
    preview_test_indices: tuple[int, ...] = (0, 9999),
) -> int:
```

Add these branches after `args = _parser().parse_args(argv)` and before `make-dummy`:

```python
    if args.command == "audit-data":
        report = audit_dataset(
            args.data_dir,
            schema=schema,
            chunk_rows=args.chunk_rows,
        )
        output = write_audit_report(report, args.output)
        print(f"Audit passed; report written to {output}")
        return 0

    if args.command == "preview-data":
        outputs = generate_fixed_previews(
            load_dataset(args.data_dir),
            args.output_dir,
            training_indices=preview_training_indices,
            test_indices=preview_test_indices,
        )
        print(f"Wrote {len(outputs)} previews to {args.output_dir}")
        return 0
```

- [ ] **Step 4: Document the data workflow**

Append to `README.md`:

```markdown
## Competition data

Place the four organizer-provided `.npy` files in `public_data/`. This directory
is ignored by Git.

Although the documented sky grid is `1424 × 176`, each stored map contains only
the 132,019 valid mask pixels. The loader reconstructs individual full maps
using the boolean survey mask and never loads the complete training or test
arrays into RAM.

Run the structural audit:

```bash
.venv/bin/wl-ood audit-data \
  --data-dir public_data \
  --output artifacts/data-audit.json
```

Create the fixed individual-map previews:

```bash
.venv/bin/wl-ood preview-data \
  --data-dir public_data \
  --output-dir artifacts/previews
```

The audit verifies test structure and finiteness but deliberately does not
calculate test-wide distribution statistics.
```

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_documentation.py -v
.venv/bin/python -m pytest
```

Expected: focused tests and the full suite pass.

- [ ] **Step 6: Commit**

```bash
git add src/wl_ood/cli.py tests/test_cli.py README.md tests/test_documentation.py
git commit -m "feat: expose reproducible data audit workflow"
```

---

### Task 6: Run and Verify the Real Data Audit

**Files:**
- Generated, ignored: `artifacts/data-audit.json`
- Generated, ignored: `artifacts/previews/train-c000-r000.png`
- Generated, ignored: `artifacts/previews/train-c100-r255.png`
- Generated, ignored: `artifacts/previews/test-00000.png`
- Generated, ignored: `artifacts/previews/test-09999.png`
- Generated, ignored: `artifacts/previews/mask.png`

**Interfaces:**
- Consumes: real arrays under `public_data/`.
- Produces: verified report and five visually inspected previews.

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: every test passes.

- [ ] **Step 2: Run the real structural audit**

Run:

```bash
.venv/bin/wl-ood audit-data \
  --data-dir public_data \
  --output artifacts/data-audit.json \
  --chunk-rows 16
```

Expected: `Audit passed`; the command may take several minutes because it streams and hashes approximately 16 GB.

- [ ] **Step 3: Inspect the report without printing test distributions**

Run:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; r=json.loads(Path("artifacts/data-audit.json").read_text()); print("schema",r["schema_version"]); print("non_finite",r["non_finite_counts"]); print("valid_pixels",r["mask"]["valid_pixels"]); print("unique_cosmologies",r["labels"]["unique_cosmologies"]); print("files",[(k,v["size_bytes"],v["sha256"]) for k,v in r["files"].items()])'
```

Expected: schema `1`, zero non-finite counts, 132,019 valid pixels, 101 unique cosmologies, and four 64-character hashes.

- [ ] **Step 4: Generate fixed previews**

Run:

```bash
.venv/bin/wl-ood preview-data \
  --data-dir public_data \
  --output-dir artifacts/previews
```

Expected: five PNG files are reported.

- [ ] **Step 5: Verify preview files and dimensions**

Run:

```bash
.venv/bin/python -c 'from pathlib import Path; from PIL import Image; files=sorted(Path("artifacts/previews").glob("*.png")); print([(p.name, Image.open(p).size) for p in files])'
```

Expected: exactly five non-empty PNGs with stable pixel dimensions.

- [ ] **Step 6: Visually inspect every preview**

Open each PNG with the local image viewer and confirm:

- the mask shape is consistent across maps;
- valid regions contain spatial structure;
- excluded regions are blank;
- images are not transposed or scrambled;
- training and test maps render without obvious file-decoding corruption.

- [ ] **Step 7: Verify repository cleanliness**

Run:

```bash
git status --short --branch
git check-ignore public_data/WIDE12H_bin2_2arcmin_mask.npy
git check-ignore artifacts/data-audit.json
```

Expected: `main` is clean; both raw data and generated report are ignored.

---

## Self-Review Results

- **Spec coverage:** Tasks cover raw-data privacy, exact local schemas, memory mapping, bounded finiteness scans, streaming hashes, map reconstruction, fixed previews, CLI commands, documentation, real-data integration, and visual verification. No test-wide distribution statistic is computed.
- **Placeholder scan:** No incomplete implementation marker or replacement token remains.
- **Type consistency:** `DatasetSchema`, `CompetitionDataset`, `load_dataset`, `audit_dataset`, `write_audit_report`, `reconstruct_map`, `generate_fixed_previews`, and injected CLI test parameters have consistent names and signatures across tasks.
