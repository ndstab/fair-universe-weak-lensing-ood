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
    _expect("unique cosmologies", unique_cosmologies, schema.unique_cosmologies)

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
