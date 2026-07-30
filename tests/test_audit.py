import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_data import FILENAMES, write_tiny_dataset
from wl_ood.audit import (
    audit_dataset,
    count_non_finite,
    sha256_file,
    validate_dataset,
    write_audit_report,
)
from wl_ood.data import DatasetSchema, load_dataset


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
