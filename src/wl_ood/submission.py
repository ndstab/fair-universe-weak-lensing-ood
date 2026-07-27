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
