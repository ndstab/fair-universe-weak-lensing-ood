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
