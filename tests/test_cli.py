import json
import zipfile

import numpy as np
import pytest

from tests.test_audit import TINY_SCHEMA
from tests.test_data import write_tiny_dataset
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
