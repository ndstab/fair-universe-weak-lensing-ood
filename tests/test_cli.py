import json
import zipfile

import numpy as np
import pytest

from tests.test_audit import TINY_SCHEMA
from tests.test_data import write_tiny_dataset
from wl_ood.cli import main
from wl_ood.data import (
    LABELS_FILENAME,
    MASK_FILENAME,
    TEST_FILENAME,
    TRAINING_FILENAME,
)
from wl_ood.ps_config import PowerSpectrumConfig


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


def write_tiny_ps_dataset(root) -> None:
    rng = np.random.default_rng(2026)
    np.save(root / MASK_FILENAME, np.ones((8, 8), dtype=bool))
    np.save(
        root / TRAINING_FILENAME,
        rng.normal(size=(4, 15, 64)).astype(np.float64),
    )
    np.save(
        root / TEST_FILENAME,
        rng.normal(size=(3, 64)).astype(np.float64),
    )
    labels = np.zeros((4, 15, 5), dtype=np.float64)
    labels[:, :, :2] = np.array(
        [[0.2, 0.7], [0.2, 0.9], [0.4, 0.7], [0.4, 0.9]],
    )[:, None, :]
    np.save(root / LABELS_FILENAME, labels)


def tiny_cli_ps_config() -> PowerSpectrumConfig:
    return PowerSpectrumConfig(
        seed=17,
        pixel_size_arcmin=60.0 * 180.0 / np.pi,
        ell_min=0.5,
        ell_max=4.0,
        n_bins=3,
    )


def test_benchmark_ps_baseline_command(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_tiny_ps_dataset(data_dir)
    run_dir = tmp_path / "run"

    assert main(
        [
            "benchmark-ps-baseline",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--seed",
            "17",
        ],
        ps_config=tiny_cli_ps_config(),
        ps_benchmark_count=2,
    ) == 0

    assert (run_dir / "benchmark.json").is_file()
    assert "benchmark" in capsys.readouterr().out.lower()


def test_validate_and_run_ps_baseline_commands(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_tiny_ps_dataset(data_dir)
    run_dir = tmp_path / "run"
    config = tiny_cli_ps_config()

    assert main(
        [
            "validate-ps-baseline",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--seed",
            "17",
        ],
        ps_config=config,
    ) == 0
    submission = tmp_path / "baseline.zip"
    assert main(
        [
            "run-ps-baseline",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--seed",
            "17",
            "--submission",
            str(submission),
        ],
        ps_config=config,
        ps_expected_test_count=3,
    ) == 0

    assert (run_dir / "validation-report.json").is_file()
    assert (run_dir / "run-manifest.json").is_file()
    assert submission.is_file()
    output = capsys.readouterr().out.lower()
    assert "validation" in output
    assert "submission" in output
