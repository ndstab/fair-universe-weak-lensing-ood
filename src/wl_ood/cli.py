"""Command-line entry points for data checks, metrics, and submissions."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from wl_ood.audit import audit_dataset, sha256_file, write_audit_report
from wl_ood.data import PHASE2_SCHEMA, DatasetSchema, load_dataset
from wl_ood.feature_cache import FeatureCache, require_run_config
from wl_ood.maps import generate_fixed_previews
from wl_ood.metric import compute_low_fpr_metric
from wl_ood.ps_config import PowerSpectrumConfig
from wl_ood.ps_pipeline import (
    benchmark_features,
    extract_test_features,
    extract_training_features,
    run_full_baseline,
    run_validation,
)
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

    audit = commands.add_parser("audit-data")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--chunk-rows", type=int, default=16)

    preview = commands.add_parser("preview-data")
    preview.add_argument("--data-dir", type=Path, required=True)
    preview.add_argument("--output-dir", type=Path, required=True)

    benchmark_ps = commands.add_parser("benchmark-ps-baseline")
    benchmark_ps.add_argument("--data-dir", type=Path, required=True)
    benchmark_ps.add_argument("--run-dir", type=Path, required=True)
    benchmark_ps.add_argument("--seed", type=int, default=20260730)

    validate_ps = commands.add_parser("validate-ps-baseline")
    validate_ps.add_argument("--data-dir", type=Path, required=True)
    validate_ps.add_argument("--run-dir", type=Path, required=True)
    validate_ps.add_argument("--seed", type=int, default=20260730)

    run_ps = commands.add_parser("run-ps-baseline")
    run_ps.add_argument("--data-dir", type=Path, required=True)
    run_ps.add_argument("--run-dir", type=Path, required=True)
    run_ps.add_argument("--seed", type=int, default=20260730)
    run_ps.add_argument("--submission", type=Path, required=True)
    return parser


def _baseline_config(
    template: PowerSpectrumConfig | None,
    seed: int,
) -> PowerSpectrumConfig:
    return replace(template or PowerSpectrumConfig(), seed=seed)


def _require_baseline_run_config(
    dataset,
    run_dir: Path,
    config: PowerSpectrumConfig,
) -> Path:
    files = {
        "training": dataset.paths.training,
        "test": dataset.paths.test,
        "mask": dataset.paths.mask,
        "labels": dataset.paths.labels,
    }
    payload = {
        "baseline": config.to_dict(),
        "dataset": {
            role: {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in files.items()
        },
        "shapes": {
            "training": list(dataset.training.shape),
            "test": list(dataset.test.shape),
            "mask": list(dataset.mask.shape),
            "labels": list(dataset.labels.shape),
        },
        "valid_pixels": int(np.count_nonzero(dataset.mask)),
    }
    return require_run_config(run_dir, payload)


def main(
    argv: Sequence[str] | None = None,
    *,
    schema: DatasetSchema = PHASE2_SCHEMA,
    preview_training_indices: tuple[tuple[int, int], ...] = ((0, 0), (100, 255)),
    preview_test_indices: tuple[int, ...] = (0, 9999),
    ps_config: PowerSpectrumConfig | None = None,
    ps_benchmark_count: int = 32,
    ps_expected_test_count: int = 10_000,
) -> int:
    args = _parser().parse_args(argv)

    if args.command == "benchmark-ps-baseline":
        dataset = load_dataset(args.data_dir)
        config = _baseline_config(ps_config, args.seed)
        _require_baseline_run_config(dataset, args.run_dir, config)
        report = benchmark_features(
            dataset,
            config,
            count=ps_benchmark_count,
        )
        output = write_audit_report(report, args.run_dir / "benchmark.json")
        print(f"CPU benchmark written to {output}")
        return 0

    if args.command == "validate-ps-baseline":
        dataset = load_dataset(args.data_dir)
        config = _baseline_config(ps_config, args.seed)
        _require_baseline_run_config(dataset, args.run_dir, config)
        train_cache = FeatureCache.open(
            args.run_dir,
            "train",
            (*dataset.training.shape[:2], config.n_bins),
        )
        extracted = extract_training_features(dataset, train_cache, config)
        output = run_validation(dataset, train_cache, args.run_dir, config)
        print(
            f"Baseline validation written to {output}; "
            f"extracted {extracted} new training rows"
        )
        return 0

    if args.command == "run-ps-baseline":
        dataset = load_dataset(args.data_dir)
        config = _baseline_config(ps_config, args.seed)
        _require_baseline_run_config(dataset, args.run_dir, config)
        validation_path = args.run_dir / "validation-report.json"
        if not validation_path.is_file():
            raise FileNotFoundError(
                "run validate-ps-baseline before public-test inference"
            )
        train_cache = FeatureCache.open(
            args.run_dir,
            "train",
            (*dataset.training.shape[:2], config.n_bins),
        )
        test_cache = FeatureCache.open(
            args.run_dir,
            "test",
            (dataset.test.shape[0], config.n_bins),
        )
        train_extracted = extract_training_features(dataset, train_cache, config)
        test_extracted = extract_test_features(dataset, test_cache, config)
        output = run_full_baseline(
            dataset,
            train_cache,
            test_cache,
            args.run_dir,
            config,
            submission_path=args.submission,
            expected_test_count=ps_expected_test_count,
        )
        print(
            f"Baseline submission written to {output}; "
            f"extracted {train_extracted} training and "
            f"{test_extracted} test rows"
        )
        return 0

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
