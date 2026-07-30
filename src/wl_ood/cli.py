"""Command-line entry points for data checks, metrics, and submissions."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from wl_ood.audit import audit_dataset, write_audit_report
from wl_ood.data import PHASE2_SCHEMA, DatasetSchema, load_dataset
from wl_ood.maps import generate_fixed_previews
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

    audit = commands.add_parser("audit-data")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--chunk-rows", type=int, default=16)

    preview = commands.add_parser("preview-data")
    preview.add_argument("--data-dir", type=Path, required=True)
    preview.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    schema: DatasetSchema = PHASE2_SCHEMA,
    preview_training_indices: tuple[tuple[int, int], ...] = ((0, 0), (100, 255)),
    preview_test_indices: tuple[int, ...] = (0, 9999),
) -> int:
    args = _parser().parse_args(argv)

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
