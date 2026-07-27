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
