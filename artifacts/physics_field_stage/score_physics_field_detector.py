#!/usr/bin/env python3
"""Score Phase-2 maps and package physics, typicality, and union submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

try:
    from .baseline_global_stats import write_submission
    from .generate_smoke_submission import validate_archive
    from .wl_physics_field import (
        SHAPE_NOISE_STD,
        ObservedPatchAccessor,
        PhysicsFieldDetector,
        deterministic_tiles,
    )
except ImportError:  # Direct execution via ``python scripts/...``.
    from baseline_global_stats import write_submission
    from generate_smoke_submission import validate_archive
    from wl_physics_field import (
        SHAPE_NOISE_STD,
        ObservedPatchAccessor,
        PhysicsFieldDetector,
        deterministic_tiles,
    )


EXPECTED_TEST_MAPS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-maps", type=Path, required=True)
    parser.add_argument("--test-maps", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-maps", type=int, default=1024)
    parser.add_argument("--held-realization-fold", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maps-per-chunk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080303)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_cdf_score(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Frozen-calibration upper-tail score; no statistic is read from ``values``."""

    reference = np.sort(np.asarray(calibration, dtype=np.float64))
    query = np.asarray(values, dtype=np.float64)
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError("calibration must be a finite non-empty vector")
    if not np.isfinite(query).all():
        raise ValueError("query contains non-finite values")
    ranks = np.searchsorted(reference, query, side="right").astype(np.float64)
    return (ranks + 0.5) / (len(reference) + 1.0)


def fit_embedding_typicality(
    calibration: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a shrinkage Mahalanobis field-embedding typicality score."""

    reference = np.asarray(calibration, dtype=np.float64)
    if reference.ndim != 2 or len(reference) < 3:
        raise ValueError("embedding calibration must contain at least three rows")
    estimator = LedoitWolf(store_precision=True, assume_centered=False).fit(reference)
    centre = estimator.location_.copy()
    precision = estimator.precision_.copy()

    def score(values: np.ndarray) -> np.ndarray:
        rows = np.asarray(values, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != reference.shape[1]:
            raise ValueError("embedding dimension differs from calibration")
        delta = rows - centre
        return np.einsum("ni,ij,nj->n", delta, precision, delta)

    return score


def stratified_calibration_rows(
    *, cosmologies: int, realizations: int, held_fold: int, count: int
) -> np.ndarray:
    """Round-robin held-realization rows across all cosmologies."""

    held_realizations = np.flatnonzero(np.arange(realizations) % 4 == held_fold)
    ordered = []
    for realization in held_realizations:
        ordered.extend(
            cosmology * realizations + int(realization)
            for cosmology in range(cosmologies)
        )
    if count > len(ordered):
        raise ValueError(f"requested {count} calibration rows, only {len(ordered)} available")
    return np.asarray(ordered[:count], dtype=np.int64)


@torch.inference_mode()
def score_maps(
    model: PhysicsFieldDetector,
    maps: np.ndarray,
    rows: np.ndarray,
    accessor: ObservedPatchAccessor,
    *,
    add_shape_noise: bool,
    noise_seed: int,
    batch_size: int,
    maps_per_chunk: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score each map independently with frozen tiles and return map summaries."""

    model.eval()
    selected_rows = np.asarray(rows, dtype=np.int64)
    tile_count = len(accessor.lookup)
    embedding_dim = model.embedding_dim
    patch_logits = np.empty((len(selected_rows), tile_count), dtype=np.float32)
    map_embeddings = np.empty((len(selected_rows), embedding_dim), dtype=np.float32)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    for chunk_start in range(0, len(selected_rows), maps_per_chunk):
        chunk_rows = selected_rows[chunk_start : chunk_start + maps_per_chunk]
        patch_parts = []
        mask_parts = []
        for row in chunk_rows:
            patches, masks = accessor.extract_all(np.asarray(maps[int(row)]))
            patch_parts.append(patches)
            mask_parts.append(masks)
        patches_np = np.concatenate(patch_parts)
        masks_np = np.concatenate(mask_parts)
        chunk_logits = []
        chunk_embeddings = []
        for start in range(0, len(patches_np), batch_size):
            values = torch.as_tensor(patches_np[start : start + batch_size], device=device)[:, None]
            masks = torch.as_tensor(masks_np[start : start + batch_size], device=device)[:, None]
            if add_shape_noise:
                values = values + SHAPE_NOISE_STD * torch.randn(
                    values.shape, generator=generator, device=device
                ) * masks
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(values, masks)
            chunk_logits.append(output["logit"].float().cpu().numpy())
            chunk_embeddings.append(output["embedding"].float().cpu().numpy())
        logits = np.concatenate(chunk_logits).reshape(len(chunk_rows), tile_count)
        embeddings = np.concatenate(chunk_embeddings).reshape(
            len(chunk_rows), tile_count, embedding_dim
        )
        stop = chunk_start + len(chunk_rows)
        patch_logits[chunk_start:stop] = logits
        map_embeddings[chunk_start:stop] = embeddings.mean(axis=1)
        if stop % 100 == 0 or stop == len(selected_rows):
            print(
                json.dumps(
                    {
                        "stage": "field_scoring",
                        "maps_scored": stop,
                        "maps_total": len(selected_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    tail_count = max(1, int(math.ceil(0.10 * tile_count)))
    return {
        "mean": patch_logits.mean(axis=1, dtype=np.float64),
        "upper_quartile": np.quantile(patch_logits, 0.75, axis=1),
        "top_fraction_mean": np.partition(patch_logits, -tail_count, axis=1)[
            :, -tail_count:
        ].mean(axis=1, dtype=np.float64),
        "embedding": map_embeddings,
        "patch_logit_std": patch_logits.std(axis=1, dtype=np.float64),
    }


def summary(values: np.ndarray) -> dict[str, object]:
    scores = np.asarray(values, dtype=np.float64)
    unique, counts = np.unique(scores, return_counts=True)
    return {
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "mean": float(scores.mean()),
        "standard_deviation": float(scores.std()),
        "unique_scores": int(len(unique)),
        "largest_tie_block": int(counts.max()),
        "quantiles": np.quantile(
            scores, (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production field scoring")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unknown field-detector checkpoint format")
    model = PhysicsFieldDetector(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    mask = np.load(args.mask)
    tiles = deterministic_tiles(mask, patch_size=64, stride=32, minimum_fraction=0.75)
    accessor = ObservedPatchAccessor(mask, tiles, patch_size=64)
    train = np.load(args.train_maps, mmap_mode="r")
    test = np.load(args.test_maps, mmap_mode="r")
    if train.shape[-1] != accessor.n_observed or test.shape[-1] != accessor.n_observed:
        raise ValueError("map arrays disagree with the survey mask")
    if test.shape != (EXPECTED_TEST_MAPS, accessor.n_observed):
        raise ValueError(f"unexpected Phase-2 test shape {test.shape}")
    calibration_rows = stratified_calibration_rows(
        cosmologies=train.shape[0],
        realizations=train.shape[1],
        held_fold=args.held_realization_fold,
        count=args.calibration_maps,
    )
    train_flat = train.reshape(-1, train.shape[-1])
    calibration_a = score_maps(
        model,
        train_flat,
        calibration_rows,
        accessor,
        add_shape_noise=True,
        noise_seed=args.seed + 1,
        batch_size=args.batch_size,
        maps_per_chunk=args.maps_per_chunk,
        device=device,
    )
    calibration_b = score_maps(
        model,
        train_flat,
        calibration_rows,
        accessor,
        add_shape_noise=True,
        noise_seed=args.seed + 2,
        batch_size=args.batch_size,
        maps_per_chunk=args.maps_per_chunk,
        device=device,
    )
    test_scores = score_maps(
        model,
        test,
        np.arange(len(test), dtype=np.int64),
        accessor,
        add_shape_noise=False,
        noise_seed=args.seed + 3,
        batch_size=args.batch_size,
        maps_per_chunk=args.maps_per_chunk,
        device=device,
    )

    physics_reference = np.concatenate((calibration_a["mean"], calibration_b["mean"]))
    physics = empirical_cdf_score(physics_reference, test_scores["mean"])

    typicality_a = fit_embedding_typicality(calibration_a["embedding"])
    typicality_b = fit_embedding_typicality(calibration_b["embedding"])
    typicality_calibration_a = typicality_b(calibration_a["embedding"])
    typicality_calibration_b = typicality_a(calibration_b["embedding"])
    typicality_reference = np.concatenate(
        (typicality_calibration_a, typicality_calibration_b)
    )
    typicality_raw = 0.5 * (
        typicality_a(test_scores["embedding"])
        + typicality_b(test_scores["embedding"])
    )
    typicality = empirical_cdf_score(typicality_reference, typicality_raw)

    physics_calibration_a = empirical_cdf_score(
        calibration_b["mean"], calibration_a["mean"]
    )
    physics_calibration_b = empirical_cdf_score(
        calibration_a["mean"], calibration_b["mean"]
    )
    typicality_calibrated_a = empirical_cdf_score(
        typicality_b(calibration_b["embedding"]), typicality_calibration_a
    )
    typicality_calibrated_b = empirical_cdf_score(
        typicality_a(calibration_a["embedding"]), typicality_calibration_b
    )
    union_reference = np.concatenate(
        (
            np.maximum(physics_calibration_a, typicality_calibrated_a),
            np.maximum(physics_calibration_b, typicality_calibrated_b),
        )
    )
    union = empirical_cdf_score(union_reference, np.maximum(physics, typicality))

    heads = {
        "physics_standalone": physics,
        "field_typicality": typicality,
        "calibrated_union": union,
    }
    for name, values in heads.items():
        if values.shape != (EXPECTED_TEST_MAPS,) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid production head {name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, values in heads.items():
        archive = args.output_dir / f"physics_field_{name}.zip"
        write_submission(archive, values)
        validate_archive(archive)
        artifacts[name] = {
            "path": str(archive),
            "sha256": sha256_file(archive),
            "score": summary(values),
        }
    scores_path = args.output_dir / "scores.npz"
    np.savez_compressed(
        scores_path,
        calibration_rows=calibration_rows,
        calibration_a_mean=calibration_a["mean"],
        calibration_b_mean=calibration_b["mean"],
        calibration_a_embedding=calibration_a["embedding"],
        calibration_b_embedding=calibration_b["embedding"],
        test_mean=test_scores["mean"],
        test_upper_quartile=test_scores["upper_quartile"],
        test_top_fraction_mean=test_scores["top_fraction_mean"],
        test_patch_logit_std=test_scores["patch_logit_std"],
        test_embedding=test_scores["embedding"],
        **heads,
    )
    manifest = {
        "format_version": 1,
        "status": "READY_PHYSICS_FIELD_PORTFOLIO",
        "protocol": {
            "test_maps_scored_independently": True,
            "test_batch_statistics_used": False,
            "calibration_source": "held challenge realization fold with two independent noise views",
            "tiles_per_map": len(tiles),
            "radial_power_matched_external_training": True,
            "full_field_aggregation": True,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "training_suites": checkpoint["training_suites"],
            "audit_report": checkpoint["audit_report"],
        },
        "sources": {
            "train_maps": {"path": str(args.train_maps)},
            "test_maps": {"path": str(args.test_maps)},
            "mask": {"path": str(args.mask), "sha256": sha256_file(args.mask)},
        },
        "artifacts": artifacts,
        "scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
