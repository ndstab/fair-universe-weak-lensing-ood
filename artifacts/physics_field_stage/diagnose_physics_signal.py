#!/usr/bin/env python3
"""Read-only diagnosis of the rejected physics-field pilot."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ood_metric import phase2_metric
import scripts.train_physics_field_detector as training
from scripts.train_physics_field_detector import aggregate_paired_fields
from scripts.wl_physics_field import (
    ObservedPatchAccessor,
    PhysicsFieldDetector,
    SHAPE_NOISE_STD,
    deterministic_tiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2026080311)
    parser.add_argument("--train-maps", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--clean-steps", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rows_for(bank: np.lib.npyio.NpzFile, suite: str) -> np.ndarray:
    return np.flatnonzero(
        (bank["suite"].astype(str) == suite)
        & (bank["split"].astype(np.int8) == 1)
    )


@torch.inference_mode()
def score_mode(
    model: PhysicsFieldDetector,
    bank: np.lib.npyio.NpzFile,
    rows: np.ndarray,
    *,
    mode: str,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dmo_scores: list[np.ndarray] = []
    hydro_scores: list[np.ndarray] = []
    first = torch.Generator(device=device).manual_seed(seed)
    second = torch.Generator(device=device).manual_seed(seed + 1)
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        dmo = torch.as_tensor(np.asarray(bank["dmo"][selected], dtype=np.float32), device=device)[:, None]
        hydro = torch.as_tensor(np.asarray(bank["hydro"][selected], dtype=np.float32), device=device)[:, None]
        if mode != "clean":
            dmo_noise = SHAPE_NOISE_STD * torch.randn(dmo.shape, generator=first, device=device)
            if mode == "shared_noise":
                hydro_noise = dmo_noise
            elif mode == "independent_noise":
                hydro_noise = SHAPE_NOISE_STD * torch.randn(hydro.shape, generator=second, device=device)
            else:
                raise ValueError(mode)
            dmo = dmo + dmo_noise
            hydro = hydro + hydro_noise
        mask = torch.ones_like(dmo)
        dmo_scores.append(model(dmo, mask)["logit"].float().cpu().numpy())
        hydro_scores.append(model(hydro, mask)["logit"].float().cpu().numpy())
    aggregate = aggregate_paired_fields(
        bank["field_id"][rows], np.concatenate(dmo_scores), np.concatenate(hydro_scores)
    )
    labels = np.concatenate((np.zeros(len(aggregate["dmo_mean"])), np.ones(len(aggregate["hydro_mean"]))))
    scores = np.concatenate((aggregate["dmo_mean"], aggregate["hydro_mean"]))
    metric, curve = phase2_metric(labels, scores)
    paired_margin = aggregate["hydro_mean"] - aggregate["dmo_mean"]
    return {
        "low_fpr": float(metric),
        "tpr_at_0p001": float(curve[0]),
        "tpr_at_0p05": float(curve[-1]),
        "paired_margin_mean": float(paired_margin.mean()),
        "paired_margin_positive_fraction": float(np.mean(paired_margin > 0.0)),
    }


@torch.inference_mode()
def smoothing_stats(bank: np.lib.npyio.NpzFile, suite: str, seed: int) -> dict[str, dict[str, float]]:
    rows = rows_for(bank, suite)
    dmo = torch.as_tensor(np.asarray(bank["dmo"][rows], dtype=np.float32))[:, None]
    hydro = torch.as_tensor(np.asarray(bank["hydro"][rows], dtype=np.float32))[:, None]
    generator = torch.Generator().manual_seed(seed)
    noise = SHAPE_NOISE_STD * torch.randn(dmo.shape, generator=generator)
    result: dict[str, dict[str, float]] = {}
    for width in (1, 3, 9, 17, 33):
        if width == 1:
            smooth_dmo, smooth_hydro, smooth_noise = dmo, hydro, noise
        else:
            smooth_dmo = F.avg_pool2d(dmo, width, stride=1, padding=width // 2)
            smooth_hydro = F.avg_pool2d(hydro, width, stride=1, padding=width // 2)
            smooth_noise = F.avg_pool2d(noise, width, stride=1, padding=width // 2)
        residual = smooth_hydro - smooth_dmo
        result[str(width)] = {
            "residual_rms": float(residual.square().mean().sqrt()),
            "noise_rms": float(smooth_noise.square().mean().sqrt()),
            "pointwise_residual_to_noise": float(
                residual.square().mean().sqrt() / smooth_noise.square().mean().sqrt()
            ),
            "field_l2_oracle_snr": float(
                residual.flatten(1).square().sum(1).sqrt().mean()
                / smooth_noise.flatten(1).square().sum(1).sqrt().mean()
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    bank = np.load(args.bank, mmap_mode="r")
    report: dict[str, object] = {"smoothing": {}, "checkpoints": {}}
    for suite_index, suite in enumerate(("kappatng", "flamingo")):
        report["smoothing"][suite] = smoothing_stats(bank, suite, args.seed + suite_index)
    for checkpoint_path in sorted(args.checkpoint_dir.glob("pilot_*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = PhysicsFieldDetector(**payload["model_kwargs"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        checkpoint_report: dict[str, object] = {}
        for suite_index, suite in enumerate(("kappatng", "flamingo")):
            checkpoint_report[suite] = {
                mode: score_mode(
                    model,
                    bank,
                    rows_for(bank, suite),
                    mode=mode,
                    seed=args.seed + 100 * suite_index,
                    batch_size=args.batch_size,
                    device=device,
                )
                for mode in ("clean", "shared_noise", "independent_noise")
            }
        report["checkpoints"][checkpoint_path.name] = checkpoint_report
    if args.clean_steps:
        if args.train_maps is None or args.mask is None:
            raise ValueError("--train-maps and --mask are required for clean training")
        mask = np.load(args.mask)
        tiles = deterministic_tiles(mask, patch_size=64, stride=32, minimum_fraction=0.75)
        accessor = ObservedPatchAccessor(mask, tiles, patch_size=64)
        maps = np.load(args.train_maps, mmap_mode="r")
        flat_maps = maps.reshape(-1, maps.shape[-1])
        realization = np.arange(len(flat_maps)) % maps.shape[1]
        train_rows = np.flatnonzero(realization % 4 != 3)
        train_args = SimpleNamespace(
            base_channels=32,
            embedding_dim=192,
            learning_rate=2.0e-4,
            weight_decay=2.0e-4,
            batch_size=32,
        )
        clean_reports: dict[str, object] = {}
        original_noise = training.SHAPE_NOISE_STD
        for suite_index, train_suite in enumerate(("kappatng", "flamingo")):
            evaluation_suite = "flamingo" if train_suite == "kappatng" else "kappatng"
            training.SHAPE_NOISE_STD = 0.0
            with contextlib.redirect_stdout(io.StringIO()):
                model, history = training.train_model(
                    train_args,
                    bank,
                    flat_maps,
                    train_rows,
                    accessor,
                    suites=(train_suite,),
                    steps=args.clean_steps,
                    seed=args.seed + 10_000 * suite_index,
                    device=device,
                )
            suite_report: dict[str, object] = {"last_history": history[-1]}
            for noise_name, noise_level in (("clean", 0.0), ("independent_noise", original_noise)):
                training.SHAPE_NOISE_STD = noise_level
                suite_report[noise_name] = {
                    "within": training.score_external(
                        model,
                        bank,
                        training._suite_rows(bank, (train_suite,), split=1),
                        seed=args.seed + 20_000,
                        batch_size=args.batch_size,
                        device=device,
                    )[0],
                    "cross": training.score_external(
                        model,
                        bank,
                        training._suite_rows(bank, (evaluation_suite,), split=1),
                        seed=args.seed + 30_000,
                        batch_size=args.batch_size,
                        device=device,
                    )[0],
                }
            clean_reports[f"{train_suite}_to_{evaluation_suite}"] = suite_report
            del model
            torch.cuda.empty_cache()
        training.SHAPE_NOISE_STD = original_noise
        report["clean_training"] = clean_reports
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
