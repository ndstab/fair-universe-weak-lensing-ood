#!/usr/bin/env python3
"""Train and audit a field-level hydro-phase detector across simulation suites."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .ood_metric import phase2_metric
    from .wl_physics_field import (
        SHAPE_NOISE_STD,
        ObservedPatchAccessor,
        PhysicsFieldDetector,
        aggregate_patch_scores,
        deterministic_tiles,
    )
except ImportError:  # Direct execution via ``python scripts/...``.
    from ood_metric import phase2_metric
    from wl_physics_field import (
        SHAPE_NOISE_STD,
        ObservedPatchAccessor,
        PhysicsFieldDetector,
        aggregate_patch_scores,
        deterministic_tiles,
    )


SUITES = ("kappatng", "flamingo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--train-maps", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-steps", type=int, default=700)
    parser.add_argument("--final-steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=192)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--held-realization-fold", type=int, default=3)
    parser.add_argument("--audit-nominal-maps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2026080302)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aggregate_paired_fields(
    field_id: np.ndarray, dmo_scores: np.ndarray, hydro_scores: np.ndarray
) -> dict[str, np.ndarray]:
    identifiers = np.asarray(field_id, dtype=np.int64)
    dmo = np.asarray(dmo_scores, dtype=np.float64)
    hydro = np.asarray(hydro_scores, dtype=np.float64)
    if identifiers.shape != dmo.shape or dmo.shape != hydro.shape:
        raise ValueError("paired field identifiers and scores are misaligned")
    unique = np.unique(identifiers)
    lookup = {value: index for index, value in enumerate(unique)}
    compact = np.asarray([lookup[value] for value in identifiers], dtype=np.int64)
    dmo_aggregates = aggregate_patch_scores(compact, dmo, map_count=len(unique))
    hydro_aggregates = aggregate_patch_scores(compact, hydro, map_count=len(unique))
    return {
        "field_id": unique,
        "dmo_mean": dmo_aggregates["mean"],
        "hydro_mean": hydro_aggregates["mean"],
        "dmo_tail": dmo_aggregates["top_fraction_mean"],
        "hydro_tail": hydro_aggregates["top_fraction_mean"],
    }


def physics_training_loss(
    *,
    dmo_logit: torch.Tensor,
    hydro_logit: torch.Tensor,
    nominal_logit: torch.Tensor,
    dmo_embedding: torch.Tensor,
    nominal_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    negative_logits = torch.cat((dmo_logit, nominal_logit))
    classification = 0.5 * (
        F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )
        + F.binary_cross_entropy_with_logits(
            hydro_logit, torch.ones_like(hydro_logit)
        )
    )
    paired_margin = F.softplus(0.75 - (hydro_logit - dmo_logit)).mean()
    dmo_mean = dmo_embedding.mean(0)
    nominal_mean = nominal_embedding.mean(0)
    dmo_std = dmo_embedding.std(0, unbiased=False)
    nominal_std = nominal_embedding.std(0, unbiased=False)
    domain_alignment = (dmo_mean - nominal_mean).square().mean() + 0.25 * (
        dmo_std - nominal_std
    ).square().mean()
    loss = classification + 0.35 * paired_margin + 0.08 * domain_alignment
    return {
        "loss": loss,
        "classification": classification,
        "paired_margin": paired_margin,
        "domain_alignment": domain_alignment,
    }


def promotion_decision(
    cross_suite_reports: dict[str, dict[str, float]],
    *,
    cross_noise_correlation: float,
    minimum_cross_suite: float = 0.50,
    mean_cross_suite: float = 0.62,
    minimum_cross_noise: float = 0.50,
) -> dict[str, object]:
    values = [float(report["low_fpr"]) for report in cross_suite_reports.values()]
    failed = []
    if not values or min(values) < minimum_cross_suite:
        failed.append("minimum_cross_suite_low_fpr")
    if not values or float(np.mean(values)) < mean_cross_suite:
        failed.append("mean_cross_suite_low_fpr")
    if cross_noise_correlation < minimum_cross_noise:
        failed.append("cross_noise_correlation")
    return {
        "passed": not failed,
        "failed_gates": failed,
        "minimum_cross_suite_low_fpr": min(values) if values else None,
        "mean_cross_suite_low_fpr": float(np.mean(values)) if values else None,
        "cross_noise_correlation": float(cross_noise_correlation),
        "thresholds": {
            "minimum_cross_suite_low_fpr": minimum_cross_suite,
            "mean_cross_suite_low_fpr": mean_cross_suite,
            "cross_noise_correlation": minimum_cross_noise,
        },
    }


def _d4_batch(
    values: torch.Tensor, masks: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.randint(0, 8, (len(values),), generator=generator, device=values.device)
    output_values = []
    output_masks = []
    for value, mask, code_tensor in zip(values, masks, codes):
        code = int(code_tensor)
        transformed_value = torch.rot90(value, code % 4, dims=(-2, -1))
        transformed_mask = torch.rot90(mask, code % 4, dims=(-2, -1))
        if code >= 4:
            transformed_value = torch.flip(transformed_value, dims=(-1,))
            transformed_mask = torch.flip(transformed_mask, dims=(-1,))
        output_values.append(transformed_value)
        output_masks.append(transformed_mask)
    return torch.stack(output_values), torch.stack(output_masks)


def paired_d4_batch(
    dmo: torch.Tensor,
    hydro: torch.Tensor,
    masks: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one sampled D4 transform to both members of each physical pair."""

    if dmo.shape != hydro.shape or dmo.shape != masks.shape:
        raise ValueError("paired D4 inputs are misaligned")
    codes = torch.randint(0, 8, (len(dmo),), generator=generator, device=dmo.device)
    dmo_output = []
    hydro_output = []
    mask_output = []
    for base, changed, mask, code_tensor in zip(dmo, hydro, masks, codes):
        code = int(code_tensor)
        base = torch.rot90(base, code % 4, dims=(-2, -1))
        changed = torch.rot90(changed, code % 4, dims=(-2, -1))
        mask = torch.rot90(mask, code % 4, dims=(-2, -1))
        if code >= 4:
            base = torch.flip(base, dims=(-1,))
            changed = torch.flip(changed, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        dmo_output.append(base)
        hydro_output.append(changed)
        mask_output.append(mask)
    return torch.stack(dmo_output), torch.stack(hydro_output), torch.stack(mask_output)


def sample_nominal(
    flat_maps: np.ndarray,
    rows: np.ndarray,
    accessor: ObservedPatchAccessor,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected_rows = rng.choice(rows, size=count, replace=len(rows) < count)
    selected_tiles = rng.integers(0, len(accessor.lookup), size=count)
    patches = []
    masks = []
    for row, tile in zip(selected_rows, selected_tiles):
        patch, mask = accessor.extract(np.asarray(flat_maps[int(row)]), int(tile))
        patches.append(patch)
        masks.append(mask)
    return np.asarray(patches, dtype=np.float32), np.asarray(masks, dtype=np.bool_)


def _suite_rows(bank: np.lib.npyio.NpzFile, suites: tuple[str, ...], split: int) -> np.ndarray:
    names = bank["suite"].astype(str)
    return np.flatnonzero(np.isin(names, suites) & (bank["split"].astype(np.int8) == split))


def train_model(
    args: argparse.Namespace,
    bank: np.lib.npyio.NpzFile,
    flat_maps: np.ndarray,
    train_rows: np.ndarray,
    accessor: ObservedPatchAccessor,
    *,
    suites: tuple[str, ...],
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[PhysicsFieldDetector, list[dict[str, float]]]:
    seed_all(seed)
    model = PhysicsFieldDetector(
        base_channels=args.base_channels, embedding_dim=args.embedding_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=args.learning_rate * 0.05
    )
    external_rows = _suite_rows(bank, suites, split=0)
    if len(external_rows) < args.batch_size:
        raise ValueError(f"insufficient external rows for {suites}")
    numpy_rng = np.random.default_rng(seed + 1)
    torch_rng = torch.Generator(device=device).manual_seed(seed + 2)
    dmo_bank = bank["dmo"]
    hydro_bank = bank["hydro"]
    history = []
    window: dict[str, float] = {}
    model.train()
    for step in range(steps):
        selected = numpy_rng.choice(external_rows, size=args.batch_size, replace=False)
        dmo = torch.as_tensor(np.asarray(dmo_bank[selected], dtype=np.float32), device=device)[:, None]
        hydro = torch.as_tensor(np.asarray(hydro_bank[selected], dtype=np.float32), device=device)[:, None]
        external_mask = torch.ones_like(dmo)
        paired_noise = SHAPE_NOISE_STD * torch.randn(
            dmo.shape, generator=torch_rng, device=device
        )
        dmo = dmo + paired_noise
        hydro = hydro + paired_noise
        dmo, hydro, external_mask_augmented = paired_d4_batch(
            dmo, hydro, external_mask, torch_rng
        )

        nominal_np, nominal_mask_np = sample_nominal(
            flat_maps, train_rows, accessor, args.batch_size, numpy_rng
        )
        nominal = torch.as_tensor(nominal_np, device=device)[:, None]
        nominal_mask = torch.as_tensor(nominal_mask_np, device=device)[:, None]
        nominal = nominal + SHAPE_NOISE_STD * torch.randn(
            nominal.shape, generator=torch_rng, device=device
        ) * nominal_mask
        nominal, nominal_mask = _d4_batch(nominal, nominal_mask, torch_rng)

        values = torch.cat((dmo, hydro, nominal))
        masks = torch.cat((external_mask_augmented, external_mask_augmented, nominal_mask))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(values, masks)
            dmo_logit, hydro_logit, nominal_logit = output["logit"].chunk(3)
            dmo_embedding, _, nominal_embedding = output["embedding"].chunk(3)
            losses = physics_training_loss(
                dmo_logit=dmo_logit,
                hydro_logit=hydro_logit,
                nominal_logit=nominal_logit,
                dmo_embedding=dmo_embedding,
                nominal_embedding=nominal_embedding,
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()
        for name, value in losses.items():
            window[name] = window.get(name, 0.0) + float(value.detach())
        if (step + 1) % 100 == 0 or step + 1 == steps:
            width = 100 if (step + 1) % 100 == 0 else steps % 100
            record = {
                "step": step + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{name: value / max(1, width) for name, value in window.items()},
            }
            history.append(record)
            print(json.dumps({"training_suites": suites, **record}, sort_keys=True), flush=True)
            window = {}
    return model, history


@torch.inference_mode()
def score_external(
    model: PhysicsFieldDetector,
    bank: np.lib.npyio.NpzFile,
    rows: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    generator_dmo = torch.Generator(device=device).manual_seed(seed)
    generator_hydro = torch.Generator(device=device).manual_seed(seed + 1)
    dmo_scores = []
    hydro_scores = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        dmo = torch.as_tensor(
            np.asarray(bank["dmo"][selected], dtype=np.float32), device=device
        )[:, None]
        hydro = torch.as_tensor(
            np.asarray(bank["hydro"][selected], dtype=np.float32), device=device
        )[:, None]
        dmo += SHAPE_NOISE_STD * torch.randn(
            dmo.shape, generator=generator_dmo, device=device
        )
        hydro += SHAPE_NOISE_STD * torch.randn(
            hydro.shape, generator=generator_hydro, device=device
        )
        mask = torch.ones_like(dmo)
        dmo_scores.append(model(dmo, mask)["logit"].float().cpu().numpy())
        hydro_scores.append(model(hydro, mask)["logit"].float().cpu().numpy())
    aggregates = aggregate_paired_fields(
        bank["field_id"][rows], np.concatenate(dmo_scores), np.concatenate(hydro_scores)
    )
    heads = {}
    for name in ("mean", "tail"):
        labels = np.concatenate(
            [
                np.zeros(len(aggregates[f"dmo_{name}"]), dtype=np.int8),
                np.ones(len(aggregates[f"hydro_{name}"]), dtype=np.int8),
            ]
        )
        scores = np.concatenate(
            [aggregates[f"dmo_{name}"], aggregates[f"hydro_{name}"]]
        )
        metric, curve = phase2_metric(labels, scores)
        heads[name] = {
            "low_fpr": float(metric),
            "tpr_at_0p001": float(curve[0]),
            "tpr_at_0p05": float(curve[-1]),
        }
    report = {"low_fpr": heads["mean"]["low_fpr"], "heads": heads}
    return report, aggregates


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = _rank(np.asarray(first, dtype=np.float64))
    y = _rank(np.asarray(second, dtype=np.float64))
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@torch.inference_mode()
def score_nominal_maps(
    model: PhysicsFieldDetector,
    flat_maps: np.ndarray,
    rows: np.ndarray,
    accessor: ObservedPatchAccessor,
    *,
    noise_seed: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    patch_values = []
    patch_masks = []
    map_ids = []
    all_scores = []
    all_ids = []

    def flush() -> None:
        if not patch_values:
            return
        values = torch.as_tensor(np.asarray(patch_values), device=device)[:, None]
        masks = torch.as_tensor(np.asarray(patch_masks), device=device)[:, None]
        values += SHAPE_NOISE_STD * torch.randn(
            values.shape, generator=generator, device=device
        ) * masks
        all_scores.append(model(values, masks)["logit"].float().cpu().numpy())
        all_ids.append(np.asarray(map_ids, dtype=np.int64))
        patch_values.clear()
        patch_masks.clear()
        map_ids.clear()

    for output_id, row in enumerate(rows):
        compressed = np.asarray(flat_maps[int(row)])
        for tile_id in range(len(accessor.lookup)):
            patch, mask = accessor.extract(compressed, tile_id)
            patch_values.append(patch)
            patch_masks.append(mask)
            map_ids.append(output_id)
            if len(patch_values) >= batch_size:
                flush()
    flush()
    aggregate = aggregate_patch_scores(
        np.concatenate(all_ids), np.concatenate(all_scores), map_count=len(rows)
    )
    return aggregate["mean"]


def save_checkpoint(
    path: Path,
    model: PhysicsFieldDetector,
    args: argparse.Namespace,
    *,
    suites: tuple[str, ...],
    training_history: list[dict[str, float]],
    report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "base_channels": args.base_channels,
                "embedding_dim": args.embedding_dim,
            },
            "training_suites": suites,
            "training_history": training_history,
            "audit_report": report,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the bounded field-detector run")
    device = torch.device("cuda")
    bank = np.load(args.bank, mmap_mode="r")
    mask = np.load(args.mask)
    tiles = deterministic_tiles(mask, patch_size=64, stride=32, minimum_fraction=0.75)
    accessor = ObservedPatchAccessor(mask, tiles, patch_size=64)
    maps = np.load(args.train_maps, mmap_mode="r")
    if maps.shape[-1] != accessor.n_observed:
        raise ValueError("challenge maps and mask disagree")
    flat_maps = maps.reshape(-1, maps.shape[-1])
    realization = np.arange(len(flat_maps)) % maps.shape[1]
    held = realization % 4 == args.held_realization_fold
    train_rows = np.flatnonzero(~held)
    audit_rows = np.flatnonzero(held)[: args.audit_nominal_maps]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cross_suite_reports = {}
    within_suite_reports = {}
    cross_noise_values = []
    pilot_rows = []
    for index, training_suite in enumerate(SUITES):
        evaluation_suite = SUITES[1 - index]
        model, history = train_model(
            args,
            bank,
            flat_maps,
            train_rows,
            accessor,
            suites=(training_suite,),
            steps=args.pilot_steps,
            seed=args.seed + 10_000 * index,
            device=device,
        )
        cross_report, _ = score_external(
            model,
            bank,
            _suite_rows(bank, (evaluation_suite,), split=1),
            seed=args.seed + 20_000 + index,
            batch_size=args.inference_batch_size,
            device=device,
        )
        within_report, _ = score_external(
            model,
            bank,
            _suite_rows(bank, (training_suite,), split=1),
            seed=args.seed + 30_000 + index,
            batch_size=args.inference_batch_size,
            device=device,
        )
        nominal_a = score_nominal_maps(
            model,
            flat_maps,
            audit_rows,
            accessor,
            noise_seed=args.seed + 40_000 + index,
            batch_size=args.inference_batch_size,
            device=device,
        )
        nominal_b = score_nominal_maps(
            model,
            flat_maps,
            audit_rows,
            accessor,
            noise_seed=args.seed + 50_000 + index,
            batch_size=args.inference_batch_size,
            device=device,
        )
        correlation = rank_correlation(nominal_a, nominal_b)
        key = f"{training_suite}_to_{evaluation_suite}"
        cross_suite_reports[key] = cross_report
        within_suite_reports[training_suite] = within_report
        cross_noise_values.append(correlation)
        pilot_report = {
            "training_suite": training_suite,
            "evaluation_suite": evaluation_suite,
            "cross_suite": cross_report,
            "within_suite": within_report,
            "cross_noise_correlation": correlation,
        }
        pilot_rows.append(pilot_report)
        save_checkpoint(
            args.output_dir / f"pilot_{training_suite}.pt",
            model,
            args,
            suites=(training_suite,),
            training_history=history,
            report=pilot_report,
        )
        print(json.dumps(pilot_report, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    gate = promotion_decision(
        cross_suite_reports,
        cross_noise_correlation=float(min(cross_noise_values)),
    )
    final_report = None
    status = "REJECTED_AT_PILOT_GATE"
    if gate["passed"]:
        model, history = train_model(
            args,
            bank,
            flat_maps,
            train_rows,
            accessor,
            suites=SUITES,
            steps=args.final_steps,
            seed=args.seed + 90_000,
            device=device,
        )
        held_reports = {
            suite: score_external(
                model,
                bank,
                _suite_rows(bank, (suite,), split=1),
                seed=args.seed + 100_000 + index,
                batch_size=args.inference_batch_size,
                device=device,
            )[0]
            for index, suite in enumerate(SUITES)
        }
        nominal_a = score_nominal_maps(
            model,
            flat_maps,
            audit_rows,
            accessor,
            noise_seed=args.seed + 110_000,
            batch_size=args.inference_batch_size,
            device=device,
        )
        nominal_b = score_nominal_maps(
            model,
            flat_maps,
            audit_rows,
            accessor,
            noise_seed=args.seed + 120_000,
            batch_size=args.inference_batch_size,
            device=device,
        )
        final_report = {
            "held_suite_reports": held_reports,
            "cross_noise_correlation": rank_correlation(nominal_a, nominal_b),
        }
        save_checkpoint(
            args.output_dir / "physics_field_final.pt",
            model,
            args,
            suites=SUITES,
            training_history=history,
            report=final_report,
        )
        status = "PROMOTED_FINAL_CHECKPOINT"

    report = {
        "format_version": 1,
        "status": status,
        "protocol": {
            "radial_power_matched": True,
            "paired_external_sources": True,
            "external_source_split_held": True,
            "cross_simulation_suite_held": True,
            "challenge_realization_fold_held": True,
            "test_rows_used": False,
            "deterministic_tiles": len(tiles),
        },
        "pilots": pilot_rows,
        "promotion_gate": gate,
        "final": final_report,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
