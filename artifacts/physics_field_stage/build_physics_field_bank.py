#!/usr/bin/env python3
"""Build paired, power-matched field patches from kappaTNG and FLAMINGO."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .wl_physics_field import TRAIN_SIGNAL_STD, radial_power_match
except ImportError:  # Direct execution via ``python scripts/...``.
    from wl_physics_field import TRAIN_SIGNAL_STD, radial_power_match


KAPPA_SOURCE_SHAPE = (1024, 1024)
KAPPA_CHALLENGE_SHAPE = (150, 150)
FLAMINGO_NSIDE = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa-ranges", type=Path, required=True)
    parser.add_argument("--flamingo-dmo", type=Path, required=True)
    parser.add_argument("--flamingo-hydro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--fields-per-split", type=int, default=48)
    parser.add_argument("--tiles-per-field", type=int, default=12)
    parser.add_argument("--kappa-runs", type=int, default=64)
    parser.add_argument("--radial-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026080301)
    return parser.parse_args()


def d4(values: np.ndarray, code: int) -> np.ndarray:
    transformed = np.rot90(values, code % 4)
    if code >= 4:
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def paired_fields(
    dmo_maps: np.ndarray,
    hydro_maps: np.ndarray,
    *,
    source_ids: np.ndarray,
    fields: int,
    tiles_per_field: int,
    patch_size: int,
    seed: int,
    power_match_after_crop: bool = False,
    radial_bins: int = 16,
) -> dict[str, np.ndarray]:
    """Draw identically transformed/cropped patches from aligned map pairs."""

    dmo = np.asarray(dmo_maps, dtype=np.float32)
    hydro = np.asarray(hydro_maps, dtype=np.float32)
    identifiers = np.asarray(source_ids)
    if dmo.shape != hydro.shape or dmo.ndim != 3 or len(dmo) != len(identifiers):
        raise ValueError("paired source maps or identifiers are misaligned")
    if patch_size > min(dmo.shape[-2:]):
        raise ValueError("patch exceeds source map")
    rng = np.random.default_rng(seed)
    dmo_output = []
    hydro_output = []
    used_sources = []
    field_ids = []
    for field_id in range(fields):
        for _ in range(tiles_per_field):
            source = int(rng.integers(0, len(dmo)))
            code = int(rng.integers(0, 8))
            base = d4(dmo[source], code)
            changed = d4(hydro[source], code)
            y = int(rng.integers(0, base.shape[0] - patch_size + 1))
            x = int(rng.integers(0, base.shape[1] - patch_size + 1))
            base_patch = base[y : y + patch_size, x : x + patch_size]
            changed_patch = changed[y : y + patch_size, x : x + patch_size]
            if power_match_after_crop:
                changed_patch = radial_power_match(
                    changed_patch, base_patch, bins=radial_bins
                )
            dmo_output.append(base_patch)
            hydro_output.append(changed_patch)
            used_sources.append(identifiers[source])
            field_ids.append(field_id)
    return {
        "dmo": np.asarray(dmo_output, dtype=np.float32),
        "hydro": np.asarray(hydro_output, dtype=np.float32),
        "source_id": np.asarray(used_sources),
        "field_id": np.asarray(field_ids, dtype=np.int64),
    }


def rescale_paired_fields(
    dmo: np.ndarray, hydro: np.ndarray, *, target_std: float = TRAIN_SIGNAL_STD
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply one DMO-derived centering/scale to both members of every pair."""

    baseline = np.asarray(dmo, dtype=np.float32)
    changed = np.asarray(hydro, dtype=np.float32)
    if baseline.shape != changed.shape:
        raise ValueError("paired arrays differ in shape")
    mean = float(np.mean(baseline, dtype=np.float64))
    standard_deviation = float(np.std(baseline, dtype=np.float64))
    if standard_deviation <= 0.0:
        raise ValueError("DMO field has zero variance")
    scale = float(target_std / standard_deviation)
    return (
        ((baseline - mean) * scale).astype(np.float32),
        ((changed - mean) * scale).astype(np.float32),
        {
            "dmo_mean": mean,
            "dmo_std": standard_deviation,
            "target_std": float(target_std),
            "scale": scale,
        },
    )


def load_kappa_plane(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float64)
    expected = KAPPA_SOURCE_SHAPE[0] * KAPPA_SOURCE_SHAPE[1]
    if values.size != expected:
        raise ValueError(f"invalid kappaTNG plane {path}: {values.size} != {expected}")
    tensor = torch.from_numpy(values.reshape(KAPPA_SOURCE_SHAPE).astype(np.float32))[None, None]
    return (
        F.interpolate(tensor, size=KAPPA_CHALLENGE_SHAPE, mode="area")[0, 0]
        .numpy()
        .astype(np.float32, copy=False)
    )


def build_kappa_sources(ranges: Path, runs: int) -> tuple[np.ndarray, np.ndarray]:
    dmo_maps = []
    hydro_maps = []
    for run in tqdm(range(1, runs + 1), desc="kappaTNG paired planes"):
        dmo = load_kappa_plane(ranges / f"dmonly_LP001_run{run:03d}_z18.f64")
        hydro = load_kappa_plane(ranges / f"fullphys_LP001_run{run:03d}_z18.f64")
        dmo_maps.append(dmo)
        hydro_maps.append(hydro)
    return np.asarray(dmo_maps), np.asarray(hydro_maps)


def load_flamingo(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle["Convergence"]
        expected = hp.nside2npix(FLAMINGO_NSIDE)
        if dataset.shape != (expected,) or dataset.dtype != np.float32:
            raise ValueError(f"unexpected FLAMINGO field {dataset.shape} {dataset.dtype}")
        return dataset[:]


def flamingo_projection(
    values: np.ndarray, longitude: float, latitude: float, patch_size: int
) -> np.ndarray:
    projector = hp.projector.GnomonicProj(
        rot=(longitude, latitude, 0.0),
        xsize=patch_size,
        ysize=patch_size,
        reso=2.0,
    )
    projected = projector.projmap(
        values,
        lambda x, y, z: hp.vec2pix(FLAMINGO_NSIDE, x, y, z, nest=False),
    )
    return np.asarray(projected, dtype=np.float32)


def build_flamingo_split(
    dmo_sky: np.ndarray,
    hydro_sky: np.ndarray,
    *,
    split: int,
    fields: int,
    tiles_per_field: int,
    patch_size: int,
    radial_bins: int,
    seed: int,
) -> dict[str, np.ndarray]:
    coarse = np.arange(hp.nside2npix(16), dtype=np.int64)
    longitude, latitude = hp.pix2ang(16, coarse, lonlat=True, nest=False)
    allowed = np.abs(latitude) <= 75.0
    if split == 0:
        allowed &= longitude < 180.0
    else:
        allowed &= longitude >= 180.0
    candidates = coarse[allowed]
    required = fields * tiles_per_field
    if required > len(candidates):
        raise ValueError(f"need {required} FLAMINGO centers, have {len(candidates)}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=required, replace=False)
    chosen_lon, chosen_lat = hp.pix2ang(16, chosen, lonlat=True, nest=False)
    dmo_patches = []
    hydro_patches = []
    for lon, lat in tqdm(
        zip(chosen_lon, chosen_lat), total=required, desc=f"FLAMINGO split {split}"
    ):
        dmo = flamingo_projection(dmo_sky, float(lon), float(lat), patch_size)
        hydro = flamingo_projection(hydro_sky, float(lon), float(lat), patch_size)
        dmo_patches.append(dmo)
        hydro_patches.append(radial_power_match(hydro, dmo, bins=radial_bins))
    return {
        "dmo": np.asarray(dmo_patches, dtype=np.float32),
        "hydro": np.asarray(hydro_patches, dtype=np.float32),
        "source_id": chosen.astype(np.int64),
        "field_id": np.repeat(np.arange(fields, dtype=np.int64), tiles_per_field),
    }


def _append_bank(
    payload: dict[str, list[np.ndarray]],
    bank: dict[str, np.ndarray],
    *,
    suite: str,
    split: int,
    field_offset: int,
) -> tuple[dict[str, float], int]:
    dmo, hydro, scale = rescale_paired_fields(bank["dmo"], bank["hydro"])
    count = len(dmo)
    payload["dmo"].append(dmo.astype(np.float16))
    payload["hydro"].append(hydro.astype(np.float16))
    payload["suite"].append(np.full(count, suite))
    payload["split"].append(np.full(count, split, dtype=np.int8))
    payload["source_id"].append(bank["source_id"].astype(np.int64))
    fields = bank["field_id"].astype(np.int64) + field_offset
    payload["field_id"].append(fields)
    return scale, int(fields.max()) + 1


def main() -> None:
    args = parse_args()
    if args.kappa_runs < 20 or args.fields_per_split < 8:
        raise ValueError("insufficient independent sources or fields")
    started = time.perf_counter()
    payload: dict[str, list[np.ndarray]] = {
        key: [] for key in ("dmo", "hydro", "suite", "split", "source_id", "field_id")
    }
    scales: dict[str, object] = {}
    field_offset = 0

    kappa_dmo, kappa_hydro = build_kappa_sources(args.kappa_ranges, args.kappa_runs)
    boundary = args.kappa_runs // 2
    for split, rows in enumerate((np.arange(boundary), np.arange(boundary, args.kappa_runs))):
        bank = paired_fields(
            kappa_dmo[rows],
            kappa_hydro[rows],
            source_ids=rows + 1,
            fields=args.fields_per_split,
            tiles_per_field=args.tiles_per_field,
            patch_size=args.patch_size,
            seed=args.seed + split,
            power_match_after_crop=True,
            radial_bins=args.radial_bins,
        )
        scales[f"kappatng_split{split}"], field_offset = _append_bank(
            payload, bank, suite="kappatng", split=split, field_offset=field_offset
        )
    del kappa_dmo, kappa_hydro

    dmo_sky = load_flamingo(args.flamingo_dmo)
    hydro_sky = load_flamingo(args.flamingo_hydro)
    for split in (0, 1):
        bank = build_flamingo_split(
            dmo_sky,
            hydro_sky,
            split=split,
            fields=args.fields_per_split,
            tiles_per_field=args.tiles_per_field,
            patch_size=args.patch_size,
            radial_bins=args.radial_bins,
            seed=args.seed + 100 + split,
        )
        scales[f"flamingo_split{split}"], field_offset = _append_bank(
            payload, bank, suite="flamingo", split=split, field_offset=field_offset
        )
    arrays = {key: np.concatenate(parts) for key, parts in payload.items()}
    metadata = {
        "format_version": 1,
        "patch_size": args.patch_size,
        "fields_per_split": args.fields_per_split,
        "tiles_per_field": args.tiles_per_field,
        "radial_bins": args.radial_bins,
        "hydro_radial_power_matched_to_dmo": True,
        "source_splits_disjoint": True,
        "target_signal_std": TRAIN_SIGNAL_STD,
        "scales": scales,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
