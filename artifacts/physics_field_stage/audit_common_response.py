#!/usr/bin/env python3
"""Audit a cross-simulation hydro response direction on actual held maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.ood_metric import phase2_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa-generic", type=Path, required=True)
    parser.add_argument("--kappa-topology", type=Path, required=True)
    parser.add_argument("--flamingo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=1)
    floor = np.median(scale[scale > 0.0]) * 1.0e-3
    return np.maximum(scale, floor)


def curve(negative: np.ndarray, positive: np.ndarray) -> dict[str, float]:
    labels = np.concatenate((np.zeros(len(negative)), np.ones(len(positive))))
    scores = np.concatenate((negative, positive))
    metric, points = phase2_metric(labels, scores)
    return {
        "low_fpr": float(metric),
        "tpr_at_0p001": float(points[0]),
        "tpr_at_0p05": float(points[-1]),
        "paired_positive_fraction": float(np.mean(positive > negative)),
        "paired_margin_mean": float(np.mean(positive - negative)),
    }


def suite_arrays(
    dmo_a: np.ndarray,
    dmo_b: np.ndarray,
    hydro_a: np.ndarray,
    hydro_b: np.ndarray,
    split: np.ndarray,
) -> dict[str, np.ndarray]:
    train = split == 0
    scale = safe_scale(np.concatenate((dmo_a[train], dmo_b[train]), axis=0))
    effect_a = (hydro_a[train] - dmo_a[train]) / scale
    effect_b = (hydro_b[train] - dmo_b[train]) / scale
    effect = 0.5 * (effect_a.mean(0) + effect_b.mean(0))
    effect_error = np.sqrt(np.mean(np.square(effect_a - effect_b), axis=0) + 1.0e-8)
    scatter = np.sqrt(0.5 * (effect_a.var(0) + effect_b.var(0)) + 1.0e-8)
    reliability = np.abs(effect) / (0.5 * effect_error + scatter + 1.0e-6)
    sign_agreement = 0.5 * (
        np.mean(np.sign(effect_a) == np.sign(effect), axis=0)
        + np.mean(np.sign(effect_b) == np.sign(effect), axis=0)
    )
    return {
        "dmo_a": dmo_a,
        "dmo_b": dmo_b,
        "hydro_a": hydro_a,
        "hydro_b": hydro_b,
        "split": split,
        "scale": scale,
        "center": 0.5 * (dmo_a[train].mean(0) + dmo_b[train].mean(0)),
        "effect": effect,
        "reliability": reliability,
        "sign_agreement": sign_agreement,
    }


def score_suite(data: dict[str, np.ndarray], weights: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    held = data["split"] == 1
    centered_dmo = (data[f"dmo_{view}"][held] - data["center"]) / data["scale"]
    centered_hydro = (data[f"hydro_{view}"][held] - data["center"]) / data["scale"]
    return centered_dmo @ weights, centered_hydro @ weights


def main() -> None:
    args = parse_args()
    kg = np.load(args.kappa_generic)
    kt = np.load(args.kappa_topology)
    fl = np.load(args.flamingo)
    k_names = np.concatenate((kg["feature_names"].astype(str), kt["feature_names"].astype(str)))
    f_names = np.concatenate((fl["generic_feature_names"].astype(str), fl["topology_feature_names"].astype(str)))
    if not np.array_equal(k_names, f_names):
        raise ValueError("KappaTNG and FLAMINGO feature names are not aligned")

    kappa = suite_arrays(
        np.concatenate((kg["dmo_a"], kt["dmo_a"]), axis=1),
        np.concatenate((kg["dmo_b"], kt["dmo_b"]), axis=1),
        np.concatenate((kg["hydro_a"], kt["hydro_a"]), axis=1),
        np.concatenate((kg["hydro_b"], kt["hydro_b"]), axis=1),
        kg["sample_split"].astype(np.int8),
    )
    flamingo = suite_arrays(
        np.concatenate((fl["generic_dmo_a"], fl["topology_dmo_a"]), axis=1),
        np.concatenate((fl["generic_dmo_b"], fl["topology_dmo_b"]), axis=1),
        np.concatenate((fl["generic_hydro_power_matched_a"], fl["topology_hydro_power_matched_a"]), axis=1),
        np.concatenate((fl["generic_hydro_power_matched_b"], fl["topology_hydro_power_matched_b"]), axis=1),
        fl["sample_split"].astype(np.int8),
    )

    same_sign = np.sign(kappa["effect"]) == np.sign(flamingo["effect"])
    stable = (
        same_sign
        & (kappa["sign_agreement"] >= 0.55)
        & (flamingo["sign_agreement"] >= 0.55)
        & np.isfinite(kappa["reliability"])
        & np.isfinite(flamingo["reliability"])
    )
    common_strength = np.sqrt(
        np.abs(kappa["effect"] * flamingo["effect"])
        * kappa["reliability"]
        * flamingo["reliability"]
    )
    common_strength[~stable] = -np.inf
    order = np.argsort(common_strength)[::-1]
    order = order[np.isfinite(common_strength[order])]

    reports: dict[str, object] = {}
    for count in (4, 8, 16, 32, 64, 128):
        columns = order[:count]
        if len(columns) != count:
            continue
        signs = np.sign(kappa["effect"][columns] + flamingo["effect"][columns])
        magnitudes = np.sqrt(
            np.abs(kappa["effect"][columns] * flamingo["effect"][columns])
        )
        weights = np.zeros(len(k_names), dtype=np.float64)
        weights[columns] = signs * magnitudes
        weights /= np.linalg.norm(weights)
        row: dict[str, object] = {
            "feature_names": k_names[columns].tolist(),
            "common_strength": common_strength[columns].tolist(),
        }
        metrics = []
        for suite_name, suite in (("kappatng", kappa), ("flamingo", flamingo)):
            row[suite_name] = {}
            for view in ("a", "b"):
                negative, positive = score_suite(suite, weights, view)
                row[suite_name][view] = curve(negative, positive)
                metrics.append(row[suite_name][view]["low_fpr"])
        row["minimum_low_fpr"] = float(min(metrics))
        row["mean_low_fpr"] = float(np.mean(metrics))
        reports[str(count)] = row

    output = {
        "protocol": {
            "actual_held_hydro_maps": True,
            "radial_power_matched": True,
            "selection_split": 0,
            "evaluation_split": 1,
            "paired_shape_noise": True,
            "test_rows_used": False,
        },
        "stable_common_features": int(len(order)),
        "reports": reports,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
