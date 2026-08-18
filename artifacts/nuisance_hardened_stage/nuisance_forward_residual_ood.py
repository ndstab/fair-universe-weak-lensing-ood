#!/usr/bin/env python3
"""Nuisance-aware forward-model goodness-of-fit scores.

The training labels describe five physical coordinates for every InD map:
``Omega_m``, ``S8``, two baryonic-feedback parameters, and photo-z bias.
For each fixed feature view this module learns

1. a low-dimensional, noise-stable representation of the map summaries;
2. an inverse model from that representation to all five coordinates; and
3. a smooth forward emulator from the five coordinates back to the
   representation.

An input map is anomalous when no parameter estimate supplied by the inverse
model can reproduce its summaries under the learned InD forward model.  All
scalers, regressors, covariance estimates, and empirical tail calibrations are
fit using training maps only.  Test maps are scored independently.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from baseline_global_stats import write_submission
from generate_smoke_submission import validate_archive
from validate_feature_detectors import (
    score_report,
    stable_feature_mask,
    tail_calibrate,
)


EXPECTED_TEST_SIZE = 10_000
REALIZATIONS_PER_COSMOLOGY = 256


def frozen_convex_blend(
    incumbent: np.ndarray,
    candidate: np.ndarray,
    *,
    candidate_weight: float,
) -> np.ndarray:
    """Blend two row-aligned frozen scores without any test-batch fitting."""
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate_weight must be between zero and one")
    incumbent_values = np.asarray(incumbent, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if incumbent_values.shape != candidate_values.shape:
        raise ValueError(
            "Incumbent and candidate scores must have the same shape, got "
            f"{incumbent_values.shape} and {candidate_values.shape}"
        )
    return (
        (1.0 - candidate_weight) * incumbent_values
        + candidate_weight * candidate_values
    )


def central_difference_jacobian(
    predictor: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    *,
    step: float = 1.0e-3,
) -> np.ndarray:
    """Differentiate a batched forward emulator with respect to its inputs."""
    values = np.asarray(parameters, dtype=np.float64)
    columns = []
    for parameter in range(values.shape[1]):
        offset = np.zeros_like(values)
        offset[:, parameter] = step
        upper = np.asarray(predictor(values + offset), dtype=np.float64)
        lower = np.asarray(predictor(values - offset), dtype=np.float64)
        columns.append((upper - lower) / (2.0 * step))
    return np.stack(columns, axis=-1)


def nuisance_hardened_quadratic(
    residuals: np.ndarray,
    jacobians: np.ndarray,
    precision: np.ndarray,
    *,
    ridge: float = 1.0e-8,
) -> np.ndarray:
    """Return the residual energy after profiling local nuisance directions."""
    values = np.asarray(residuals, dtype=np.float64)
    tangents = np.asarray(jacobians, dtype=np.float64)
    metric = np.asarray(precision, dtype=np.float64)
    metric_tangents = np.einsum("de,nep->ndp", metric, tangents, optimize=True)
    fisher = np.einsum(
        "ndp,ndq->npq", tangents, metric_tangents, optimize=True
    )
    if ridge:
        fisher = fisher + float(ridge) * np.eye(fisher.shape[-1])[None, :, :]
    score_gradient = np.einsum(
        "ndp,nd->np", metric_tangents, values, optimize=True
    )
    profiled_step = np.linalg.solve(fisher, score_gradient[..., None])[..., 0]
    raw_energy = np.einsum(
        "nd,de,ne->n", values, metric, values, optimize=True
    )
    removed_energy = np.einsum(
        "np,np->n", score_gradient, profiled_step, optimize=True
    )
    return np.maximum(raw_energy - removed_energy, 0.0)


@dataclass(frozen=True)
class ViewSpec:
    name: str
    selected: np.ndarray
    components: int


class ForwardResidualModel:
    """Estimate physical coordinates and score the unexplained feature residual."""

    def __init__(
        self,
        *,
        components: int,
        trees: int,
        seed: int,
        ridge_alpha: float,
        polynomial_degree: int,
        refine_neighbors: int,
    ) -> None:
        self.components = components
        self.trees = trees
        self.seed = seed
        self.ridge_alpha = ridge_alpha
        self.polynomial_degree = polynomial_degree
        self.refine_neighbors = refine_neighbors

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ForwardResidualModel":
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.float32)
        self.feature_scaler = StandardScaler().fit(values)
        scaled = self.feature_scaler.transform(values).astype(np.float32)
        n_components = min(self.components, scaled.shape[1], len(scaled) - 1)
        self.reducer = PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=4,
            random_state=self.seed,
        ).fit(scaled)
        latent = self.reducer.transform(scaled).astype(np.float32)

        self.label_scaler = StandardScaler().fit(targets)
        normalized_labels = self.label_scaler.transform(targets).astype(np.float32)
        self.inverse = ExtraTreesRegressor(
            n_estimators=self.trees,
            max_features=0.8,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=self.seed + 1,
        ).fit(latent, normalized_labels)

        self.polynomial = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=True,
        ).fit(normalized_labels)
        design = self.polynomial.transform(normalized_labels).astype(np.float32)
        self.forward = Ridge(
            alpha=self.ridge_alpha,
            fit_intercept=False,
        ).fit(design, latent)
        forward_at_truth = self.forward.predict(design).astype(np.float32)
        self.residual_scale = np.std(latent - forward_at_truth, axis=0).clip(1e-3)
        self.training_normalized_labels = normalized_labels
        self.training_forward_latent = forward_at_truth
        if self.refine_neighbors > 0:
            self.neighbor_index = NearestNeighbors(
                n_neighbors=min(self.refine_neighbors, len(normalized_labels)),
                algorithm="auto",
                n_jobs=-1,
            ).fit(normalized_labels)
        else:
            self.neighbor_index = None
        return self

    def latent(self, features: np.ndarray) -> np.ndarray:
        scaled = self.feature_scaler.transform(
            np.asarray(features, dtype=np.float32)
        ).astype(np.float32)
        return self.reducer.transform(scaled).astype(np.float32)

    def predict_normalized_labels_from_latent(self, latent: np.ndarray) -> np.ndarray:
        prediction = self.inverse.predict(latent).astype(np.float32)
        # Extra-trees already stays inside the componentwise training support;
        # clipping makes that final-eligibility property explicit and stable.
        return np.clip(prediction, -4.0, 4.0)

    def forward_latent_from_normalized_labels(
        self, normalized_labels: np.ndarray
    ) -> np.ndarray:
        design = self.polynomial.transform(
            np.asarray(normalized_labels, dtype=np.float64)
        )
        return np.asarray(self.forward.predict(design), dtype=np.float64)

    def residual_and_labels(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latent = self.latent(features)
        normalized_labels = self.predict_normalized_labels_from_latent(latent)
        design = self.polynomial.transform(normalized_labels).astype(np.float32)
        expected = self.forward.predict(design).astype(np.float32)
        if self.neighbor_index is not None:
            candidate_indices = self.neighbor_index.kneighbors(
                normalized_labels, return_distance=False
            )
            refined_labels = np.empty_like(normalized_labels)
            refined_expected = np.empty_like(expected)
            # Bound the temporary tensor: 512 x 256 x 96 float32 is about 50 MB.
            for start in range(0, len(latent), 512):
                stop = min(start + 512, len(latent))
                indices = candidate_indices[start:stop]
                candidates = self.training_forward_latent[indices]
                delta = (
                    latent[start:stop, None, :] - candidates
                ) / self.residual_scale[None, None, :]
                distance = np.einsum("bkd,bkd->bk", delta, delta)
                best = np.argmin(distance, axis=1)
                rows = np.arange(stop - start)
                refined_expected[start:stop] = candidates[rows, best]
                refined_labels[start:stop] = self.training_normalized_labels[
                    indices[rows, best]
                ]

            continuous_delta = (latent - expected) / self.residual_scale[None, :]
            refined_delta = (latent - refined_expected) / self.residual_scale[None, :]
            continuous_distance = np.einsum(
                "nd,nd->n", continuous_delta, continuous_delta
            )
            refined_distance = np.einsum("nd,nd->n", refined_delta, refined_delta)
            use_refined = refined_distance < continuous_distance
            expected[use_refined] = refined_expected[use_refined]
            normalized_labels[use_refined] = refined_labels[use_refined]
        labels = self.label_scaler.inverse_transform(normalized_labels).astype(np.float32)
        return latent - expected, labels

    def residual_labels_and_jacobian(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        residual, labels = self.residual_and_labels(features)
        normalized_labels = self.label_scaler.transform(labels)
        jacobian = central_difference_jacobian(
            self.forward_latent_from_normalized_labels,
            normalized_labels,
        )
        return residual, labels, jacobian


class ResidualScorer:
    """Shared-covariance and robust-coordinate scores calibrated on InD residuals."""

    def fit(self, residuals: np.ndarray) -> "ResidualScorer":
        calibration = np.asarray(residuals, dtype=np.float64)
        self.oas = OAS(assume_centered=False).fit(calibration)
        self.center = np.median(calibration, axis=0)
        mad = np.median(np.abs(calibration - self.center), axis=0)
        standard = calibration.std(axis=0)
        self.scale = np.maximum(1.4826 * mad, 0.20 * standard).clip(1e-6)
        self.tail_count = max(1, min(12, calibration.shape[1] // 4))
        return self

    def score(
        self,
        residuals: np.ndarray,
        jacobians: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        values = np.asarray(residuals, dtype=np.float64)
        delta = values - self.oas.location_[None, :]
        chi2 = np.einsum("ni,ij,nj->n", delta, self.oas.precision_, delta)
        robust = np.abs((values - self.center[None, :]) / self.scale[None, :])
        top = np.partition(
            robust,
            kth=robust.shape[1] - self.tail_count,
            axis=1,
        )[:, -self.tail_count :]
        scores = {
            "chi2": chi2,
            "coordinate_tail": np.mean(top, axis=1),
        }
        if jacobians is not None:
            scores["nuisance_hardened_chi2"] = nuisance_hardened_quadratic(
                delta,
                jacobians,
                self.oas.precision_,
                ridge=0.0,
            )
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-a", type=Path, required=True)
    parser.add_argument("--train-b", type=Path, required=True)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--incumbent-scores", type=Path)
    parser.add_argument("--incumbent-key", default="candidate")
    parser.add_argument("--blend-score")
    parser.add_argument("--blend-weight", type=float)
    parser.add_argument("--blend-name", default="incumbent75_hand_forward25")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--trees", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--polynomial-degree", type=int, default=3)
    parser.add_argument("--refine-neighbors", type=int, default=256)
    parser.add_argument(
        "--views",
        nargs="+",
        default=("spectrum", "hand", "scattering", "all"),
    )
    return parser.parse_args()


def make_views(names: np.ndarray, stable: np.ndarray) -> list[ViewSpec]:
    names = names.astype(str)
    definitions = {
        "radial": (np.char.startswith(names, "ps_log_radial_"), 24),
        "spectrum": (np.char.startswith(names, "ps_"), 48),
        "wavelet8": (np.char.startswith(names, "wavelet_"), 8),
        "wavelet40": (np.char.startswith(names, "wavelet_"), 40),
        "hand": (~np.char.startswith(names, "scat_"), 64),
        "scattering": (np.char.startswith(names, "scat_"), 64),
        "all": (np.ones(len(names), dtype=bool), 96),
    }
    output = []
    for name, (selected, components) in definitions.items():
        selected = selected & stable
        if int(selected.sum()) >= 8:
            output.append(ViewSpec(name, selected, components))
    return output


def fit_view(
    spec: ViewSpec,
    x_a: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    args: argparse.Namespace,
    offset: int,
) -> ForwardResidualModel:
    model = ForwardResidualModel(
        components=spec.components,
        trees=args.trees,
        seed=args.seed + offset,
        ridge_alpha=args.ridge_alpha,
        polynomial_degree=args.polynomial_degree,
        refine_neighbors=args.refine_neighbors,
    )
    return model.fit(x_a[indices][:, spec.selected], labels[indices])


def validate(
    args: argparse.Namespace,
    x_a: np.ndarray,
    x_b: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    views: list[ViewSpec],
    proxy: np.lib.npyio.NpzFile,
    *,
    scores_output: Path | None = None,
) -> dict:
    x_proxy = proxy["features"].astype(np.float32)
    proxy_groups = proxy["source_index"] // REALIZATIONS_PER_COSMOLOGY
    proxy_types = proxy["proxy_type"].astype(str)
    splitter = GroupKFold(n_splits=args.folds)
    fold_reports = []
    score_arrays: dict[str, np.ndarray] = {}
    for fold, (fit_indices, val_indices) in enumerate(splitter.split(x_a, groups=groups)):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        fit_group_values = np.unique(groups[fit_indices])
        proxy_val = ~np.isin(proxy_groups, fit_group_values)
        calibrated_ind: dict[str, np.ndarray] = {}
        calibrated_proxy: dict[str, np.ndarray] = {}
        report: dict = {"fold": fold, "scores": {}, "inverse_r2": {}}

        for view_number, spec in enumerate(views):
            model = fit_view(
                spec, x_a, labels, fit_indices, args, fold * 100 + view_number * 10
            )
            calibration_residual, _, calibration_jacobian = (
                model.residual_labels_and_jacobian(
                x_b[fit_indices][:, spec.selected]
                )
            )
            ind_residual, ind_labels, ind_jacobian = (
                model.residual_labels_and_jacobian(
                x_b[val_indices][:, spec.selected]
                )
            )
            proxy_residual, _, proxy_jacobian = (
                model.residual_labels_and_jacobian(
                x_proxy[proxy_val][:, spec.selected]
                )
            )
            report["inverse_r2"][spec.name] = [
                float(value)
                for value in r2_score(
                    labels[val_indices], ind_labels, multioutput="raw_values"
                )
            ]
            scorer = ResidualScorer().fit(calibration_residual)
            calibration_raw = scorer.score(
                calibration_residual, calibration_jacobian
            )
            ind_raw = scorer.score(ind_residual, ind_jacobian)
            proxy_raw = scorer.score(proxy_residual, proxy_jacobian)
            for score_name in calibration_raw:
                name = f"{spec.name}_{score_name}"
                calibrated_ind[name] = tail_calibrate(
                    ind_raw[score_name], calibration_raw[score_name]
                )
                calibrated_proxy[name] = tail_calibrate(
                    proxy_raw[score_name], calibration_raw[score_name]
                )
                report["scores"][name] = score_report(
                    calibrated_ind[name], calibrated_proxy[name], proxy_types[proxy_val]
                )
                print(
                    f"fold={fold} {name}: {report['scores'][name]['all']:.6f}",
                    flush=True,
                )

        for suffix in ("chi2", "nuisance_hardened_chi2", "coordinate_tail"):
            members = [
                f"{spec.name}_{suffix}"
                for spec in views
                if f"{spec.name}_{suffix}" in calibrated_ind
            ]
            if len(members) < 2:
                continue
            ensemble_name = f"conditional_{suffix}_mean"
            ind_score = np.mean([calibrated_ind[name] for name in members], axis=0)
            proxy_score = np.mean([calibrated_proxy[name] for name in members], axis=0)
            calibrated_ind[ensemble_name] = ind_score
            calibrated_proxy[ensemble_name] = proxy_score
            report["scores"][ensemble_name] = score_report(
                ind_score, proxy_score, proxy_types[proxy_val]
            )
            print(
                f"fold={fold} {ensemble_name}: "
                f"{report['scores'][ensemble_name]['all']:.6f}",
                flush=True,
            )
        if scores_output is not None:
            score_arrays[f"fold{fold}_ind_indices"] = val_indices
            score_arrays[f"fold{fold}_proxy_positions"] = np.flatnonzero(proxy_val)
            score_arrays[f"fold{fold}_proxy_type"] = proxy_types[proxy_val]
            for name, values in calibrated_ind.items():
                score_arrays[f"fold{fold}_ind_{name}"] = values
            for name, values in calibrated_proxy.items():
                score_arrays[f"fold{fold}_proxy_{name}"] = values
        fold_reports.append(report)

    names = list(fold_reports[0]["scores"])
    summary = {
        name: {
            "mean": float(np.mean([fold["scores"][name]["all"] for fold in fold_reports])),
            "worst": float(np.min([fold["scores"][name]["all"] for fold in fold_reports])),
            "folds": [float(fold["scores"][name]["all"]) for fold in fold_reports],
        }
        for name in names
    }
    if scores_output is not None:
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores_output, **score_arrays)
    return {"fold_reports": fold_reports, "summary": summary}


def produce(
    args: argparse.Namespace,
    x_a: np.ndarray,
    x_b: np.ndarray,
    labels: np.ndarray,
    views: list[ViewSpec],
    test: np.lib.npyio.NpzFile,
) -> dict:
    x_test = test["features"].astype(np.float32)
    if len(x_test) != EXPECTED_TEST_SIZE:
        raise ValueError(f"Expected {EXPECTED_TEST_SIZE} test rows, got {len(x_test)}")
    calibrated: dict[str, np.ndarray] = {}
    raw_test: dict[str, np.ndarray] = {}
    all_indices = np.arange(len(x_a))
    for view_number, spec in enumerate(views):
        model = fit_view(spec, x_a, labels, all_indices, args, view_number * 10)
        calibration_residual, _, calibration_jacobian = (
            model.residual_labels_and_jacobian(x_b[:, spec.selected])
        )
        test_residual, _, test_jacobian = model.residual_labels_and_jacobian(
            x_test[:, spec.selected]
        )
        scorer = ResidualScorer().fit(calibration_residual)
        calibration_raw = scorer.score(calibration_residual, calibration_jacobian)
        test_raw = scorer.score(test_residual, test_jacobian)
        for score_name in calibration_raw:
            name = f"{spec.name}_{score_name}"
            raw_test[name] = test_raw[score_name]
            calibrated[name] = tail_calibrate(test_raw[score_name], calibration_raw[score_name])
            print(f"built {name}", flush=True)

    for suffix in ("chi2", "nuisance_hardened_chi2", "coordinate_tail"):
        members = [
            f"{spec.name}_{suffix}"
            for spec in views
            if f"{spec.name}_{suffix}" in calibrated
        ]
        if len(members) >= 2:
            calibrated[f"conditional_{suffix}_mean"] = np.mean(
                [calibrated[name] for name in members], axis=0
            )

    if args.output_dir is None:
        raise ValueError("--output-dir is required with --test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_to_package = calibrated
    incumbent_values: np.ndarray | None = None
    blended_values: np.ndarray | None = None
    if args.incumbent_scores is not None:
        if args.blend_score not in calibrated or args.blend_weight is None:
            raise ValueError(
                "--incumbent-scores requires a valid --blend-score and --blend-weight"
            )
        incumbent_cache = np.load(args.incumbent_scores)
        if args.incumbent_key not in incumbent_cache.files:
            raise ValueError(
                f"Missing incumbent key {args.incumbent_key!r} in "
                f"{args.incumbent_scores}"
            )
        incumbent_values = np.asarray(
            incumbent_cache[args.incumbent_key], dtype=np.float64
        )
        blended_values = frozen_convex_blend(
            incumbent_values,
            calibrated[args.blend_score],
            candidate_weight=args.blend_weight,
        )
        scores_to_package = {args.blend_name: blended_values}
    artifacts = {}
    for name, scores in scores_to_package.items():
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite scores in {name}")
        path = args.output_dir / f"{name}.zip"
        write_submission(path, scores)
        validate_archive(path)
        artifacts[name] = str(path)
    np.savez_compressed(
        args.output_dir / "scores.npz",
        **{f"raw_{name}": score for name, score in raw_test.items()},
        **{f"calibrated_{name}": score for name, score in calibrated.items()},
        **(
            {"incumbent": incumbent_values, "blended_candidate": blended_values}
            if incumbent_values is not None and blended_values is not None
            else {}
        ),
    )
    return {
        "artifacts": artifacts,
        "blend": (
            {
                "incumbent_scores": str(args.incumbent_scores),
                "incumbent_key": args.incumbent_key,
                "candidate_score": args.blend_score,
                "candidate_weight": args.blend_weight,
            }
            if incumbent_values is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.proxy is None and args.test is None:
        raise ValueError("At least one of --proxy or --test is required")
    started = time.perf_counter()
    train_a = np.load(args.train_a)
    train_b = np.load(args.train_b)
    names = train_a["feature_names"].astype(str)
    if not np.array_equal(names, train_b["feature_names"].astype(str)):
        raise ValueError("Feature names differ between train caches")
    x_a = train_a["features"].astype(np.float32)
    x_b = train_b["features"].astype(np.float32)
    labels = train_a["labels"].astype(np.float32)
    if x_a.shape != x_b.shape or labels.shape != (len(x_a), 5):
        raise ValueError("Unexpected train feature or label shape")
    stable = stable_feature_mask(x_a, x_b)
    requested = set(args.views)
    views = [view for view in make_views(names, stable) if view.name in requested]
    if not views:
        raise ValueError(f"No valid requested views: {sorted(requested)}")
    print(
        "views: " + ", ".join(f"{view.name}={view.selected.sum()}" for view in views),
        flush=True,
    )
    output: dict = {
        "config": {
            "trees": args.trees,
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "polynomial_degree": args.polynomial_degree,
            "refine_neighbors": args.refine_neighbors,
            "views": [view.name for view in views],
        },
        "views": {view.name: int(view.selected.sum()) for view in views},
    }
    groups = np.arange(len(x_a)) // REALIZATIONS_PER_COSMOLOGY
    if args.proxy is not None:
        proxy = np.load(args.proxy)
        if not np.array_equal(names, proxy["feature_names"].astype(str)):
            raise ValueError("Feature names differ in proxy cache")
        output["validation"] = validate(
            args,
            x_a,
            x_b,
            labels,
            groups,
            views,
            proxy,
            scores_output=args.scores_output,
        )
    if args.test is not None:
        test = np.load(args.test)
        if not np.array_equal(names, test["feature_names"].astype(str)):
            raise ValueError("Feature names differ in test cache")
        output["production"] = produce(args, x_a, x_b, labels, views, test)
    output["elapsed_seconds"] = time.perf_counter() - started
    if args.validation_output is not None:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
