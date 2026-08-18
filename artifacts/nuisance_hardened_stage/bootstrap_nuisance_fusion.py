#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np

from validate_feature_detectors import score_report
from validate_high_order_profile import curve_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--incumbent-scores", type=Path, required=True)
    parser.add_argument("--name", default="hand_chi2")
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = np.load(args.candidate_scores)
    incumbent = np.load(args.incumbent_scores)
    folds = sorted(
        int(match.group(1))
        for key in candidate.files
        if (match := re.fullmatch(r"fold(\d+)_ind_indices", key))
    )
    ind_candidate = np.empty_like(incumbent["audit_ind"], dtype=np.float64)
    proxy_candidate = np.empty_like(incumbent["audit_proxy"], dtype=np.float64)
    for fold in folds:
        ind_candidate[candidate[f"fold{fold}_ind_indices"]] = candidate[
            f"fold{fold}_ind_{args.name}"
        ]
        proxy_candidate[candidate[f"fold{fold}_proxy_positions"]] = candidate[
            f"fold{fold}_proxy_{args.name}"
        ]
    base_ind = incumbent["audit_ind"].astype(np.float64)
    base_proxy = incumbent["audit_proxy"].astype(np.float64)
    blended_ind = (1.0 - args.weight) * base_ind + args.weight * ind_candidate
    blended_proxy = (
        (1.0 - args.weight) * base_proxy + args.weight * proxy_candidate
    )
    base_curve = curve_components(base_ind, base_proxy)
    blend_curve = curve_components(blended_ind, blended_proxy)
    ind_group = np.arange(len(base_ind)) // 256
    proxy_group = incumbent["proxy_source_index"].astype(np.int64) // 256
    groups = np.unique(ind_group)
    ind_positions = {group: np.flatnonzero(ind_group == group) for group in groups}
    proxy_positions = {group: np.flatnonzero(proxy_group == group) for group in groups}
    rng = np.random.default_rng(args.seed)
    delta_all = []
    delta_low = []
    for _ in range(args.replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        ind_select = np.concatenate([ind_positions[group] for group in sampled])
        proxy_select = np.concatenate([proxy_positions[group] for group in sampled])
        sampled_base = curve_components(base_ind[ind_select], base_proxy[proxy_select])
        sampled_blend = curve_components(
            blended_ind[ind_select], blended_proxy[proxy_select]
        )
        delta_all.append(sampled_blend["all"] - sampled_base["all"])
        delta_low.append(
            sampled_blend["low_fpr_mean"] - sampled_base["low_fpr_mean"]
        )
    proxy_type = incumbent["proxy_type"].astype(str)
    print(
        json.dumps(
            {
                "name": args.name,
                "weight": args.weight,
                "replicates": args.replicates,
                "base": base_curve,
                "blend": blend_curve,
                "delta": {
                    "all": float(blend_curve["all"] - base_curve["all"]),
                    "low_fpr": float(
                        blend_curve["low_fpr_mean"] - base_curve["low_fpr_mean"]
                    ),
                },
                "bootstrap": {
                    "all_lower_2p5": float(np.quantile(delta_all, 0.025)),
                    "all_median": float(np.median(delta_all)),
                    "all_upper_97p5": float(np.quantile(delta_all, 0.975)),
                    "low_fpr_lower_2p5": float(np.quantile(delta_low, 0.025)),
                    "low_fpr_median": float(np.median(delta_low)),
                    "low_fpr_upper_97p5": float(np.quantile(delta_low, 0.975)),
                },
                "base_by_proxy": score_report(base_ind, base_proxy, proxy_type),
                "blend_by_proxy": score_report(
                    blended_ind, blended_proxy, proxy_type
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
