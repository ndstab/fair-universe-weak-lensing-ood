#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np

from validate_high_order_profile import curve_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--incumbent-scores", type=Path, required=True)
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
    names = sorted(
        key.removeprefix(f"fold{folds[0]}_ind_")
        for key in candidate.files
        if key.startswith(f"fold{folds[0]}_ind_")
        and key != f"fold{folds[0]}_ind_indices"
    )
    rows = []
    for name in names:
        for weight in np.arange(0.05, 0.55, 0.05):
            fold_rows = []
            for fold in folds:
                ind_indices = candidate[f"fold{fold}_ind_indices"].astype(np.int64)
                proxy_positions = candidate[f"fold{fold}_proxy_positions"].astype(
                    np.int64
                )
                base_ind = incumbent["audit_ind"][ind_indices]
                base_proxy = incumbent["audit_proxy"][proxy_positions]
                ind = candidate[f"fold{fold}_ind_{name}"]
                proxy = candidate[f"fold{fold}_proxy_{name}"]
                base_curve = curve_components(base_ind, base_proxy)
                standalone = curve_components(ind, proxy)
                blended = curve_components(
                    (1.0 - weight) * base_ind + weight * ind,
                    (1.0 - weight) * base_proxy + weight * proxy,
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "base": base_curve,
                        "standalone": standalone,
                        "blend": blended,
                        "delta_all": float(blended["all"] - base_curve["all"]),
                        "delta_low_fpr": float(
                            blended["low_fpr_mean"]
                            - base_curve["low_fpr_mean"]
                        ),
                    }
                )
            rows.append(
                {
                    "name": name,
                    "weight": float(weight),
                    "mean_delta_all": float(
                        np.mean([row["delta_all"] for row in fold_rows])
                    ),
                    "mean_delta_low_fpr": float(
                        np.mean([row["delta_low_fpr"] for row in fold_rows])
                    ),
                    "minimum_delta_all": float(
                        np.min([row["delta_all"] for row in fold_rows])
                    ),
                    "minimum_delta_low_fpr": float(
                        np.min([row["delta_low_fpr"] for row in fold_rows])
                    ),
                    "folds": fold_rows,
                }
            )
    rows.sort(
        key=lambda row: (row["mean_delta_all"], row["mean_delta_low_fpr"]),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "best_overall": rows[:12],
                "positive_both": [
                    row
                    for row in rows
                    if row["minimum_delta_all"] > 0.0
                    and row["minimum_delta_low_fpr"] > 0.0
                ][:12],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
