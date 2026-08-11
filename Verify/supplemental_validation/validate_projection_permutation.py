from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "wnp_tc_analysis" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    block_permutation_projection,
    block_permutation_scalar,
    period_indices,
    projection_scores,
)


def main() -> None:
    years = np.arange(1966, 2026)
    early, late = period_indices(years)
    rows = []
    n_trials = 200
    nperm = 199
    for trial in range(n_trials):
        rng = np.random.default_rng(710000 + trial)
        fields = rng.dirichlet(np.ones(80), size=len(years))
        _, oos, _ = projection_scores(fields, years)
        _, old_p = block_permutation_scalar(
            oos, early, late, block=3, nperm=nperm, seed=720000 + trial, years=years
        )
        _, new_p = block_permutation_projection(
            fields, years, block=3, nperm=nperm, seed=720000 + trial
        )
        rows.append({"trial": trial, "fixed_score_p": old_p, "refit_pattern_p": new_p})

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent
    table.to_csv(out / "projection_null_calibration.csv", index=False)
    summary = {
        "n_trials": n_trials,
        "n_permutations_per_trial": nperm,
        "alpha_0.10_fixed_score": float((table["fixed_score_p"] <= 0.10).mean()),
        "alpha_0.10_refit_pattern": float((table["refit_pattern_p"] <= 0.10).mean()),
        "alpha_0.05_fixed_score": float((table["fixed_score_p"] <= 0.05).mean()),
        "alpha_0.05_refit_pattern": float((table["refit_pattern_p"] <= 0.05).mean()),
        "alpha_0.01_fixed_score": float((table["fixed_score_p"] <= 0.01).mean()),
        "alpha_0.01_refit_pattern": float((table["refit_pattern_p"] <= 0.01).mean()),
        "median_fixed_score_p": float(table["fixed_score_p"].median()),
        "median_refit_pattern_p": float(table["refit_pattern_p"].median()),
    }
    (out / "projection_null_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
