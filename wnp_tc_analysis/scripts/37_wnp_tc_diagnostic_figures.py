from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, FIGURES, ensure_dirs


def run():
    ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    annual = pd.read_csv(DATA / "wnp_tc_redistribution_index_annual.csv")
    summary = pd.read_csv(DATA / "wnp_tc_genesis_propagation_summary.csv")
    land = pd.read_csv(DATA / "wnp_tc_landfall_unique_summary.csv")
    robust = pd.read_csv(DATA / "wnp_tc_robustness_matrix.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    ax = axes[0, 0]
    for agency, color in zip(["PRIMARY", "USA", "TOKYO", "CMA"], ["black", "#3B6FB6", "#D17A22", "#4B9B6D"]):
        d = annual[(annual.agency == agency) & (annual.weighting == "track_point")]
        ax.plot(d.year, d.index_oos, lw=1.0, label=agency, color=color, alpha=0.9)
    ax.axhline(0, color="0.5", lw=0.6); ax.legend(frameon=False, ncol=2)
    ax.set(xlabel="Year", ylabel="OOS redistribution index")

    ax = axes[0, 1]
    s = summary[summary.catalog.isin(["PRIMARY", "USA", "TOKYO", "CMA"])]
    x = np.arange(len(s)); width = 0.35
    ax.bar(x - width/2, s.genesis_projection_fraction * 100, width, label="Genesis")
    ax.bar(x + width/2, s.propagation_projection_fraction * 100, width, label="Propagation")
    ax.set_xticks(x, s.catalog); ax.set_ylabel("Projection contribution (%)"); ax.legend(frameon=False)

    ax = axes[1, 0]
    r = robust[(robust.catalog.isin(["PRIMARY", "PRIMARY_TY"])) & (robust.weighting == "track_point")]
    order = ["grid_primary", "block_sensitivity", "end_2024_drop_1995", "start_1967_drop_1996", "exclude_2020_2025", "typhoon_threshold"]
    vals = [r.loc[r.analysis == k, "block_permutation_p"].median() for k in order]
    ax.barh(np.arange(len(order)), vals, color="#6C8EBF")
    ax.axvline(0.05, color="#B23A48", ls="--"); ax.set_yticks(np.arange(len(order)), order)
    ax.set_xlabel("Block-permutation p")

    ax = axes[1, 1]
    l = land[land.assignment_rule.isin(["first_any", "first_named"])]
    labels = l.agency + "\n" + l.assignment_rule
    ax.bar(np.arange(len(l)), l.north_share_change_percentage_points, color="#7A9E7E")
    ax.set_xticks(np.arange(len(l)), labels, rotation=45, ha="right")
    ax.set_ylabel("North-share change (percentage points)")
    for i, p in enumerate(l.north_share_block_p):
        ax.text(i, l.north_share_change_percentage_points.iloc[i], f"p={p:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    for label, ax in zip("abcd", axes.flat):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold")
    fig.savefig(FIGURES / "wnp_tc_diagnostic_evidence_overview.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    run()
