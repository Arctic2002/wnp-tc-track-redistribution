"""Publication rendering for the primary-record activity diagnostic (S4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .figure_typography import scale_figure_typography


PROJECT = Path(__file__).resolve().parents[2]


def make_figure(output: Path) -> None:
    """Render S4 with the saved Mann–Kendall p and BH-FDR q values visible."""
    annual = pd.read_csv(PROJECT / "data" / "processed" / "p1_annual.csv")
    trends = pd.read_csv(PROJECT / "data" / "processed" / "p1_trends.csv")
    trends = trends.loc[trends.scope.eq("annual")].set_index("var")
    panels = [
        ("genesis_count", "Genesis Frequency"),
        ("super_count", "Super-Typhoon Frequency"),
        ("lmi_mean", "Mean LMI (kt)"),
        ("ace_total", "ACE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    for index, (column, ylabel) in enumerate(panels):
        ax = axes.flat[index]
        data = annual.loc[:, ["season", column]].dropna()
        row = trends.loc[column]
        ax.plot(data.season, data[column], color="#4C72B0", lw=1.5)
        slope = float(row.sen_slope)
        intercept = float(data[column].mean()) - slope * float(data.season.mean())
        ax.plot(data.season, intercept + slope * data.season, color="#7C3E50", ls="--", lw=1.2)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.text(
            0.98,
            0.96,
            f"MK p={float(row.p_raw):.4f}\nBH q={float(row.p_fdr):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#5F2637",
        )
        ax.text(0.02, 0.98, "abcd"[index], transform=ax.transAxes, va="top", fontweight="bold")
        ax.spines["top"].set_visible(False)
        if index % 2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines["left"].set_visible(False)
        else:
            ax.spines["right"].set_visible(False)
    scale_figure_typography(fig, scale=1.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=320, bbox_inches="tight")
    plt.close(fig)
