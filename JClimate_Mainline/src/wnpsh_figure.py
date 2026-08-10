"""Publication rendering for the fixed-contour WNPSH metrics figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import SUPPLEMENTARY_FIGURES, WORK
from .figure_typography import scale_figure_typography


def make_figure(output: Path | None = None) -> None:
    """Render S9 solely from existing annual values and saved trend estimates."""
    output = output or SUPPLEMENTARY_FIGURES / "FigS09_wnpsh_fixed_contour_metrics"
    annual = pd.read_csv(WORK / "data" / "jcli_eddy_wnpsh_annual.csv")
    trends = pd.read_csv(WORK / "analysis" / "07_wnpsh_dynamic_metric" / "wnpsh_metric_trends.csv")
    panels = [
        ("wpsh_area", "WNPSH Area\n(10$^6$ km²)"),
        ("wpsh_intensity", "WNPSH\nIntensity"),
        ("ridge_line", "Ridge Latitude\n(°N)"),
        ("west_ridge_point", "West Ridge Point\n(°E)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), sharex=True, constrained_layout=True)
    for i, (column, ylabel) in enumerate(panels):
        ax = axes.flat[i]
        old = annual.loc[annual.year.le(1981)]
        recent = annual.loc[annual.year.ge(1982)]
        ax.plot(old.year, old[column], color="0.70", lw=1.2, label="1940–1981 (context only)")
        ax.plot(recent.year, recent[column], color="#4C72B0", lw=1.5, label="1982–2025 (trend period)")
        row = trends.loc[(trends.label.eq(column)) & trends.start.eq(1982)].iloc[0]
        slope = float(row.sen_slope_per_year)
        intercept = float(recent[column].mean()) - slope * float(recent.year.mean())
        ax.plot(recent.year, intercept + slope * recent.year, color="#7C3E50", ls="--", lw=1.2,
                label="Theil–Sen trend (1982–2025)")
        ax.axvline(1982, color="0.55", ls=":", lw=1.0)
        ax.set_ylabel(ylabel)
        ax.text(
            0.98,
            0.96,
            f"Sen={float(row.sen_slope_per_decade):+.3g} decade$^{{-1}}$\nBH q={float(row.mk_p_fdr_bh):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#5F2637",
        )
        if i % 2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines["left"].set_visible(False)
        else:
            ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.text(0.02, 0.98, "abcd"[i], transform=ax.transAxes, va="top", fontweight="bold")
    fig.supxlabel("Year", y=0.055)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    # S9 is read at reduced supplementary-figure scale; apply one additional
    # modest step beyond the release-wide typography scale.
    scale_figure_typography(fig, scale=1.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=320, bbox_inches="tight")
    plt.close(fig)
