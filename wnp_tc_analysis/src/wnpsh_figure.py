"""Publication rendering for the fixed-contour WNPSH metrics figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import SUPPLEMENTARY_FIGURES, WORK


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "font.size": 10.5,
        "axes.labelsize": 10.8,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _outer_frame(axes) -> None:
    """Keep only the perimeter spines of the 2 x 2 panel grid."""
    nrows, ncols = axes.shape
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            ax.spines["left"].set_visible(col == 0)
            ax.spines["right"].set_visible(col == ncols - 1)
            ax.spines["top"].set_visible(row == 0)
            ax.spines["bottom"].set_visible(row == nrows - 1)
            if col == ncols - 1:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.tick_params(
                    axis="y",
                    left=False,
                    labelleft=False,
                    right=True,
                    labelright=True,
                )
            else:
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
            if row == 0:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)


def make_figure(output: Path | None = None) -> None:
    """Render S8 from the frozen annual values and trend estimates."""
    output = output or SUPPLEMENTARY_FIGURES / "FigS08_wnpsh_fixed_contour_metrics"
    annual = pd.read_csv(WORK / "data" / "wnp_tc_eddy_wnpsh_annual.csv")
    trends = pd.read_csv(
        WORK / "analysis" / "07_wnpsh_dynamic_metric" / "wnpsh_metric_trends.csv"
    )
    panels = [
        ("wpsh_area", "WNPSH area\n(10$^6$ km²)"),
        ("wpsh_intensity", "WNPSH intensity"),
        ("ridge_line", "Ridge latitude (°N)"),
        ("west_ridge_point", "Western ridge point (°E)"),
    ]
    end_year = max(
        int(annual.loc[annual[column].notna(), "year"].max())
        for column, _ in panels
    )
    display_start_year = 1960
    first_tick = ((display_start_year + 9) // 10) * 10
    last_tick = (end_year // 10) * 10
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.2, 6.6),
        sharex=True,
        constrained_layout=True,
    )
    for i, (column, ylabel) in enumerate(panels):
        ax = axes.flat[i]
        old = annual.loc[annual.year.le(1981)]
        recent = annual.loc[annual.year.ge(1982)]
        ax.plot(old.year, old[column], color="0.68", lw=1.15, label="1966–1981")
        ax.plot(
            recent.year,
            recent[column],
            color="#4C72B0",
            lw=1.45,
            label="1982–2025",
        )
        row = trends.loc[(trends.label.eq(column)) & trends.start.eq(1982)].iloc[0]
        slope = float(row.sen_slope_per_year)
        intercept = float(recent[column].mean()) - slope * float(recent.year.mean())
        ax.plot(
            recent.year,
            intercept + slope * recent.year,
            color="#7C3E50",
            ls="--",
            lw=1.2,
            label="Trend",
        )
        ax.axvline(1982, color="0.50", ls=":", lw=0.9)
        ax.set_ylabel(ylabel)
        ax.text(
            0.015,
            0.985,
            "abcd"[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            zorder=20,
        )
    for ax in axes[1, :]:
        ax.set_xlabel("Year")
        ax.set_xlim(display_start_year, end_year)
        ax.set_xticks(list(range(first_tick, last_tick + 1, 10)))
    _outer_frame(axes)
    axes[0, 0].legend(
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.045, 0.98),
        handlelength=1.4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output.with_suffix(f".{suffix}"),
            dpi=360,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
