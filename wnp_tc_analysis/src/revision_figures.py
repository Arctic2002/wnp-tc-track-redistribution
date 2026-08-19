from __future__ import annotations

import argparse
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import MAIN_FIGURES, SUPPLEMENTARY_FIGURES, WORK
from .figure_typography import scale_figure_typography


COLORS = {"USA": "#4C72B0", "TOKYO": "#7E9CC4", "CMA": "#C76D6D"}
DISPLAY = {"USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}


def panel_labels(axes):
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).flat):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold")


def open_panel_axes(axes):
    """Use publication-style open axes; the right column carries its y axis."""
    for row in np.atleast_2d(axes):
        for column, ax in enumerate(row):
            ax.spines["top"].set_visible(False)
            if column == 0:
                ax.spines["right"].set_visible(False)
            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)


def save(fig, folder, stem, *, scale=1.10):
    folder.mkdir(parents=True, exist_ok=True)
    scale_figure_typography(fig, scale=scale)
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2(folder):
    annual = pd.read_csv(WORK / "analysis" / "03_common_storms" / "core_crossagency_annual.csv")
    red = pd.read_csv(WORK / "analysis" / "03_common_storms" / "core_crossagency_recheck_redistribution.csv")
    land = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_latitude_summary.csv")
    land = land.loc[(land["start"].eq(1966)) & (land["end"].eq(2025)) & land["metric"].eq("mean_lat") &
                    land["definition"].eq("first_landfall")]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), constrained_layout=True)
    for agency, color in COLORS.items():
        part = annual.loc[annual["agency"].eq(agency)]
        axes[0, 0].plot(part["season"], part["n_tc"], color=color, lw=1.1, label=DISPLAY[agency])
        axes[0, 1].plot(part["season"], part["mean_lmi_lat_common"], color=color, lw=1.1,
                        label=DISPLAY[agency])
    axes[0, 0].set(xlabel="Year", ylabel="TC count")
    axes[0, 1].set(xlabel="Year", ylabel="Mean LMI\nlatitude (°N)")
    axes[0, 0].legend(frameon=False, ncol=3)
    x = np.arange(3)
    red = red.set_index("agency").loc[["USA", "TOKYO", "CMA"]].reset_index()
    bars = axes[1, 0].bar(x, red["total_variation"], color=[COLORS[a] for a in red["agency"]])
    axes[1, 0].set(xticks=x, xticklabels=[DISPLAY[a] for a in red["agency"]], ylabel="Total-variation\ndistance")
    for bar, p in zip(bars, red["block_permutation_p"]):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .003, f"p={p:.4f}", ha="center", fontsize=8)
    land = land.set_index("agency").loc[["USA", "TOKYO", "CMA"]].reset_index()
    effect = land["period_difference"].to_numpy()
    lo = land["period_ci_low"].to_numpy(); hi = land["period_ci_high"].to_numpy()
    axes[1, 1].errorbar(x, effect, yerr=[effect-lo, hi-effect], fmt="none", ecolor="0.25", capsize=4)
    axes[1, 1].scatter(x, effect, s=55, c=[COLORS[a] for a in land["agency"]], zorder=3)
    axes[1, 1].axhline(0, color="0.45", lw=.8)
    axes[1, 1].set(xticks=x, xticklabels=[DISPLAY[a] for a in land["agency"]], ylabel="First-landfall latitude\nchange (°)")
    axes[1, 1].set_xlim(-0.3, 2.3)
    axes[1, 1].set_ylim(-0.1, float(hi.max() + 0.35))
    for i, row in enumerate(land.itertuples()):
        axes[1, 1].text(i, row.period_ci_high + .08, f"p={row.period_block_p:.4f}", ha="center", fontsize=7.5)
    open_panel_axes(axes)
    panel_labels(axes)
    save(fig, folder, "Fig02_cross_agency_core_v2", scale=1.32)


def figure6(folder):
    direct = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_latitude_summary.csv")
    direct = direct.loc[(direct["start"].eq(1966)) & (direct["end"].eq(2025)) & direct["metric"].eq("mean_lat")]
    common = pd.read_csv(WORK / "analysis" / "03_common_storms" / "common_storm_landfall_summary.csv")
    common = common.loc[common["metric"].eq("mean_landfall_latitude")]
    coast = pd.read_csv(WORK / "analysis" / "06_landfall_grouping" / "coast_grouping_sensitivity.csv")
    coast = coast.query("taiwan_rule == 'taiwan_north' and denominator == 'named_only'")
    cut = pd.read_csv(WORK / "analysis" / "02_cutpoint_sensitivity" / "cutpoint_sensitivity.csv")
    cut = cut.query("scheme == 'full_record' and metric == 'first_landfall_latitude_difference'")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    offsets = {"all_events": -.13, "first_landfall": .13}
    top_legends = []
    for definition, marker in [("all_events", "o"), ("first_landfall", "s")]:
        part = direct.loc[direct["definition"].eq(definition)].set_index("agency").loc[["USA", "TOKYO", "CMA"]]
        x = np.arange(3) + offsets[definition]
        eff = part["period_difference"].to_numpy(); lo = part["period_ci_low"].to_numpy(); hi = part["period_ci_high"].to_numpy()
        axes[0, 0].errorbar(x, eff, yerr=[eff-lo, hi-eff], fmt=marker, ms=5, capsize=3,
                            color="0.25", label="All events" if definition == "all_events" else "First landfall")
    axes[0, 0].axhline(0, color="0.5", lw=.8)
    axes[0, 0].set(xticks=np.arange(3), xticklabels=["USA", "JMA", "CMA"], ylabel="Latitude change (°)")
    top_legends.append(axes[0, 0].legend(frameon=False, loc="lower right"))

    offsets2 = {"all_events": -.13, "first_landfall": .13}
    for definition, marker in [("all_events", "o"), ("first_landfall", "s")]:
        part = common.loc[common["definition"].eq(definition)].set_index("agency").loc[["USA", "TOKYO", "CMA"]]
        x = np.arange(3) + offsets2[definition]
        eff = part["late_minus_early"].to_numpy(); lo = part["ci_low"].to_numpy(); hi = part["ci_high"].to_numpy()
        axes[0, 1].errorbar(x, eff, yerr=[eff-lo, hi-eff], fmt=marker, ms=5, capsize=3,
                            color="0.25", label="All events" if definition == "all_events" else "First landfall")
    axes[0, 1].axhline(0, color="0.5", lw=.8)
    axes[0, 1].set(xticks=np.arange(3), xticklabels=["USA", "JMA", "CMA"], ylabel="Common-storm latitude change (°)")
    top_legends.append(axes[0, 1].legend(frameon=False, loc="lower right"))

    rules = ["all_events", "first_any", "first_named"]
    labels = ["All events", "First", "First named"]
    for i, agency in enumerate(["USA", "TOKYO", "CMA"]):
        part = coast.loc[coast["agency"].eq(agency)].set_index("event_rule").loc[rules]
        axes[1, 0].plot(np.arange(3), part["change_percentage_points"], marker="o", color=COLORS[agency], label=DISPLAY[agency])
    axes[1, 0].axhline(0, color="0.5", lw=.8)
    axes[1, 0].set(xticks=np.arange(3), xticklabels=labels, ylabel="North-share change (percentage points)")
    axes[1, 0].tick_params(axis="x", rotation=0)
    axes[1, 0].legend(frameon=False, ncol=3)
    axes[1, 0].set_ylim(bottom=2.0)

    for agency, color in COLORS.items():
        part = cut.loc[cut["agency"].eq(agency)]
        axes[1, 1].plot(part["cutpoint"], part["effect"], marker="o", ms=3, color=color, label=DISPLAY[agency])
        significant = part["p_value"] < .05
        axes[1, 1].scatter(part.loc[significant, "cutpoint"], part.loc[significant, "effect"],
                           s=28, facecolors="none", edgecolors=color, linewidths=1.1)
    axes[1, 1].axvline(1996, color="0.4", ls="--", lw=.9)
    axes[1, 1].axhline(0, color="0.5", lw=.8)
    axes[1, 1].set(xlabel="First year of late period", ylabel="First-landfall latitude change (°)")
    axes[1, 1].set_ylim(bottom=0.60)
    fig.canvas.draw()
    for ax, legend in zip(axes[0, :], top_legends):
        axis_height_pt = ax.get_window_extent().height * 72.0 / fig.dpi
        legend.set_bbox_to_anchor(
            (1.0, 12.7 / axis_height_pt), transform=ax.transAxes
        )
    open_panel_axes(axes)
    panel_labels(axes)
    save(fig, folder, "Fig06_landfall_latitude_and_grouping", scale=1.27)


def figure_s3(folder):
    table = pd.read_csv(
        WORK
        / "analysis"
        / "06_landfall_grouping"
        / "coast_grouping_threshold_sensitivity.csv"
    )
    table = table.query(
        "taiwan_rule == 'taiwan_north' and denominator == 'named_only'"
    )
    rules = ["all_events", "first_any", "first_named"]
    titles = ["All events", "First any coast", "First named coast"]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.4, 3.8), sharex=True, constrained_layout=True
    )
    axes = np.asarray(axes).reshape(1, -1)
    for ax, rule, title in zip(axes.flat, rules, titles):
        for agency, color in COLORS.items():
            part = table.loc[
                table["event_rule"].eq(rule) & table["agency"].eq(agency)
            ].sort_values("threshold_km")
            ax.plot(
                part["threshold_km"],
                part["change_percentage_points"],
                color=color,
                lw=1.35,
                label=DISPLAY[agency],
            )
            significant = part["q_bh"] < 0.05
            ax.scatter(
                part.loc[significant, "threshold_km"],
                part.loc[significant, "change_percentage_points"],
                s=42,
                facecolors=color,
                edgecolors=color,
                zorder=3,
            )
            ax.scatter(
                part.loc[~significant, "threshold_km"],
                part.loc[~significant, "change_percentage_points"],
                s=42,
                facecolors="white",
                edgecolors=color,
                zorder=3,
            )
        ax.axhline(0, color="0.5", lw=.8)
        ax.set_title(title, pad=4)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(
            bottom={"all_events": 6.0, "first_any": 1.0, "first_named": 4.0}[rule]
        )
    axes[0, 0].set_ylabel("North-share change\n(percentage points)")
    axes[0, -1].set_ylabel("North-share change\n(percentage points)")
    for ax in axes.flat:
        ax.set_xlabel("Coastline-segment threshold (km)")
    axes[0, 1].legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
    )
    open_panel_axes(axes)
    panel_labels(axes)
    save(
        fig,
        folder,
        "FigS13_exclusive_coast_threshold_sensitivity",
        scale=1.18,
    )


def publish_landfall_supplementary(folder):
    figure_s3(folder)
    sources = {
        "FigS12_cutpoint_sensitivity": (
            WORK / "analysis" / "02_cutpoint_sensitivity" / "cutpoint_sensitivity"
        ),
        "FigS11_landfall_latitude_diagnostic": (
            WORK
            / "analysis"
            / "01_landfall_latitude"
            / "landfall_latitude_diagnostic"
        ),
    }
    for destination_stem, source in sources.items():
        for suffix in (".png", ".pdf"):
            shutil.copy2(
                source.with_suffix(suffix),
                folder / f"{destination_stem}{suffix}",
            )


def supplementary(folder):
    path = pd.read_csv(WORK / "analysis" / "04_track_density_sensitivity" / "path_definition_sensitivity.csv")
    path = path.query("grid_deg == 2.5 and block_years == 3")
    definitions = ["track_point", "storm_equal", "binary_occupancy", "line_length",
                   "pre_landfall_track_point", "full_life_track_point"]
    labels = ["Points", "Storm equal", "Binary", "Line length", "Pre-landfall", "Full life"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    width = .24; base = np.arange(len(definitions))
    for i, agency in enumerate(["USA", "TOKYO", "CMA"]):
        part = path.loc[path["agency"].eq(agency)].set_index("definition").loc[definitions]
        axes[0].bar(base + (i-1)*width, part["total_variation"], width=width, color=COLORS[agency], label=DISPLAY[agency])
        axes[1].bar(base + (i-1)*width, part["block_permutation_p"], width=width, color=COLORS[agency])
    axes[0].set(xticks=base, xticklabels=labels, ylabel="Total-variation distance")
    axes[1].set(xticks=base, xticklabels=labels, ylabel="Block-permutation p")
    axes[1].axhline(.05, color="0.45", ls=":", lw=1)
    axes[1].set_ylim(0, .055)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    # Place the key above panel a so it cannot cover the high pre-landfall bars.
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    open_panel_axes(axes)
    panel_labels(axes)
    save(fig, folder, "FigS06_path_definition_sensitivity", scale=1.22)

    climate = pd.read_csv(WORK / "analysis" / "05_climate_mode_adjustment" / "hellinger_period_effect_adjusted.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    x = np.arange(3)
    axes[0].bar(x, climate["partial_r2"], color=[COLORS[a] for a in climate["agency"]])
    axes[1].bar(x, climate["block_permutation_p"], color=[COLORS[a] for a in climate["agency"]])
    axes[0].set(xticks=x, xticklabels=[DISPLAY[a] for a in climate["agency"]], ylabel="Partial R²")
    axes[1].set(xticks=x, xticklabels=[DISPLAY[a] for a in climate["agency"]], ylabel="Block-permutation p")
    axes[1].axhline(.05, color="0.45", ls=":", lw=1)
    axes[1].set_ylim(0, .055)
    open_panel_axes(axes)
    panel_labels(axes)
    save(fig, folder, "FigS07_climate_mode_adjustment", scale=1.20)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landfall-only", action="store_true")
    args = parser.parse_args()
    main = MAIN_FIGURES
    supp = SUPPLEMENTARY_FIGURES
    figure2(main)
    figure6(main)
    publish_landfall_supplementary(supp)
    if not args.landfall_only:
        supplementary(supp)
    print({"main": [p.name for p in main.glob("*v2.*")], "supp": [p.name for p in supp.glob("*v2.*")]})


if __name__ == "__main__":
    run()
