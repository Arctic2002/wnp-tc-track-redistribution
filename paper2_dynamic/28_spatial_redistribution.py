"""Quantify path and landfall redistribution for the supervisor revision.

Primary comparison: 1966-1995 versus 1996-2025 (equal 30-year halves).
The path test uses annual *relative* density, so it tests spatial composition
rather than merely reproducing changes in annual storm counts or track length.

Outputs
-------
revision_outputs/data/p2_redistribution_summary.csv
revision_outputs/data/p2_redistribution_grid.csv
revision_outputs/data/p2_redistribution_coast.csv
revision_outputs/data/p2_redistribution_regional_trends.csv
revision_outputs/data/p2_redistribution_annual.npz
revision_outputs/figures/revision_spatial_redistribution_diagnostic.png/.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.utils import load_config
from paper2_dynamic.revision_stats import (
    add_family_fdr,
    compositional_change_test,
    trend_row,
)


START, END = 1966, 2025
PRIMARY_SPLIT = 1995
N_PERM = 9999
NORTH_COASTS = ["China_E", "Taiwan", "Korea", "Japan"]
SOUTH_COASTS = ["China_S", "Vietnam", "Philippines"]

# Match the model-series palette used in FigS05_model_coefficients.
MODEL_BLUE = "#4C72B0"
MODEL_LIGHT_BLUE = "#7E9CC4"
MODEL_RED = "#C0504D"


def _period_indices(years, early_start, early_end, late_start, late_end):
    early = np.flatnonzero((years >= early_start) & (years <= early_end))
    late = np.flatnonzero((years >= late_start) & (years <= late_end))
    if len(early) != len(late):
        raise ValueError("redistribution comparisons require equal-length periods")
    return early, late


def _annual_path_fields(tracks, years, lon_edges, lat_edges):
    """Return point-weighted and storm-equal annual relative-density fields."""
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    point_fields, storm_fields = [], []
    for year in years:
        annual = tracks.loc[tracks["season"] == year]
        point = np.histogram2d(
            annual["lon"], annual["lat"], bins=[lon_edges, lat_edges]
        )[0].T
        if point.sum() == 0:
            raise ValueError(f"no eligible path points in {year}")
        point_fields.append(point / point.sum())

        storm = np.zeros(shape, dtype=float)
        for _, group in annual.groupby("sid"):
            one = np.histogram2d(
                group["lon"], group["lat"], bins=[lon_edges, lat_edges]
            )[0].T
            if one.sum() > 0:
                storm += one / one.sum()
        if storm.sum() == 0:
            raise ValueError(f"no eligible storm paths in {year}")
        storm_fields.append(storm / storm.sum())
    return np.asarray(point_fields), np.asarray(storm_fields)


def _coast_compositions(landfalls, years, coasts, kind):
    rows = []
    for year in years:
        group = landfalls.loc[landfalls["season"] == year]
        if kind == "event":
            count = group["coast"].value_counts().reindex(coasts, fill_value=0)
        elif kind == "storm":
            count = group.groupby("coast")["sid"].nunique().reindex(coasts, fill_value=0)
        else:
            raise ValueError(kind)
        if count.sum() == 0:
            raise ValueError(f"no landfalls in {year}")
        rows.append((count / count.sum()).to_numpy(float))
    return np.asarray(rows)


def _north_named_share(landfalls, years, kind):
    values = []
    for year in years:
        group = landfalls.loc[landfalls["season"] == year]
        if kind == "event":
            count = group["coast"].value_counts()
        else:
            count = group.groupby("coast")["sid"].nunique()
        north = count.reindex(NORTH_COASTS, fill_value=0).sum()
        south = count.reindex(SOUTH_COASTS, fill_value=0).sum()
        values.append(north / (north + south))
    return np.asarray(values)


def _append_summary(summary, analysis, weighting, period, years, early, late, result):
    summary.append(
        {
            "analysis": analysis,
            "weighting": weighting,
            "period_definition": period,
            "early_start": int(years[early].min()),
            "early_end": int(years[early].max()),
            "late_start": int(years[late].min()),
            "late_end": int(years[late].max()),
            "n_years_each": int(len(early)),
            "total_variation": result["tv"],
            "block_permutation_p": result["global_p"],
            "n_permutations": result["nperm"],
            "block_years": result["block"],
        }
    )


def _diagnostic_figure(
    out_dir,
    years,
    lon_edges,
    lat_edges,
    path_result,
    coast_names,
    coast_result,
    north_share,
    north_result,
    save_diagnostic=True,
    save_formal=False,
):
    fig = plt.figure(figsize=(11.0, 7.9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.10, 1.0])
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[2, 0])
    axes[1, 1] = fig.add_subplot(gs[2, 1])
    cbar_host = fig.add_subplot(gs[1, 0])
    cbar_host.set_axis_off()
    cax = cbar_host.inset_axes([1 / 8, 1 / 6, 3 / 4, 2 / 3])
    panel_labels = ["a", "b", "c", "d"]

    ax = axes[0, 0]
    change = path_result["change"].reshape(len(lat_edges) - 1, len(lon_edges) - 1)
    vmax = np.nanmax(np.abs(change)) * 100
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        change * 100,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
    )
    out_path = Path(out_dir)
    project_root = out_path.parent if out_path.name == "figures" else out_path.parents[1]
    land_path = (
        project_root
        / "data"
        / "raw"
        / "GSHHG"
        / "GSHHS_shp"
        / "i"
        / "GSHHS_i_L1.shp"
    )
    if land_path.exists():
        import geopandas as gpd

        land = gpd.read_file(land_path, bbox=(lon_edges[0], lat_edges[0], lon_edges[-1], lat_edges[-1]))
        land.plot(
            ax=ax,
            facecolor="#E2E2E2",
            edgecolor="#4A4A4A",
            linewidth=0.35,
            zorder=3,
        )
        ax.set_xlim(lon_edges[0], lon_edges[-1])
        ax.set_ylim(lat_edges[0], lat_edges[-1])
    fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        label="Late - early share (percentage points/cell)",
    )
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")

    ax = axes[0, 1]
    early_lat = path_result["early_mean"].reshape(change.shape).sum(axis=1)
    late_lat = path_result["late_mean"].reshape(change.shape).sum(axis=1)
    centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    ax.plot(
        early_lat * 100, centers, marker="o", color=MODEL_BLUE,
        label="1966-1995",
    )
    ax.plot(
        late_lat * 100, centers, marker="o", color=MODEL_LIGHT_BLUE,
        label="1996-2025",
    )
    ax.set(xlabel="Share of track points (%)", ylabel="Latitude (°N)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    x = np.arange(len(coast_names))
    width = 0.38
    ax.bar(
        x - width / 2, coast_result["early_mean"] * 100, width,
        color=MODEL_BLUE, label="1966-1995",
    )
    ax.bar(
        x + width / 2, coast_result["late_mean"] * 100, width,
        color=MODEL_LIGHT_BLUE, label="1996-2025",
    )
    display_coasts = {
        "China_E": "East China",
        "China_S": "South China",
        "Japan": "Japan",
        "Korea": "Korea",
        "Other": "Other",
        "Philippines": "Philippines",
        "Taiwan": "Taiwan",
        "Vietnam": "Vietnam",
    }
    ax.set_xticks(
        x,
        [display_coasts.get(name, name) for name in coast_names],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Share of landfall events (%)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(
        years, north_share * 100, color=MODEL_BLUE,
        lw=1.2, marker="o", ms=2.5,
    )
    fit = np.polyfit(years, north_share * 100, 1)
    ax.plot(
        years, np.polyval(fit, years), color=MODEL_RED,
        lw=1.5, label="Linear fit",
    )
    ax.axvline(PRIMARY_SPLIT + 0.5, color="0.5", ls="--", lw=0.8)
    ax.set(xlabel="Year", ylabel="Northern share of named-coast events (%)")
    ax.legend(frameon=False, loc="upper left")

    for ax in (axes[0, 1], axes[1, 1]):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.tick_params(
            axis="y", which="both", left=False, labelleft=False,
            right=True, labelright=True,
        )

    for label, panel in zip(panel_labels, axes.flat):
        panel.text(
            0.01,
            0.99,
            label,
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    for suffix in ("png", "pdf"):
        if save_diagnostic:
            fig.savefig(out_dir / f"revision_spatial_redistribution_diagnostic.{suffix}", dpi=300)
        if save_formal:
            fig.savefig(out_dir / f"FigS25_spatial_redistribution.{suffix}", dpi=300)
    plt.close(fig)


def main():
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    output_root = processed.parents[1] / "revision_outputs"
    output_data = output_root / "data"
    figures = output_root / "figures"
    output_data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    years = np.arange(START, END + 1)
    block = cfg["statistics"]["bootstrap_block"]
    seed = cfg["statistics"]["random_seed"]

    tracks = pd.read_csv(processed / "tracks.csv")
    tracks = tracks.loc[
        tracks["season"].between(START, END)
        & (tracks["wind"] >= cfg["ts_threshold_kt"])
        & ((tracks["nature"] == "TS") | tracks["nature"].isna())
    ].copy()
    region = cfg["regions"]["tc"]
    tracks = tracks.loc[
        tracks["lon"].between(region["lon_min"], region["lon_max"])
        & tracks["lat"].between(region["lat_min"], region["lat_max"])
    ]
    bin_width = cfg["grids"]["density_bin"]
    lon_edges = np.arange(region["lon_min"], region["lon_max"] + bin_width, bin_width)
    lat_edges = np.arange(region["lat_min"], region["lat_max"] + bin_width, bin_width)
    point_fields, storm_fields = _annual_path_fields(
        tracks, years, lon_edges, lat_edges
    )

    storms = pd.read_csv(processed / "storms.csv", usecols=["sid", "season"])
    landfalls = pd.read_csv(processed / "landfalls.csv").merge(
        storms, on="sid", how="left", validate="many_to_one"
    )
    landfalls = landfalls.loc[landfalls["season"].between(START, END)].copy()
    coasts = sorted(landfalls["coast"].dropna().unique())
    coast_event = _coast_compositions(landfalls, years, coasts, "event")
    coast_storm = _coast_compositions(landfalls, years, coasts, "storm")
    north_event = _north_named_share(landfalls, years, "event")
    north_storm = _north_named_share(landfalls, years, "storm")

    comparisons = [
        ("primary_equal_halves", START, PRIMARY_SPLIT, PRIMARY_SPLIT + 1, END),
        ("endpoint_20_years", 1966, 1985, 2006, 2025),
        ("intensity_era_equal_halves", 1982, 2003, 2004, 2025),
    ]
    summary = []
    primary = {}
    analyses = {
        ("path_density", "track_point"): point_fields.reshape(len(years), -1),
        ("path_density", "storm_equal"): storm_fields.reshape(len(years), -1),
        ("landfall_coast", "event"): coast_event,
        ("landfall_coast", "storm_incidence"): coast_storm,
        ("north_vs_south_named_coast", "event"): np.column_stack(
            [north_event, 1 - north_event]
        ),
        ("north_vs_south_named_coast", "storm_incidence"): np.column_stack(
            [north_storm, 1 - north_storm]
        ),
    }
    for period, es, ee, ls, le in comparisons:
        subset = np.flatnonzero((years >= es) & (years <= le))
        subset_years = years[subset]
        early, late = _period_indices(subset_years, es, ee, ls, le)
        for (analysis, weighting), values in analyses.items():
            result = compositional_change_test(
                values[subset],
                early,
                late,
                nperm=N_PERM,
                block=block,
                seed=seed,
            )
            _append_summary(
                summary,
                analysis,
                weighting,
                period,
                subset_years,
                early,
                late,
                result,
            )
            if period == "primary_equal_halves":
                primary[(analysis, weighting)] = result
    pd.DataFrame(summary).to_csv(output_data / "p2_redistribution_summary.csv", index=False)

    grid_rows = []
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    for weighting in ("track_point", "storm_equal"):
        result = primary[("path_density", weighting)]
        for k, (i, j) in enumerate(
            np.ndindex(len(lat_centers), len(lon_centers))
        ):
            grid_rows.append(
                {
                    "weighting": weighting,
                    "lat": lat_centers[i],
                    "lon": lon_centers[j],
                    "early_share": result["early_mean"][k],
                    "late_share": result["late_mean"][k],
                    "late_minus_early": result["change"][k],
                    "cell_p_raw": result["cell_p"][k],
                    "cell_q_bh": result["cell_q_bh"][k],
                    "cell_p_max_abs": result["cell_p_max"][k],
                }
            )
    pd.DataFrame(grid_rows).to_csv(output_data / "p2_redistribution_grid.csv", index=False)

    coast_rows = []
    for weighting in ("event", "storm_incidence"):
        result = primary[("landfall_coast", weighting)]
        for i, coast in enumerate(coasts):
            coast_rows.append(
                {
                    "weighting": weighting,
                    "coast": coast,
                    "early_share": result["early_mean"][i],
                    "late_share": result["late_mean"][i],
                    "late_minus_early": result["change"][i],
                    "cell_p_raw": result["cell_p"][i],
                    "cell_q_bh": result["cell_q_bh"][i],
                    "cell_p_max_abs": result["cell_p_max"][i],
                }
            )
    pd.DataFrame(coast_rows).to_csv(output_data / "p2_redistribution_coast.csv", index=False)

    annual_path = []
    for year in years:
        group = tracks.loc[tracks["season"] == year]
        n = len(group)
        annual_path.append(
            {
                "season": year,
                "mean_lat": group["lat"].mean(),
                "mean_lon": group["lon"].mean(),
                "share_nw": ((group["lat"] >= 20) & (group["lon"] < 140)).sum() / n,
                "share_ne": ((group["lat"] >= 20) & (group["lon"] >= 140)).sum() / n,
                "share_sw": ((group["lat"] < 20) & (group["lon"] < 140)).sum() / n,
                "share_se": ((group["lat"] < 20) & (group["lon"] >= 140)).sum() / n,
            }
        )
    annual_path = pd.DataFrame(annual_path)
    trend_rows = []
    for start in (1966, 1982):
        subset = annual_path.loc[annual_path["season"] >= start]
        family = f"path_regional_{start}_{END}"
        for variable in ["mean_lat", "mean_lon", "share_nw", "share_ne", "share_sw", "share_se"]:
            trend_rows.append(
                trend_row(
                    subset["season"],
                    subset[variable],
                    label=variable,
                    family=family,
                    cfg=cfg,
                    extra={"analysis": "path", "weighting": "track_point"},
                )
            )
    for kind, values in (("event", coast_event), ("storm_incidence", coast_storm)):
        family = f"landfall_coast_share_{kind}_{START}_{END}"
        for i, coast in enumerate(coasts):
            trend_rows.append(
                trend_row(
                    years,
                    values[:, i],
                    label=coast,
                    family=family,
                    cfg=cfg,
                    extra={"analysis": "landfall_coast", "weighting": kind},
                )
            )
        north = north_event if kind == "event" else north_storm
        trend_rows.append(
            trend_row(
                years,
                north,
                label="northern_named_coast_share",
                family=f"north_vs_south_named_coast_{kind}",
                cfg=cfg,
                extra={"analysis": "north_vs_south_named_coast", "weighting": kind},
            )
        )
    add_family_fdr(trend_rows).to_csv(
        output_data / "p2_redistribution_regional_trends.csv", index=False
    )

    np.savez_compressed(
        output_data / "p2_redistribution_annual.npz",
        years=years,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        point_relative_density=point_fields,
        storm_equal_relative_density=storm_fields,
        coast_names=np.asarray(coasts),
        coast_event_share=coast_event,
        coast_storm_incidence_share=coast_storm,
        north_named_event_share=north_event,
        north_named_storm_share=north_storm,
    )

    _diagnostic_figure(
        figures,
        years,
        lon_edges,
        lat_edges,
        primary[("path_density", "track_point")],
        coasts,
        primary[("landfall_coast", "event")],
        north_event,
        primary[("north_vs_south_named_coast", "event")],
    )
    p_path = primary[("path_density", "track_point")]["global_p"]
    p_coast = primary[("landfall_coast", "event")]["global_p"]
    p_north = primary[("north_vs_south_named_coast", "event")]["global_p"]
    print(
        "spatial redistribution written: "
        f"path TV={primary[('path_density', 'track_point')]['tv']:.3f}, p={p_path:.4f}; "
        f"all-coast TV p={p_coast:.4f}; north/south named-coast p={p_north:.4f}"
    )


if __name__ == "__main__":
    main()
