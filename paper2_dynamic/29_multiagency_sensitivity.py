"""USA/JMA/CMA sensitivity checks using original agency reports in IBTrACS.

No cross-agency wind conversion is used for frequency.  USA uses its native
1-min 34-kt threshold; JMA and CMA use their native grade/category definitions.
LMI latitude is based on each agency's own maximum-wind time, for which a
constant wind-duration conversion would not change the argmax.

All products are written below ``revision_outputs`` so the original processed
tables and figures remain untouched.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.landfall_gshhg import attribute_coast, build_land_mask, densify
from core.utils import load_config
from paper2_dynamic.agency_data import (
    AGENCIES,
    agency_flag,
    build_agency_catalog,
    native_ts_mask,
    read_ibtracs_agencies,
)
from paper2_dynamic.revision_stats import (
    add_family_fdr,
    compositional_change_test,
    trend_row,
)


START, END = 1966, 2025
LMI_START = 1982
N_PERM = 9999
NORTH_COASTS = ["China_E", "Taiwan", "Korea", "Japan"]
SOUTH_COASTS = ["China_S", "Vietnam", "Philippines"]


def _annual_path_composition(points, years, lon_edges, lat_edges):
    rows = []
    points = points.loc[
        points["lon"].between(lon_edges[0], lon_edges[-1])
        & points["lat"].between(lat_edges[0], lat_edges[-1])
    ]
    for year in years:
        group = points.loc[points["season"] == year]
        field = np.histogram2d(
            group["lon"], group["lat"], bins=[lon_edges, lat_edges]
        )[0].T
        if field.sum() == 0:
            raise ValueError(f"no agency path points in {year}")
        rows.append((field / field.sum()).ravel())
    return np.asarray(rows)


def _vector_land_state(mask, lon0, lat1, res, lat, lon):
    """Vectorized counterpart of is_land; outside-domain points are invalid."""
    col = ((lon - lon0) / res).astype(int)
    row = ((lat1 - lat) / res).astype(int)
    inside = (
        (row >= 0)
        & (row < mask.shape[0])
        & (col >= 0)
        & (col < mask.shape[1])
    )
    land = np.zeros(len(lat), dtype=bool)
    land[inside] = mask[row[inside], col[inside]] == 1
    return inside, land


def _detect_landfalls(tracks, agency, cfg, mask_info):
    mask, lon0, lat1, res = mask_info
    events = []
    for sid, group in tracks.groupby("sid", sort=False):
        group = group.sort_values("iso_time").reset_index(drop=True)
        last_event = None
        for k in range(len(group) - 1):
            a, b = group.iloc[k], group.iloc[k + 1]
            hours = (b.iso_time - a.iso_time).total_seconds() / 3600
            if not (0 < hours <= 12):
                continue
            fraction, lat, lon = densify(a, b, res / 2)
            inside, land = _vector_land_state(mask, lon0, lat1, res, lat, lon)
            crossing = np.flatnonzero(
                inside[1:] & inside[:-1] & land[1:] & ~land[:-1]
            )
            if not len(crossing):
                continue
            j = int(crossing[0] + 1)
            event_time = a.iso_time + (b.iso_time - a.iso_time) * float(fraction[j])
            if (
                last_event is not None
                and (event_time - last_event).total_seconds() / 3600
                < cfg["landfall_min_separation_h"]
            ):
                continue
            events.append(
                {
                    "agency": agency,
                    "sid": sid,
                    "season": int(a.season),
                    "time": event_time,
                    "lat": float(lat[j]),
                    "lon": float(lon[j]),
                    "coast": attribute_coast(float(lat[j]), float(lon[j])),
                }
            )
            last_event = event_time
    return pd.DataFrame(events)


def _coast_annual(events, years, coasts):
    composition, totals, north_share = [], [], []
    for year in years:
        group = events.loc[events["season"] == year]
        count = group["coast"].value_counts().reindex(coasts, fill_value=0)
        total = int(count.sum())
        if total == 0:
            raise ValueError(f"no agency landfall events in {year}")
        composition.append((count / total).to_numpy(float))
        totals.append(total)
        north = count.reindex(NORTH_COASTS, fill_value=0).sum()
        south = count.reindex(SOUTH_COASTS, fill_value=0).sum()
        north_share.append(north / (north + south))
    return np.asarray(composition), np.asarray(totals), np.asarray(north_share)


def _safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def _figure(out_dir, annual, redistribution, save_diagnostic=True, save_formal=False):
    agencies = list(AGENCIES)
    # Extend the FigS05 blue palette consistently across the three agencies.
    colors = {"USA": "#4C72B0", "TOKYO": "#7E9CC4", "CMA": "#C76D6D"}
    display = {"USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    for agency in agencies:
        group = annual.loc[annual["agency"] == agency]
        axes[0, 0].plot(group["season"], group["n_tc"], label=display[agency], color=colors[agency])
        axes[0, 1].plot(
            group["season"],
            group["mean_lmi_lat_common"],
            label=display[agency],
            color=colors[agency],
        )
        axes[1, 1].plot(
            group["season"],
            group["north_named_event_share"] * 100,
            label=display[agency],
            color=colors[agency],
        )
    axes[0, 0].set(ylabel="TC count")
    axes[0, 1].set(ylabel="Latitude (°N)")
    axes[1, 1].set(
        ylabel="Share (%)",
        xlabel="Year",
    )
    for ax in (axes[0, 1], axes[1, 1]):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.tick_params(
            axis="y", which="both", left=False, labelleft=False,
            right=True, labelright=True,
        )
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.legend(frameon=False)
        ax.set_xlabel("Year")

    path = redistribution.loc[redistribution["analysis"] == "path_density"]
    x = np.arange(len(path))
    axes[1, 0].bar(x, path["total_variation"], color=[colors[a] for a in path["agency"]])
    axes[1, 0].set_xticks(x, [display[a] for a in path["agency"]])
    axes[1, 0].set_ylabel("Total-variation distance")
    for i, row in enumerate(path.itertuples()):
        axes[1, 0].text(i, row.total_variation, f"p={row.block_permutation_p:.3f}", ha="center", va="bottom")

    for label, panel in zip(("a", "b", "c", "d"), axes.flat):
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
            fig.savefig(out_dir / f"revision_multiagency_sensitivity_diagnostic.{suffix}", dpi=300)
        if save_formal:
            fig.savefig(out_dir / f"FigS26_multiagency_sensitivity.{suffix}", dpi=300)
    plt.close(fig)


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["processed"]).parents[1]
    output_data = root / "revision_outputs" / "data"
    output_figures = root / "revision_outputs" / "figures"
    output_data.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    raw_path = Path(cfg["paths"]["raw"]) / "IBTrACS.WP.v04r01.csv"
    source = read_ibtracs_agencies(raw_path, start=1945, end=END)
    catalogs = {agency: build_agency_catalog(source, agency) for agency in AGENCIES}
    years = np.arange(START, END + 1)
    lmi_years = np.arange(LMI_START, END + 1)
    seed = cfg["statistics"]["random_seed"]
    block = cfg["statistics"]["bootstrap_block"]
    early = np.arange(30)
    late = np.arange(30, 60)

    common_lmi_sids = set.intersection(
        *(set(catalogs[agency]["lmi"]["sid"]) for agency in AGENCIES)
    )
    lmi_full, lmi_common = {}, {}
    frequency = {}
    for agency, catalog in catalogs.items():
        frequency[agency] = (
            catalog["frequency"].set_index("season")["n_tc"].reindex(years, fill_value=0)
        )
        lmi = catalog["lmi"].loc[catalog["lmi"]["season"].between(LMI_START, END)]
        lmi_full[agency] = lmi.groupby("season")["lmi_lat"].agg(["mean", "count"]).reindex(lmi_years)
        common = lmi.loc[lmi["sid"].isin(common_lmi_sids)]
        lmi_common[agency] = common.groupby("season")["lmi_lat"].agg(["mean", "count"]).reindex(lmi_years)

    region = cfg["regions"]["tc"]
    width = cfg["grids"]["density_bin"]
    lon_edges = np.arange(region["lon_min"], region["lon_max"] + width, width)
    lat_edges = np.arange(region["lat_min"], region["lat_max"] + width, width)
    path_composition = {
        agency: _annual_path_composition(catalog["ts_points"], years, lon_edges, lat_edges)
        for agency, catalog in catalogs.items()
    }
    path_results = {
        agency: compositional_change_test(
            value,
            early,
            late,
            nperm=N_PERM,
            block=block,
            seed=seed,
        )
        for agency, value in path_composition.items()
    }

    mask_info = build_land_mask(cfg)
    landfalls = pd.concat(
        [
            _detect_landfalls(catalog["tracks"], agency, cfg, mask_info)
            for agency, catalog in catalogs.items()
        ],
        ignore_index=True,
    )
    landfalls.to_csv(output_data / "p2_multiagency_landfalls.csv", index=False)
    coasts = sorted(landfalls["coast"].unique())
    coast_composition, coast_total, north_share = {}, {}, {}
    for agency in AGENCIES:
        coast_composition[agency], coast_total[agency], north_share[agency] = _coast_annual(
            landfalls.loc[landfalls["agency"] == agency], years, coasts
        )

    coast_results = {
        agency: compositional_change_test(
            coast_composition[agency],
            early,
            late,
            nperm=N_PERM,
            block=block,
            seed=seed,
        )
        for agency in AGENCIES
    }
    north_results = {
        agency: compositional_change_test(
            np.column_stack([north_share[agency], 1 - north_share[agency]]),
            early,
            late,
            nperm=N_PERM,
            block=block,
            seed=seed,
        )
        for agency in AGENCIES
    }

    annual_rows = []
    for agency in AGENCIES:
        for year in years:
            row = {
                "agency": agency,
                "season": year,
                "n_tc": int(frequency[agency].loc[year]),
                "mean_lmi_lat_full": np.nan,
                "n_lmi_full": 0,
                "mean_lmi_lat_common": np.nan,
                "n_lmi_common": 0,
                "landfall_events": int(coast_total[agency][year - START]),
                "north_named_event_share": north_share[agency][year - START],
            }
            if year >= LMI_START:
                row.update(
                    {
                        "mean_lmi_lat_full": lmi_full[agency].loc[year, "mean"],
                        "n_lmi_full": int(lmi_full[agency].loc[year, "count"]),
                        "mean_lmi_lat_common": lmi_common[agency].loc[year, "mean"],
                        "n_lmi_common": int(lmi_common[agency].loc[year, "count"]),
                    }
                )
            annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(output_data / "p2_multiagency_annual.csv", index=False)

    trend_rows = []
    metrics = [
        ("n_tc", START, "frequency"),
        ("mean_lmi_lat_full", LMI_START, "lmi_full_catalog"),
        ("mean_lmi_lat_common", LMI_START, "lmi_common_storms"),
        ("north_named_event_share", START, "north_named_landfall_share"),
    ]
    for variable, start, metric in metrics:
        for end in (2024, 2025):
            family = f"multiagency_{metric}_{start}_{end}"
            for agency in AGENCIES:
                subset = annual.loc[
                    (annual["agency"] == agency)
                    & annual["season"].between(start, end)
                ]
                trend_rows.append(
                    trend_row(
                        subset["season"],
                        subset[variable],
                        label=agency,
                        family=family,
                        cfg=cfg,
                        extra={"analysis": metric, "agency": agency, "variable": variable},
                    )
                )
    trends = add_family_fdr(trend_rows)
    trends.to_csv(output_data / "p2_multiagency_trends.csv", index=False)

    redistribution_rows = []
    for agency in AGENCIES:
        for analysis, result in [
            ("path_density", path_results[agency]),
            ("landfall_coast", coast_results[agency]),
            ("north_vs_south_named_coast", north_results[agency]),
        ]:
            redistribution_rows.append(
                {
                    "agency": agency,
                    "analysis": analysis,
                    "early_start": 1966,
                    "early_end": 1995,
                    "late_start": 1996,
                    "late_end": 2025,
                    "total_variation": result["tv"],
                    "block_permutation_p": result["global_p"],
                    "n_permutations": result["nperm"],
                    "block_years": result["block"],
                }
            )
    redistribution = pd.DataFrame(redistribution_rows)
    redistribution.to_csv(output_data / "p2_multiagency_redistribution.csv", index=False)

    grid_rows = []
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    for agency, result in path_results.items():
        for k, (i, j) in enumerate(np.ndindex(len(lat_centers), len(lon_centers))):
            grid_rows.append(
                {
                    "agency": agency,
                    "lat": lat_centers[i],
                    "lon": lon_centers[j],
                    "early_share": result["early_mean"][k],
                    "late_share": result["late_mean"][k],
                    "late_minus_early": result["change"][k],
                    "cell_p_raw": result["cell_p"][k],
                    "cell_q_bh": result["cell_q_bh"][k],
                }
            )
    pd.DataFrame(grid_rows).to_csv(output_data / "p2_multiagency_path_grid.csv", index=False)

    agreement_rows = []
    for left, right in combinations(AGENCIES, 2):
        agreement_rows.extend(
            [
                {
                    "agency_left": left,
                    "agency_right": right,
                    "metric": "annual_frequency",
                    "correlation": _safe_corr(frequency[left], frequency[right]),
                },
                {
                    "agency_left": left,
                    "agency_right": right,
                    "metric": "annual_lmi_lat_common",
                    "correlation": _safe_corr(
                        lmi_common[left]["mean"], lmi_common[right]["mean"]
                    ),
                },
                {
                    "agency_left": left,
                    "agency_right": right,
                    "metric": "path_change_map",
                    "correlation": _safe_corr(
                        path_results[left]["change"], path_results[right]["change"]
                    ),
                },
                {
                    "agency_left": left,
                    "agency_right": right,
                    "metric": "coast_change_vector",
                    "correlation": _safe_corr(
                        coast_results[left]["change"], coast_results[right]["change"]
                    ),
                },
            ]
        )
    pd.DataFrame(agreement_rows).to_csv(
        output_data / "p2_multiagency_agreement.csv", index=False
    )

    metadata_rows = []
    for agency, catalog in catalogs.items():
        info = AGENCIES[agency]
        pos = agency_flag(source, agency, intensity=False) & source[info["lat"]].notna()
        inten = agency_flag(source, agency, intensity=True) & source[info["wind"]].notna()
        ts = native_ts_mask(source, agency)
        metadata_rows.append(
            {
                "agency": agency,
                "wind_averaging": {"USA": "1-min", "TOKYO": "10-min", "CMA": "2-min"}[agency],
                "frequency_definition": {
                    "USA": "original 1-min wind >=34 kt; explicit non-tropical stages excluded",
                    "TOKYO": "original native grades 3/4/5/9",
                    "CMA": "original native categories 2-6",
                }[agency],
                "first_original_position_year": int(source.loc[pos, "SEASON"].min()),
                "last_original_position_year": int(source.loc[pos, "SEASON"].max()),
                "first_original_wind_year": int(source.loc[inten, "SEASON"].min()),
                "last_original_wind_year": int(source.loc[inten, "SEASON"].max()),
                "original_position_reports": int(pos.sum()),
                "original_intensity_reports": int(inten.sum()),
                "native_ts_reports": int(ts.sum()),
                "eligible_storms": len(catalog["eligible_sids"]),
                "common_lmi_storms_1982_2025": int(
                    catalog["lmi"]["sid"].isin(common_lmi_sids).sum()
                ),
            }
        )
    pd.DataFrame(metadata_rows).to_csv(
        output_data / "p2_multiagency_metadata.csv", index=False
    )

    _figure(output_figures, annual, redistribution)
    print(
        "multi-agency sensitivity written; "
        + "; ".join(
            f"{a}: path p={path_results[a]['global_p']:.4f}, "
            f"coast p={coast_results[a]['global_p']:.4f}"
            for a in AGENCIES
        )
    )


if __name__ == "__main__":
    main()
