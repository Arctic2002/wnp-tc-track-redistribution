"""Trial the Feng (2024) translation-speed decomposition.

The agency-aligned samples use original 6-hourly USA/JMA/CMA positions, the
whole recorded lifecycle, all months, and the 0-60N, 100-180E WNP domain over
1980-2023.  A separate project-aligned sensitivity uses the manuscript's USA-
wind TS stage, pre-first-landfall sample over 0-40N and 1966-2025.

The exact three-term identity is retained:

    basin anomaly = track-density term + regional-speed term + covariance term

Products are revision diagnostics and do not overwrite the existing Paper II
translation-speed metric or figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.utils import haversine, load_config
from paper2_dynamic.agency_data import build_agency_catalog, read_ibtracs_agencies
from paper2_dynamic.revision_stats import add_family_fdr, trend_row


FENG_START, FENG_END = 1980, 2023
PROJECT_START, PROJECT_END = 1966, 2025


def _segments(tracks, *, exact_six_hour=True):
    """Create segment speeds and assign each segment to its starting location."""
    rows = []
    for sid, group in tracks.groupby("sid", sort=False):
        group = group.sort_values("iso_time")
        if len(group) < 2:
            continue
        lat1 = group["lat"].to_numpy(float)[:-1]
        lon1 = group["lon"].to_numpy(float)[:-1]
        lat2 = group["lat"].to_numpy(float)[1:]
        lon2 = group["lon"].to_numpy(float)[1:]
        dt = np.diff(group["iso_time"].to_numpy()) / np.timedelta64(1, "h")
        if exact_six_hour:
            valid = np.isfinite(dt) & np.isclose(dt, 6.0, atol=0.01)
        else:
            valid = np.isfinite(dt) & (dt > 0) & (dt <= 12)
        if not valid.any():
            continue
        dlon = (lon2 - lon1 + 180) % 360 - 180
        midlat = (lat1 + lat2) / 2
        total = haversine(lat1, lon1, lat2, lon2) / dt
        meridional = 6371.0 * np.abs(np.deg2rad(lat2 - lat1)) / dt
        zonal = (
            6371.0
            * np.cos(np.deg2rad(midlat))
            * np.abs(np.deg2rad(dlon))
            / dt
        )
        part = pd.DataFrame(
            {
                "sid": sid,
                "season": group["season"].to_numpy()[:-1],
                "iso_time": group["iso_time"].to_numpy()[:-1],
                "lat": lat1,
                "lon": lon1 % 360,
                "dt_h": dt,
                "speed_total": total,
                "speed_meridional": meridional,
                "speed_zonal": zonal,
            }
        )
        rows.append(part.loc[valid])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _project_tracks(cfg):
    """Reproduce the manuscript's TS/pre-first-landfall sampling protocol."""
    processed = Path(cfg["paths"]["processed"])
    tracks = pd.read_csv(processed / "tracks.csv", parse_dates=["iso_time"])
    landfalls = pd.read_csv(processed / "landfalls.csv", parse_dates=["time"])
    first = landfalls.groupby("sid")["time"].min().to_dict()
    tracks = tracks.loc[
        tracks["season"].between(PROJECT_START, PROJECT_END)
        & (tracks["wind"] >= cfg["ts_threshold_kt"])
        & ((tracks["nature"] == "TS") | tracks["nature"].isna())
    ].copy()
    keep = []
    for sid, group in tracks.groupby("sid", sort=False):
        landfall_time = first.get(sid)
        if landfall_time is not None:
            group = group.loc[group["iso_time"] <= landfall_time]
        keep.append(group)
    return pd.concat(keep, ignore_index=True)


def decompose_segments(segments, years, *, lat_max, speed_column):
    """Return exact annual three-term decomposition and latitude-band details."""
    years = np.asarray(years, dtype=int)
    edges = np.arange(0, lat_max + 1, 1.0)
    data = segments.loc[
        segments["season"].isin(years)
        & segments["lon"].between(100, 180)
        & segments["lat"].between(0, lat_max, inclusive="left")
        & np.isfinite(segments[speed_column])
    ].copy()
    data["band"] = pd.cut(
        data["lat"], edges, labels=False, right=False, include_lowest=True
    )
    nbands = len(edges) - 1
    counts = np.zeros((len(years), nbands), dtype=float)
    regional = np.full((len(years), nbands), np.nan, dtype=float)
    mean_lat = np.full(len(years), np.nan)
    n_segments = np.zeros(len(years), dtype=int)
    for yi, year in enumerate(years):
        group = data.loc[data["season"] == year]
        n_segments[yi] = len(group)
        mean_lat[yi] = group["lat"].mean()
        by_band = group.groupby("band", observed=True)[speed_column].agg(["size", "mean"])
        index = by_band.index.to_numpy(int)
        counts[yi, index] = by_band["size"].to_numpy(float)
        regional[yi, index] = by_band["mean"].to_numpy(float)
    if np.any(n_segments == 0):
        missing = years[n_segments == 0].tolist()
        raise ValueError(f"no valid speed segments in years: {missing}")
    fraction = counts / counts.sum(axis=1, keepdims=True)
    active = counts.sum(axis=0) > 0
    fraction = fraction[:, active]
    regional = regional[:, active]
    centers = ((edges[:-1] + edges[1:]) / 2)[active]

    regional_mean = np.nanmean(regional, axis=0)
    regional_filled = np.where(np.isnan(regional), regional_mean[None, :], regional)
    fraction_mean = fraction.mean(axis=0)
    speed_anomaly = regional_filled - regional_mean
    fraction_anomaly = fraction - fraction_mean

    baseline = float(np.sum(regional_mean * fraction_mean))
    shift = np.sum(regional_mean[None, :] * fraction_anomaly, axis=1)
    regional_term = np.sum(speed_anomaly * fraction_mean[None, :], axis=1)
    covariance = np.sum(speed_anomaly * fraction_anomaly, axis=1)
    basin = np.sum(regional_filled * fraction, axis=1)
    closure = basin - (baseline + shift + regional_term + covariance)
    direct = data.groupby("season")[speed_column].mean().reindex(years).to_numpy(float)

    annual = pd.DataFrame(
        {
            "season": years,
            "n_segments": n_segments,
            "mean_track_lat": mean_lat,
            "basin_speed": basin,
            "direct_speed": direct,
            "baseline": baseline,
            "track_shift": shift,
            "regional_speed": regional_term,
            "covariance": covariance,
            "closure_error": closure,
        }
    )
    bands = pd.DataFrame(
        {
            "lat_center": centers,
            "climatological_fraction": fraction_mean,
            "climatological_speed": regional_mean,
        }
    )
    # Lightweight descriptive slopes for the full set of 1-degree bands.
    bands["fraction_ols_per_decade"] = [
        np.polyfit(years, fraction[:, i], 1)[0] * 10 for i in range(fraction.shape[1])
    ]
    bands["speed_ols_per_decade"] = [
        np.polyfit(years, regional_filled[:, i], 1)[0] * 10
        for i in range(regional_filled.shape[1])
    ]
    return annual, bands


def _component_trends(annual, cfg, dataset, speed_type, start, end):
    subset = annual.loc[annual["season"].between(start, end)]
    family = f"speed_decomposition_{dataset}_{speed_type}_{start}_{end}"
    rows = []
    mean_basin = subset["basin_speed"].mean()
    for variable in ["basin_speed", "track_shift", "regional_speed", "covariance"]:
        row = trend_row(
            subset["season"],
            subset[variable],
            label=variable,
            family=family,
            cfg=cfg,
            extra={
                "dataset": dataset,
                "speed_type": speed_type,
                "variable": variable,
                "trend_window_start": start,
                "trend_window_end": end,
            },
        )
        row["period_change_percent_of_mean_basin_speed"] = (
            row["sen_slope_per_year"] * (end - start) / mean_basin * 100
            if np.isfinite(row["sen_slope_per_year"])
            else np.nan
        )
        rows.append(row)
    return rows


def _diagnostic_figure(
    out_dir,
    annual_all,
    trends,
    existing_speed,
    save_diagnostic=True,
    save_formal=False,
):
    # Extend the FigS05 blue palette consistently across the three agencies.
    colors = {"USA": "#4C72B0", "TOKYO": "#7E9CC4", "CMA": "#C76D6D"}
    display = {"USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    agency = annual_all.loc[
        (annual_all["sample"] == "feng_full_lifecycle")
        & (annual_all["speed_type"] == "total")
    ]
    for name in ["USA", "TOKYO", "CMA"]:
        group = agency.loc[agency["agency"] == name]
        axes[0, 0].plot(group["season"], group["basin_speed"], label=display[name], color=colors[name])
    axes[0, 0].set(ylabel="km h$^{-1}$", xlabel="Year")
    axes[0, 0].legend(frameon=False)

    jma = agency.loc[agency["agency"] == "TOKYO"]
    for variable, label in [
        ("track_shift", "Track-density term"),
        ("regional_speed", "Regional-speed term"),
        ("covariance", "Covariance term"),
    ]:
        axes[0, 1].plot(jma["season"], jma[variable], label=label)
    axes[0, 1].axhline(0, color="0.5", lw=0.8)
    axes[0, 1].set(ylabel="Speed anomaly (km h$^{-1}$)", xlabel="Year")
    axes[0, 1].legend(frameon=False)

    component = trends.loc[
        (trends["speed_type"] == "total")
        & trends["variable"].isin(["track_shift", "regional_speed", "covariance"])
        & trends["dataset"].isin(["USA_feng", "TOKYO_feng", "CMA_feng"])
    ]
    pivot = component.pivot(index="dataset", columns="variable", values="period_change_percent_of_mean_basin_speed")
    pivot = pivot.reindex(["USA_feng", "TOKYO_feng", "CMA_feng"])
    pivot.plot.bar(ax=axes[1, 0], color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[1, 0].axhline(0, color="0.4", lw=0.8)
    axes[1, 0].set(ylabel="% of mean basin speed", xlabel="")
    axes[1, 0].set_xticklabels(["USA", "JMA", "CMA"])
    axes[1, 0].tick_params(axis="x", rotation=0)
    handles, _ = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(
        handles,
        ["Covariance", "Regional-speed term", "Track-density term"],
        frameon=False,
        fontsize=8,
    )
    significant = component.loc[component["mk_p_fdr_bh"] < 0.05]
    for row in significant.itertuples():
        x_index = ["USA_feng", "TOKYO_feng", "CMA_feng"].index(row.dataset)
        var_index = ["covariance", "regional_speed", "track_shift"].index(row.variable)
        value = row.period_change_percent_of_mean_basin_speed
        axes[1, 0].text(
            x_index + (var_index - 1) * 0.25,
            value + (0.18 if value >= 0 else -0.32),
            "*",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=12,
        )

    project = annual_all.loc[
        (annual_all["sample"] == "project_ts_prelandfall")
        & (annual_all["speed_type"] == "total")
    ]
    axes[1, 1].plot(project["season"], project["basin_speed"], label="Point-weighted decomposition")
    axes[1, 1].plot(existing_speed["season"], existing_speed["speed"], label="Existing storm-equal metric")
    axes[1, 1].set(ylabel="km h$^{-1}$", xlabel="Year")
    axes[1, 1].legend(frameon=False)

    for ax in (axes[0, 1], axes[1, 1]):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.tick_params(
            axis="y", which="both", left=False, labelleft=False,
            right=True, labelright=True,
        )

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
            fig.savefig(out_dir / f"revision_speed_decomposition_trial.{suffix}", dpi=300)
        if save_formal:
            fig.savefig(out_dir / f"FigS27_speed_decomposition.{suffix}", dpi=300)
    plt.close(fig)


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["processed"]).parents[1]
    output_data = root / "revision_outputs" / "data"
    output_figures = root / "revision_outputs" / "figures"
    output_data.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    raw = read_ibtracs_agencies(
        Path(cfg["paths"]["raw"]) / "IBTrACS.WP.v04r01.csv",
        start=FENG_START,
        end=PROJECT_END,
    )
    samples = {}
    for agency in ("USA", "TOKYO", "CMA"):
        catalog = build_agency_catalog(raw, agency)
        samples[(agency, "feng_full_lifecycle")] = (
            _segments(catalog["tracks"], exact_six_hour=True),
            np.arange(FENG_START, FENG_END + 1),
            60,
        )
    project_tracks = _project_tracks(cfg)
    samples[("USA", "project_ts_prelandfall")] = (
        _segments(project_tracks, exact_six_hour=False),
        np.arange(PROJECT_START, PROJECT_END + 1),
        40,
    )

    annual_outputs, band_outputs = [], []
    decomposed = {}
    speed_columns = {
        "total": "speed_total",
        "meridional": "speed_meridional",
        "zonal": "speed_zonal",
    }
    for (agency, sample), (segments, years, lat_max) in samples.items():
        for speed_type, column in speed_columns.items():
            annual, bands = decompose_segments(
                segments, years, lat_max=lat_max, speed_column=column
            )
            annual["agency"] = agency
            annual["sample"] = sample
            annual["speed_type"] = speed_type
            bands["agency"] = agency
            bands["sample"] = sample
            bands["speed_type"] = speed_type
            annual_outputs.append(annual)
            band_outputs.append(bands)
            decomposed[(agency, sample, speed_type)] = annual
    annual_all = pd.concat(annual_outputs, ignore_index=True)
    bands_all = pd.concat(band_outputs, ignore_index=True)
    annual_all.to_csv(output_data / "p2_speed_decomposition_annual.csv", index=False)
    bands_all.to_csv(output_data / "p2_speed_decomposition_bands.csv", index=False)

    trend_rows = []
    for agency in ("USA", "TOKYO", "CMA"):
        dataset = f"{agency}_feng"
        annual = decomposed[(agency, "feng_full_lifecycle", "total")]
        trend_rows.extend(
            _component_trends(annual, cfg, dataset, "total", FENG_START, FENG_END)
        )
    # JMA is the Feng-aligned primary dataset for component directions.
    for speed_type in ("meridional", "zonal"):
        annual = decomposed[("TOKYO", "feng_full_lifecycle", speed_type)]
        trend_rows.extend(
            _component_trends(
                annual, cfg, "TOKYO_feng", speed_type, FENG_START, FENG_END
            )
        )
    project = decomposed[("USA", "project_ts_prelandfall", "total")]
    for start in (1966, 1982):
        trend_rows.extend(
            _component_trends(
                project,
                cfg,
                f"PROJECT_ts_prelandfall_{start}",
                "total",
                start,
                PROJECT_END,
            )
        )

    # Mean track latitude is a separate, explicitly named family.
    for agency in ("USA", "TOKYO", "CMA"):
        annual = decomposed[(agency, "feng_full_lifecycle", "total")]
        trend_rows.append(
            trend_row(
                annual["season"],
                annual["mean_track_lat"],
                label=agency,
                family=f"mean_track_lat_{FENG_START}_{FENG_END}",
                cfg=cfg,
                extra={
                    "dataset": f"{agency}_feng",
                    "speed_type": "position",
                    "variable": "mean_track_lat",
                    "trend_window_start": FENG_START,
                    "trend_window_end": FENG_END,
                    "period_change_percent_of_mean_basin_speed": np.nan,
                },
            )
        )
    trends = add_family_fdr(trend_rows)
    trends.to_csv(output_data / "p2_speed_decomposition_trends.csv", index=False)

    existing = pd.read_csv(Path(cfg["paths"]["processed"]) / "p2_stats.csv")
    existing = existing.loc[
        (existing["scope"] == "annual")
        & existing["season"].between(PROJECT_START, PROJECT_END),
        ["season", "speed"],
    ]
    project_total = project[["season", "basin_speed"]]
    matched = existing.merge(project_total, on="season", how="inner")
    comparison = pd.DataFrame(
        [
            {
                "metric": "annual_correlation_existing_vs_point_weighted",
                "value": np.corrcoef(matched["speed"], matched["basin_speed"])[0, 1],
            },
            {
                "metric": "max_absolute_decomposition_closure_error",
                "value": annual_all["closure_error"].abs().max(),
            },
            {
                "metric": "max_absolute_direct_vs_reconstructed_speed",
                "value": (annual_all["direct_speed"] - annual_all["basin_speed"]).abs().max(),
            },
        ]
    )
    comparison.to_csv(output_data / "p2_speed_decomposition_qc.csv", index=False)

    _diagnostic_figure(output_figures, annual_all, trends, existing)
    key = trends.loc[
        (trends["dataset"] == "TOKYO_feng")
        & (trends["speed_type"] == "total")
        & trends["variable"].isin(["basin_speed", "track_shift", "regional_speed"]),
        ["variable", "sen_slope_per_decade", "mk_p_fdr_bh", "period_change_percent_of_mean_basin_speed"],
    ]
    print("speed decomposition written; JMA/Feng-aligned total-speed terms:")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
