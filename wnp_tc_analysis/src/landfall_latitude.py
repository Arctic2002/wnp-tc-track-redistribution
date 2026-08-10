from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point
from shapely.prepared import prep

from .common import PROJECT, REPORTS, WORK, load_config
from .figure_typography import scale_figure_typography

from core.landfall_gshhg import attribute_coast
from paper2_dynamic.agency_data import (
    AGENCIES,
    agency_flag,
    build_agency_catalog,
    native_ts_mask,
    read_ibtracs_agencies,
)

from .stats import bh_fdr, block_bootstrap_many, block_permutation_many, trend_summary


ANALYSIS = WORK / "analysis" / "01_landfall_latitude"
AUTHORITY_PATH = (
    PROJECT
    / "data"
    / "processed"
    / "exclusive_coast"
    / "classified_exact_vector_events_admin0_corrected.csv"
)
AGENCY_LABEL = {"USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}
COLORS = {"USA": "#4069A1", "TOKYO": "#7F9FC5", "CMA": "#C66D6D"}


class LocalPolygonSet:
    """Small vectorized land predicate without constructing polygon unions."""

    def __init__(self, geometries):
        self.geometries = geometries

    def covers(self, point):
        return bool(np.any(shapely.covers(self.geometries, point)))


def geometry_points(geometry):
    """Return point-like boundary intersections, including overlap endpoints."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, (MultiPoint, GeometryCollection)):
        out = []
        for part in geometry.geoms:
            out.extend(geometry_points(part))
        return out
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return [Point(coords[0]), Point(coords[-1])] if coords else []
    if isinstance(geometry, MultiLineString):
        out = []
        for part in geometry.geoms:
            out.extend(geometry_points(part))
        return out
    return []


def sea_to_land_fractions(line: LineString, intersection, prepared_land) -> list[float]:
    fractions = sorted(line.project(p, normalized=True) for p in geometry_points(intersection))
    unique = []
    for value in fractions:
        value = float(np.clip(value, 0.0, 1.0))
        if not unique or abs(value - unique[-1]) > 1e-8:
            unique.append(value)
    if not unique:
        return []
    bounds = [0.0, *unique, 1.0]
    states = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        fraction = (left + right) / 2 if right - left > 1e-10 else min(1.0, left + 1e-8)
        states.append(bool(prepared_land.covers(line.interpolate(fraction, normalized=True))))
    return [fraction for i, fraction in enumerate(unique) if not states[i] and states[i + 1]]


def build_segments(tracks: pd.DataFrame, max_gap_hours: float) -> pd.DataFrame:
    frame = tracks.sort_values(["sid", "iso_time"]).copy()
    for column in ["sid", "iso_time", "lat", "lon", "wind"]:
        frame[f"next_{column}"] = frame.groupby("sid")[column].shift(-1)
    frame["gap_hours"] = (frame.next_iso_time - frame.iso_time).dt.total_seconds() / 3600
    return frame.loc[
        frame.sid.eq(frame.next_sid)
        & frame.gap_hours.gt(0)
        & frame.gap_hours.le(max_gap_hours)
        & frame[["lat", "lon", "next_lat", "next_lon"]].notna().all(axis=1)
    ].copy()


def refine_candidate_crossings(tracks, agency, candidates, coast_geometries, coast_tree, cfg):
    """Refine conservative raster candidates to exact vector-coast crossings.

    The 0.02-degree mask is used only to identify a small set of candidate
    segments. Final position, time and wind are recomputed from the exact
    intersection with the GSHHG high-resolution polygon boundary.
    """
    segments = build_segments(tracks, cfg["landfall"]["maximum_track_gap_hours"])
    by_sid = {sid: group for sid, group in segments.groupby("sid", sort=False)}
    events, unresolved, cache = [], [], {}
    for candidate in candidates.sort_values("time").itertuples():
        group = by_sid.get(candidate.sid)
        if group is None:
            unresolved.append({"agency": agency, "sid": candidate.sid, "time": candidate.time, "reason": "sid_not_in_track_segments"})
            continue
        matched = group.loc[group.iso_time.le(candidate.time) & group.next_iso_time.ge(candidate.time)]
        if matched.empty:
            unresolved.append({"agency": agency, "sid": candidate.sid, "time": candidate.time, "reason": "candidate_not_bracketed"})
            continue
        row = matched.iloc[0]
        cache_key = (candidate.sid, row.iso_time, row.next_iso_time)
        if cache_key not in cache:
            line = LineString([(row.lon, row.lat), (row.next_lon, row.next_lat)])
            polygon_indices = coast_tree.query(line)
            if len(polygon_indices):
                local_geometries = coast_geometries[polygon_indices]
                intersections = shapely.intersection(shapely.boundary(local_geometries), line)
                intersection = GeometryCollection([item for item in intersections if not item.is_empty])
                fractions = sea_to_land_fractions(line, intersection, LocalPolygonSet(local_geometries))
            else:
                fractions = []
            cache[cache_key] = (line, fractions)
        line, fractions = cache[cache_key]
        if not fractions:
            unresolved.append({"agency": agency, "sid": candidate.sid, "time": candidate.time, "reason": "no_exact_sea_to_land_intersection"})
            continue
        approximate_fraction = (candidate.time - row.iso_time) / (row.next_iso_time - row.iso_time)
        fraction = min(fractions, key=lambda value: abs(value - approximate_fraction))
        event_time = row.iso_time + (row.next_iso_time - row.iso_time) * fraction
        lat = row.lat + fraction * (row.next_lat - row.lat)
        lon = row.lon + fraction * (row.next_lon - row.lon)
        wind = np.nan
        if pd.notna(row.wind) and pd.notna(row.next_wind):
            wind = row.wind + fraction * (row.next_wind - row.wind)
        events.append({
            "agency": agency, "sid": row.sid, "season": int(row.season),
            "time": event_time, "lat": float(lat), "lon": float(lon), "wind": float(wind),
            "coast": attribute_coast(float(lat), float(lon)), "segment_start": row.iso_time,
            "segment_end": row.next_iso_time, "start_lat": row.lat, "start_lon": row.lon,
            "end_lat": row.next_lat, "end_lon": row.next_lon, "segment_fraction": fraction,
            "candidate_source": "0.02-degree raster sea-to-land screen",
        })
    events = pd.DataFrame(events)
    if events.empty:
        return events, pd.DataFrame(unresolved)
    events = events.sort_values("time").drop_duplicates(["sid", "time"])
    kept = []
    minimum = cfg["landfall"]["minimum_event_separation_hours"]
    for sid, group in events.sort_values("time").groupby("sid", sort=False):
        last = None
        for index, row in group.iterrows():
            if last is None or (row.time - last).total_seconds() / 3600 >= minimum:
                kept.append(index)
                last = row.time
    events = events.loc[kept].sort_values(["season", "sid", "time"]).reset_index(drop=True)
    return events, pd.DataFrame(unresolved)


def definitions(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = events.sort_values(["sid", "time"])
    return {
        "first_landfall": ordered.drop_duplicates("sid", keep="first"),
        "all_events": ordered,
        "unique_storm_coast": ordered.drop_duplicates(["sid", "coast"], keep="first"),
    }


def attach_native_stage_and_exclusive_coast(
    events: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Attach agency-native TS state and the current exclusive coast authority."""
    authority_path = AUTHORITY_PATH
    authority = pd.read_csv(
        authority_path,
        usecols=[
            "agency",
            "sid",
            "segment_start",
            "segment_end",
            "coast_exclusive",
            "nearest_segment",
            "nearest_distance_km",
        ],
        parse_dates=["segment_start", "segment_end"],
    )
    keys = ["agency", "sid", "segment_start", "segment_end"]
    if authority.duplicated(keys).any():
        raise RuntimeError("Exclusive-coast authority contains duplicate event keys")

    out = events.copy()
    out["segment_start"] = pd.to_datetime(out["segment_start"])
    out["segment_end"] = pd.to_datetime(out["segment_end"])
    out = out.merge(authority, on=keys, how="left", validate="one_to_one")
    if out["coast_exclusive"].isna().any():
        raise RuntimeError("Exact events did not fully match exclusive-coast authority")

    chunks = []
    for agency, info in AGENCIES.items():
        positioned = (
            agency_flag(source, agency, intensity=False)
            & source[info["lat"]].notna()
            & source[info["lon"]].notna()
        )
        state = source.loc[positioned, ["SID", "ISO_TIME"]].copy()
        state.columns = ["sid", "state_time"]
        state["native_ts"] = native_ts_mask(source, agency).loc[positioned].to_numpy(bool)
        state["intensity_original"] = (
            agency_flag(source, agency, intensity=True)
            .loc[positioned]
            .to_numpy(bool)
        )
        state = state.sort_values(["sid", "state_time"]).drop_duplicates(
            ["sid", "state_time"], keep="last"
        )
        group = out.loc[out["agency"].eq(agency)].copy()
        for side, time_column in (("start", "segment_start"), ("end", "segment_end")):
            renamed = state.rename(
                columns={
                    "state_time": time_column,
                    "native_ts": f"{side}_native_ts",
                    "intensity_original": f"{side}_intensity_original",
                }
            )
            group = group.merge(
                renamed,
                on=["sid", time_column],
                how="left",
                validate="many_to_one",
            )
        chunks.append(group)
    out = pd.concat(chunks, ignore_index=True)
    endpoint_columns = [
        "start_native_ts",
        "end_native_ts",
        "start_intensity_original",
        "end_intensity_original",
    ]
    if out[endpoint_columns].isna().any().any():
        raise RuntimeError("At least one exact event lacks an agency-native endpoint state")
    if not (
        out["start_intensity_original"].astype(bool)
        & out["end_intensity_original"].astype(bool)
    ).all():
        raise RuntimeError("At least one endpoint is not an original/verified intensity report")
    out["pre_crossing_native_ts"] = out["start_native_ts"].astype(bool)
    out["either_endpoint_native_ts"] = (
        out["start_native_ts"].astype(bool) | out["end_native_ts"].astype(bool)
    )
    out["both_endpoints_native_ts"] = (
        out["start_native_ts"].astype(bool) & out["end_native_ts"].astype(bool)
    )
    out["coast_legacy"] = out["coast"]
    out["coast"] = out["coast_exclusive"]
    return out.sort_values(["agency", "season", "sid", "time"]).reset_index(drop=True)


def annual_metrics(events: pd.DataFrame, years: np.ndarray) -> dict[str, float]:
    rows = []
    for year in years:
        values = events.loc[events.season.eq(year), "lat"].to_numpy(float)
        rows.append({
            "year": year, "n_events": len(values), "n_storms": events.loc[events.season.eq(year), "sid"].nunique(),
            "mean_lat": np.mean(values) if len(values) else np.nan,
            "median_lat": np.median(values) if len(values) else np.nan,
            "q25_lat": np.quantile(values, 0.25) if len(values) else np.nan,
            "q75_lat": np.quantile(values, 0.75) if len(values) else np.nan,
        })
    return rows


def summarize(annual: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    metrics = ["mean_lat", "median_lat", "q25_lat", "q75_lat"]
    for (agency, definition), group in annual.groupby(["agency", "definition"]):
        group = group.sort_values("year")
        for start, end in [(1966, 2025), (1966, 2024), (1982, 2025), (1982, 2024)]:
            period = group.loc[group.year.between(start, end)]
            for metric in metrics:
                trend = trend_summary(period.year, period[metric])
                row = {"agency": agency, "definition": definition, "metric": metric, "start": start, "end": end, **trend,
                       "period_difference": np.nan, "period_ci_low": np.nan, "period_ci_high": np.nan, "period_block_p": np.nan}
                rows.append(row)
    out = pd.DataFrame(rows)
    for end in [2024, 2025]:
        target = out.loc[(out.start == 1966) & (out.end == end)]
        series = []
        for r in target.itertuples():
            d = annual.loc[annual.agency.eq(r.agency) & annual.definition.eq(r.definition) & annual.year.between(1966, end)].sort_values("year")
            series.append(d[r.metric].to_numpy(float))
        matrix = np.column_stack(series)
        years = np.arange(1966, end + 1)
        early_idx = np.flatnonzero(years <= 1995)
        late_idx = np.flatnonzero(years >= 1996)
        difference, p = block_permutation_many(matrix, early_idx, late_idx, block=3, nperm=cfg["n_permutations"], seed=cfg["random_seed"])
        lo, hi = block_bootstrap_many(matrix[early_idx], matrix[late_idx], block=3, nboot=cfg["n_bootstrap"], seed=cfg["random_seed"])
        out.loc[target.index, "period_difference"] = difference
        out.loc[target.index, "period_ci_low"] = lo
        out.loc[target.index, "period_ci_high"] = hi
        out.loc[target.index, "period_block_p"] = p
    out["mk_fdr_family"] = (
        "landfall_latitude_trend_"
        + out.definition
        + "_"
        + out.metric
        + "_"
        + out.start.astype(str)
        + "_"
        + out.end.astype(str)
    )
    out["mk_q_bh"] = np.nan
    for _, idx in out.groupby("mk_fdr_family").groups.items():
        out.loc[idx, "mk_q_bh"] = bh_fdr(out.loc[idx, "mk_p"])
    out["period_fdr_family"] = np.where(
        out.period_block_p.notna(),
        "landfall_latitude_period_"
        + out.definition
        + "_"
        + out.metric
        + "_1966_"
        + out.end.astype(str),
        "",
    )
    out["period_q_bh"] = np.nan
    for family, idx in out.loc[out.period_block_p.notna()].groupby("period_fdr_family").groups.items():
        out.loc[idx, "period_q_bh"] = bh_fdr(out.loc[idx, "period_block_p"])
    return out


def leave_one_coast_out(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    years = np.arange(1966, 2026)
    rows, series = [], []
    for agency, agency_events in events.groupby("agency"):
        for definition, defined in definitions(agency_events).items():
            if definition not in {"first_landfall", "all_events"}:
                continue
            for excluded in ["none", *sorted(defined.coast.unique())]:
                sample = defined if excluded == "none" else defined.loc[defined.coast.ne(excluded)]
                annual = pd.DataFrame(annual_metrics(sample, years))
                early = np.flatnonzero(annual.year.between(1966, 1995))
                late = np.flatnonzero(annual.year.between(1996, 2025))
                rows.append({"agency": agency, "definition": definition, "excluded_coast": excluded, "n_events": len(sample)})
                series.append(annual.mean_lat.to_numpy(float))
    matrix = np.column_stack(series)
    difference, p = block_permutation_many(matrix, early, late, block=3, nperm=cfg["n_permutations"], seed=cfg["random_seed"])
    out = pd.DataFrame(rows)
    out["period_difference_mean_lat"] = difference
    out["block_permutation_p"] = p
    return out


def compare_raster(exact: pd.DataFrame) -> pd.DataFrame:
    approximate = pd.read_csv(WORK / "data/upstream_revision/p2_multiagency_landfalls.csv", parse_dates=["time"])
    exact = exact.copy()
    rows = []
    for (agency, sid), group in approximate.groupby(["agency", "sid"]):
        candidates = exact.loc[exact.agency.eq(agency) & exact.sid.eq(sid)]
        for r in group.itertuples():
            if candidates.empty:
                rows.append({"agency": agency, "sid": sid, "approx_time": r.time, "matched": False})
                continue
            delta = (candidates.time - r.time).abs()
            match = candidates.loc[delta.idxmin()]
            rows.append({"agency": agency, "sid": sid, "approx_time": r.time, "matched": True,
                         "exact_time": match.time, "time_difference_hours": abs((match.time-r.time).total_seconds())/3600,
                         "lat_difference_deg": match.lat-r.lat, "lon_difference_deg": match.lon-r.lon})
    return pd.DataFrame(rows)


def title_decision(summary: pd.DataFrame, leave_out: pd.DataFrame) -> tuple[str, dict]:
    main = summary.loc[(summary.start == 1966) & (summary.end == 2025) & summary.metric.eq("mean_lat") & summary.definition.isin(["first_landfall", "all_events"])]
    no2025 = summary.loc[(summary.start == 1966) & (summary.end == 2024) & summary.metric.eq("mean_lat") & summary.definition.isin(["first_landfall", "all_events"])]
    recent = summary.loc[(summary.start == 1982) & (summary.end == 2025) & summary.metric.eq("mean_lat") & summary.definition.isin(["first_landfall", "all_events"])]
    direction_all = bool((main.period_difference > 0).all())
    significant_agencies = main.loc[main.period_q_bh < 0.05, "agency"].nunique()
    no2025_no_reversal = bool((no2025.period_difference > 0).all())
    recent_direction = bool((recent.sen_slope_per_decade >= 0).all())
    leave = leave_out.loc[leave_out.excluded_coast.ne("none")]
    not_single_coast = bool((leave.period_difference_mean_lat > 0).groupby([leave.agency, leave.definition]).mean().ge(0.75).all())
    passed = direction_all and significant_agencies >= 2 and no2025_no_reversal and recent_direction and not_single_coast
    title = ("西北太平洋1966—2025年热带气旋路径空间重分配与登陆纬度北移" if passed
             else "西北太平洋1966—2025年热带气旋路径空间重分配与登陆海岸构成变化")
    return title, {"direction_all": direction_all, "significant_agencies": int(significant_agencies),
                   "no2025_no_reversal": no2025_no_reversal, "recent_direction": recent_direction,
                   "not_single_coast": not_single_coast, "rule_A_passed": passed}


def make_figure(annual, summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for ax, definition, label in [(axes[0, 0], "first_landfall", "First landfall"), (axes[0, 1], "all_events", "All landfall events")]:
        for agency in AGENCIES:
            d = annual.loc[annual.agency.eq(agency) & annual.definition.eq(definition)]
            ax.plot(d.year, d.mean_lat, lw=1.05, color=COLORS[agency], label=AGENCY_LABEL[agency])
        ax.set(xlabel="Year", ylabel="Annual mean landfall latitude (°N)", title=label)
        ax.legend(frameon=False, ncol=3)
    key = summary.loc[(summary.start == 1966) & (summary.end == 2025) & summary.metric.eq("mean_lat") & summary.definition.isin(["first_landfall", "all_events"])]
    x = np.arange(len(key)); y = key.period_difference.to_numpy(); lo = y-key.period_ci_low.to_numpy(); hi = key.period_ci_high.to_numpy()-y
    axes[1, 0].errorbar(x, y, yerr=[lo, hi], fmt="o", capsize=3, color="#333333")
    axes[1, 0].axhline(0, color="0.5", lw=0.8); axes[1, 0].set_xticks(x, [f"{AGENCY_LABEL[a]}\n{d.replace('_',' ')}" for a,d in zip(key.agency,key.definition)], rotation=20)
    axes[1, 0].set_ylabel("Late minus early mean latitude (°)")
    axes[1, 0].set_ylim(bottom=0)
    sensitivity = summary.loc[(summary.start == 1966) & summary.metric.eq("mean_lat") & summary.definition.eq("first_landfall")]
    for agency in AGENCIES:
        d = sensitivity.loc[sensitivity.agency.eq(agency)].sort_values("end")
        axes[1, 1].plot(d.end, d.period_difference, "o-", color=COLORS[agency], label=AGENCY_LABEL[agency])
    axes[1, 1].axhline(0, color="0.5", lw=0.8)
    # These are two complete-calendar-year endpoints, not a continuous monthly
    # series; show them as such rather than rendering fractional year ticks.
    endpoints = sorted(sensitivity.end.unique())
    axes[1, 1].set(
        xticks=endpoints,
        xticklabels=[f"Through Dec. {int(year)}" for year in endpoints],
        xlabel="Analysis endpoint",
        ylabel="First-landfall latitude difference (°)",
    )
    axes[1, 1].set_xlim(min(endpoints) - 0.08, max(endpoints) + 0.08)
    axes[1, 1].set_ylim(bottom=0.50)
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
    for ax in axes[:, 0]:
        ax.spines["right"].set_visible(False)
    for ax in axes[:, 1]:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    for label, ax in zip("abcd", axes.flat):
        ax.text(0.015, 0.985, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    scale_figure_typography(fig, scale=1.20)
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def update_registry(summary):
    path = REPORTS / "results_registry.csv"
    registry = pd.read_csv(path)
    registry = registry.loc[~registry.claim_id.str.startswith("landfall_latitude_", na=False)]
    rows = []
    key = summary.loc[(summary.start == 1966) & (summary.end == 2025) & summary.metric.eq("mean_lat") & summary.definition.isin(["first_landfall", "all_events"])]
    for r in key.itertuples():
        rows.append({
            "manuscript_section": "Results direct landfall latitude", "claim_id": f"landfall_latitude_{r.definition}_{r.agency}",
            "metric": "annual_mean_landfall_latitude_period_difference", "dataset": AGENCY_LABEL[r.agency], "period": "1966-1995 vs 1996-2025",
            "spatial_domain": "WNP GSHHG land crossings", "method": "exact segment-coastline intersection; annual aggregation; 3-yr block permutation",
            "estimate": r.period_difference, "uncertainty": f"moving-block 95% CI [{r.period_ci_low}, {r.period_ci_high}]",
            "p_value": r.period_block_p, "q_value": r.period_q_bh, "source_script": "src/landfall_latitude.py",
            "source_output": "analysis/01_landfall_latitude/landfall_latitude_summary.csv", "status": "recalculated_new_result",
            "notes": f"definition={r.definition}; FDR family={r.period_fdr_family}",
        })
    pd.concat([registry, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def run(*, reuse_existing: bool = False):
    started = time.time()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    source = read_ibtracs_agencies(PROJECT / "data/raw/IBTrACS.WP.v04r01.csv", start=1966, end=2025)
    full_lifecycle_path = ANALYSIS / "landfall_events_exact_full_lifecycle.csv"
    if reuse_existing:
        source_path = (
            full_lifecycle_path
            if full_lifecycle_path.exists()
            else ANALYSIS / "landfall_events_exact.csv"
        )
        full_lifecycle = pd.read_csv(
            source_path,
            parse_dates=["time", "segment_start", "segment_end"],
        )
        unresolved_path = ANALYSIS / "unresolved_candidates.csv"
        unresolved = (
            pd.read_csv(unresolved_path)
            if unresolved_path.exists()
            else pd.DataFrame()
        )
    else:
        catalogs = {agency: build_agency_catalog(source, agency) for agency in AGENCIES}
        candidate_events = pd.read_csv(WORK / "data/upstream_revision/p2_multiagency_landfalls.csv", parse_dates=["time"])
        candidate_events = candidate_events.loc[candidate_events.season.between(1966, 2025)]
        coast = gpd.read_file(cfg["landfall"]["coastline"], bbox=(80, 0, 180, 65))
        coast_geometries = coast.geometry.to_numpy()
        coast_tree = shapely.STRtree(coast_geometries)
        all_events, all_unresolved = [], []
        for agency, catalog in catalogs.items():
            events_agency, unresolved_agency = refine_candidate_crossings(
                catalog["tracks"], agency,
                candidate_events.loc[candidate_events.agency.eq(agency)],
                coast_geometries, coast_tree, cfg,
            )
            all_events.append(events_agency)
            all_unresolved.append(unresolved_agency)
        full_lifecycle = pd.concat(all_events, ignore_index=True)
        unresolved = pd.concat(all_unresolved, ignore_index=True)
        unresolved.to_csv(ANALYSIS / "unresolved_candidates.csv", index=False)

    full_lifecycle.to_csv(full_lifecycle_path, index=False)
    stage_audit = attach_native_stage_and_exclusive_coast(full_lifecycle, source)
    stage_audit.to_csv(ANALYSIS / "landfall_events_exact_stage_audit.csv", index=False)
    events = stage_audit.loc[stage_audit["pre_crossing_native_ts"]].copy()
    events.to_csv(ANALYSIS / "landfall_events_exact.csv", index=False)
    years = np.arange(1966, 2026)
    annual_rows = []
    for agency, group in events.groupby("agency"):
        for definition, sample in definitions(group).items():
            for row in annual_metrics(sample, years):
                annual_rows.append({"agency": agency, "definition": definition, **row})
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(ANALYSIS / "landfall_latitude_annual.csv", index=False)
    summary = summarize(annual, cfg)
    summary.to_csv(ANALYSIS / "landfall_latitude_summary.csv", index=False)
    leave = leave_one_coast_out(events, cfg)
    leave.to_csv(ANALYSIS / "leave_one_coast_out.csv", index=False)
    comparison = compare_raster(events)
    comparison.to_csv(ANALYSIS / "comparison_with_raster_candidates.csv", index=False)
    samples = events.groupby(["agency", "coast"], group_keys=False).head(2).head(30)
    samples.to_csv(ANALYSIS / "manual_validation_samples.csv", index=False)
    title, criteria = title_decision(summary, leave)
    key = summary.loc[(summary.start == 1966) & (summary.end == 2025) & summary.metric.eq("mean_lat") & summary.definition.isin(["first_landfall", "all_events"])]
    criteria_lines = "\n".join(f"- {name}: `{value}`" for name, value in criteria.items())
    results_lines = "\n".join(
        f"- {AGENCY_LABEL[r.agency]} / {r.definition}: Δ={r.period_difference:.3f}°，95% CI [{r.period_ci_low:.3f}, {r.period_ci_high:.3f}]，p={r.period_block_p:.4f}，q={r.period_q_bh:.4f}。"
        for r in key.itertuples())
    (ANALYSIS / "title_evidence_boundary.md").write_text(
        f"# 标题证据边界\n\n## 直接登陆纬度P0结果\n\n{results_lines}\n\n## 规则A判定\n\n{criteria_lines}\n\n留一海岸检验只用于判断北移方向是否由单一岸段驱动，相关置换p值作诊断，不替代预设完整样本的族内FDR检验。卫星时期趋势门禁仅检查方向，趋势显著性及置信区间须在正文单独披露。\n\n## 推荐标题\n\n**{title}**\n\n规则A未通过时不得在标题中使用无条件“登陆纬度北移”。\n",
        encoding="utf-8")
    (ANALYSIS / "method.md").write_text(
        "# 直接登陆纬度方法\n\n使用IBTrACS v04r01中USA、JMA（TOKYO字段）和CMA原始机构位置。仅连接同一SID内时间间隔大于0且不超过12 h的相邻点。为控制内存，先用既有0.02°GSHHG海陆掩膜识别海到陆候选线段；该步骤只作保守筛选，不提供最终坐标。随后将每条候选轨迹线段与GSHHG高分辨率L1陆地多边形边界精确求交，并按线段顺序检查交点前后的海陆状态；最终登陆时间、纬度、经度和风速全部按精确交点在线段上的比例插值。12 h内的重复交点合并；再入海后超过12 h的登陆保留。未能得到矢量海到陆交点的候选单列于`unresolved_candidates.csv`，不以近似点填补。正式登陆事件要求海岸交点前一个机构原生时次已达到热带风暴或以上；交点前后任一端及两端的阶段标记另存于审计表，不进入正文主统计。离散海岸归属统一读取当前互斥GSHHG岸段权威表。统计采用首次合格登陆、全部合格登陆事件和唯一风暴—海岸三种定义，以年度均值、中位数和四分位数为统计单位。时期差使用3年分块置换，置信区间使用3年移动块自助；趋势使用Theil—Sen和Hamed—Rao修正MK。BH-FDR按登陆定义、统计量和分析端点分别成族，每族包含USA、JMA和CMA三个机构检验。\n",
        encoding="utf-8")
    make_figure(annual, summary, ANALYSIS / "landfall_latitude_diagnostic")
    elapsed = time.time() - started
    (WORK / "outputs/logs/01_landfall_latitude.log").write_text(
        f"elapsed_seconds={elapsed:.3f}\nfull_lifecycle_events={len(full_lifecycle)}\nqualified_native_ts_events={len(events)}\nunresolved_candidates={len(unresolved)}\nUSA={sum(events.agency.eq('USA'))}\nJMA={sum(events.agency.eq('TOKYO'))}\nCMA={sum(events.agency.eq('CMA'))}\n",
        encoding="utf-8")
    print(key[["agency", "definition", "period_difference", "period_ci_low", "period_ci_high", "period_block_p", "period_q_bh"]].to_string(index=False))
    print(title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse the existing exact crossing table and rerun stage/statistical layers.",
    )
    run(reuse_existing=parser.parse_args().reuse_existing)
