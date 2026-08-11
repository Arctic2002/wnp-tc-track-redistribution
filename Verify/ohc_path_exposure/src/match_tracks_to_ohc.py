"""Match track points to contemporaneous monthly ORAS5 ocean cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from common import (
    atomic_replace,
    ensure_output_target,
    load_config,
    project_path,
    require_execute,
    resolve_config_path,
    validate_method_record,
)
from land_mask import build_land_mask, classify_points


def spherical_xyz(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    lon = np.deg2rad(np.asarray(longitude, dtype=float) % 360.0)
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def chord_to_great_circle_km(chord: np.ndarray, radius_km: float) -> np.ndarray:
    clipped = np.clip(np.asarray(chord, dtype=float) / 2.0, 0.0, 1.0)
    return 2.0 * float(radius_km) * np.arcsin(clipped)


def nearest_ocean_cells(
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    values: np.ndarray,
    region_mask: np.ndarray,
    radius_km: float,
) -> dict[str, np.ndarray]:
    valid = (
        np.asarray(region_mask, dtype=bool)
        & np.isfinite(grid_lat)
        & np.isfinite(grid_lon)
        & np.isfinite(values)
    )
    y_index, x_index = np.where(valid)
    if y_index.size == 0:
        raise ValueError("monthly OHC slice has no finite ocean cells")
    tree = cKDTree(spherical_xyz(grid_lat[valid], grid_lon[valid]))
    chord, local = tree.query(spherical_xyz(query_lat, query_lon), k=1)
    selected_y = y_index[local]
    selected_x = x_index[local]
    return {
        "grid_y": selected_y.astype(int),
        "grid_x": selected_x.astype(int),
        "grid_lat": grid_lat[selected_y, selected_x].astype(float),
        "grid_lon": (grid_lon[selected_y, selected_x] % 360.0).astype(float),
        "match_km": chord_to_great_circle_km(chord, radius_km),
        "ohc300": values[selected_y, selected_x].astype(float),
    }


def shifted_year_month(timestamp: pd.Timestamp, month_lag: int) -> tuple[int, int]:
    shifted = timestamp.to_period("M") - int(month_lag)
    return int(shifted.year), int(shifted.month)


def match_group(
    group: pd.DataFrame,
    prepared_dir: Path,
    variable: str,
    ohc_year: int,
    ohc_month: int,
    radius_km: float,
    max_distance_km: float,
) -> pd.DataFrame:
    output = group.copy()
    for column in ["grid_y", "grid_x", "grid_lat", "grid_lon", "match_km", "ohc300"]:
        output[column] = np.nan
    source_path = prepared_dir / f"oras5_ohc300_{ohc_year}.nc"
    if not source_path.is_file():
        output["match_status"] = "missing_ohc_file"
        return output
    with xr.open_dataset(source_path, decode_times=True) as dataset:
        required = {variable, "nav_lat", "nav_lon", "region_mask"}
        missing = required.difference(dataset.variables)
        if missing:
            raise ValueError(f"{source_path.name}: missing variables {sorted(missing)}")
        times = pd.to_datetime(dataset["time"].values)
        candidates = np.flatnonzero(times.month == ohc_month)
        if len(candidates) != 1:
            raise ValueError(f"{source_path.name}: month {ohc_month} occurs {len(candidates)} times")
        index = int(candidates[0])
        values = np.asarray(dataset[variable].isel(time=index).values)
        grid_lat = np.asarray(dataset["nav_lat"].values)
        grid_lon = np.asarray(dataset["nav_lon"].values)
        region_mask = np.asarray(dataset["region_mask"].values)

    matched = nearest_ocean_cells(
        output["lat"].to_numpy(float),
        output["lon"].to_numpy(float),
        grid_lat,
        grid_lon,
        values,
        region_mask,
        radius_km,
    )
    for column, data in matched.items():
        output[column] = data
    output["match_status"] = np.where(
        output["match_km"] <= max_distance_km, "matched", "too_far"
    )
    far = output["match_status"].eq("too_far")
    output.loc[far, ["grid_y", "grid_x", "grid_lat", "grid_lon", "ohc300"]] = np.nan
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(args.execute)
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    validate_method_record(config_path, config)
    match_cfg = config["matching"]
    ohc_cfg = config["ohc"]
    track_path = project_path(config["tracks"]["output_csv"])
    prepared_dir = project_path(ohc_cfg["prepared_dir"])
    points = pd.read_csv(track_path, parse_dates=["iso_time"])
    required = {"agency", "definition", "sid", "season", "iso_time", "lat", "lon"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"track table missing columns: {sorted(missing)}")

    domain = ohc_cfg["domain"]
    points["calendar_year"] = points["iso_time"].dt.year.astype("Int64")
    points["ohc_year"] = pd.NA
    points["ohc_month"] = pd.NA
    points["point_is_land"] = pd.NA
    points["match_status"] = "outside_domain"
    inside = (
        points["lon"].between(float(domain["west"]), float(domain["east"]))
        & points["lat"].between(float(domain["south"]), float(domain["north"]))
    )
    mask_info = build_land_mask(
        project_path(match_cfg["gshhg_land_shapefile"]),
        domain,
        float(match_cfg["land_mask_resolution_deg"]),
    )
    land_code = classify_points(
        points.loc[inside, "lat"].to_numpy(float),
        points.loc[inside, "lon"].to_numpy(float),
        mask_info,
    )
    if (land_code < 0).any():
        raise AssertionError("a point inside the OHC domain fell outside the GSHHG mask")
    points.loc[inside, "point_is_land"] = land_code.astype(bool)
    eligible = points.loc[inside].copy()
    shifted = eligible["iso_time"].apply(
        lambda value: shifted_year_month(pd.Timestamp(value), int(match_cfg["month_lag"]))
    )
    eligible["ohc_year"] = [item[0] for item in shifted]
    eligible["ohc_month"] = [item[1] for item in shifted]

    study_start = int(config["study"]["start_year"])
    study_end = int(config["study"]["end_year"])
    in_time_window = eligible["ohc_year"].between(study_start, study_end)
    outside_time = eligible.loc[~in_time_window].copy()
    for column in ["grid_y", "grid_x", "grid_lat", "grid_lon", "match_km", "ohc300"]:
        outside_time[column] = np.nan
    outside_time["match_status"] = "outside_ohc_time_window"

    matched_groups: list[pd.DataFrame] = []
    for (year, month), group in eligible.loc[in_time_window].groupby(["ohc_year", "ohc_month"], sort=True):
        matched_groups.append(
            match_group(
                group,
                prepared_dir,
                ohc_cfg["canonical_variable"],
                int(year),
                int(month),
                float(match_cfg["earth_radius_km"]),
                float(match_cfg["max_match_distance_km"]),
            )
        )
    if not outside_time.empty:
        matched_groups.append(outside_time)
    matched_inside = pd.concat(matched_groups, ignore_index=False) if matched_groups else eligible
    result = points.copy()
    columns = [
        "ohc_year", "ohc_month", "grid_y", "grid_x", "grid_lat", "grid_lon",
        "match_km", "ohc300", "match_status",
    ]
    for column in columns:
        if column not in result:
            result[column] = np.nan
    result.loc[matched_inside.index, columns] = matched_inside[columns].to_numpy()
    result["ohc_year"] = pd.to_numeric(result["ohc_year"], errors="coerce").astype("Int64")
    result["ohc_month"] = pd.to_numeric(result["ohc_month"], errors="coerce").astype("Int64")
    result["grid_y"] = pd.to_numeric(result["grid_y"], errors="coerce").astype("Int64")
    result["grid_x"] = pd.to_numeric(result["grid_x"], errors="coerce").astype("Int64")

    output_path = project_path(match_cfg["output_csv"])
    ensure_output_target(output_path, overwrite=args.overwrite)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    result.to_csv(temp, index=False, compression="gzip")
    atomic_replace(temp, output_path)

    qc = (
        result.assign(track_season=pd.to_numeric(result["season"], errors="coerce").astype("Int64"))
        .groupby(["agency", "definition", "track_season", "calendar_year", "match_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    qc["total"] = qc.drop(columns=["agency", "definition", "track_season", "calendar_year"]).sum(axis=1)
    land_qc = (
        result.assign(
            track_season=pd.to_numeric(result["season"], errors="coerce").astype("Int64"),
            matched_land=(result["match_status"].eq("matched") & result["point_is_land"].eq(True)).astype(int),
            matched_ocean=(result["match_status"].eq("matched") & result["point_is_land"].eq(False)).astype(int),
        )
        .groupby(["agency", "definition", "track_season", "calendar_year"], as_index=False, dropna=False)
        .agg(matched_land_points=("matched_land", "sum"), matched_ocean_points=("matched_ocean", "sum"))
    )
    qc = qc.merge(
        land_qc,
        on=["agency", "definition", "track_season", "calendar_year"],
        how="left",
        validate="one_to_one",
    )
    qc_path = output_path.parent / "match_qc.csv"
    ensure_output_target(qc_path, overwrite=args.overwrite)
    temp_qc = qc_path.with_suffix(".csv.tmp")
    qc.to_csv(temp_qc, index=False)
    atomic_replace(temp_qc, qc_path)


if __name__ == "__main__":
    main()
