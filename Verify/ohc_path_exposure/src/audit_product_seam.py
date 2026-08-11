"""Audit the ORAS5 consolidated/operational product boundary without correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import (
    atomic_replace,
    coordinate_grids_equivalent,
    ensure_output_target,
    load_config,
    project_path,
    require_execute,
    resolve_config_path,
    validate_method_record,
    write_json,
)


def robust_z(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size < 3:
        return float("nan")
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    if mad == 0.0:
        return 0.0 if value == median else float("inf")
    return (float(value) - median) / (1.4826 * mad)


def empirical_absolute_percentile(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return float("nan")
    return float((np.count_nonzero(np.abs(reference) <= abs(float(value))) + 1) / (reference.size + 1))


def read_year(path: Path, variable: str) -> dict:
    with xr.open_dataset(path, decode_times=True) as dataset:
        required = {variable, "nav_lat", "nav_lon", "region_mask"}
        missing = required.difference(dataset.variables)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(missing)}")
        times = pd.to_datetime(dataset["time"].values)
        order = np.argsort(times.month)
        values = np.asarray(dataset[variable].isel(time=order).values, dtype=float)
        latitude = np.asarray(dataset["nav_lat"].values, dtype=float)
        longitude = np.asarray(dataset["nav_lon"].values, dtype=float) % 360.0
        mask = np.asarray(dataset["region_mask"].values, dtype=bool)
        units = str(dataset[variable].attrs.get("units", "")).strip()
    if sorted(times.month.tolist()) != list(range(1, 13)):
        raise ValueError(f"{path.name}: expected one field for every calendar month")
    monthly_mean = np.array([np.nanmean(month[mask]) for month in values], dtype=float)
    finite = np.isfinite(values) & mask[None, :, :]
    return {
        "monthly_mean": monthly_mean,
        "annual_mean": float(monthly_mean.mean()),
        "latitude": latitude,
        "longitude": longitude,
        "mask": mask,
        "finite_any": finite.any(axis=0),
        "finite_all": finite.all(axis=0),
        "finite_monthly_counts": finite.sum(axis=(1, 2)),
        "units": units,
    }


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
    ohc = config["ohc"]
    audit = config["seam_audit"]
    prepared = project_path(ohc["prepared_dir"])
    years = list(range(int(config["study"]["start_year"]), int(config["study"]["end_year"]) + 1))
    records = {year: read_year(prepared / f"oras5_ohc300_{year}.nc", ohc["canonical_variable"]) for year in years}

    first = records[years[0]]
    coordinate_atol = float(ohc["coordinate_equality_atol_degrees"])
    structural_errors: list[str] = []
    footprint_differences: list[dict] = []
    coordinate_differences: list[dict] = []
    for year in years[1:]:
        current = records[year]
        if current["units"] != first["units"]:
            structural_errors.append(f"units differ in {year}: {current['units']!r} vs {first['units']!r}")
        latitude_difference = float(np.nanmax(np.abs(current["latitude"] - first["latitude"])))
        longitude_difference = float(np.nanmax(np.abs(current["longitude"] - first["longitude"])))
        if latitude_difference > 0.0 or longitude_difference > 0.0:
            coordinate_differences.append({
                "year": year,
                "max_abs_latitude_difference_degrees_vs_first_year": latitude_difference,
                "max_abs_longitude_difference_degrees_vs_first_year": longitude_difference,
            })
        if not coordinate_grids_equivalent(current["latitude"], first["latitude"], coordinate_atol):
            structural_errors.append(f"latitude grid differs in {year}")
        if not coordinate_grids_equivalent(current["longitude"], first["longitude"], coordinate_atol):
            structural_errors.append(f"longitude grid differs in {year}")
        if not np.array_equal(current["mask"], first["mask"]):
            structural_errors.append(f"region mask differs in {year}")
        changed_any = int(np.count_nonzero(current["finite_any"] != first["finite_any"]))
        changed_all = int(np.count_nonzero(current["finite_all"] != first["finite_all"]))
        if changed_any or changed_all:
            footprint_differences.append({
                "year": year,
                "changed_finite_any_cells_vs_first_year": changed_any,
                "changed_finite_all_cells_vs_first_year": changed_all,
            })

    left = int(audit["left_year"])
    right = int(audit["right_year"])
    if right != left + 1:
        raise ValueError("seam audit requires adjacent years")
    annual_differences = np.array(
        [records[year + 1]["annual_mean"] - records[year]["annual_mean"] for year in years[:-1]],
        dtype=float,
    )
    seam_index = years[:-1].index(left)
    annual_reference = np.delete(annual_differences, seam_index)
    seam_annual = float(records[right]["annual_mean"] - records[left]["annual_mean"])
    year_end_differences = np.array(
        [records[year + 1]["monthly_mean"][0] - records[year]["monthly_mean"][11] for year in years[:-1]],
        dtype=float,
    )
    seasonal_reference = np.delete(year_end_differences, seam_index)
    seam_dec_jan = float(records[right]["monthly_mean"][0] - records[left]["monthly_mean"][11])
    annual_z = robust_z(seam_annual, annual_reference)
    dec_jan_z = robust_z(seam_dec_jan, seasonal_reference)
    annual_percentile = empirical_absolute_percentile(seam_annual, annual_reference)
    dec_jan_percentile = empirical_absolute_percentile(seam_dec_jan, seasonal_reference)
    seam_changed_any = int(np.count_nonzero(records[left]["finite_any"] != records[right]["finite_any"]))
    seam_changed_all = int(np.count_nonzero(records[left]["finite_all"] != records[right]["finite_all"]))
    seam_latitude_difference = float(
        np.nanmax(np.abs(records[right]["latitude"] - records[left]["latitude"]))
    )
    seam_longitude_difference = float(
        np.nanmax(np.abs(records[right]["longitude"] - records[left]["longitude"]))
    )

    rows = []
    for year in years:
        row = {
            "year": year,
            "annual_domain_mean_ohc300": records[year]["annual_mean"],
            "finite_any_month_cells": int(records[year]["finite_any"].sum()),
            "finite_all_months_cells": int(records[year]["finite_all"].sum()),
            "finite_monthly_min_cells": int(records[year]["finite_monthly_counts"].min()),
            "finite_monthly_max_cells": int(records[year]["finite_monthly_counts"].max()),
        }
        row.update({f"month_{month:02d}_mean": value for month, value in enumerate(records[year]["monthly_mean"], start=1)})
        rows.append(row)
    output_csv = project_path(audit["output_csv"])
    ensure_output_target(output_csv, overwrite=args.overwrite)
    temp_csv = output_csv.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temp_csv, index=False)
    atomic_replace(temp_csv, output_csv)
    payload = {
        "status": "FAIL_STRUCTURAL" if structural_errors else "STRUCTURAL_CHECKS_PASSED",
        "structural_errors": structural_errors,
        "coordinate_tolerance_degrees": coordinate_atol,
        "coordinate_differences_vs_first_year": coordinate_differences,
        "footprint_differences_vs_first_year": footprint_differences,
        "left_year": left,
        "right_year": right,
        "annual_mean_jump": seam_annual,
        "annual_jump_robust_z": annual_z,
        "annual_jump_empirical_absolute_percentile": annual_percentile,
        "december_to_january_jump": seam_dec_jan,
        "december_to_january_robust_z": dec_jan_z,
        "december_to_january_empirical_absolute_percentile": dec_jan_percentile,
        "seam_changed_finite_any_cells": seam_changed_any,
        "seam_changed_finite_all_cells": seam_changed_all,
        "seam_max_abs_latitude_difference_degrees": seam_latitude_difference,
        "seam_max_abs_longitude_difference_degrees": seam_longitude_difference,
        "numerical_decision": audit["numerical_decision"],
        "automatic_bias_correction_applied": False,
    }
    write_json(project_path(audit["output_json"]), payload, overwrite=args.overwrite)
    if structural_errors:
        raise AssertionError("ORAS5 stream boundary has structural inconsistencies")


if __name__ == "__main__":
    main()
