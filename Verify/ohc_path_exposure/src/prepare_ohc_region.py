"""Crop yearly ORAS5 OHC300 archives to the fixed WNP native-grid domain."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import (
    PACKAGE_ROOT,
    atomic_replace,
    ensure_output_target,
    load_config,
    project_path,
    require_execute,
    resolve_config_path,
    sha256_file,
    validate_code_review_gate,
)


def _archive_for_year(config: dict, year: int) -> tuple[str, Path]:
    ohc = config["ohc"]
    if year <= int(ohc["consolidated_end_year"]):
        stream = "consolidated"
        root = project_path(ohc["consolidated_dir"])
    else:
        stream = "operational"
        root = project_path(ohc["operational_dir"])
    name = f"oras5_{stream}_ocean_heat_content_for_the_upper_300m_{year}.zip"
    return stream, root / name


def _safe_extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    member_path = Path(member)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe archive member: {member}")
    with archive.open(member, "r") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def regional_subset(
    dataset: xr.Dataset,
    native_variable: str,
    canonical_variable: str,
    domain: dict,
) -> xr.Dataset:
    required = {native_variable, "nav_lat", "nav_lon"}
    missing = required.difference(dataset.variables)
    if missing:
        raise ValueError(f"missing ORAS5 variables: {sorted(missing)}")

    latitude = dataset["nav_lat"]
    longitude = dataset["nav_lon"] % 360.0
    mask = (
        (longitude >= float(domain["west"]))
        & (longitude <= float(domain["east"]))
        & (latitude >= float(domain["south"]))
        & (latitude <= float(domain["north"]))
    )
    locations = np.argwhere(np.asarray(mask))
    if locations.size == 0:
        raise ValueError("fixed domain contains no ORAS5 grid cells")
    y0, x0 = locations.min(axis=0)
    y1, x1 = locations.max(axis=0)

    subset = dataset[[native_variable]].isel(
        y=slice(int(y0), int(y1) + 1),
        x=slice(int(x0), int(x1) + 1),
    )
    submask = mask.isel(
        y=slice(int(y0), int(y1) + 1),
        x=slice(int(x0), int(x1) + 1),
    )
    subset = subset.rename({native_variable: canonical_variable})
    if "time_counter" in subset.dims:
        subset = subset.rename({"time_counter": "time"})
    elif "time" not in subset.dims:
        raise ValueError("OHC variable has no time dimension")
    subset = subset.assign_coords(
        nav_lat=latitude.isel(y=slice(int(y0), int(y1) + 1), x=slice(int(x0), int(x1) + 1)),
        nav_lon=longitude.isel(y=slice(int(y0), int(y1) + 1), x=slice(int(x0), int(x1) + 1)),
    )
    subset[canonical_variable] = subset[canonical_variable].where(submask)
    subset["region_mask"] = submask.astype("int8")
    return subset


def prepare_year(config: dict, year: int, output_dir: Path, scratch_root: Path, overwrite: bool) -> dict:
    ohc = config["ohc"]
    stream, archive_path = _archive_for_year(config, year)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    final_path = output_dir / f"oras5_ohc300_{year}.nc"
    ensure_output_target(final_path, overwrite=overwrite)
    native = ohc["native_variable"]
    canonical = ohc["canonical_variable"]

    monthly: list[xr.Dataset] = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(
        prefix=f"year_{year}_", dir=scratch_root
    ) as temp_name:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".nc"))
        expected = int(ohc["expected_months_per_year"])
        if len(members) != expected:
            raise ValueError(f"{year}: expected {expected} NetCDF members, found {len(members)}")
        temp_dir = Path(temp_name)
        for index, member in enumerate(members):
            extracted = temp_dir / f"month_{index + 1:02d}.nc"
            _safe_extract_member(archive, member, extracted)
            with xr.open_dataset(extracted, decode_times=True) as source:
                piece = regional_subset(source, native, canonical, ohc["domain"]).load()
            if piece.sizes.get("time") != 1:
                raise ValueError(f"{year}/{member}: expected one monthly time step")
            monthly.append(piece)

    combined = xr.concat(
        monthly,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
    ).sortby("time")
    times = pd.to_datetime(combined["time"].values)
    if set(times.year) != {year} or sorted(times.month.tolist()) != list(range(1, 13)):
        raise ValueError(f"{year}: decoded times are not the 12 calendar months")
    units = str(combined[canonical].attrs.get("units", "")).strip()
    normalized_units = units.lower().replace(" ", "").replace("^", "")
    if normalized_units not in {"j/m2", "jm-2", "jm**-2"}:
        raise ValueError(f"{year}: unexpected or missing OHC units {units!r}")
    combined[canonical].attrs["source_units"] = units
    combined[canonical].attrs["units"] = ohc["canonical_units"]
    finite = np.isfinite(np.asarray(combined[canonical].values))
    region = np.asarray(combined["region_mask"].values, dtype=bool)
    finite_any = finite.any(axis=0) & region
    finite_all = finite.all(axis=0) & region
    combined.attrs.update(
        analysis_domain="100E-180E, 0N-50N; ORAS5 native curvilinear grid",
        source_archive=archive_path.name,
        source_stream=stream,
        interpolation="none",
    )
    temp_output = final_path.with_suffix(".nc.tmp")
    encoding = {canonical: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    combined.encoding.pop("unlimited_dims", None)
    combined.to_netcdf(temp_output, engine="netcdf4", encoding=encoding)
    atomic_replace(temp_output, final_path)
    return {
        "year": year,
        "stream": stream,
        "source_zip": str(archive_path.relative_to(project_path("."))),
        "source_sha256": sha256_file(archive_path),
        "months": int(combined.sizes["time"]),
        "grid_cells": int(combined["region_mask"].sum()),
        "finite_any_month_cells": int(finite_any.sum()),
        "finite_all_months_cells": int(finite_all.sum()),
        "finite_monthly_min_cells": int((finite & region[None, :, :]).sum(axis=(1, 2)).min()),
        "finite_monthly_max_cells": int((finite & region[None, :, :]).sum(axis=(1, 2)).max()),
        "units_seen": units,
        "units_written": ohc["canonical_units"],
        "output_file": str(final_path.relative_to(project_path("."))),
        "output_sha256": sha256_file(final_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(args.execute)
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    validate_code_review_gate(config_path, config)
    start = int(config["study"]["start_year"])
    end = int(config["study"]["end_year"])
    years = args.years or list(range(start, end + 1))
    if any(year < start or year > end for year in years):
        raise ValueError("requested years fall outside the fixed study range")

    output_dir = project_path(config["ohc"]["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = PACKAGE_ROOT / "results" / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    rows = [prepare_year(config, year, output_dir, scratch_root, args.overwrite) for year in years]
    manifest = PACKAGE_ROOT / "results" / "ohc_region_manifest.csv"
    ensure_output_target(manifest, overwrite=args.overwrite)
    temp_manifest = manifest.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temp_manifest, index=False)
    atomic_replace(temp_manifest, manifest)


if __name__ == "__main__":
    main()
