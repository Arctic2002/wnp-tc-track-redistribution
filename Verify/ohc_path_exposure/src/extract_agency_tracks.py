"""Build independently implemented PRIMARY/USA/JMA/CMA track definitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    atomic_replace,
    ensure_output_target,
    load_config,
    project_path,
    require_execute,
    resolve_config_path,
    validate_code_review_gate,
)


NON_TROPICAL_USA = {
    "EX", "ET", "PT", "SS", "SD", "IN", "DS", "DB", "LO", "WV", "MD"
}

AGENCIES = {
    "USA": {
        "flag_index": 0,
        "lat": "USA_LAT",
        "lon": "USA_LON",
        "wind": "USA_WIND",
        "class": "USA_STATUS",
    },
    "JMA": {
        "flag_index": 1,
        "lat": "TOKYO_LAT",
        "lon": "TOKYO_LON",
        "wind": "TOKYO_WIND",
        "class": "TOKYO_GRADE",
    },
    "CMA": {
        "flag_index": 2,
        "lat": "CMA_LAT",
        "lon": "CMA_LON",
        "wind": "CMA_WIND",
        "class": "CMA_CAT",
    },
}

BASE_COLUMNS = ["SID", "SEASON", "ISO_TIME", "TRACK_TYPE", "IFLAG", "NATURE"]
AGENCY_COLUMNS = sorted(
    {column for item in AGENCIES.values() for column in (item["lat"], item["lon"], item["wind"], item["class"])}
)


def agency_flag(frame: pd.DataFrame, agency: str, *, intensity: bool) -> pd.Series:
    index = int(AGENCIES[agency]["flag_index"])
    allowed = {"O", "V"} if intensity else {"O", "I", "V"}
    return frame["IFLAG"].fillna("").astype(str).str.get(index).isin(allowed)


def native_ts_mask(frame: pd.DataFrame, agency: str) -> pd.Series:
    info = AGENCIES[agency]
    original = agency_flag(frame, agency, intensity=True)
    located = frame[info["lat"]].notna() & frame[info["lon"]].notna()
    if agency == "USA":
        status = frame[info["class"]].fillna("").astype(str).str.strip().str.upper()
        classified = (frame[info["wind"]] >= 34.0) & ~status.isin(NON_TROPICAL_USA)
    elif agency == "JMA":
        classified = frame[info["class"]].isin([3, 4, 5, 9])
    elif agency == "CMA":
        classified = frame[info["class"]].isin([2, 3, 4, 5, 6])
    else:
        raise ValueError(f"unknown agency: {agency}")
    return original & located & classified


def read_ibtracs(path: Path, start: int, end: int, synoptic_hours: list[int]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=BASE_COLUMNS + AGENCY_COLUMNS,
        skiprows=[1],
        low_memory=False,
    )
    frame["SEASON"] = pd.to_numeric(frame["SEASON"], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    for column in AGENCY_COLUMNS:
        if column not in {"USA_STATUS"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["SEASON"].between(start, end)
        & frame["ISO_TIME"].notna()
        & frame["ISO_TIME"].dt.hour.isin(synoptic_hours)
        & ~frame["TRACK_TYPE"].fillna("").str.lower().str.contains("spur")
    )
    return frame.loc[valid].copy()


def _standardize_output(frame: pd.DataFrame, agency: str, definition: str, info: dict) -> pd.DataFrame:
    output = frame[["SID", "SEASON", "ISO_TIME", info["lat"], info["lon"], info["wind"]]].copy()
    output.columns = ["sid", "season", "iso_time", "lat", "lon", "wind"]
    output.insert(0, "definition", definition)
    output.insert(0, "agency", agency)
    output["lon"] = pd.to_numeric(output["lon"], errors="coerce") % 360.0
    output["lat"] = pd.to_numeric(output["lat"], errors="coerce")
    output["wind"] = pd.to_numeric(output["wind"], errors="coerce")
    output = output.dropna(subset=["sid", "season", "iso_time", "lat", "lon"])
    return output.sort_values(["sid", "iso_time"]).drop_duplicates(
        ["agency", "definition", "sid", "iso_time"], keep="last"
    )


def build_agency_definitions(frame: pd.DataFrame, agency: str) -> list[pd.DataFrame]:
    info = AGENCIES[agency]
    ts_mask = native_ts_mask(frame, agency)
    eligible = set(frame.loc[ts_mask, "SID"].dropna().astype(str))
    ts = _standardize_output(frame.loc[ts_mask], agency, "ts_only", info)
    full_mask = (
        agency_flag(frame, agency, intensity=False)
        & frame["SID"].astype(str).isin(eligible)
        & frame[info["lat"]].notna()
        & frame[info["lon"]].notna()
    )
    full = _standardize_output(frame.loc[full_mask], agency, "eligible_full_track", info)
    return [ts, full]


def build_primary_definitions(path: Path, start: int, end: int, synoptic_hours: list[int]) -> list[pd.DataFrame]:
    frame = pd.read_csv(path, parse_dates=["iso_time"])
    required = {"sid", "season", "iso_time", "lat", "lon", "wind", "nature"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"PRIMARY source missing columns: {sorted(missing)}")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    valid = (
        frame["season"].between(start, end)
        & frame["iso_time"].notna()
        & frame["iso_time"].dt.hour.isin(synoptic_hours)
    )
    frame = frame.loc[valid].copy()
    ts_mask = (pd.to_numeric(frame["wind"], errors="coerce") >= 34.0) & (
        frame["nature"].eq("TS") | frame["nature"].isna()
    )
    eligible = set(frame.loc[ts_mask, "sid"].dropna().astype(str))

    def standardize(part: pd.DataFrame, definition: str) -> pd.DataFrame:
        output = part[["sid", "season", "iso_time", "lat", "lon", "wind"]].copy()
        output.insert(0, "definition", definition)
        output.insert(0, "agency", "PRIMARY")
        output["lon"] = pd.to_numeric(output["lon"], errors="coerce") % 360.0
        output["lat"] = pd.to_numeric(output["lat"], errors="coerce")
        output = output.dropna(subset=["sid", "season", "iso_time", "lat", "lon"])
        return output.sort_values(["sid", "iso_time"]).drop_duplicates(
            ["agency", "definition", "sid", "iso_time"], keep="last"
        )

    return [
        standardize(frame.loc[ts_mask], "ts_only"),
        standardize(frame.loc[frame["sid"].astype(str).isin(eligible)], "eligible_full_track"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(args.execute)
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    validate_code_review_gate(config_path, config)
    track_cfg = config["tracks"]
    start = int(track_cfg["start_year"])
    end = int(track_cfg["end_year"])
    hours = [int(value) for value in track_cfg["synoptic_hours"]]

    outputs = build_primary_definitions(project_path(track_cfg["primary_csv"]), start, end, hours)
    source = read_ibtracs(project_path(track_cfg["ibtracs_csv"]), start, end, hours)
    for agency in AGENCIES:
        outputs.extend(build_agency_definitions(source, agency))
    combined = pd.concat(outputs, ignore_index=True)
    if combined.duplicated(["agency", "definition", "sid", "iso_time"]).any():
        raise AssertionError("duplicate track identity after extraction")
    if not combined["lon"].between(0.0, 360.0, inclusive="left").all():
        raise AssertionError("longitude normalization failed")

    output_path = project_path(track_cfg["output_csv"])
    ensure_output_target(output_path, overwrite=args.overwrite)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    combined.to_csv(temp, index=False, compression="gzip")
    atomic_replace(temp, output_path)

    qc = (
        combined.assign(year=combined["season"].astype(int))
        .groupby(["agency", "definition", "year"], as_index=False)
        .agg(storms=("sid", "nunique"), points=("sid", "size"))
    )
    qc_path = output_path.parent / "track_extraction_qc.csv"
    ensure_output_target(qc_path, overwrite=args.overwrite)
    temp_qc = qc_path.with_suffix(".csv.tmp")
    qc.to_csv(temp_qc, index=False)
    atomic_replace(temp_qc, qc_path)


if __name__ == "__main__":
    main()
