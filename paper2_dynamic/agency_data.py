"""Read original agency reports from the full IBTrACS WP table.

The IBTrACS interpolation flag is essential here.  Agency positions retain
O/I/V (original position); agency intensity and native TS classification retain
O/V only.  P reports are filled/interpolated by IBTrACS and are not treated as
independent agency observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AGENCIES = {
    "USA": {
        "flag_index": 0,
        "lat": "USA_LAT",
        "lon": "USA_LON",
        "wind": "USA_WIND",
        "class": "USA_STATUS",
    },
    "TOKYO": {
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

BASE_COLUMNS = [
    "SID",
    "SEASON",
    "ISO_TIME",
    "TRACK_TYPE",
    "IFLAG",
    "NATURE",
]
AGENCY_COLUMNS = [
    "USA_LAT",
    "USA_LON",
    "USA_STATUS",
    "USA_WIND",
    "TOKYO_LAT",
    "TOKYO_LON",
    "TOKYO_GRADE",
    "TOKYO_WIND",
    "CMA_LAT",
    "CMA_LON",
    "CMA_CAT",
    "CMA_WIND",
]


def read_ibtracs_agencies(path, start=1945, end=2025):
    """Read only the columns required for USA/JMA/CMA comparisons."""
    frame = pd.read_csv(
        Path(path),
        usecols=BASE_COLUMNS + AGENCY_COLUMNS,
        skiprows=[1],  # official list CSV stores units in its second row
        low_memory=False,
    )
    frame["SEASON"] = pd.to_numeric(frame["SEASON"], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    numeric = [
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "TOKYO_LAT",
        "TOKYO_LON",
        "TOKYO_GRADE",
        "TOKYO_WIND",
        "CMA_LAT",
        "CMA_LON",
        "CMA_CAT",
        "CMA_WIND",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["SEASON"].between(start, end)
        & frame["ISO_TIME"].notna()
        & ~frame["TRACK_TYPE"].fillna("").str.lower().str.contains("spur")
    ].copy()
    # Match the project's main-synoptic-time protocol.
    frame = frame.loc[frame["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])]
    frame["IFLAG"] = frame["IFLAG"].fillna("").astype(str)
    return frame


def agency_flag(frame, agency, *, intensity=False):
    """Mask original agency reports using the documented IFLAG character."""
    index = AGENCIES[agency]["flag_index"]
    allowed = ["O", "V"] if intensity else ["O", "I", "V"]
    return frame["IFLAG"].str.get(index).isin(allowed)


def native_ts_mask(frame, agency):
    """Identify TS-or-higher reports in each agency's native convention."""
    info = AGENCIES[agency]
    original = agency_flag(frame, agency, intensity=True)
    located = frame[info["lat"]].notna() & frame[info["lon"]].notna()
    if agency == "USA":
        # Early JTWC status entries are often blank in IBTrACS.  Use the native
        # 1-min 34-kt threshold and only exclude explicitly non-tropical stages.
        non_tropical = {
            "EX",
            "ET",
            "PT",
            "SS",
            "SD",
            "IN",
            "DS",
            "DB",
            "LO",
            "WV",
            "MD",
        }
        status = frame[info["class"]].fillna("").astype(str).str.strip()
        classified = (frame[info["wind"]] >= 34) & ~status.isin(non_tropical)
    elif agency == "TOKYO":
        # JMA grades: 3 TS, 4 STS, 5 TY, 9 TC of TS intensity or higher.
        classified = frame[info["class"]].isin([3, 4, 5, 9])
    elif agency == "CMA":
        # CMA categories 2-6 are TS through super typhoon.
        classified = frame[info["class"]].isin([2, 3, 4, 5, 6])
    else:
        raise ValueError(agency)
    return original & located & classified


def build_agency_catalog(frame, agency):
    """Return TS points, full original-position tracks, LMI table and counts."""
    info = AGENCIES[agency]
    ts_mask = native_ts_mask(frame, agency)
    eligible = frame.loc[ts_mask, "SID"].unique()

    ts = frame.loc[ts_mask, ["SID", "SEASON", "ISO_TIME", info["lat"], info["lon"], info["wind"]]].copy()
    ts.columns = ["sid", "season", "iso_time", "lat", "lon", "wind"]
    ts["lon"] = ts["lon"] % 360
    ts = ts.sort_values(["sid", "iso_time"]).drop_duplicates(["sid", "iso_time"], keep="last")

    pos_mask = (
        agency_flag(frame, agency, intensity=False)
        & frame["SID"].isin(eligible)
        & frame[info["lat"]].notna()
        & frame[info["lon"]].notna()
    )
    tracks = frame.loc[
        pos_mask,
        ["SID", "SEASON", "ISO_TIME", info["lat"], info["lon"], info["wind"]],
    ].copy()
    tracks.columns = ["sid", "season", "iso_time", "lat", "lon", "wind"]
    tracks["lon"] = tracks["lon"] % 360
    tracks = tracks.sort_values(["sid", "iso_time"]).drop_duplicates(
        ["sid", "iso_time"], keep="last"
    )

    with_wind = ts.loc[ts["wind"].notna()].copy()
    index = with_wind.groupby("sid")["wind"].idxmax()
    lmi = with_wind.loc[index, ["sid", "season", "iso_time", "lat", "lon", "wind"]].copy()
    lmi.columns = ["sid", "season", "lmi_time", "lmi_lat", "lmi_lon", "lmi_wind"]
    lmi["agency"] = agency

    frequency = (
        ts.groupby("season")["sid"].nunique().rename("n_tc").reset_index()
    )
    frequency["agency"] = agency
    return {
        "agency": agency,
        "ts_points": ts,
        "tracks": tracks,
        "lmi": lmi,
        "frequency": frequency,
        "eligible_sids": set(eligible),
    }

