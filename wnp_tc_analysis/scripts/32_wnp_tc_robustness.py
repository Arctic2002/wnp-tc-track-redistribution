from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    DATA,
    END,
    START,
    annual_fields,
    ensure_dirs,
    global_tv_permutation,
    load_agency_tracks,
    load_primary_tracks,
)

NPERM = 1999


def indices_for(years: np.ndarray, early: tuple[int, int], late: tuple[int, int]):
    e = np.flatnonzero((years >= early[0]) & (years <= early[1]))
    l = np.flatnonzero((years >= late[0]) & (years <= late[1]))
    if len(e) == 0 or len(l) == 0:
        raise ValueError((early, late))
    return e, l


def one_test(rows, catalog, weighting, width, block, years, fields, early, late, analysis, ref=None):
    e, l = indices_for(years, early, late)
    tv, p, change = global_tv_permutation(
        fields, e, l, block=block, nperm=NPERM, years=years
    )
    corr = np.nan
    if ref is not None and len(ref) == len(change) and np.std(ref) > 0 and np.std(change) > 0:
        corr = float(np.corrcoef(ref, change)[0, 1])
    rows.append(
        {
            "analysis": analysis,
            "catalog": catalog,
            "weighting": weighting,
            "grid_deg": width,
            "block_years": block,
            "early_start": early[0],
            "early_end": early[1],
            "late_start": late[0],
            "late_end": late[1],
            "n_early": len(e),
            "n_late": len(l),
            "tv": tv,
            "block_permutation_p": p,
            "n_permutations": NPERM,
            "spatial_corr_with_primary_same_grid": corr,
        }
    )
    return change


def run() -> None:
    ensure_dirs()
    base_years = np.arange(START, END + 1)
    catalogs = {"PRIMARY": load_primary_tracks()}
    catalogs.update(load_agency_tracks())
    rows: list[dict] = []

    # Grid and block sensitivity for the primary catalog.
    for width in (1.0, 2.5, 5.0):
        point, storm, *_ = annual_fields(catalogs["PRIMARY"], base_years, width)
        refs = {}
        for weighting, fields in (("track_point", point), ("storm_equal", storm)):
            refs[weighting] = one_test(
                rows, "PRIMARY", weighting, width, 3, base_years, fields,
                (1966, 1995), (1996, 2025), "grid_primary", None,
            )
            for block in (2, 4, 5):
                one_test(
                    rows, "PRIMARY", weighting, width, block, base_years, fields,
                    (1966, 1995), (1996, 2025), "block_sensitivity", refs[weighting],
                )

    # Cross-agency primary setting.
    agency_fields = {}
    for catalog, tracks in catalogs.items():
        point, storm, *_ = annual_fields(tracks, base_years, 2.5)
        agency_fields[catalog] = {"track_point": point, "storm_equal": storm}
        for weighting, fields in agency_fields[catalog].items():
            one_test(
                rows, catalog, weighting, 2.5, 3, base_years, fields,
                (1966, 1995), (1996, 2025), "cross_agency_primary", None,
            )

    # Equal-sized endpoint alternatives; a center year is omitted in the two 59-year cases.
    endpoint_defs = [
        ("end_2024_drop_1995", np.arange(1966, 2025), (1966, 1994), (1996, 2024)),
        ("start_1967_drop_1996", np.arange(1967, 2026), (1967, 1995), (1997, 2025)),
        ("exclude_2020_2025", np.arange(1966, 2020), (1966, 1992), (1993, 2019)),
    ]
    for label, years, early, late in endpoint_defs:
        for catalog, tracks in catalogs.items():
            point, storm, *_ = annual_fields(tracks, years, 2.5)
            for weighting, fields in (("track_point", point), ("storm_equal", storm)):
                one_test(rows, catalog, weighting, 2.5, 3, years, fields, early, late, label)

    # Leave-one-decade-out influence checks. Unequal sample sizes are retained and recorded.
    removals = {
        "drop_1970s": (1970, 1979),
        "drop_1980s": (1980, 1989),
        "drop_1990s": (1990, 1999),
        "drop_2000s": (2000, 2009),
        "drop_2010s": (2010, 2019),
        "drop_2020_2025": (2020, 2025),
    }
    for label, (a, b) in removals.items():
        keep = ~((base_years >= a) & (base_years <= b))
        years = base_years[keep]
        for catalog in catalogs:
            for weighting, full in agency_fields[catalog].items():
                fields = full[keep]
                one_test(
                    rows, catalog, weighting, 2.5, 3, years, fields,
                    (1966, 1995), (1996, 2025), label,
                )

    # Stronger-storm sensitivity for the primary catalog only.
    ty = load_primary_tracks(threshold=64.0)
    point, storm, *_ = annual_fields(ty, base_years, 2.5)
    for weighting, fields in (("track_point", point), ("storm_equal", storm)):
        one_test(
            rows, "PRIMARY_TY", weighting, 2.5, 3, base_years, fields,
            (1966, 1995), (1996, 2025), "typhoon_threshold",
        )

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "wnp_tc_robustness_matrix.csv", index=False)
    out.loc[out["analysis"].str.startswith("drop_")].to_csv(
        DATA / "wnp_tc_leave_decade_out.csv", index=False
    )
    out.loc[out["analysis"].isin([x[0] for x in endpoint_defs])].to_csv(
        DATA / "wnp_tc_endpoint_sensitivity.csv", index=False
    )
    print(out.groupby(["analysis", "weighting"])["block_permutation_p"].agg(["count", "min", "median", "max"]).to_string())


if __name__ == "__main__":
    run()
