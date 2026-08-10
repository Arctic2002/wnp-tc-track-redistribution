from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from common import (
    DATA,
    EARLY,
    END,
    LATE,
    START,
    annual_fields,
    block_permutation_projection,
    ensure_dirs,
    load_agency_tracks,
    load_primary_tracks,
    period_indices,
    projection_scores,
    sen_mk,
    write_json,
)


def run() -> None:
    ensure_dirs()
    years = np.arange(START, END + 1)
    catalogs = {"PRIMARY": load_primary_tracks()}
    catalogs.update(load_agency_tracks())
    annual_rows: list[dict] = []
    trend_rows: list[dict] = []
    pattern_payload: dict[str, np.ndarray] = {"years": years}
    e, l = period_indices(years)

    for agency, tracks in catalogs.items():
        point, storm, nstorms, npoints, lon_edges, lat_edges = annual_fields(tracks, years, 2.5)
        for weighting, fields in (("track_point", point), ("storm_equal", storm)):
            full, oos, pattern = projection_scores(fields, years)
            observed, p = block_permutation_projection(fields, years, block=3, nperm=9999)
            trend = sen_mk(years, oos)
            trend_rows.append(
                {
                    "agency": agency,
                    "weighting": weighting,
                    "early_mean_oos": float(np.mean(oos[e])),
                    "late_mean_oos": float(np.mean(oos[l])),
                    "late_minus_early_oos": observed,
                    "block_permutation_p": p,
                    **trend,
                }
            )
            for i, year in enumerate(years):
                annual_rows.append(
                    {
                        "year": int(year),
                        "agency": agency,
                        "weighting": weighting,
                        "index_full": float(full[i]),
                        "index_oos": float(oos[i]),
                        "n_storms": int(nstorms[i]),
                        "n_track_points": int(npoints[i]),
                    }
                )
            prefix = f"{agency}_{weighting}"
            pattern_payload[f"{prefix}_annual_fields"] = fields
            pattern_payload[f"{prefix}_pattern"] = pattern
        pattern_payload[f"{agency}_lon_edges"] = lon_edges
        pattern_payload[f"{agency}_lat_edges"] = lat_edges

    annual = pd.DataFrame(annual_rows)
    trends = pd.DataFrame(trend_rows)
    agreement_rows = []
    for weighting in ("track_point", "storm_equal"):
        pivot = annual.loc[annual.weighting == weighting].pivot(index="year", columns="agency", values="index_oos")
        for a, b in combinations(pivot.columns, 2):
            agreement_rows.append({
                "weighting": weighting,
                "catalog_a": a,
                "catalog_b": b,
                "annual_index_correlation": float(pivot[a].corr(pivot[b])),
                "change_pattern_correlation": float(np.corrcoef(
                    pattern_payload[f"{a}_{weighting}_pattern"],
                    pattern_payload[f"{b}_{weighting}_pattern"],
                )[0, 1]),
            })
    annual.to_csv(DATA / "wnp_tc_redistribution_index_annual.csv", index=False)
    trends.to_csv(DATA / "wnp_tc_redistribution_index_summary.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(DATA / "wnp_tc_redistribution_index_agreement.csv", index=False)
    np.savez_compressed(DATA / "wnp_tc_redistribution_pattern.npz", **pattern_payload)
    write_json(
        DATA / "wnp_tc_redistribution_index_metadata.json",
        {
            "period": [START, END],
            "early": list(EARLY),
            "late": list(LATE),
            "grid_degrees": 2.5,
            "projection": "2*(p-midpoint).pattern/(pattern.pattern)",
            "oos": "leave target year out of its period centroid and pattern",
            "permutation_test": "refit the leave-one-year-out projection pattern within every block permutation",
            "permutations": 9999,
            "block_years": 3,
        },
    )
    print(trends.to_string(index=False))


if __name__ == "__main__":
    run()
