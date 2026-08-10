"""Integrate completed OHC exposure CSV outputs without recomputing the analysis."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "results"
OUTPUT_ROOT = PACKAGE_ROOT / "integration" / "results"

IDENTITY = [
    "comparison", "period1", "period2", "agency", "definition",
    "year_assignment", "land_treatment", "weighting", "state_key",
    "n_years_p1", "n_years_p2",
]
COMPONENTS = [
    "delta_exposure",
    "redistribution_component",
    "ocean_component",
    "covariability_component",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def long_estimates(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    estimate = summary.melt(
        id_vars=IDENTITY,
        value_vars=COMPONENTS,
        var_name="component",
        value_name="estimate_j_m2",
    )
    interval = bootstrap[IDENTITY + ["component", "q025", "q50", "q975"]].copy()
    merged = estimate.merge(interval, on=IDENTITY + ["component"], validate="one_to_one")
    merged = merged.rename(columns={
        "q025": "q025_j_m2",
        "q50": "q50_j_m2",
        "q975": "q975_j_m2",
    })
    merged["point_sign"] = np.sign(merged["estimate_j_m2"]).astype(int)
    merged["interval_sign"] = np.select(
        [merged["q975_j_m2"] < 0.0, merged["q025_j_m2"] > 0.0],
        ["negative", "positive"],
        default="crosses_zero",
    )
    return merged


def factor_sensitivity(long: pd.DataFrame) -> pd.DataFrame:
    contrasts = [
        ("definition", "ts_only", "eligible_full_track"),
        ("year_assignment", "calendar_year", "season_year_aligned_only"),
        ("land_treatment", "ocean_points_only", "nearest_ocean_all_points"),
        ("weighting", "storm_normalized_equal_year", "track_point_equal_year"),
    ]
    rows: list[pd.DataFrame] = []
    for factor, base_value, alternative_value in contrasts:
        pair_keys = [column for column in IDENTITY if column != factor] + ["component"]
        base = long.loc[long[factor].eq(base_value), pair_keys + ["estimate_j_m2"]].rename(
            columns={"estimate_j_m2": "base_estimate_j_m2"}
        )
        alternative = long.loc[
            long[factor].eq(alternative_value), pair_keys + ["estimate_j_m2"]
        ].rename(columns={"estimate_j_m2": "alternative_estimate_j_m2"})
        paired = base.merge(alternative, on=pair_keys, validate="one_to_one")
        paired["shift_j_m2"] = paired["alternative_estimate_j_m2"] - paired["base_estimate_j_m2"]
        grouped = paired.groupby(["comparison", "component"])["shift_j_m2"]
        result = grouped.agg(
            n_pairs="size",
            mean_shift_j_m2="mean",
            median_shift_j_m2="median",
            minimum_shift_j_m2="min",
            maximum_shift_j_m2="max",
        ).reset_index()
        absolute = paired.assign(absolute_shift_j_m2=paired["shift_j_m2"].abs()).groupby(
            ["comparison", "component"]
        )["absolute_shift_j_m2"].agg(
            median_absolute_shift_j_m2="median",
            maximum_absolute_shift_j_m2="max",
        ).reset_index()
        result = result.merge(absolute, on=["comparison", "component"], validate="one_to_one")
        result.insert(0, "contrast", f"{alternative_value} minus {base_value}")
        result.insert(0, "factor", factor)
        rows.append(result)

    agency_keys = [column for column in IDENTITY if column != "agency"] + ["component"]
    agency_ranges = long.groupby(agency_keys)["estimate_j_m2"].agg(
        lambda values: float(values.max() - values.min())
    ).rename("agency_range_j_m2").reset_index()
    grouped = agency_ranges.groupby(["comparison", "component"])["agency_range_j_m2"]
    result = grouped.agg(
        n_pairs="size",
        mean_shift_j_m2="mean",
        median_shift_j_m2="median",
        minimum_shift_j_m2="min",
        maximum_shift_j_m2="max",
        median_absolute_shift_j_m2="median",
        maximum_absolute_shift_j_m2="max",
    ).reset_index()
    result.insert(0, "contrast", "maximum minus minimum across agencies")
    result.insert(0, "factor", "agency")
    rows.append(result)
    return pd.concat(rows, ignore_index=True)


def match_summary(matched_path: Path, qc_path: Path) -> pd.DataFrame:
    qc = pd.read_csv(qc_path)
    grouped = qc.groupby(["agency", "definition"], as_index=False)[[
        "matched", "outside_domain", "too_far", "total",
        "matched_land_points", "matched_ocean_points",
    ]].sum()
    grouped["matched_percent"] = 100.0 * grouped["matched"] / grouped["total"]
    grouped["land_percent_of_matched"] = (
        100.0 * grouped["matched_land_points"] / grouped["matched"]
    )

    matched = pd.read_csv(matched_path, low_memory=False)
    valid = matched.loc[matched["match_status"].eq("matched") & matched["ohc300"].notna()].copy()
    valid["cross_year"] = pd.to_datetime(valid["iso_time"]).dt.year.ne(
        pd.to_numeric(valid["season"], errors="raise")
    )
    diagnostics = valid.groupby(["agency", "definition"]).agg(
        matched_points_from_point_table=("sid", "size"),
        unique_storms=("sid", "nunique"),
        cross_year_points=("cross_year", "sum"),
    ).reset_index()
    diagnostics["cross_year_percent"] = (
        100.0 * diagnostics["cross_year_points"] / diagnostics["matched_points_from_point_table"]
    )
    return grouped.merge(diagnostics, on=["agency", "definition"], validate="one_to_one")


def spatial_contributions(path: Path) -> pd.DataFrame:
    usecols = [
        "comparison", "agency", "definition", "year_assignment",
        "land_treatment", "weighting", "ohc_month", "grid_lat", "grid_lon",
        "weight_p1", "weight_p2", "redistribution_cell", "ocean_cell",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200000):
        selected = chunk.loc[
            chunk["definition"].eq("ts_only")
            & chunk["year_assignment"].eq("calendar_year")
            & chunk["land_treatment"].eq("ocean_points_only")
            & chunk["weighting"].eq("storm_normalized_equal_year")
        ].copy()
        if selected.empty:
            continue
        selected["weight_change"] = selected["weight_p2"] - selected["weight_p1"]
        selected["latitude_band"] = pd.cut(
            selected["grid_lat"],
            [-np.inf, 10.0, 20.0, 30.0, 40.0, 50.0, np.inf],
            right=False,
            labels=["<10N", "10-20N", "20-30N", "30-40N", "40-50N", ">=50N"],
        ).astype(str)
        selected["latitude_zone"] = np.where(
            selected["grid_lat"] < 20.0, "south_of_20N", "20N_and_north"
        )
        selected["longitude_band"] = pd.cut(
            selected["grid_lon"],
            [99.999, 120.0, 140.0, 160.0, 180.001],
            right=False,
            labels=["100-120E", "120-140E", "140-160E", "160-180E"],
        ).astype(str)
        specifications = [
            ("month", selected["ohc_month"].astype(int).map(lambda value: f"{value:02d}")),
            ("latitude_band", selected["latitude_band"]),
            ("latitude_zone", selected["latitude_zone"]),
            ("longitude_band", selected["longitude_band"]),
        ]
        for aggregation, labels in specifications:
            work = selected.assign(bin=labels)
            grouped = work.groupby(
                ["comparison", "agency", "bin"], as_index=False
            )[["redistribution_cell", "ocean_cell", "weight_change"]].sum()
            grouped.insert(2, "aggregation", aggregation)
            parts.append(grouped)
    if not parts:
        raise ValueError("no component rows matched the fixed integration scope")
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(
        ["comparison", "agency", "aggregation", "bin"], as_index=False
    )[["redistribution_cell", "ocean_cell", "weight_change"]].sum()
    return combined.sort_values(["comparison", "agency", "aggregation", "bin"])


def validate_spatial_sums(spatial: pd.DataFrame, summary: pd.DataFrame) -> None:
    latitude = spatial.loc[spatial["aggregation"].eq("latitude_zone")].groupby(
        ["comparison", "agency"], as_index=False
    )[["redistribution_cell", "ocean_cell", "weight_change"]].sum()
    expected = summary.loc[
        summary["definition"].eq("ts_only")
        & summary["year_assignment"].eq("calendar_year")
        & summary["land_treatment"].eq("ocean_points_only")
        & summary["weighting"].eq("storm_normalized_equal_year"),
        ["comparison", "agency", "redistribution_component", "ocean_component"],
    ]
    checked = latitude.merge(expected, on=["comparison", "agency"], validate="one_to_one")
    if not np.allclose(
        checked["redistribution_cell"], checked["redistribution_component"], rtol=1e-12, atol=1e-4
    ):
        raise AssertionError("latitude aggregation does not reproduce redistribution totals")
    if not np.allclose(checked["ocean_cell"], checked["ocean_component"], rtol=1e-12, atol=1e-4):
        raise AssertionError("latitude aggregation does not reproduce ocean totals")
    if not np.allclose(checked["weight_change"], 0.0, rtol=0.0, atol=1e-12):
        raise AssertionError("aggregated path-weight changes do not close to zero")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("integration requires explicit --execute")

    paths = {
        "summary": SOURCE_ROOT / "decomposition_summary.csv",
        "bootstrap": SOURCE_ROOT / "decomposition_bootstrap.csv",
        "annual": SOURCE_ROOT / "annual_exposure.csv",
        "components": SOURCE_ROOT / "decomposition_components.csv.gz",
        "match_qc": SOURCE_ROOT / "match_qc.csv",
        "matched_points": SOURCE_ROOT / "matched_points.csv.gz",
        "seam": PACKAGE_ROOT / "qa" / "product_seam_audit.json",
        "output_manifest": PACKAGE_ROOT / "qa" / "output_manifest.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    summary = pd.read_csv(paths["summary"])
    bootstrap = pd.read_csv(paths["bootstrap"])
    annual = pd.read_csv(paths["annual"])
    if len(summary) != 128 or len(bootstrap) != 512 or len(annual) != 3840:
        raise AssertionError("source result tables do not have the accepted dimensions")
    long = long_estimates(summary, bootstrap)

    main = long.loc[
        long["definition"].eq("ts_only")
        & long["year_assignment"].eq("calendar_year")
        & long["land_treatment"].eq("ocean_points_only")
        & long["weighting"].eq("storm_normalized_equal_year")
    ].copy()
    robustness = long.groupby(["comparison", "component"]).agg(
        n=("component", "size"),
        point_negative=("point_sign", lambda values: int((values < 0).sum())),
        point_positive=("point_sign", lambda values: int((values > 0).sum())),
        interval_negative=("interval_sign", lambda values: int((values == "negative").sum())),
        interval_positive=("interval_sign", lambda values: int((values == "positive").sum())),
        interval_crosses_zero=("interval_sign", lambda values: int((values == "crosses_zero").sum())),
    ).reset_index()
    primary = long.loc[long["agency"].eq("PRIMARY") & long["definition"].eq("ts_only")]
    primary_ranges = primary.groupby(["comparison", "component"]).agg(
        n=("component", "size"),
        estimate_min_j_m2=("estimate_j_m2", "min"),
        estimate_max_j_m2=("estimate_j_m2", "max"),
        interval_negative=("interval_sign", lambda values: int((values == "negative").sum())),
        interval_positive=("interval_sign", lambda values: int((values == "positive").sum())),
        interval_crosses_zero=("interval_sign", lambda values: int((values == "crosses_zero").sum())),
    ).reset_index()
    factors = factor_sensitivity(long)
    matches = match_summary(paths["matched_points"], paths["match_qc"])
    spatial = spatial_contributions(paths["components"])
    validate_spatial_sums(spatial, summary)

    source_manifest = pd.DataFrame([
        {
            "name": name,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    ])
    outputs = {
        "source_manifest.csv": source_manifest,
        "main_estimates.csv": main,
        "robustness_counts.csv": robustness,
        "primary_sensitivity_ranges.csv": primary_ranges,
        "factor_sensitivity.csv": factors,
        "match_summary.csv": matches,
        "spatiotemporal_contributions.csv": spatial,
    }
    for name, frame in outputs.items():
        if frame.isna().any().any():
            raise AssertionError(f"{name} contains missing values")
        write_csv(frame, OUTPUT_ROOT / name, args.overwrite)

    deliverables = [
        Path(__file__).resolve(),
        PACKAGE_ROOT / "integration" / "README.md",
        *[OUTPUT_ROOT / name for name in outputs],
    ]
    report = PACKAGE_ROOT / "integration" / "INTEGRATED_RESULTS_REPORT_20260801.md"
    if report.is_file():
        deliverables.append(report)
    integration_manifest = pd.DataFrame([
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in deliverables
    ])
    write_csv(integration_manifest, OUTPUT_ROOT / "integration_manifest.csv", args.overwrite)


if __name__ == "__main__":
    main()
