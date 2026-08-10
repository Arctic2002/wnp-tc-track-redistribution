"""Exact symmetric decomposition of path-month OHC300 exposure."""

from __future__ import annotations

import argparse
import math
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
    validate_code_review_gate,
    validate_seam_review_gate,
)


def comparison_list(config: dict) -> list[dict]:
    study = config["study"]
    return [study["primary_comparison"], *study.get("sensitivity_comparisons", [])]


def state_table(matched: pd.DataFrame) -> pd.DataFrame:
    source = matched.loc[matched["match_status"].eq("matched")].copy()
    keys = (
        source[["ohc_month", "grid_y", "grid_x", "grid_lat", "grid_lon"]]
        .dropna()
        .drop_duplicates(["ohc_month", "grid_y", "grid_x"])
        .sort_values(["ohc_month", "grid_y", "grid_x"])
        .reset_index(drop=True)
    )
    keys[["ohc_month", "grid_y", "grid_x"]] = keys[["ohc_month", "grid_y", "grid_x"]].astype(int)
    usage = (
        source.groupby(["ohc_month", "grid_y", "grid_x"], as_index=False)
        .agg(
            source_ocean_points=("point_is_land", lambda values: int(values.eq(False).sum())),
            source_land_points=("point_is_land", lambda values: int(values.eq(True).sum())),
        )
    )
    usage[["ohc_month", "grid_y", "grid_x"]] = usage[["ohc_month", "grid_y", "grid_x"]].astype(int)
    usage["used_ocean_points_only"] = usage["source_ocean_points"] > 0
    usage["used_nearest_ocean_all_points"] = (
        usage["source_ocean_points"] + usage["source_land_points"]
    ) > 0
    usage["sensitivity_only_land_source_key"] = (
        usage["source_ocean_points"].eq(0) & usage["source_land_points"].gt(0)
    )
    keys = keys.merge(
        usage,
        on=["ohc_month", "grid_y", "grid_x"],
        how="left",
        validate="one_to_one",
    )
    keys.insert(0, "key_id", np.arange(len(keys), dtype=int))
    return keys


def attach_key_ids(matched: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    return matched.merge(
        keys[["key_id", "ohc_month", "grid_y", "grid_x"]],
        on=["ohc_month", "grid_y", "grid_x"],
        how="left",
        validate="many_to_one",
    )


def load_field_matrix(
    prepared_dir: Path,
    variable: str,
    years: list[int],
    keys: pd.DataFrame,
    coordinate_atol_degrees: float,
) -> np.ndarray:
    months = keys["ohc_month"].to_numpy(int) - 1
    y_index = keys["grid_y"].to_numpy(int)
    x_index = keys["grid_x"].to_numpy(int)
    matrix = np.empty((len(years), len(keys)), dtype="float64")
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    for row, year in enumerate(years):
        path = prepared_dir / f"oras5_ohc300_{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, decode_times=True) as dataset:
            times = pd.to_datetime(dataset["time"].values)
            if sorted(times.month.tolist()) != list(range(1, 13)):
                raise ValueError(f"{path.name}: incomplete or duplicate months")
            order = np.argsort(times.month)
            values = np.asarray(dataset[variable].isel(time=order).values, dtype="float64")
            latitude = np.asarray(dataset["nav_lat"].values, dtype="float64")
            longitude = np.asarray(dataset["nav_lon"].values, dtype="float64") % 360.0
        if reference_lat is None:
            reference_lat, reference_lon = latitude, longitude
        elif not (
            coordinate_grids_equivalent(reference_lat, latitude, coordinate_atol_degrees)
            and coordinate_grids_equivalent(reference_lon, longitude, coordinate_atol_degrees)
        ):
            raise ValueError(f"{path.name}: native grid differs from the first year")
        matrix[row] = values[months, y_index, x_index]
    if not np.isfinite(matrix).all():
        locations = np.argwhere(~np.isfinite(matrix))
        examples = []
        for year_row_index, key_index in locations[:20]:
            key = keys.iloc[int(key_index)]
            examples.append(
                {
                    "year": years[int(year_row_index)],
                    "month": int(key["ohc_month"]),
                    "grid_y": int(key["grid_y"]),
                    "grid_x": int(key["grid_x"]),
                    "used_ocean_points_only": bool(key["used_ocean_points_only"]),
                    "used_nearest_ocean_all_points": bool(key["used_nearest_ocean_all_points"]),
                    "sensitivity_only_land_source_key": bool(key["sensitivity_only_land_source_key"]),
                }
            )
        raise ValueError(
            f"period counterfactual field has {len(locations)} non-finite state values; "
            f"first locations={examples}"
        )
    return matrix


def annual_weight_matrix(
    frame: pd.DataFrame,
    years: list[int],
    n_keys: int,
    weighting: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.zeros((len(years), n_keys), dtype="float64")
    storms = np.zeros(len(years), dtype=int)
    points = np.zeros(len(years), dtype=int)
    effective_storms = np.zeros(len(years), dtype="float64")
    storm_point_totals = frame.groupby("sid")["sid"].size()
    for row, year in enumerate(years):
        annual = frame.loc[frame["analysis_year"].eq(year)].copy()
        if annual.empty:
            raise ValueError(f"no matched track points in {year}")
        storms[row] = int(annual["sid"].nunique())
        points[row] = int(len(annual))
        base_storm_weights = 1.0 / annual["sid"].map(storm_point_totals).to_numpy(float)
        effective_storms[row] = float(base_storm_weights.sum())
        if weighting == "storm_normalized_equal_year":
            annual["row_weight"] = base_storm_weights
            annual["row_weight"] /= float(annual["row_weight"].sum())
        elif weighting == "track_point_equal_year":
            annual["row_weight"] = 1.0 / float(points[row])
        else:
            raise ValueError(f"unknown weighting: {weighting}")
        grouped = annual.groupby("key_id")["row_weight"].sum()
        key_index = grouped.index.to_numpy(int)
        weights[row, key_index] = grouped.to_numpy(float)
        if not np.isclose(weights[row].sum(), 1.0, atol=1e-12):
            raise AssertionError(f"{year}: annual path weights do not sum to one")
    return weights, storms, points, effective_storms


def analysis_view(
    matched: pd.DataFrame,
    year_assignment: str,
    land_treatment: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    frame = matched.copy()
    if land_treatment == "ocean_points_only":
        frame = frame.loc[frame["point_is_land"].eq(False)].copy()
    elif land_treatment != "nearest_ocean_all_points":
        raise ValueError(f"unknown land treatment: {land_treatment}")

    if year_assignment == "calendar_year":
        frame["analysis_year"] = frame["ohc_year"]
    elif year_assignment == "season_year_aligned_only":
        frame = frame.loc[frame["ohc_year"].eq(frame["season"])].copy()
        frame["analysis_year"] = frame["season"]
    else:
        raise ValueError(f"unknown year assignment: {year_assignment}")
    frame["analysis_year"] = pd.to_numeric(frame["analysis_year"], errors="raise").astype(int)
    return frame.loc[frame["analysis_year"].between(start_year, end_year)].copy()


def coverage_gaps(
    frame: pd.DataFrame,
    years: list[int],
    expected_combinations: list[tuple[str, str]],
) -> list[dict]:
    expected = set(years)
    gaps: list[dict] = []
    for agency, definition in expected_combinations:
        group = frame.loc[(frame["agency"] == agency) & (frame["definition"] == definition)]
        present = set(pd.to_numeric(group["analysis_year"], errors="raise").astype(int))
        missing = sorted(expected.difference(present))
        if missing:
            gaps.append({"agency": agency, "definition": definition, "missing_years": missing})
    return gaps


def period_statistics(weights: np.ndarray, fields: np.ndarray) -> dict[str, np.ndarray | float]:
    if weights.shape != fields.shape:
        raise ValueError("weight and field matrices must have identical shape")
    annual_exposure = np.einsum("ij,ij->i", weights, fields)
    mean_weight = weights.mean(axis=0)
    mean_field = fields.mean(axis=0)
    climatological_exposure = float(mean_weight @ mean_field)
    observed_exposure = float(annual_exposure.mean())
    return {
        "mean_weight": mean_weight,
        "mean_field": mean_field,
        "annual_exposure": annual_exposure,
        "climatological_exposure": climatological_exposure,
        "observed_exposure": observed_exposure,
        "covariability": observed_exposure - climatological_exposure,
    }


def symmetric_decomposition(
    weights1: np.ndarray,
    fields1: np.ndarray,
    weights2: np.ndarray,
    fields2: np.ndarray,
    atol: float,
    rtol: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    p1 = period_statistics(weights1, fields1)
    p2 = period_statistics(weights2, fields2)
    w1 = np.asarray(p1["mean_weight"])
    w2 = np.asarray(p2["mean_weight"])
    h1 = np.asarray(p1["mean_field"])
    h2 = np.asarray(p2["mean_field"])
    redistribution_cell = (w2 - w1) * (h2 + h1) / 2.0
    ocean_cell = (w2 + w1) / 2.0 * (h2 - h1)
    redistribution = float(redistribution_cell.sum())
    ocean = float(ocean_cell.sum())
    covariability = float(p2["covariability"] - p1["covariability"])
    delta_climatological = float(p2["climatological_exposure"] - p1["climatological_exposure"])
    delta_observed = float(p2["observed_exposure"] - p1["observed_exposure"])
    climatological_residual = delta_climatological - redistribution - ocean
    observed_residual = delta_observed - redistribution - ocean - covariability
    scale = max(
        1.0,
        abs(delta_observed),
        abs(redistribution),
        abs(ocean),
        abs(covariability),
    )
    tolerance = float(atol) + float(rtol) * scale
    if abs(climatological_residual) > tolerance or abs(observed_residual) > tolerance:
        raise AssertionError(
            f"decomposition closure failed: climatological={climatological_residual}, "
            f"observed={observed_residual}, tolerance={tolerance}"
        )
    summary = {
        "exposure_p1": float(p1["observed_exposure"]),
        "exposure_p2": float(p2["observed_exposure"]),
        "delta_exposure": delta_observed,
        "climatological_exposure_p1": float(p1["climatological_exposure"]),
        "climatological_exposure_p2": float(p2["climatological_exposure"]),
        "delta_climatological_exposure": delta_climatological,
        "redistribution_component": redistribution,
        "ocean_component": ocean,
        "covariability_p1": float(p1["covariability"]),
        "covariability_p2": float(p2["covariability"]),
        "covariability_component": covariability,
        "climatological_closure_residual": climatological_residual,
        "observed_closure_residual": observed_residual,
        "closure_tolerance": tolerance,
    }
    arrays = {
        "weight_p1": w1,
        "weight_p2": w2,
        "field_p1": h1,
        "field_p2": h2,
        "redistribution_cell": redistribution_cell,
        "ocean_cell": ocean_cell,
        "annual_exposure_p1": np.asarray(p1["annual_exposure"]),
        "annual_exposure_p2": np.asarray(p2["annual_exposure"]),
    }
    return summary, arrays


def circular_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0 or block_length <= 0:
        raise ValueError("length and block_length must be positive")
    blocks = math.ceil(length / block_length)
    starts = rng.integers(0, length, size=blocks)
    sample = np.concatenate(
        [(start + np.arange(block_length, dtype=int)) % length for start in starts]
    )
    return sample[:length]


def build_annual_cross_products(
    weights1: np.ndarray,
    fields1: np.ndarray,
    weights2: np.ndarray,
    fields2: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "cross11": weights1 @ fields1.T,
        "cross12": weights1 @ fields2.T,
        "cross21": weights2 @ fields1.T,
        "cross22": weights2 @ fields2.T,
        "observed1_by_year": np.einsum("ij,ij->i", weights1, fields1),
        "observed2_by_year": np.einsum("ij,ij->i", weights2, fields2),
    }


def cross_product_components(
    products: dict[str, np.ndarray],
    count1: np.ndarray,
    count2: np.ndarray,
) -> dict[str, float]:
    """Compute one resample from year weights and annual cross-products."""
    w1h1 = float(count1 @ products["cross11"] @ count1)
    w1h2 = float(count1 @ products["cross12"] @ count2)
    w2h1 = float(count2 @ products["cross21"] @ count1)
    w2h2 = float(count2 @ products["cross22"] @ count2)
    observed1 = float(count1 @ products["observed1_by_year"])
    observed2 = float(count2 @ products["observed2_by_year"])
    redistribution = 0.5 * (w2h2 + w2h1 - w1h2 - w1h1)
    ocean = 0.5 * (w2h2 - w2h1 + w1h2 - w1h1)
    covariability = (observed2 - w2h2) - (observed1 - w1h1)
    return {
        "delta_exposure": observed2 - observed1,
        "redistribution_component": redistribution,
        "ocean_component": ocean,
        "covariability_component": covariability,
    }


def bootstrap_summary(
    weights1: np.ndarray,
    fields1: np.ndarray,
    weights2: np.ndarray,
    fields2: np.ndarray,
    block_length: int,
    replicates: int,
    seed: int,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    """Bootstrap exact components from annual cross-products.

    The expensive state dimension is reduced once to four year-by-year
    cross-product matrices.  Each replicate then uses only block-sample year
    weights; this is algebraically identical to rebuilding mean fields and
    weights for every draw.
    """
    rng = np.random.default_rng(seed)
    names = ["delta_exposure", "redistribution_component", "ocean_component", "covariability_component"]
    draws = {name: np.empty(replicates, dtype=float) for name in names}
    products = build_annual_cross_products(weights1, fields1, weights2, fields2)
    for replicate in range(replicates):
        index1 = circular_block_indices(len(weights1), block_length, rng)
        index2 = circular_block_indices(len(weights2), block_length, rng)
        count1 = np.bincount(index1, minlength=len(weights1)).astype(float) / len(index1)
        count2 = np.bincount(index2, minlength=len(weights2)).astype(float) / len(index2)
        values = cross_product_components(products, count1, count2)
        residual = values["delta_exposure"] - sum(
            values[name] for name in names if name != "delta_exposure"
        )
        scale = max(1.0, *(abs(values[name]) for name in names))
        if abs(residual) > float(atol) + float(rtol) * scale:
            raise AssertionError(f"bootstrap decomposition failed to close: {residual}")
        for name in names:
            draws[name][replicate] = values[name]
    rows = []
    for name in names:
        q025, q50, q975 = np.quantile(draws[name], [0.025, 0.5, 0.975])
        rows.append({"component": name, "q025": q025, "q50": q50, "q975": q975})
    return pd.DataFrame(rows)


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
    validate_seam_review_gate(config_path, config)
    if int(config["matching"]["month_lag"]) != 0:
        raise ValueError("this decomposition specification is fixed to contemporaneous month_lag=0")

    matched_path = project_path(config["matching"]["output_csv"])
    matched = pd.read_csv(matched_path, parse_dates=["iso_time"])
    matched = matched.loc[
        matched["match_status"].eq("matched") & matched["ohc300"].notna()
    ].copy()
    matched["season"] = pd.to_numeric(matched["season"], errors="raise").astype(int)
    for column in ["ohc_year", "ohc_month", "grid_y", "grid_x"]:
        matched[column] = pd.to_numeric(matched[column], errors="raise").astype(int)
    land_flag = matched["point_is_land"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if land_flag.isna().any():
        raise ValueError("matched points contain missing or invalid point_is_land values")
    matched["point_is_land"] = land_flag.astype(bool)
    keys = state_table(matched)
    matched = attach_key_ids(matched, keys)
    if matched["key_id"].isna().any():
        raise AssertionError("matched point did not receive a state key")
    matched["key_id"] = matched["key_id"].astype(int)

    all_years = list(range(int(config["study"]["start_year"]), int(config["study"]["end_year"]) + 1))
    year_row = {year: index for index, year in enumerate(all_years)}
    cfg = config["decomposition"]
    expected_combinations = sorted(
        map(tuple, matched[["agency", "definition"]].drop_duplicates().to_numpy().tolist())
    )
    analysis_specs: list[tuple[str, str, pd.DataFrame]] = []
    preflight_errors: list[dict] = []
    for year_assignment in cfg["year_assignment_modes"]:
        for land_treatment in cfg["land_treatments"]:
            view = analysis_view(
                matched,
                year_assignment,
                land_treatment,
                all_years[0],
                all_years[-1],
            )
            gaps = coverage_gaps(view, all_years, expected_combinations)
            if gaps:
                preflight_errors.append({
                    "year_assignment": year_assignment,
                    "land_treatment": land_treatment,
                    "coverage_gaps": gaps,
                })
            analysis_specs.append((year_assignment, land_treatment, view))
    if preflight_errors:
        raise ValueError(f"annual coverage preflight failed: {preflight_errors}")

    field_matrix = load_field_matrix(
        project_path(config["ohc"]["prepared_dir"]),
        config["ohc"]["canonical_variable"],
        all_years,
        keys,
        float(config["ohc"]["coordinate_equality_atol_degrees"]),
    )
    recorded = matched["ohc300"].to_numpy(float)
    extracted = field_matrix[
        matched["ohc_year"].map(year_row).to_numpy(int), matched["key_id"].to_numpy(int)
    ]
    if not np.allclose(recorded, extracted, rtol=1e-7, atol=1e-3, equal_nan=False):
        raise AssertionError("matched-point OHC values do not reproduce the prepared field")

    summary_rows: list[dict] = []
    component_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    annual_rows: list[dict] = []
    for year_assignment, land_treatment, view in analysis_specs:
        for agency, definition in expected_combinations:
            subset = view.loc[(view["agency"] == agency) & (view["definition"] == definition)]
            for weighting in cfg["weightings"]:
                weights_all, storm_counts, point_counts, effective_storm_counts = annual_weight_matrix(
                    subset, all_years, len(keys), weighting
                )
                for comparison in comparison_list(config):
                    start1, end1 = map(int, comparison["period1"])
                    start2, end2 = map(int, comparison["period2"])
                    years1 = list(range(start1, end1 + 1))
                    years2 = list(range(start2, end2 + 1))
                    rows1 = np.array([year_row[year] for year in years1], dtype=int)
                    rows2 = np.array([year_row[year] for year in years2], dtype=int)
                    summary, arrays = symmetric_decomposition(
                        weights_all[rows1], field_matrix[rows1],
                        weights_all[rows2], field_matrix[rows2],
                        float(cfg["closure_atol"]), float(cfg["closure_rtol"]),
                    )
                    identity = {
                        "comparison": comparison["label"],
                        "period1": f"{start1}-{end1}",
                        "period2": f"{start2}-{end2}",
                        "agency": agency,
                        "definition": definition,
                        "year_assignment": year_assignment,
                        "land_treatment": land_treatment,
                        "weighting": weighting,
                        "state_key": cfg["state_key"],
                        "n_years_p1": len(years1),
                        "n_years_p2": len(years2),
                    }
                    summary_rows.append({**identity, **summary})

                    active = (
                        (arrays["weight_p1"] != 0.0)
                        | (arrays["weight_p2"] != 0.0)
                        | (arrays["redistribution_cell"] != 0.0)
                        | (arrays["ocean_cell"] != 0.0)
                    )
                    components = keys.loc[active].copy()
                    for name in ["weight_p1", "weight_p2", "field_p1", "field_p2", "redistribution_cell", "ocean_cell"]:
                        components[name] = arrays[name][active]
                    for name, value in identity.items():
                        components[name] = value
                    component_frames.append(components)

                    boot = bootstrap_summary(
                        weights_all[rows1], field_matrix[rows1],
                        weights_all[rows2], field_matrix[rows2],
                        int(cfg["block_length_years"]), int(cfg["bootstrap_replicates"]),
                        int(cfg["random_seed"]), float(cfg["closure_atol"]), float(cfg["closure_rtol"]),
                    )
                    for name, value in identity.items():
                        boot[name] = value
                    bootstrap_frames.append(boot)

                annual_exposure = np.einsum("ij,ij->i", weights_all, field_matrix)
                for index, year in enumerate(all_years):
                    annual_rows.append({
                        "year": year,
                        "agency": agency,
                        "definition": definition,
                        "year_assignment": year_assignment,
                        "land_treatment": land_treatment,
                        "weighting": weighting,
                        "ohc300_exposure": annual_exposure[index],
                        "storms": storm_counts[index],
                        "effective_storms": effective_storm_counts[index],
                        "matched_points": point_counts[index],
                    })

    outputs = [
        (project_path(cfg["output_summary_csv"]), pd.DataFrame(summary_rows), None),
        (project_path(cfg["output_components_csv"]), pd.concat(component_frames, ignore_index=True), "gzip"),
        (project_path(cfg["output_bootstrap_csv"]), pd.concat(bootstrap_frames, ignore_index=True), None),
        (project_path("Verify/ohc_path_exposure/results/annual_exposure.csv"), pd.DataFrame(annual_rows), None),
    ]
    for path, frame, compression in outputs:
        ensure_output_target(path, overwrite=args.overwrite)
        temp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temp, index=False, compression=compression)
        atomic_replace(temp, path)


if __name__ == "__main__":
    main()
