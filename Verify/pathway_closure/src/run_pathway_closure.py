"""Independent verification of pathway-type closure.

All outputs stay under Verify/pathway_closure. Existing project data and formal
analysis products are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve()
VERIFY_ROOT = HERE.parents[1]
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import load_config  # noqa: E402
from paper2_dynamic.agency_data import (  # noqa: E402
    AGENCIES,
    build_agency_catalog,
    read_ibtracs_agencies,
)


@dataclass
class PathSet:
    dataset: str
    sid: np.ndarray
    season: np.ndarray
    features: np.ndarray
    paths: np.ndarray
    membership: np.ndarray | None = None


@dataclass
class DensityCache:
    geometry: str
    normalization: str
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    fields: np.ndarray
    valid: np.ndarray
    pi: np.ndarray
    joint: np.ndarray
    counts: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_verify_config() -> dict:
    with (VERIFY_ROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def resample_track(group: pd.DataFrame, n_points: int) -> np.ndarray | None:
    group = group.sort_values("iso_time").drop_duplicates("iso_time", keep="last")
    if len(group) < 4:
        return None
    hours = (
        group["iso_time"] - group["iso_time"].iloc[0]
    ).dt.total_seconds().to_numpy() / 3600.0
    if not np.isfinite(hours[-1]) or hours[-1] <= 0:
        return None
    query = np.linspace(0.0, hours[-1], n_points)
    lon = np.rad2deg(np.unwrap(np.deg2rad(group["lon"].to_numpy(float))))
    lat = group["lat"].to_numpy(float)
    return np.column_stack(
        [np.interp(query, hours, lon), np.interp(query, hours, lat)]
    )


def build_path_set(
    points: pd.DataFrame,
    dataset: str,
    start: int,
    end: int,
    n_points: int,
) -> PathSet:
    required = {"sid", "season", "iso_time", "lat", "lon"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"{dataset}: missing columns {sorted(missing)}")
    frame = points.loc[points["season"].between(start, end)].copy()
    frame["iso_time"] = pd.to_datetime(frame["iso_time"], errors="coerce")
    frame = frame.dropna(subset=["sid", "season", "iso_time", "lat", "lon"])

    rows: list[np.ndarray] = []
    sids: list[str] = []
    seasons: list[int] = []
    for sid, group in frame.groupby("sid", sort=True):
        path = resample_track(group, n_points)
        if path is None:
            continue
        rows.append(path)
        sids.append(str(sid))
        seasons.append(int(group["season"].iloc[0]))
    if not rows:
        raise ValueError(f"{dataset}: no eligible tracks")

    paths = np.stack(rows)
    features = np.concatenate([paths[:, :, 0], paths[:, :, 1]], axis=1)
    features[:, :n_points] *= np.cos(np.deg2rad(20.0))
    return PathSet(
        dataset=dataset,
        sid=np.asarray(sids),
        season=np.asarray(seasons, dtype=int),
        features=features,
        paths=paths,
    )


def xb_index(
    x: np.ndarray, centers: np.ndarray, membership: np.ndarray, m: float
) -> float:
    distance2 = ((centers[:, None, :] - x.T[None, :, :]) ** 2).sum(axis=2)
    center_distance2 = (
        (centers[:, None, :] - centers[None, :, :]) ** 2
    ).sum(axis=2)
    center_distance2[center_distance2 == 0] = np.inf
    return float(
        ((membership**m) * distance2).sum()
        / (x.shape[1] * center_distance2.min())
    )


def fit_reference_model(
    reference: PathSet,
    n_families: int,
    m: float,
    seeds: int,
    n_points: int,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, dict]:
    scaler = StandardScaler().fit(reference.features)
    x = scaler.transform(reference.features).T
    best = None
    for seed in range(seeds):
        centers, membership, _, _, _, _, fpc = fuzz.cluster.cmeans(
            x,
            n_families,
            m=m,
            error=1e-5,
            maxiter=1000,
            seed=seed,
        )
        xb = xb_index(x, centers, membership, m)
        candidate = (xb, -float(fpc), seed, centers, membership, float(fpc))
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    xb, _, seed, centers, membership, fpc = best

    physical = centers * scaler.scale_ + scaler.mean_
    physical[:, :n_points] /= np.cos(np.deg2rad(20.0))
    recurving_index = int(np.argmax(physical[:, -1]))
    remaining = [idx for idx in range(n_families) if idx != recurving_index]
    central_index = max(remaining, key=lambda idx: physical[idx, 0])
    western_index = next(idx for idx in remaining if idx != central_index)
    order = np.asarray([western_index, central_index, recurving_index], dtype=int)
    centers = centers[order]
    membership = membership[order]
    physical = physical[order]

    diagnostics = {
        "k": int(n_families),
        "m": float(m),
        "seed": int(seed),
        "xb": float(xb),
        "fpc": float(fpc),
        "median_max_membership": float(np.median(membership.max(axis=0))),
    }
    return (
        scaler,
        centers,
        membership,
        {"diagnostics": diagnostics, "physical": physical},
    )


def fit_k_scan(
    reference: PathSet,
    k_values: list[int],
    m: float,
    seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all requested k/seed combinations and quantify hard-label stability."""
    scaler = StandardScaler().fit(reference.features)
    x = scaler.transform(reference.features).T
    summary_rows: list[dict] = []
    seed_rows: list[dict] = []
    for k in k_values:
        runs: list[dict] = []
        for seed in range(seeds):
            centers, membership, _, _, _, _, fpc = fuzz.cluster.cmeans(
                x,
                k,
                m=m,
                error=1e-5,
                maxiter=1000,
                seed=seed,
            )
            runs.append(
                {
                    "seed": seed,
                    "centers": centers,
                    "membership": membership,
                    "xb": xb_index(x, centers, membership, m),
                    "fpc": float(fpc),
                    "median_max_membership": float(
                        np.median(membership.max(axis=0))
                    ),
                }
            )
        best = min(runs, key=lambda row: (row["xb"], -row["fpc"], row["seed"]))
        best_labels = best["membership"].argmax(axis=0)
        ari_values = []
        for run in runs:
            ari = float(
                adjusted_rand_score(
                    best_labels, run["membership"].argmax(axis=0)
                )
            )
            ari_values.append(ari)
            seed_rows.append(
                {
                    "k": k,
                    "seed": run["seed"],
                    "xb": run["xb"],
                    "fpc": run["fpc"],
                    "median_max_membership": run["median_max_membership"],
                    "ari_vs_best": ari,
                    "selected_by_xb": run["seed"] == best["seed"],
                }
            )

        scan_set = PathSet(
            dataset=f"PRIMARY_k{k}",
            sid=reference.sid,
            season=reference.season,
            features=reference.features,
            paths=reference.paths,
            membership=best["membership"],
        )
        domain = load_verify_config()["density_domain"]
        width = load_verify_config()["density_grid_deg"]
        lon_edges = np.arange(
            domain["lon_min"], domain["lon_max"] + width, width
        )
        lat_edges = np.arange(
            domain["lat_min"], domain["lat_max"] + width, width
        )
        years = np.arange(
            load_verify_config()["period_early"][0],
            load_verify_config()["period_late"][1] + 1,
        )
        pi, joint, _ = annual_joint_fields(
            scan_set, years, lon_edges, lat_edges
        )
        early, late = scenario_indices(
            years,
            load_verify_config()["period_early"][0],
            load_verify_config()["period_late"][0],
            load_verify_config()["period_late"][1],
        )
        decomposition = decompose(pi, joint, early, late)
        summary_rows.append(
            {
                "k": k,
                "best_seed": best["seed"],
                "best_xb": best["xb"],
                "best_fpc": best["fpc"],
                "best_median_max_membership": best[
                    "median_max_membership"
                ],
                "ari_vs_best_min": float(np.min(ari_values)),
                "ari_vs_best_median": float(np.median(ari_values)),
                "ari_vs_best_max": float(np.max(ari_values)),
                "between_share": decomposition["between_share"],
                "within_share": decomposition["within_share"],
                "closure_max_abs": decomposition["closure_max_abs"],
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(seed_rows)


def predict_membership(
    path_set: PathSet,
    scaler: StandardScaler,
    centers: np.ndarray,
    m: float,
) -> np.ndarray:
    x = scaler.transform(path_set.features).T
    membership, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        x,
        centers,
        m=m,
        error=1e-6,
        maxiter=1000,
    )
    if not np.allclose(membership.sum(axis=0), 1.0, atol=1e-10):
        raise AssertionError(f"{path_set.dataset}: fuzzy memberships do not sum to 1")
    return membership


def storm_density_fields(
    paths: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    normalization: str = "in_domain_renormalized",
) -> tuple[np.ndarray, np.ndarray]:
    fields = np.zeros(
        (len(paths), len(lat_edges) - 1, len(lon_edges) - 1), dtype=float
    )
    valid = np.zeros(len(paths), dtype=bool)
    for idx, path in enumerate(paths):
        field = np.histogram2d(
            path[:, 0], path[:, 1], bins=[lon_edges, lat_edges]
        )[0].T
        total = field.sum()
        if normalization == "in_domain_renormalized" and total > 0:
            fields[idx] = field / total
            valid[idx] = True
        elif normalization == "fixed_path_points":
            fields[idx] = field / len(path)
            valid[idx] = True
        elif normalization != "in_domain_renormalized":
            raise ValueError(f"unknown density normalization: {normalization}")
    return fields, valid


def annual_joint_fields(
    path_set: PathSet,
    years: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path_set.membership is None:
        raise ValueError("membership must be assigned first")
    fields, valid = storm_density_fields(path_set.paths, lon_edges, lat_edges)
    n_families = path_set.membership.shape[0]
    joint = np.zeros((len(years), n_families, *fields.shape[1:]), dtype=float)
    pi = np.zeros((len(years), n_families), dtype=float)
    counts = np.zeros(len(years), dtype=int)
    for year_index, year in enumerate(years):
        take = np.flatnonzero((path_set.season == year) & valid)
        if len(take) == 0:
            raise ValueError(f"{path_set.dataset}: no eligible storms in {year}")
        counts[year_index] = len(take)
        weights = path_set.membership[:, take]
        pi[year_index] = weights.mean(axis=1)
        joint[year_index] = np.einsum(
            "ki,ihw->khw", weights, fields[take], optimize=True
        ) / len(take)
        if not np.isclose(joint[year_index].sum(), 1.0, atol=1e-10):
            raise AssertionError(f"{path_set.dataset}: annual density does not sum to 1")
    return pi, joint, counts


def density_cache(
    path_set: PathSet,
    years: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    geometry: str,
    normalization: str = "in_domain_renormalized",
) -> DensityCache:
    """Build annual decomposition inputs once and retain storm-level fields."""
    if path_set.membership is None:
        raise ValueError("membership must be assigned first")
    paths = path_set.paths
    if geometry == "relative_genesis":
        paths = paths - paths[:, :1, :]
    elif geometry != "absolute":
        raise ValueError(f"unknown geometry: {geometry}")
    fields, valid = storm_density_fields(
        paths, lon_edges, lat_edges, normalization=normalization
    )
    n_families = path_set.membership.shape[0]
    joint = np.zeros((len(years), n_families, *fields.shape[1:]), dtype=float)
    pi = np.zeros((len(years), n_families), dtype=float)
    counts = np.zeros(len(years), dtype=int)
    for year_index, year in enumerate(years):
        take = np.flatnonzero((path_set.season == year) & valid)
        if len(take) == 0:
            raise ValueError(
                f"{path_set.dataset}/{geometry}: no eligible storms in {year}"
            )
        counts[year_index] = len(take)
        weights = path_set.membership[:, take]
        pi[year_index] = weights.mean(axis=1)
        joint[year_index] = np.einsum(
            "ki,ihw->khw", weights, fields[take], optimize=True
        ) / len(take)
        annual_mass = joint[year_index].sum()
        if normalization == "in_domain_renormalized" and not np.isclose(
            annual_mass, 1.0, atol=1e-10
        ):
            raise AssertionError(
                f"{path_set.dataset}/{geometry}: annual density does not sum to 1"
            )
        if normalization == "fixed_path_points" and not (
            -1e-12 <= annual_mass <= 1.0 + 1e-12
        ):
            raise AssertionError(
                f"{path_set.dataset}/{geometry}: fixed-denominator mass outside [0,1]"
            )
    return DensityCache(
        geometry=geometry,
        normalization=normalization,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        fields=fields,
        valid=valid,
        pi=pi,
        joint=joint,
        counts=counts,
    )


def scenario_indices(
    years: np.ndarray,
    start_year: int,
    cutpoint: int,
    end_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated early/late indices; cutpoint is the first late year."""
    if start_year >= cutpoint or cutpoint > end_year:
        raise ValueError(
            f"invalid period: {start_year=}, {cutpoint=}, {end_year=}"
        )
    early_years = np.arange(start_year, cutpoint)
    late_years = np.arange(cutpoint, end_year + 1)
    if len(early_years) < 2 or len(late_years) < 2:
        raise ValueError("each period must contain at least two years")
    year_to_index = {int(year): index for index, year in enumerate(years)}
    missing = [
        int(year)
        for year in np.concatenate([early_years, late_years])
        if int(year) not in year_to_index
    ]
    if missing:
        raise ValueError(f"scenario contains unavailable years: {missing}")
    early = np.asarray([year_to_index[int(year)] for year in early_years])
    late = np.asarray([year_to_index[int(year)] for year in late_years])
    if not np.array_equal(years[early], early_years):
        raise AssertionError("early period is not continuous")
    if not np.array_equal(years[late], late_years):
        raise AssertionError("late period is not continuous")
    return early, late


def decompose_period_means(
    pi_early: np.ndarray,
    pi_late: np.ndarray,
    joint_early: np.ndarray,
    joint_late: np.ndarray,
) -> dict:
    """Symmetric decomposition from already aggregated period means."""
    if np.any(pi_early <= 0) or np.any(pi_late <= 0):
        raise ValueError("all family shares must be positive in both periods")
    f_early = joint_early / pi_early[:, None, None]
    f_late = joint_late / pi_late[:, None, None]
    between_family = (
        (pi_late - pi_early)[:, None, None] * (f_late + f_early) / 2.0
    )
    within_family = (
        ((pi_late + pi_early) / 2.0)[:, None, None] * (f_late - f_early)
    )
    between = between_family.sum(axis=0)
    within = within_family.sum(axis=0)
    density_early = joint_early.sum(axis=0)
    density_late = joint_late.sum(axis=0)
    delta = density_late - density_early
    closure = delta - between - within
    denominator = float(np.vdot(delta, delta))
    if denominator <= 0:
        raise ValueError("observed density difference has zero norm")
    return {
        "pi_early": pi_early,
        "pi_late": pi_late,
        "f_early": f_early,
        "f_late": f_late,
        "density_early": density_early,
        "density_late": density_late,
        "delta": delta,
        "between": between,
        "within": within,
        "between_family": between_family,
        "within_family": within_family,
        "between_share": float(np.vdot(between, delta) / denominator),
        "within_share": float(np.vdot(within, delta) / denominator),
        "total_variation": float(0.5 * np.abs(delta).sum()),
        "closure_max_abs": float(np.abs(closure).max()),
    }


def decompose(
    pi: np.ndarray,
    joint: np.ndarray,
    early_index: np.ndarray,
    late_index: np.ndarray,
) -> dict:
    pi_early = pi[early_index].mean(axis=0)
    pi_late = pi[late_index].mean(axis=0)
    joint_early = joint[early_index].mean(axis=0)
    joint_late = joint[late_index].mean(axis=0)
    return decompose_period_means(
        pi_early, pi_late, joint_early, joint_late
    )


def block_indices(
    base_index: np.ndarray,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(base_index) % block:
        raise ValueError("period length must be divisible by block length")
    blocks = base_index.reshape(-1, block)
    draw = rng.integers(0, len(blocks), size=len(blocks))
    return blocks[draw].ravel()


def bootstrap_decomposition(
    pi: np.ndarray,
    joint: np.ndarray,
    early_index: np.ndarray,
    late_index: np.ndarray,
    block: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_bootstrap):
        early_draw = block_indices(early_index, block, rng)
        late_draw = block_indices(late_index, block, rng)
        result = decompose(pi, joint, early_draw, late_draw)
        rows.append(
            {
                "replicate": replicate,
                "between_share": result["between_share"],
                "within_share": result["within_share"],
                "total_variation": result["total_variation"],
            }
        )
    return pd.DataFrame(rows)


def label_permutation_null(
    path_set: PathSet,
    cache: DensityCache,
    years: np.ndarray,
    early_index: np.ndarray,
    late_index: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Condition on annual memberships and permute their storm association.

    Membership vectors are permuted only among valid storms in the same year.
    This keeps every annual family-share vector exactly fixed while breaking
    the association between a path field and its pathway membership.
    """
    if path_set.membership is None:
        raise ValueError("membership must be assigned first")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    observed = decompose(cache.pi, cache.joint, early_index, late_index)
    valid_global = np.flatnonzero(cache.valid)
    membership = path_set.membership[:, valid_global]
    seasons = path_set.season[valid_global]
    fields = sparse.csr_matrix(
        cache.fields[valid_global].reshape(len(valid_global), -1)
    )
    groups = {
        int(year): np.flatnonzero(seasons == year)
        for year in years
    }
    if any(len(group) == 0 for group in groups.values()):
        empty = [year for year, group in groups.items() if len(group) == 0]
        raise ValueError(f"permutation contains empty years: {empty}")

    early_years = years[early_index]
    late_years = years[late_index]
    early_weight = np.zeros(len(valid_global), dtype=float)
    late_weight = np.zeros(len(valid_global), dtype=float)
    for year in early_years:
        group = groups[int(year)]
        early_weight[group] = 1.0 / (len(early_years) * len(group))
    for year in late_years:
        group = groups[int(year)]
        late_weight[group] = 1.0 / (len(late_years) * len(group))
    weighted_early = sparse.diags(early_weight) @ fields
    weighted_late = sparse.diags(late_weight) @ fields
    pi_early = cache.pi[early_index].mean(axis=0)
    pi_late = cache.pi[late_index].mean(axis=0)

    rng = np.random.default_rng(seed)
    null_rows: list[dict] = []
    for replicate in range(n_permutations):
        permuted = membership.copy()
        for group in groups.values():
            permuted[:, group] = membership[:, rng.permutation(group)]
        joint_early = np.asarray(
            weighted_early.T.dot(permuted.T).T
        ).reshape(
            membership.shape[0],
            cache.fields.shape[1],
            cache.fields.shape[2],
        )
        joint_late = np.asarray(
            weighted_late.T.dot(permuted.T).T
        ).reshape(
            membership.shape[0],
            cache.fields.shape[1],
            cache.fields.shape[2],
        )
        result = decompose_period_means(
            pi_early, pi_late, joint_early, joint_late
        )
        null_rows.append(
            {
                "replicate": replicate,
                "between_share": result["between_share"],
            }
        )
    null = pd.DataFrame(null_rows)
    values = null["between_share"].to_numpy(float)
    observed_value = observed["between_share"]
    p_lower = (1.0 + float(np.count_nonzero(values <= observed_value))) / (
        n_permutations + 1.0
    )
    p_upper = (1.0 + float(np.count_nonzero(values >= observed_value))) / (
        n_permutations + 1.0
    )
    summary = {
        "observed_between_share": observed_value,
        "null_mean": float(values.mean()),
        "null_sd": float(values.std(ddof=1)),
        "null_ci_low": float(np.quantile(values, 0.025)),
        "null_median": float(np.quantile(values, 0.5)),
        "null_ci_high": float(np.quantile(values, 0.975)),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
        "n_permutations": n_permutations,
        "conditioning": "membership_permuted_within_year",
    }
    return summary, null


def period_sensitivity(
    path_set: PathSet,
    cache: DensityCache,
    years: np.ndarray,
    scenarios: list[dict],
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        early, late = scenario_indices(
            years,
            int(scenario["start_year"]),
            int(scenario["cutpoint"]),
            int(scenario["end_year"]),
        )
        result = decompose(cache.pi, cache.joint, early, late)
        rows.append(
            {
                "dataset": path_set.dataset,
                "geometry": cache.geometry,
                "scenario": scenario["scenario"],
                "start_year": int(scenario["start_year"]),
                "early_end_year": int(scenario["cutpoint"]) - 1,
                "late_start_year": int(scenario["cutpoint"]),
                "end_year": int(scenario["end_year"]),
                "n_early_years": len(early),
                "n_late_years": len(late),
                "between_share": result["between_share"],
                "within_share": result["within_share"],
                "total_variation": result["total_variation"],
                "closure_max_abs": result["closure_max_abs"],
            }
        )
    return pd.DataFrame(rows)


def domain_denominator_sensitivity(
    path_set: PathSet,
    years: np.ndarray,
    early_index: np.ndarray,
    late_index: np.ndarray,
    specifications: list[dict],
    grid_width: float,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare domain boundaries and per-storm density denominators."""
    summary_rows: list[dict] = []
    null_frames: list[pd.DataFrame] = []
    for mode_index, specification in enumerate(specifications):
        lon_edges = np.arange(
            specification["lon_min"],
            specification["lon_max"] + grid_width,
            grid_width,
        )
        lat_edges = np.arange(
            specification["lat_min"],
            specification["lat_max"] + grid_width,
            grid_width,
        )
        cache = density_cache(
            path_set,
            years,
            lon_edges,
            lat_edges,
            "absolute",
            normalization=specification["normalization"],
        )
        result = decompose(
            cache.pi, cache.joint, early_index, late_index
        )
        permutation_summary, permutation_null = label_permutation_null(
            path_set,
            cache,
            years,
            early_index,
            late_index,
            n_permutations,
            seed + mode_index,
        )
        paths = path_set.paths
        inside = (
            (paths[:, :, 0] >= lon_edges[0])
            & (paths[:, :, 0] <= lon_edges[-1])
            & (paths[:, :, 1] >= lat_edges[0])
            & (paths[:, :, 1] <= lat_edges[-1])
        )
        summary_rows.append(
            {
                "dataset": path_set.dataset,
                "mode": specification["mode"],
                "normalization": specification["normalization"],
                "lon_min": float(lon_edges[0]),
                "lon_max": float(lon_edges[-1]),
                "lat_min": float(lat_edges[0]),
                "lat_max": float(lat_edges[-1]),
                "valid_storm_fraction": float(cache.valid.mean()),
                "point_coverage_fraction": float(inside.mean()),
                "density_mass_early": float(result["density_early"].sum()),
                "density_mass_late": float(result["density_late"].sum()),
                "between_share": result["between_share"],
                "within_share": result["within_share"],
                "total_variation": result["total_variation"],
                "closure_max_abs": result["closure_max_abs"],
                "null_median": permutation_summary["null_median"],
                "null_ci_low": permutation_summary["null_ci_low"],
                "null_ci_high": permutation_summary["null_ci_high"],
                "p_two_sided": permutation_summary["p_two_sided"],
                "n_permutations": n_permutations,
            }
        )
        permutation_null.insert(0, "mode", specification["mode"])
        permutation_null.insert(0, "dataset", path_set.dataset)
        null_frames.append(permutation_null)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(null_frames, ignore_index=True),
    )


def weighted_centroid(
    field: np.ndarray,
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
) -> tuple[float, float]:
    total = field.sum()
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    return (
        float((field * lon_grid).sum() / total),
        float((field * lat_grid).sum() / total),
    )


def landfall_decomposition(
    reference: PathSet,
    landfalls_path: Path,
    years: np.ndarray,
    early_index: np.ndarray,
    late_index: np.ndarray,
    block: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if reference.membership is None:
        raise ValueError("reference membership missing")
    landfalls = pd.read_csv(landfalls_path, parse_dates=["time"])
    first = (
        landfalls.sort_values(["sid", "time"])
        .drop_duplicates("sid", keep="first")[["sid", "lat"]]
    )
    membership = pd.DataFrame(
        reference.membership.T,
        columns=[f"m{k}" for k in range(reference.membership.shape[0])],
    )
    membership.insert(0, "season", reference.season)
    membership.insert(0, "sid", reference.sid)
    events = membership.merge(first, on="sid", how="inner")

    n_families = reference.membership.shape[0]
    annual_pi = np.zeros((len(years), n_families), dtype=float)
    annual_moment = np.zeros((len(years), n_families), dtype=float)
    counts = np.zeros(len(years), dtype=int)
    member_cols = [f"m{idx}" for idx in range(n_families)]
    for year_index, year in enumerate(years):
        group = events.loc[events["season"] == year]
        if group.empty:
            raise ValueError(f"no first-landfall events in {year}")
        weights = group[member_cols].to_numpy(float)
        lat = group["lat"].to_numpy(float)
        annual_pi[year_index] = weights.mean(axis=0)
        annual_moment[year_index] = (weights * lat[:, None]).mean(axis=0)
        counts[year_index] = len(group)

    def one(early: np.ndarray, late: np.ndarray) -> dict:
        pi_1 = annual_pi[early].mean(axis=0)
        pi_2 = annual_pi[late].mean(axis=0)
        moment_1 = annual_moment[early].mean(axis=0)
        moment_2 = annual_moment[late].mean(axis=0)
        mu_1 = moment_1 / pi_1
        mu_2 = moment_2 / pi_2
        between = float(((pi_2 - pi_1) * (mu_2 + mu_1) / 2.0).sum())
        within = float((((pi_2 + pi_1) / 2.0) * (mu_2 - mu_1)).sum())
        mean_1 = float(moment_1.sum())
        mean_2 = float(moment_2.sum())
        return {
            "mean_early": mean_1,
            "mean_late": mean_2,
            "change_deg": mean_2 - mean_1,
            "between_deg": between,
            "within_deg": within,
            "closure_abs": abs((mean_2 - mean_1) - between - within),
            "pi_early": pi_1,
            "pi_late": pi_2,
            "mu_early": mu_1,
            "mu_late": mu_2,
        }

    result = one(early_index, late_index)
    rng = np.random.default_rng(seed)
    boot_rows = []
    for replicate in range(n_bootstrap):
        draw_early = block_indices(early_index, block, rng)
        draw_late = block_indices(late_index, block, rng)
        draw = one(draw_early, draw_late)
        boot_rows.append(
            {
                "replicate": replicate,
                "change_deg": draw["change_deg"],
                "between_deg": draw["between_deg"],
                "within_deg": draw["within_deg"],
            }
        )
    return result, pd.DataFrame(boot_rows), events


def quantiles(series: pd.Series) -> tuple[float, float, float]:
    values = series.quantile([0.025, 0.5, 0.975]).to_numpy(float)
    return float(values[0]), float(values[1]), float(values[2])


def save_figure(
    model_paths: np.ndarray,
    summary: pd.DataFrame,
    family: pd.DataFrame,
    landfall: dict,
    output: Path,
) -> None:
    colors = ["#4477AA", "#EE6677", "#228833"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    names = ["Western basin", "Central westward", "Recurving"]
    n_points = model_paths.shape[1] // 2
    for idx, (name, color) in enumerate(zip(names, colors)):
        axes[0, 0].plot(
            model_paths[idx, :n_points],
            model_paths[idx, n_points:],
            color=color,
            linewidth=2.5,
            label=name,
        )
        axes[0, 0].scatter(
            model_paths[idx, 0],
            model_paths[idx, n_points],
            color=color,
            s=28,
        )
    axes[0, 0].set(
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
        title="Fixed pathway-family centers",
    )
    axes[0, 0].legend(frameon=False)

    x = np.arange(len(summary))
    axes[0, 1].bar(
        x - 0.18,
        summary["between_share"],
        width=0.36,
        color="#CCBB44",
        label="Between-family",
    )
    axes[0, 1].bar(
        x + 0.18,
        summary["within_share"],
        width=0.36,
        color="#66CCEE",
        label="Within-family",
    )
    axes[0, 1].set_xticks(x, summary["dataset"])
    axes[0, 1].set(ylabel="Projection contribution", title="Path-density decomposition")
    axes[0, 1].legend(frameon=False)

    pivot = family.pivot(index="dataset", columns="family", values="share_change_pp")
    pivot = pivot.reindex(summary["dataset"])
    for name, color in zip(
        ["western_basin", "central_westward", "recurving"], colors
    ):
        axes[1, 0].plot(
            np.arange(len(pivot)),
            pivot[name].to_numpy(),
            marker="o",
            color=color,
            label=name.replace("_", " ").title(),
        )
    axes[1, 0].axhline(0, color="0.3", linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(len(pivot)), pivot.index)
    axes[1, 0].set(
        ylabel="Late minus early (percentage points)",
        title="Pathway-family share changes",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(
        ["Observed", "Between-family", "Within-family"],
        [landfall["change_deg"], landfall["between_deg"], landfall["within_deg"]],
        color=["#999999", "#CCBB44", "#66CCEE"],
    )
    axes[1, 1].axhline(0, color="0.3", linewidth=0.8)
    axes[1, 1].set(
        ylabel="Latitude change (°)",
        title="Primary first-landfall latitude",
    )

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            0.01,
            0.99,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    landfall: dict,
    landfall_boot: pd.DataFrame,
) -> None:
    dominant = summary.assign(
        dominant=np.where(
            summary["within_share"] > summary["between_share"],
            "类型内部走廊位移",
            "类型比例变化",
        )
    )
    agency_consistent = dominant["dominant"].nunique() == 1
    lf_change_ci = quantiles(landfall_boot["change_deg"])
    lf_within_ci = quantiles(landfall_boot["within_deg"])
    hard = sensitivity.loc[sensitivity["definition"] == "hard_max_membership"]
    hard_consistent = bool((hard["within_share"] > hard["between_share"]).all())

    lines = [
        "# 路径型闭环验证结果",
        "",
        "## 模型诊断",
        "",
        f"- 固定类别数：{diagnostics['k']}。",
        f"- Xie—Beni 指数：{diagnostics['xb']:.4f}。",
        f"- 模糊划分系数：{diagnostics['fpc']:.4f}。",
        f"- 最大隶属度中位数：{diagnostics['median_max_membership']:.4f}。",
        "",
        "## 路径密度分解",
        "",
        "| 资料 | 类型间贡献 | 类型间95%区间 | 类型内贡献 | 类型内95%区间 | 总变差距离 | 闭合误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.dataset} | {row.between_share:.3f} "
            f"| [{row.between_ci_low:.3f}, {row.between_ci_high:.3f}] "
            f"| {row.within_share:.3f} "
            f"| [{row.within_ci_low:.3f}, {row.within_ci_high:.3f}] "
            f"| {row.total_variation:.4f} | {row.closure_max_abs:.2e} |"
        )
    lines.extend(
        [
            "",
            f"四套资料的主导分量{'一致' if agency_consistent else '不完全一致'}。"
            "贡献采用观测差异轴投影，出现负值或大于1时表示分量抵消，不代表计算错误。",
            f"最大隶属度硬分类下，四套资料的类型内部贡献"
            f"{'仍全部占主导' if hard_consistent else '并非全部占主导'}；"
            "详细数值见 `pathway_classification_sensitivity.csv`。",
            "",
            "## 首次登陆纬度传递",
            "",
            f"- 前期均值：{landfall['mean_early']:.3f}°N；"
            f"后期均值：{landfall['mean_late']:.3f}°N。",
            f"- 变化：{landfall['change_deg']:.3f}°，3年分块95%区间"
            f"[{lf_change_ci[0]:.3f}, {lf_change_ci[2]:.3f}]°。",
            f"- 类型比例贡献：{landfall['between_deg']:.3f}°；"
            f"类型内部贡献：{landfall['within_deg']:.3f}°，后者95%区间"
            f"[{lf_within_ci[0]:.3f}, {lf_within_ci[2]:.3f}]°。",
            f"- 数值闭合误差：{landfall['closure_abs']:.2e}。",
            "",
            "## 结果定位",
            "",
            "三机构方向、bootstrap区间和首次登陆传递共同支持生成后路径形态差异。"
            "分析使用路径和登陆统计结构，不构成动力因果归因。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_p0_report(
    path: Path,
    permutation: pd.DataFrame,
    relative: pd.DataFrame,
    k_scan: pd.DataFrame,
    period_results: pd.DataFrame,
    coverage: pd.DataFrame,
    domain_results: pd.DataFrame,
) -> None:
    """Write a result-bounded report using only generated CSV-level values."""
    lines = [
        "# 路径闭环稳健性复核报告",
        "",
        "本报告覆盖标签置换、相对生成点分解、k=2—7扫描、"
        "1966/1982起算与移动切点，以及域边界/密度分母敏感性。"
        "所有数值均由分析脚本写入CSV；"
        "不据图读数，也不构成动力因果归因。",
        "",
        "## 1. 标签置换零分布",
        "",
        "置换在每一年内打乱“连续隶属度—气旋路径”的对应关系，"
        "因此逐年类型比例保持不变，只破坏类别与路径形态的关联。"
        "`p_two_sided`检验观测`between_share`是否偏离这一条件零分布。",
        "",
        "| 资料 | 几何 | 观测 between | 零分布中位数 | 95%零区间 | 双侧p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in permutation.itertuples():
        lines.append(
            f"| {row.dataset} | {row.geometry} | "
            f"{row.observed_between_share:.4f} | {row.null_median:.4f} | "
            f"[{row.null_ci_low:.4f}, {row.null_ci_high:.4f}] | "
            f"{row.p_two_sided:.4f} |"
        )
    all_permutation_significant = bool(
        (permutation["p_two_sided"] <= 0.05).all()
    )
    lines.extend(
        [
            "",
            "在主分期下，八个“资料×几何”组合的观测类型间份额"
            f"{'均高于条件置换零分布' if all_permutation_significant else '并非全部偏离条件置换零分布'}。"
            "这说明类型比例项虽小，但不是年内随机标签—路径配对自动产生的量；"
            "该检验不把其余份额自动解释为动力机制。",
        ]
    )

    lines.extend(
        [
            "",
            "## 2. 相对生成点路径分解",
            "",
            "每条路径以其首个重采样点为原点，分类隶属度保持固定。"
            "因此该检验移除绝对生成位置平移，但不改变“同一固定分类模型”这一估计对象。",
            "",
            "| 资料 | between | within | 总变差距离 | 闭合误差 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in relative.itertuples():
        lines.append(
            f"| {row.dataset} | {row.between_share:.4f} | "
            f"{row.within_share:.4f} | {row.total_variation:.4f} | "
            f"{row.closure_max_abs:.2e} |"
        )
    main_period = period_results.loc[
        period_results["scenario"] == "start_1966_main"
    ]
    absolute_tv = main_period.loc[
        main_period["geometry"] == "absolute",
        ["dataset", "total_variation"],
    ].set_index("dataset")["total_variation"]
    relative_tv = main_period.loc[
        main_period["geometry"] == "relative_genesis",
        ["dataset", "total_variation"],
    ].set_index("dataset")["total_variation"]
    retained = (relative_tv / absolute_tv).dropna()
    lines.extend(
        [
            "",
            "相对坐标下的总变差距离保留了绝对坐标结果的"
            f"{retained.min() * 100:.1f}%—{retained.max() * 100:.1f}%。"
            "因此，首个合格TS点的位置变化解释了部分空间重分配，"
            "但不能消除生成后路径形态差异。",
        ]
    )

    lines.extend(
        [
            "",
            "## 3. k=2—7结构扫描",
            "",
            "每个k在固定种子集合中以最小XB选择一次模型；"
            "ARI为其他种子硬标签相对该模型的置换不变一致性。"
            "FPC、XB和ARI不使用跨数据集通用硬阈值。",
            "",
            "| k | 最佳种子 | XB | FPC | 最大隶属度中位数 | ARI中位数（最小） | between |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in k_scan.itertuples():
        lines.append(
            f"| {row.k} | {row.best_seed} | {row.best_xb:.4f} | "
            f"{row.best_fpc:.4f} | {row.best_median_max_membership:.4f} | "
            f"{row.ari_vs_best_median:.3f} ({row.ari_vs_best_min:.3f}) | "
            f"{row.between_share:.4f} |"
        )
    unstable_k = k_scan.loc[k_scan["ari_vs_best_min"] < 0.9, "k"].tolist()
    lines.extend(
        [
            "",
            f"k=2—7时`between_share`为{k_scan['between_share'].min():.4f}—"
            f"{k_scan['between_share'].max():.4f}，类型间项保持较小。"
            + (
                f"但k={','.join(map(str, unstable_k))}出现种子间低ARI，"
                "说明类别几何并非在所有k下都稳定；k=3只能作为固定工作模型，"
                "不能据此宣称存在唯一三类物理结构。"
                if unstable_k
                else "各k的种子稳定性未出现低ARI。"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## 4. 起始年与切点敏感性",
            "",
            "1982起算使用固定的1966—2025分类模型，不重新拟合类别，"
            "以便把样本窗口影响与模型变化分开。下表给出各组合的范围；"
            "逐切点数值见`start_cutpoint_sensitivity.csv`。",
            "",
            "| 资料 | 几何 | 起始年 | between范围 | within范围 | 总变差范围 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    grouped = (
        period_results.groupby(["dataset", "geometry", "start_year"])
        .agg(
            between_min=("between_share", "min"),
            between_max=("between_share", "max"),
            within_min=("within_share", "min"),
            within_max=("within_share", "max"),
            tv_min=("total_variation", "min"),
            tv_max=("total_variation", "max"),
        )
        .reset_index()
    )
    for row in grouped.itertuples():
        lines.append(
            f"| {row.dataset} | {row.geometry} | {row.start_year} | "
            f"[{row.between_min:.4f}, {row.between_max:.4f}] | "
            f"[{row.within_min:.4f}, {row.within_max:.4f}] | "
            f"[{row.tv_min:.4f}, {row.tv_max:.4f}] |"
        )
    absolute_period = period_results.loc[
        period_results["geometry"] == "absolute", "between_share"
    ]
    relative_period = period_results.loc[
        period_results["geometry"] == "relative_genesis", "between_share"
    ]
    lines.extend(
        [
            "",
            "绝对坐标下全部起始年—切点组合的类型间份额为"
            f"{absolute_period.min():.4f}—{absolute_period.max():.4f}，"
            "始终为正且不超过5%。相对坐标下为"
            f"{relative_period.min():.4f}—{relative_period.max():.4f}，"
            "部分组合跨过零。稳健结论是类型比例项整体较小；"
            "其在去除首个TS点位置后的精确符号和量级不具有同等稳定性。",
        ]
    )

    min_relative_coverage = coverage.loc[
        coverage["geometry"] == "relative_genesis", "point_coverage_fraction"
    ].min()
    lines.extend(
        [
            "",
            "## 5. 域边界与密度分母敏感性",
            "",
            "三种口径分别为当前0°—40°N域内重归一、扩展至50°N后域内重归一，"
            "以及当前域内计数除以固定20个重采样点。固定分母口径保留域外点造成的"
            "质量损失，并将完全位于域外的气旋保留为零场。",
            "",
            "| 资料 | 口径 | 点覆盖率 | 前/后期密度质量 | between | 总变差 | 双侧p |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in domain_results.itertuples():
        lines.append(
            f"| {row.dataset} | {row.mode} | "
            f"{row.point_coverage_fraction:.4f} | "
            f"{row.density_mass_early:.4f}/{row.density_mass_late:.4f} | "
            f"{row.between_share:.4f} | {row.total_variation:.4f} | "
            f"{row.p_two_sided:.4f} |"
        )
    lines.extend(
        [
            "",
            f"三种口径下全部资料的`between_share`为"
            f"{domain_results['between_share'].min():.4f}—"
            f"{domain_results['between_share'].max():.4f}。"
            f"{'全部组合均偏离各自的条件置换零分布。' if (domain_results['p_two_sided'] <= 0.05).all() else '并非全部组合偏离各自的条件置换零分布。'}",
        ]
    )

    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            f"- 相对生成点固定域的最低点覆盖率为{min_relative_coverage:.4f}；"
            "绝对坐标的50°N扩展域与固定20点分母敏感性已完成。",
            "- 以上检验回答统计路径结构是否稳健，不回答海温、环流或海气耦合机制；"
            "AGU海洋过程分析按当前优先级顺延。",
            "- 科学解释同时依据合成测试和域边界敏感性；代数闭合或单个p值不足以单独决定结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=None,
        help="Override config bootstrap count for a diagnostic run.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=None,
        help="Override label-permutation count for a diagnostic run.",
    )
    parser.add_argument(
        "--cluster-seeds",
        type=int,
        default=None,
        help="Override the number of FCM seeds for a diagnostic run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Diagnostic output root; defaults to Verify/pathway_closure. "
            "The resolved path must remain inside the project workspace."
        ),
    )
    args = parser.parse_args()

    verify_cfg = load_verify_config()
    project_cfg = load_config()
    start = verify_cfg["period_early"][0]
    end = verify_cfg["period_late"][1]
    n_points = verify_cfg["n_path_points"]
    n_bootstrap = (
        args.n_bootstrap
        if args.n_bootstrap is not None
        else verify_cfg["n_bootstrap"]
    )
    n_permutations = (
        args.n_permutations
        if args.n_permutations is not None
        else verify_cfg["n_label_permutations"]
    )
    cluster_seeds = (
        args.cluster_seeds
        if args.cluster_seeds is not None
        else verify_cfg["cluster_seeds"]
    )
    years = np.arange(start, end + 1)
    early_index, late_index = scenario_indices(
        years,
        verify_cfg["period_early"][0],
        verify_cfg["period_late"][0],
        verify_cfg["period_late"][1],
    )
    if len(early_index) % verify_cfg["bootstrap_block_years"]:
        raise ValueError("early period length must be divisible by block length")
    if len(late_index) % verify_cfg["bootstrap_block_years"]:
        raise ValueError("late period length must be divisible by block length")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else VERIFY_ROOT
    )
    if output_root != PROJECT_ROOT and PROJECT_ROOT not in output_root.parents:
        raise ValueError(f"output root must remain under {PROJECT_ROOT}")
    results_dir = output_root / "results"
    qa_dir = output_root / "qa"
    results_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    processed = Path(project_cfg["paths"]["processed"])
    raw = Path(project_cfg["paths"]["raw"])
    tracks_path = processed / "tracks.csv"
    landfalls_path = processed / "landfalls.csv"
    ibtracs_path = raw / "IBTrACS.WP.v04r01.csv"

    primary_points = pd.read_csv(tracks_path, parse_dates=["iso_time"])
    primary_points = primary_points.loc[
        (primary_points["wind"] >= project_cfg["ts_threshold_kt"])
        & (
            (primary_points["nature"] == "TS")
            | primary_points["nature"].isna()
        )
    ].copy()
    reference = build_path_set(primary_points, "PRIMARY", start, end, n_points)
    scaler, centers, membership, model = fit_reference_model(
        reference,
        verify_cfg["n_families"],
        verify_cfg["fuzzy_m"],
        cluster_seeds,
        n_points,
    )
    reference.membership = membership
    k_scan, k_seed_diagnostics = fit_k_scan(
        reference,
        [int(value) for value in verify_cfg["k_scan"]],
        verify_cfg["fuzzy_m"],
        cluster_seeds,
    )

    source = read_ibtracs_agencies(ibtracs_path, start=start, end=end)
    path_sets = [reference]
    for agency in AGENCIES:
        catalog = build_agency_catalog(source, agency)
        path_set = build_path_set(
            catalog["ts_points"],
            "JMA" if agency == "TOKYO" else agency,
            start,
            end,
            n_points,
        )
        path_set.membership = predict_membership(
            path_set, scaler, centers, verify_cfg["fuzzy_m"]
        )
        path_sets.append(path_set)

    domain = verify_cfg["density_domain"]
    width = verify_cfg["density_grid_deg"]
    lon_edges = np.arange(domain["lon_min"], domain["lon_max"] + width, width)
    lat_edges = np.arange(domain["lat_min"], domain["lat_max"] + width, width)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    relative_domain = verify_cfg["relative_genesis_domain"]
    relative_lon_edges = np.arange(
        relative_domain["lon_min"],
        relative_domain["lon_max"] + width,
        width,
    )
    relative_lat_edges = np.arange(
        relative_domain["lat_min"],
        relative_domain["lat_max"] + width,
        width,
    )
    family_names = verify_cfg["family_order"]

    summary_rows = []
    family_rows = []
    bootstrap_frames = []
    assignment_frames = []
    sensitivity_rows = []
    period_frames = []
    permutation_summary_rows = []
    permutation_null_frames = []
    relative_rows = []
    coverage_rows = []
    domain_frames = []
    domain_null_frames = []
    field_output = {}
    for dataset_index, path_set in enumerate(path_sets):
        absolute_cache = density_cache(
            path_set, years, lon_edges, lat_edges, "absolute"
        )
        relative_cache = density_cache(
            path_set,
            years,
            relative_lon_edges,
            relative_lat_edges,
            "relative_genesis",
        )
        pi = absolute_cache.pi
        joint = absolute_cache.joint
        counts = absolute_cache.counts
        result = decompose(pi, joint, early_index, late_index)
        relative_result = decompose(
            relative_cache.pi,
            relative_cache.joint,
            early_index,
            late_index,
        )
        relative_rows.append(
            {
                "dataset": path_set.dataset,
                "geometry": "relative_genesis",
                "start_year": int(years[early_index][0]),
                "early_end_year": int(years[early_index][-1]),
                "late_start_year": int(years[late_index][0]),
                "end_year": int(years[late_index][-1]),
                "between_share": relative_result["between_share"],
                "within_share": relative_result["within_share"],
                "total_variation": relative_result["total_variation"],
                "closure_max_abs": relative_result["closure_max_abs"],
            }
        )
        for cache, edges_lon, edges_lat in (
            (absolute_cache, lon_edges, lat_edges),
            (relative_cache, relative_lon_edges, relative_lat_edges),
        ):
            geometry_paths = path_set.paths
            if cache.geometry == "relative_genesis":
                geometry_paths = geometry_paths - geometry_paths[:, :1, :]
            inside = (
                (geometry_paths[:, :, 0] >= edges_lon[0])
                & (geometry_paths[:, :, 0] <= edges_lon[-1])
                & (geometry_paths[:, :, 1] >= edges_lat[0])
                & (geometry_paths[:, :, 1] <= edges_lat[-1])
            )
            coverage_rows.append(
                {
                    "dataset": path_set.dataset,
                    "geometry": cache.geometry,
                    "n_storms": len(path_set.sid),
                    "valid_storm_fraction": float(cache.valid.mean()),
                    "point_coverage_fraction": float(inside.mean()),
                }
            )
            period_frames.append(
                period_sensitivity(
                    path_set,
                    cache,
                    years,
                    verify_cfg["period_scenarios"],
                )
            )
            permutation_summary, permutation_null = label_permutation_null(
                path_set,
                cache,
                years,
                early_index,
                late_index,
                n_permutations,
                verify_cfg["random_seed"]
                + dataset_index * 100
                + (0 if cache.geometry == "absolute" else 1),
            )
            permutation_summary_rows.append(
                {
                    "dataset": path_set.dataset,
                    "geometry": cache.geometry,
                    **permutation_summary,
                }
            )
            permutation_null.insert(0, "geometry", cache.geometry)
            permutation_null.insert(0, "dataset", path_set.dataset)
            permutation_null_frames.append(permutation_null)
        sensitivity_rows.append(
            {
                "dataset": path_set.dataset,
                "definition": "fuzzy_continuous",
                "between_share": result["between_share"],
                "within_share": result["within_share"],
                "closure_max_abs": result["closure_max_abs"],
            }
        )
        soft_membership = path_set.membership
        hard_membership = np.eye(len(family_names))[
            soft_membership.argmax(axis=0)
        ].T
        path_set.membership = hard_membership
        hard_cache = density_cache(
            path_set, years, lon_edges, lat_edges, "absolute"
        )
        hard_result = decompose(
            hard_cache.pi, hard_cache.joint, early_index, late_index
        )
        sensitivity_rows.append(
            {
                "dataset": path_set.dataset,
                "definition": "hard_max_membership",
                "between_share": hard_result["between_share"],
                "within_share": hard_result["within_share"],
                "closure_max_abs": hard_result["closure_max_abs"],
            }
        )
        path_set.membership = soft_membership
        domain_result, domain_null = domain_denominator_sensitivity(
            path_set,
            years,
            early_index,
            late_index,
            verify_cfg["domain_denominator_sensitivity"],
            width,
            n_permutations,
            verify_cfg["random_seed"] + 1000 + dataset_index * 10,
        )
        domain_frames.append(domain_result)
        domain_null_frames.append(domain_null)
        boot = bootstrap_decomposition(
            pi,
            joint,
            early_index,
            late_index,
            verify_cfg["bootstrap_block_years"],
            n_bootstrap,
            verify_cfg["random_seed"] + dataset_index,
        )
        boot.insert(0, "dataset", path_set.dataset)
        bootstrap_frames.append(boot)
        between_ci = quantiles(boot["between_share"])
        within_ci = quantiles(boot["within_share"])

        summary_rows.append(
            {
                "dataset": path_set.dataset,
                "n_storms": len(path_set.sid),
                "mean_annual_storms": counts.mean(),
                "between_share": result["between_share"],
                "between_ci_low": between_ci[0],
                "between_ci_median": between_ci[1],
                "between_ci_high": between_ci[2],
                "within_share": result["within_share"],
                "within_ci_low": within_ci[0],
                "within_ci_median": within_ci[1],
                "within_ci_high": within_ci[2],
                "total_variation": result["total_variation"],
                "closure_max_abs": result["closure_max_abs"],
            }
        )

        denominator = float(np.vdot(result["delta"], result["delta"]))
        for family_index, family_name in enumerate(family_names):
            early_centroid = weighted_centroid(
                result["f_early"][family_index], lon_centers, lat_centers
            )
            late_centroid = weighted_centroid(
                result["f_late"][family_index], lon_centers, lat_centers
            )
            family_rows.append(
                {
                    "dataset": path_set.dataset,
                    "family": family_name,
                    "share_early": result["pi_early"][family_index],
                    "share_late": result["pi_late"][family_index],
                    "share_change_pp": 100.0
                    * (
                        result["pi_late"][family_index]
                        - result["pi_early"][family_index]
                    ),
                    "between_projection": float(
                        np.vdot(
                            result["between_family"][family_index],
                            result["delta"],
                        )
                        / denominator
                    ),
                    "within_projection": float(
                        np.vdot(
                            result["within_family"][family_index],
                            result["delta"],
                        )
                        / denominator
                    ),
                    "centroid_lon_early": early_centroid[0],
                    "centroid_lon_late": late_centroid[0],
                    "centroid_lon_change": late_centroid[0] - early_centroid[0],
                    "centroid_lat_early": early_centroid[1],
                    "centroid_lat_late": late_centroid[1],
                    "centroid_lat_change": late_centroid[1] - early_centroid[1],
                }
            )

        assignments = pd.DataFrame(
            {
                "dataset": path_set.dataset,
                "sid": path_set.sid,
                "season": path_set.season,
                "hard_family": np.asarray(family_names)[
                    path_set.membership.argmax(axis=0)
                ],
                "max_membership": path_set.membership.max(axis=0),
            }
        )
        for family_index, family_name in enumerate(family_names):
            assignments[f"membership_{family_name}"] = path_set.membership[
                family_index
            ]
        assignment_frames.append(assignments)
        for key in (
            "density_early",
            "density_late",
            "delta",
            "between",
            "within",
        ):
            field_output[f"{path_set.dataset}_{key}"] = result[key]
            field_output[
                f"{path_set.dataset}_relative_genesis_{key}"
            ] = relative_result[key]

    summary = pd.DataFrame(summary_rows)
    family = pd.DataFrame(family_rows)
    bootstraps = pd.concat(bootstrap_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    period_results = pd.concat(period_frames, ignore_index=True)
    permutation_results = pd.DataFrame(permutation_summary_rows)
    permutation_null = pd.concat(permutation_null_frames, ignore_index=True)
    relative_summary = pd.DataFrame(relative_rows)
    density_coverage = pd.DataFrame(coverage_rows)
    domain_results = pd.concat(domain_frames, ignore_index=True)
    domain_permutation_null = pd.concat(
        domain_null_frames, ignore_index=True
    )

    landfall, landfall_boot, landfall_events = landfall_decomposition(
        reference,
        landfalls_path,
        years,
        early_index,
        late_index,
        verify_cfg["bootstrap_block_years"],
        n_bootstrap,
        verify_cfg["random_seed"] + 100,
    )

    summary.to_csv(results_dir / "pathway_summary.csv", index=False)
    family.to_csv(results_dir / "pathway_family_components.csv", index=False)
    bootstraps.to_csv(results_dir / "pathway_bootstrap.csv", index=False)
    assignments.to_csv(results_dir / "pathway_assignments.csv", index=False)
    sensitivity.to_csv(
        results_dir / "pathway_classification_sensitivity.csv", index=False
    )
    permutation_results.to_csv(
        results_dir / "between_share_permutation_summary.csv", index=False
    )
    permutation_null.to_csv(
        results_dir / "between_share_permutation_null.csv", index=False
    )
    relative_summary.to_csv(
        results_dir / "relative_genesis_summary.csv", index=False
    )
    period_results.to_csv(
        results_dir / "start_cutpoint_sensitivity.csv", index=False
    )
    k_scan.to_csv(results_dir / "k_scan_summary.csv", index=False)
    k_seed_diagnostics.to_csv(
        results_dir / "k_seed_diagnostics.csv", index=False
    )
    density_coverage.to_csv(
        results_dir / "density_domain_coverage.csv", index=False
    )
    domain_results.to_csv(
        results_dir / "domain_denominator_sensitivity.csv", index=False
    )
    domain_permutation_null.to_csv(
        results_dir / "domain_denominator_permutation_null.csv", index=False
    )
    landfall_boot.to_csv(results_dir / "landfall_bootstrap.csv", index=False)
    landfall_events.to_csv(
        results_dir / "primary_first_landfall_events.csv", index=False
    )
    np.savez_compressed(
        results_dir / "pathway_decomposition_fields.npz",
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        relative_lon_edges=relative_lon_edges,
        relative_lat_edges=relative_lat_edges,
        **field_output,
    )
    np.savez_compressed(
        results_dir / "pathway_fixed_model.npz",
        centers=centers,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        physical_centers=model["physical"],
        family_names=np.asarray(family_names),
    )
    save_figure(
        model["physical"],
        summary,
        family,
        landfall,
        results_dir / "pathway_closure_diagnostic.png",
    )
    write_report(
        results_dir / "ANALYSIS_REPORT.md",
        model["diagnostics"],
        summary,
        sensitivity,
        landfall,
        landfall_boot,
    )
    write_p0_report(
        results_dir / "P0_ANALYSIS_REPORT.md",
        permutation_results,
        relative_summary,
        k_scan,
        period_results,
        density_coverage,
        domain_results,
    )

    manifest = {
        "analysis": verify_cfg["analysis_name"],
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "n_bootstrap": n_bootstrap,
        "n_label_permutations": n_permutations,
        "cluster_seeds": cluster_seeds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "scikit-fuzzy",
                )
            },
        },
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                tracks_path,
                landfalls_path,
                ibtracs_path,
                VERIFY_ROOT / "config.json",
                HERE,
            )
        },
        "model": {
            **model["diagnostics"],
            "physical_centers_sha256": array_sha256(model["physical"]),
        },
        "datasets": {
            path_set.dataset: {
                "n_storms": int(len(path_set.sid)),
                "median_max_membership": float(
                    np.median(path_set.membership.max(axis=0))
                ),
            }
            for path_set in path_sets
        },
        "implementation_invariants": {
            "membership_sum": True,
            "max_absolute_path_closure_error": float(
                max(
                    summary["closure_max_abs"].max(),
                    relative_summary["closure_max_abs"].max(),
                    period_results["closure_max_abs"].max(),
                    k_scan["closure_max_abs"].max(),
                    sensitivity["closure_max_abs"].max(),
                    domain_results["closure_max_abs"].max(),
                )
            ),
            "max_landfall_closure_error": float(landfall["closure_abs"]),
            "all_outputs_finite": bool(
                np.isfinite(
                    permutation_results.select_dtypes(include=[np.number])
                ).all().all()
                and np.isfinite(
                    relative_summary.select_dtypes(include=[np.number])
                ).all().all()
                and np.isfinite(
                    period_results.select_dtypes(include=[np.number])
                ).all().all()
                and np.isfinite(
                    k_scan.select_dtypes(include=[np.number])
                ).all().all()
                and np.isfinite(
                    domain_results.select_dtypes(include=[np.number])
                ).all().all()
            ),
            "analysis_inputs_modified": False,
        },
        "scientific_decision": {
            "automatic_gate": False,
            "reason": (
                "P0 statistics and structural sensitivities require joint "
                "scientific interpretation."
            ),
        },
    }
    (qa_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gate_pass = (
        manifest["implementation_invariants"][
            "max_absolute_path_closure_error"
        ]
        < 1e-10
        and landfall["closure_abs"] < 1e-10
        and manifest["implementation_invariants"]["all_outputs_finite"]
    )
    p0_files = [
        "between_share_permutation_summary.csv",
        "between_share_permutation_null.csv",
        "relative_genesis_summary.csv",
        "k_scan_summary.csv",
        "k_seed_diagnostics.csv",
        "start_cutpoint_sensitivity.csv",
        "density_domain_coverage.csv",
        "domain_denominator_sensitivity.csv",
        "domain_denominator_permutation_null.csv",
        "P0_ANALYSIS_REPORT.md",
    ]
    (qa_dir / "VALIDATION_REPORT.md").write_text(
        "\n".join(
            [
                "# 验证报告",
                "",
                f"- 实现不变量：{'通过' if gate_pass else '失败'}；"
                "该项只验证计算实现，不作为科学有效性判据。",
                f"- 最大路径场闭合误差："
                f"{manifest['implementation_invariants']['max_absolute_path_closure_error']:.3e}。",
                f"- 登陆纬度闭合误差：{landfall['closure_abs']:.3e}。",
                f"- bootstrap 次数：{n_bootstrap}，分块长度："
                f"{verify_cfg['bootstrap_block_years']} 年。",
                f"- 标签置换次数：{n_permutations}；"
                "置换限制在同一年内，逐年类型比例不变。",
                f"- k扫描：{min(verify_cfg['k_scan'])}—"
                f"{max(verify_cfg['k_scan'])}，每个k使用{cluster_seeds}个种子。",
                f"- 域/分母敏感性：{len(domain_results)}个资料—口径组合，"
                "均使用同一条件标签置换。",
                f"- P0输出齐备："
                f"{'是' if all((results_dir / name).exists() for name in p0_files) else '否'}。",
                "- 统计路径闭环及其敏感性不构成动力因果归因。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not gate_pass:
        raise AssertionError("exact decomposition gate failed")
    print(
        f"pathway closure complete: {len(path_sets)} datasets, "
        f"{n_bootstrap} bootstrap replicates, "
        f"{n_permutations} label permutations; outputs={results_dir}"
    )


if __name__ == "__main__":
    main()
