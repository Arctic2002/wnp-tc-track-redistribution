from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent
DATA = WORK / "data"
RESULTS = WORK / "results"
FIGURES = WORK / "outputs" / "figures"
DOCS = WORK / "docs"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

START = 1966
END = 2025
EARLY = (1966, 1995)
LATE = (1996, 2025)
REGION = (100.0, 180.0, 0.0, 40.0)
SEED = 202406


def ensure_dirs() -> None:
    for path in (DATA, RESULTS, FIGURES, DOCS):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grid_edges(width: float) -> tuple[np.ndarray, np.ndarray]:
    lon0, lon1, lat0, lat1 = REGION
    return np.arange(lon0, lon1 + width, width), np.arange(lat0, lat1 + width, width)


def load_primary_tracks(start: int = START, end: int = END, threshold: float = 34.0) -> pd.DataFrame:
    p = PROJECT / "data" / "processed" / "tracks.csv"
    d = pd.read_csv(p)
    d = d.loc[
        d["season"].between(start, end)
        & (d["wind"] >= threshold)
        & ((d["nature"] == "TS") | d["nature"].isna())
        & d["lon"].between(REGION[0], REGION[1])
        & d["lat"].between(REGION[2], REGION[3])
    ].copy()
    return d.sort_values(["season", "sid", "iso_time"])


def load_agency_tracks(start: int = START, end: int = END) -> dict[str, pd.DataFrame]:
    mod = importlib.import_module("paper2_dynamic.agency_data")
    raw = mod.read_ibtracs_agencies(PROJECT / "data" / "raw" / "IBTrACS.WP.v04r01.csv", start, end)
    out: dict[str, pd.DataFrame] = {}
    for agency in ("USA", "TOKYO", "CMA"):
        cat = mod.build_agency_catalog(raw, agency)
        d = cat["ts_points"].copy()
        d = d.loc[
            d["season"].between(start, end)
            & d["lon"].between(REGION[0], REGION[1])
            & d["lat"].between(REGION[2], REGION[3])
        ]
        out[agency] = d.sort_values(["season", "sid", "iso_time"])
    return out


def annual_fields(
    tracks: pd.DataFrame,
    years: np.ndarray,
    width: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_edges, lat_edges = grid_edges(width)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    point_fields: list[np.ndarray] = []
    storm_fields: list[np.ndarray] = []
    nstorms: list[int] = []
    npoints: list[int] = []
    for year in years:
        annual = tracks.loc[tracks["season"] == year]
        if annual.empty:
            raise ValueError(f"no eligible tracks in {year}")
        point = np.histogram2d(annual["lon"], annual["lat"], bins=[lon_edges, lat_edges])[0].T
        if point.sum() == 0:
            raise ValueError(f"no in-domain points in {year}")
        point_fields.append((point / point.sum()).reshape(-1))
        storm = np.zeros(shape, dtype=float)
        valid_storms = 0
        for _, group in annual.groupby("sid"):
            one = np.histogram2d(group["lon"], group["lat"], bins=[lon_edges, lat_edges])[0].T
            if one.sum() > 0:
                storm += one / one.sum()
                valid_storms += 1
        if storm.sum() == 0:
            raise ValueError(f"no valid storms in {year}")
        storm_fields.append((storm / storm.sum()).reshape(-1))
        nstorms.append(valid_storms)
        npoints.append(len(annual))
    return (
        np.asarray(point_fields),
        np.asarray(storm_fields),
        np.asarray(nstorms),
        np.asarray(npoints),
        lon_edges,
        lat_edges,
    )


def period_indices(years: np.ndarray, early: tuple[int, int] = EARLY, late: tuple[int, int] = LATE):
    e = np.flatnonzero((years >= early[0]) & (years <= early[1]))
    l = np.flatnonzero((years >= late[0]) & (years <= late[1]))
    if len(e) != len(l) or len(e) == 0:
        raise ValueError(f"periods must be nonempty and equal: {early}, {late}")
    return e, l


def temporal_blocks(length: int, block: int, years: np.ndarray | None = None) -> list[np.ndarray]:
    """Return position blocks without allowing a block to cross a year gap."""
    if block <= 0:
        raise ValueError("block must be positive")
    positions = np.arange(length)
    if years is None:
        runs = [positions]
    else:
        years = np.asarray(years, dtype=int)
        if len(years) != length or np.any(np.diff(years) <= 0):
            raise ValueError("years must be strictly increasing and match the data length")
        runs = np.split(positions, np.flatnonzero(np.diff(years) != 1) + 1)
    return [run[i : i + block] for run in runs for i in range(0, len(run), block)]


def _projection_scores_for_indices(
    fields: np.ndarray,
    early_idx: np.ndarray,
    late_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projection scores for explicit period memberships."""
    fields = np.asarray(fields, dtype=float)
    e = np.asarray(early_idx, dtype=int)
    l = np.asarray(late_idx, dtype=int)
    if len(e) < 2 or len(l) < 2:
        raise ValueError("each projection period needs at least two observations")
    em = fields[e].mean(axis=0)
    lm = fields[l].mean(axis=0)
    pattern = lm - em
    mid = 0.5 * (lm + em)
    denom = float(pattern @ pattern)
    if denom <= 0:
        raise ValueError("zero redistribution pattern")
    full = 2.0 * ((fields - mid) @ pattern) / denom

    # Vectorized leave-one-out centroids.  Each target year is excluded only
    # from the centroid of the period to which that year belongs.
    em_loo = np.broadcast_to(em, fields.shape).copy()
    lm_loo = np.broadcast_to(lm, fields.shape).copy()
    em_loo[e] = (fields[e].sum(axis=0) - fields[e]) / (len(e) - 1)
    lm_loo[l] = (fields[l].sum(axis=0) - fields[l]) / (len(l) - 1)
    patterns = lm_loo - em_loo
    den = np.einsum("ij,ij->i", patterns, patterns)
    oos = np.full(len(fields), np.nan)
    valid = den > 0
    oos[valid] = (
        2.0
        * np.einsum(
            "ij,ij->i",
            fields[valid] - 0.5 * (lm_loo[valid] + em_loo[valid]),
            patterns[valid],
        )
        / den[valid]
    )
    return full, oos, pattern


def projection_scores(fields: np.ndarray, years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-sample and leave-one-year-out projection scores.

    The coefficient is scaled so the two period centroids lie near -1 and +1.
    For the out-of-sample score, the target year is excluded from its period
    centroid and pattern before projection.
    """
    e, l = period_indices(years)
    em = fields[e].mean(axis=0)
    lm = fields[l].mean(axis=0)
    pattern = lm - em
    mid = 0.5 * (lm + em)
    denom = float(pattern @ pattern)
    if denom <= 0:
        raise ValueError("zero redistribution pattern")
    full = 2.0 * ((fields - mid) @ pattern) / denom
    oos = np.full(len(years), np.nan)
    for i in range(len(years)):
        ee = e[e != i]
        ll = l[l != i]
        em_i = fields[ee].mean(axis=0)
        lm_i = fields[ll].mean(axis=0)
        d_i = lm_i - em_i
        den_i = float(d_i @ d_i)
        if den_i > 0:
            oos[i] = 2.0 * ((fields[i] - 0.5 * (lm_i + em_i)) @ d_i) / den_i
    return full, oos, pattern


def block_permutation_projection(
    fields: np.ndarray,
    years: np.ndarray,
    block: int = 3,
    nperm: int = 9999,
    seed: int = SEED,
) -> tuple[float, float]:
    """Test OOS period separation while refitting the pattern each permutation.

    The projection direction is estimated from the same period labels that are
    being tested.  Consequently, a valid randomization repeats that estimation
    inside every permutation; permuting already-computed scores would hold an
    observed-label pattern fixed and understate the null variability.
    """
    fields = np.asarray(fields, dtype=float)
    years = np.asarray(years, dtype=int)
    e, l = period_indices(years)
    _, observed_scores, _ = projection_scores(fields, years)
    observed = float(observed_scores[l].mean() - observed_scores[e].mean())
    blocks = temporal_blocks(len(fields), block, years)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        _, trial_scores, _ = _projection_scores_for_indices(fields[order], e, l)
        trial = float(trial_scores[l].mean() - trial_scores[e].mean())
        exceed += abs(trial) >= abs(observed)
    return observed, (exceed + 1) / (nperm + 1)


def block_permutation_scalar(
    values: np.ndarray,
    early_idx: np.ndarray,
    late_idx: np.ndarray,
    block: int = 3,
    nperm: int = 9999,
    seed: int = SEED,
    years: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = float(values[late_idx].mean() - values[early_idx].mean())
    blocks = temporal_blocks(len(values), block, years)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        diff = values[order[late_idx]].mean() - values[order[early_idx]].mean()
        exceed += abs(diff) >= abs(observed)
    return observed, (exceed + 1) / (nperm + 1)


def global_tv_permutation(
    fields: np.ndarray,
    early_idx: np.ndarray,
    late_idx: np.ndarray,
    block: int,
    nperm: int,
    seed: int = SEED,
    years: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    change = fields[late_idx].mean(axis=0) - fields[early_idx].mean(axis=0)
    observed = float(0.5 * np.abs(change).sum())
    blocks = temporal_blocks(len(fields), block, years)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        diff = fields[order[late_idx]].mean(axis=0) - fields[order[early_idx]].mean(axis=0)
        exceed += 0.5 * np.abs(diff).sum() >= observed
    return observed, (exceed + 1) / (nperm + 1), change


def sen_mk(years: np.ndarray, values: np.ndarray) -> dict[str, float | str | int]:
    from scipy.stats import theilslopes
    import pymannkendall as mk

    mask = np.isfinite(years) & np.isfinite(values)
    x = np.asarray(years)[mask].astype(float)
    y = np.asarray(values)[mask].astype(float)
    ts = theilslopes(y, x, alpha=0.95)
    m = mk.hamed_rao_modification_test(y)
    return {
        "n": len(y),
        "slope_per_decade": float(ts.slope * 10),
        "slope_ci_low_per_decade": float(ts.low_slope * 10),
        "slope_ci_high_per_decade": float(ts.high_slope * 10),
        "mk_p": float(m.p),
        "mk_trend": str(m.trend),
    }
