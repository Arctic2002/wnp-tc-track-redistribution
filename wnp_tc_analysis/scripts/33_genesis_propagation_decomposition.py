from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, END, START, annual_fields, ensure_dirs, grid_edges, load_agency_tracks, load_primary_tracks

SEED = 202406
NBOOT = 999


def storm_table(tracks: pd.DataFrame, field_width: float = 2.5, gen_lon_width: float = 10.0, gen_lat_width: float = 5.0):
    lon_edges, lat_edges = grid_edges(field_width)
    gen_lon_edges, gen_lat_edges = grid_edges(gen_lon_width)[0], grid_edges(gen_lat_width)[1]
    rows = []
    fields = []
    for sid, g in tracks.sort_values("iso_time").groupby("sid"):
        first = g.iloc[0]
        lon_bin = np.digitize(first["lon"], gen_lon_edges) - 1
        lat_bin = np.digitize(first["lat"], gen_lat_edges) - 1
        if not (0 <= lon_bin < len(gen_lon_edges) - 1 and 0 <= lat_bin < len(gen_lat_edges) - 1):
            continue
        h = np.histogram2d(g["lon"], g["lat"], bins=[lon_edges, lat_edges])[0].T.reshape(-1)
        if h.sum() == 0:
            continue
        fields.append(h / h.sum())
        rows.append({"sid": sid, "year": int(first["season"]), "gen_bin": int(lat_bin * (len(gen_lon_edges) - 1) + lon_bin)})
    return pd.DataFrame(rows), np.asarray(fields), lon_edges, lat_edges


def weighted_period_components(meta, q, sampled_years, min_storms=3):
    # Each sampled year has equal weight, and storms within a year share that year's weight.
    weights = np.zeros(len(meta), dtype=float)
    for year in sampled_years:
        idx = np.flatnonzero(meta["year"].to_numpy() == year)
        if len(idx):
            weights[idx] += 1.0 / (len(sampled_years) * len(idx))
    bins = np.sort(meta.loc[weights > 0, "gen_bin"].unique())
    g = {}
    cond = {}
    n = {}
    for b in bins:
        idx = np.flatnonzero((meta["gen_bin"].to_numpy() == b) & (weights > 0))
        n[b] = len(idx)
        wb = weights[idx]
        g[b] = float(wb.sum())
        cond[b] = np.average(q[idx], axis=0, weights=wb)
    return g, cond, n


def decompose(meta, q, early_years, late_years, min_storms=3):
    ge0, qe0, ne = weighted_period_components(meta, q, early_years, min_storms)
    gl0, ql0, nl = weighted_period_components(meta, q, late_years, min_storms)
    shared = sorted(b for b in set(ge0) & set(gl0) if ne[b] >= min_storms and nl[b] >= min_storms)
    if not shared:
        raise ValueError("no shared genesis bins")
    cov_e = sum(ge0[b] for b in shared)
    cov_l = sum(gl0[b] for b in shared)
    ge = np.array([ge0[b] for b in shared], float); ge /= ge.sum()
    gl = np.array([gl0[b] for b in shared], float); gl /= gl.sum()
    qe = np.stack([qe0[b] for b in shared])
    ql = np.stack([ql0[b] for b in shared])
    pe = ge @ qe
    pl = gl @ ql
    genesis = 0.5 * ((gl - ge) @ qe + (gl - ge) @ ql)
    propagation = 0.5 * (ge @ (ql - qe) + gl @ (ql - qe))
    total = pl - pe
    closure = total - genesis - propagation
    denom = float(total @ total)
    return {
        "shared_bins": len(shared),
        "early_coverage": cov_e,
        "late_coverage": cov_l,
        "total_tv": 0.5 * float(np.abs(total).sum()),
        "genesis_l1": 0.5 * float(np.abs(genesis).sum()),
        "propagation_l1": 0.5 * float(np.abs(propagation).sum()),
        "genesis_projection_fraction": float(genesis @ total / denom) if denom else np.nan,
        "propagation_projection_fraction": float(propagation @ total / denom) if denom else np.nan,
        "closure_max_abs": float(np.max(np.abs(closure))),
        "early_field": pe,
        "late_field": pl,
        "total_field": total,
        "genesis_field": genesis,
        "propagation_field": propagation,
    }


def circular_block_sample(years: np.ndarray, rng, block=3):
    n = len(years)
    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        out.extend(years[(start + np.arange(block)) % n].tolist())
    return np.asarray(out[:n])


def run_one(catalog, tracks, gen_lon_width=10.0, gen_lat_width=5.0, bootstrap=True):
    meta, q, lon_edges, lat_edges = storm_table(tracks, 2.5, gen_lon_width, gen_lat_width)
    early = np.arange(1966, 1996)
    late = np.arange(1996, 2026)
    result = decompose(meta, q, early, late)
    summary = {k: v for k, v in result.items() if not k.endswith("_field")}
    summary.update({"catalog": catalog, "gen_lon_width": gen_lon_width, "gen_lat_width": gen_lat_width, "n_storms": len(meta)})
    boot_rows = []
    if bootstrap:
        rng = np.random.default_rng(SEED)
        for i in range(NBOOT):
            try:
                b = decompose(
                    meta,
                    q,
                    circular_block_sample(early, rng),
                    circular_block_sample(late, rng),
                )
            except ValueError:
                continue
            boot_rows.append(
                {
                    "catalog": catalog,
                    "replicate": i,
                    "total_tv": b["total_tv"],
                    "genesis_projection_fraction": b["genesis_projection_fraction"],
                    "propagation_projection_fraction": b["propagation_projection_fraction"],
                    "early_coverage": b["early_coverage"],
                    "late_coverage": b["late_coverage"],
                }
            )
    fields = {k: v for k, v in result.items() if k.endswith("_field")}
    fields.update({"lon_edges": lon_edges, "lat_edges": lat_edges})
    return summary, boot_rows, fields


def run() -> None:
    ensure_dirs()
    catalogs = {"PRIMARY": load_primary_tracks()}
    catalogs.update(load_agency_tracks())
    summaries, boots = [], []
    payload = {}
    for catalog, tracks in catalogs.items():
        summary, boot, fields = run_one(catalog, tracks, bootstrap=True)
        summaries.append(summary); boots.extend(boot)
        for key, value in fields.items():
            payload[f"{catalog}_{key}"] = value
    for glon, glat in ((5.0, 5.0), (10.0, 10.0)):
        summary, _, _ = run_one("PRIMARY", catalogs["PRIMARY"], glon, glat, bootstrap=False)
        summary["catalog"] = f"PRIMARY_sensitivity_{glon:g}x{glat:g}"
        summaries.append(summary)
    s = pd.DataFrame(summaries)
    b = pd.DataFrame(boots)
    if not b.empty:
        for catalog, group in b.groupby("catalog"):
            for variable in ("total_tv", "genesis_projection_fraction", "propagation_projection_fraction"):
                lo, hi = np.quantile(group[variable].dropna(), [0.025, 0.975])
                s.loc[s["catalog"] == catalog, f"{variable}_boot_lo"] = lo
                s.loc[s["catalog"] == catalog, f"{variable}_boot_hi"] = hi
    s.to_csv(DATA / "wnp_tc_genesis_propagation_summary.csv", index=False)
    b.to_csv(DATA / "wnp_tc_genesis_propagation_bootstrap.csv", index=False)
    np.savez_compressed(DATA / "wnp_tc_genesis_propagation_fields.npz", **payload)
    print(s.to_string(index=False))


if __name__ == "__main__":
    run()
