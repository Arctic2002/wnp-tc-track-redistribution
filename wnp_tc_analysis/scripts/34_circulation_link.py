from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend

from common import DATA, FIGURES, PROJECT, SEED, ensure_dirs, sen_mk

# Field permutation is computationally expensive. Use 499 preregistered
# screening permutations here; scalar relations are tested separately in J5.
NPERM = 499
BLOCK = 3


def standardize(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def block_orders(n, nperm=NPERM, block=BLOCK):
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    rng = np.random.default_rng(SEED)
    for _ in range(nperm):
        yield np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def field_test(x, y, *, inferential=True):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    yc = y - y.mean(axis=0)
    xss = float(np.sum(xc**2))
    yss = np.sum(yc**2, axis=0)
    dot = np.tensordot(xc, yc, axes=(0, 0))
    beta = dot / xss
    denom = np.sqrt(xss * yss)
    r = np.divide(dot, denom, out=np.zeros_like(beta), where=denom > 0)
    observed_global = float(np.nanmean(r**2))
    if not inferential:
        missing = np.full(r.shape, np.nan)
        return beta, r, missing, missing.copy(), observed_global, np.nan
    exceed = np.zeros(r.shape, dtype=np.int32)
    exceed_global = 0
    for order in block_orders(len(x)):
        dotp = np.tensordot(xc[order], yc, axes=(0, 0))
        rp = np.divide(dotp, denom, out=np.zeros_like(beta), where=denom > 0)
        exceed += np.abs(rp) >= np.abs(r)
        exceed_global += float(np.nanmean(rp**2)) >= observed_global
    p = (exceed + 1) / (NPERM + 1)
    from paper2_dynamic.revision_stats import bh_fdr
    q = bh_fdr(p.reshape(-1)).reshape(p.shape)
    return beta, r, p, q, observed_global, (exceed_global + 1) / (NPERM + 1)


def _coarsen_2d(a, factor=10):
    nlat = (a.shape[-2] // factor) * factor
    nlon = (a.shape[-1] // factor) * factor
    return a[:nlat, :nlon].reshape(nlat // factor, factor, nlon // factor, factor).mean(axis=(1, 3))


def direct_annual(path, variables, level=None):
    """Read monthly hyperslabs directly and accumulate 2.5-degree annual means."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], float)
        lon = np.asarray(ds.variables["longitude"][:], float)
        nlat = (len(lat) // 10) * 10
        nlon = (len(lon) // 10) * 10
        latc = lat[:nlat].reshape(-1, 10).mean(axis=1)
        lonc = lon[:nlon].reshape(-1, 10).mean(axis=1)
        tv = ds.variables["time"]
        dates = nc.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"), only_use_cftime_datetimes=False)
        selected = [(i, int(d.year)) for i, d in enumerate(dates) if 1966 <= d.year <= 2025 and d.month in (6, 7, 8, 9, 10)]
        years = np.arange(1966, 2026)
        sums = {v: np.zeros((len(years), len(latc), len(lonc)), float) for v in variables}
        metadata = {
            name: {
                "units": str(getattr(ds.variables[name], "units", "")),
                "standard_name": str(getattr(ds.variables[name], "standard_name", "")),
                "long_name": str(getattr(ds.variables[name], "long_name", "")),
            }
            for name in variables
        }
        counts = np.zeros(len(years), int)
        level_index = None
        if level is not None:
            levels = np.asarray(ds.variables["level"][:])
            level_index = int(np.flatnonzero(levels == level)[0])
        by_year = {year: [i for i, y in selected if y == year] for year in years}
        for year, indices in by_year.items():
            yi = year - 1966
            counts[yi] = len(indices)
            for name in variables:
                var = ds.variables[name]
                raw = var[indices, level_index, :, :] if level_index is not None and "level" in var.dimensions else var[indices, :, :]
                sums[name][yi] = _coarsen_2d(np.asarray(raw, float).mean(axis=0)) * len(indices)
        if not np.all(counts == 5):
            raise ValueError(f"incomplete typhoon-season months: {counts}")
        return (
            years,
            latc,
            lonc,
            {name: value / counts[:, None, None] for name, value in sums.items()},
            metadata,
        )


def geopotential_to_height(values, metadata, gravity=9.80665):
    """Convert CF geopotential to geopotential height using declared metadata."""
    values = np.asarray(values, float)
    standard_name = str(metadata.get("standard_name", "")).strip().lower()
    units = str(metadata.get("units", "")).strip().lower().replace(" ", "")
    geopotential_units = {
        "m**2s**-2",
        "m^2s^-2",
        "m2s-2",
        "m2/s2",
        "m²s⁻²",
    }
    height_units = {"m", "gpm", "geopotentialmetre", "geopotentialmeter"}
    if standard_name == "geopotential":
        if units and units not in geopotential_units:
            raise ValueError(f"geopotential has unexpected units: {metadata.get('units')}")
        return values / gravity
    if standard_name == "geopotential_height" or units in height_units:
        return values
    if units in geopotential_units:
        return values / gravity
    raise ValueError(f"cannot determine geopotential-height conversion from {metadata}")


def region_mean(values, lat, lon, lon0, lon1, lat0, lat1):
    mask_lat = (lat >= lat0) & (lat <= lat1)
    mask_lon = (lon >= lon0) & (lon <= lon1)
    return values[:, mask_lat][:, :, mask_lon].mean(axis=(1, 2))


def run() -> None:
    ensure_dirs()
    idx = pd.read_csv(DATA / "wnp_tc_redistribution_index_annual.csv")
    idx = idx.loc[(idx.agency == "PRIMARY") & (idx.weighting == "track_point")].sort_values("year")
    years = idx["year"].to_numpy(int)
    x = standardize(idx["index_oos"].to_numpy(float))

    sy, lat, lon, sdata, _ = direct_annual(
        PROJECT / "data" / "interim" / "steering.nc", ["u_steer", "v_steer"]
    )
    zy, zlat, zlon, zdata, zmetadata = direct_annual(
        PROJECT / "data" / "interim" / "era5_wnp_dynamic_plev.nc", ["z"], level=500
    )
    if not (np.array_equal(sy, years) and np.array_equal(zy, years)):
        raise ValueError("annual field years do not align")
    if not (np.allclose(lat, zlat) and np.allclose(lon, zlon)):
        raise ValueError("steering and height grids do not align")
    u = sdata["u_steer"]
    v = sdata["v_steer"]
    zheight = geopotential_to_height(zdata["z"], zmetadata["z"])
    ze = zheight - zheight.mean(axis=2, keepdims=True)

    payload = {
        "years": years,
        "latitude": lat,
        "longitude": lon,
        "index_oos_z": x,
        "z_input_units": np.asarray(zmetadata["z"]["units"]),
        "z_input_standard_name": np.asarray(zmetadata["z"]["standard_name"]),
        "z_height_units": np.asarray("m"),
        "z_background_definition": np.asarray("latitude-wise 80-180E sector mean removed"),
    }
    summary_rows = []
    for timescale, xx, arrays in (
        ("raw", x, {"u": u, "v": v, "z_eddy": ze}),
        (
            "detrended",
            detrend(x),
            {
                "u": detrend(u, axis=0),
                "v": detrend(v, axis=0),
                "z_eddy": detrend(ze, axis=0),
            },
        ),
    ):
        for variable, arr in arrays.items():
            inferential = timescale == "detrended"
            beta, r, p, q, stat, gp = field_test(xx, arr, inferential=inferential)
            for name, value in (("beta", beta), ("r", r), ("p", p), ("q", q)):
                payload[f"{timescale}_{variable}_{name}"] = value
            summary_rows.append(
                {
                    "timescale": timescale,
                    "variable": variable,
                    "inference_role": "inferential_detrended" if inferential else "descriptive_raw",
                    "global_mean_r2": stat,
                    "global_block_permutation_p": gp,
                    "n_cells_fdr_q_lt_0_05": int(np.sum(q < 0.05)) if inferential else np.nan,
                    "max_abs_r": float(np.nanmax(np.abs(r))),
                }
            )

    scalar = pd.DataFrame({
        "year": years,
        "redistribution_index_oos_z": x,
        "eddy_wnpsh_mean_m": region_mean(ze, lat, lon, 110, 160, 15, 35),
        "corridor_u_steer_ms": region_mean(u, lat, lon, 110, 160, 10, 30),
        "corridor_v_steer_ms": region_mean(v, lat, lon, 110, 160, 10, 30),
    })
    fixed = pd.read_csv(PROJECT / "data" / "processed" / "p2_wnpsh.csv").rename(columns={"season": "year"})
    scalar = scalar.merge(fixed, on="year", how="left")
    scalar.to_csv(DATA / "wnp_tc_eddy_wnpsh_annual.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(DATA / "wnp_tc_circulation_regression_summary.csv", index=False)
    np.savez_compressed(DATA / "wnp_tc_circulation_fields.npz", **payload)

    trend_rows = []
    for col in ["eddy_wnpsh_mean_m", "corridor_u_steer_ms", "corridor_v_steer_ms", "west_ridge_point", "ridge_line"]:
        trend_rows.append({"variable": col, **sen_mk(years, scalar[col].to_numpy(float))})
    pd.DataFrame(trend_rows).to_csv(DATA / "wnp_tc_circulation_scalar_trends.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(years, x, color="#345995", lw=1.2)
    axes[0, 0].axhline(0, color="0.4", lw=0.7)
    axes[0, 0].set(xlabel="Year", ylabel="OOS redistribution index (z)")
    lon = payload["longitude"]; lat = payload["latitude"]
    m = axes[0, 1].pcolormesh(lon, lat, payload["raw_z_eddy_beta"], cmap="RdBu_r", shading="auto")
    fig.colorbar(m, ax=axes[0, 1], label="Eddy Z500 regression (m per SD)")
    step = 2
    axes[1, 0].quiver(lon[::step], lat[::step], payload["raw_u_beta"][::step, ::step], payload["raw_v_beta"][::step, ::step])
    axes[1, 0].set(xlabel="Longitude", ylabel="Latitude")
    m2 = axes[1, 1].pcolormesh(lon, lat, payload["detrended_z_eddy_beta"], cmap="RdBu_r", shading="auto")
    fig.colorbar(m2, ax=axes[1, 1], label="Detrended eddy Z500 regression")
    for label, ax in zip("abcd", axes.flat):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold")
    fig.savefig(FIGURES / "wnp_tc_diagnostic_circulation_link.png", dpi=220)
    plt.close(fig)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(trend_rows).to_string(index=False))


if __name__ == "__main__":
    run()
