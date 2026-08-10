"""绘制反事实试验结果对照图。

面板 (a)–(e) 为相对路径密度差异场（后期减前期，×1000）。
面板 (f) 为两期引导流气候态差异本身，是主判据不通过的直接原因。

图件字体遵循项目规范使用无衬线字体；标注用英文以避免缺字。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.linewidth"] = 0.6

EXP = Path(__file__).resolve().parent.parent
ROOT = EXP.parent.parent


def steering_diff():
    """主季（6–10月）两期引导流气候态差异与月际标准差。"""
    src = Path("/tmp/steering.nc")
    if not src.is_file():
        src = ROOT / "data" / "interim" / "steering.nc"
    ds = Dataset(src)
    lat = np.asarray(ds.variables["latitude"][:])
    lon = np.asarray(ds.variables["longitude"][:])
    u = np.asarray(ds.variables["u_steer"][:], dtype=np.float32)
    v = np.asarray(ds.variables["v_steer"][:], dtype=np.float32)
    ds.close()
    n = u.shape[0]
    yrs = 1940 + np.arange(n) // 12
    mons = 1 + np.arange(n) % 12
    season = np.isin(mons, [6, 7, 8, 9, 10])
    p1 = season & (yrs >= 1966) & (yrs <= 1995)
    p2 = season & (yrs >= 1996) & (yrs <= 2025)
    du = u[p2].mean(0) - u[p1].mean(0)
    dv = v[p2].mean(0) - v[p1].mean(0)
    sd = np.sqrt(np.concatenate([u[p1], u[p2]]).std(0) ** 2
                 + np.concatenate([v[p1], v[p2]]).std(0) ** 2)
    return lat, lon, du, dv, sd


def main() -> None:
    z = np.load(EXP / "results" / "fields.npz")
    lon_e, lat_e = z["lon_edges"], z["lat_edges"]
    ext = [lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]]

    panels = [
        ("a", "Observed  (P2 - P1)", z["d_obs"]),
        ("b", "Model, all factors", z["d_full"]),
        ("c", "Genesis location only", z["d_loc"]),
        ("d", "Mean steering only", z["d_steer"]),
        ("e", "In-domain lifetime only", z["d_life"]),
    ]
    vmax = max(np.abs(p[2]).max() for p in panels) * 1000

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), constrained_layout=True)
    for ax, (tag, title, fld) in zip(axes.ravel()[:5], panels):
        im = ax.imshow(fld * 1000, origin="lower", extent=ext, aspect="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"({tag}) {title}", fontsize=9.5, pad=4)
        ax.set_xticks([100, 120, 140, 160, 180])
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Longitude (°E)", fontsize=8)
        ax.set_ylabel("Latitude (°N)", fontsize=8)
    cb = fig.colorbar(im, ax=axes.ravel()[:5].tolist(), shrink=0.85, pad=0.01)
    cb.set_label("Relative track-density difference (×10$^{-3}$)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # (f) 引导流气候态差异：主判据不通过的直接原因
    ax = axes.ravel()[5]
    lat, lon, du, dv, sd = steering_diff()
    my = (lat >= 0) & (lat <= 40)
    mx = (lon >= 100) & (lon <= 180)
    mag = np.sqrt(du[np.ix_(my, mx)] ** 2 + dv[np.ix_(my, mx)] ** 2)
    ratio = mag / sd[np.ix_(my, mx)]
    im2 = ax.imshow(ratio, origin="lower", aspect="auto",
                    extent=[100, 180, 0, 40], cmap="Greys", vmin=0, vmax=1)
    ax.set_title("(f) Steering change / its own variability", fontsize=9.5, pad=4)
    ax.set_xticks([100, 120, 140, 160, 180])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    cb2 = fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.01)
    cb2.set_label("|ΔV| / σ", fontsize=8)
    cb2.ax.tick_params(labelsize=7.5)
    ax.text(0.5, -0.28, f"basin mean = {np.nanmean(ratio):.2f}σ",
            transform=ax.transAxes, ha="center", fontsize=8.5)

    out = EXP / "figures" / "counterfactual_fields.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    print(f"  (f) 面板全域均值 = {np.nanmean(ratio):.3f}σ, 最大 = {np.nanmax(ratio):.3f}σ")


if __name__ == "__main__":
    main()
