"""Paper II 全部图件（P2-1..P2-14）。

严格按 Docs/03 §4 的绘制要点与 §5 强制枚举模式：一图一函数、各存各的 PNG，
FIGURES 注册表 + main() 循环，单图失败不影响其余。所有图复用 paperfig/figstyle。
图件只读取已验收产物，不重新计算筛选口径。

注意：动力场读取 dynamic 框文件 era5_wnp_dynamic_plev.nc（覆盖 80–180°E/0–65°N，
含副高/引导所需层次），而非 thermo 框 era5_wnp_plev.nc。
"""
import os
import glob
import shutil
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pymannkendall as mk
from scipy import stats
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from core.utils import load_config
from core.stats_utils import theil_sen_ci
import paperfig.figstyle as fs

cfg = load_config()
P = cfg["paths"]["processed"]
I = cfg["paths"]["interim"]
RAW = cfg["paths"]["raw"]
SEASON = cfg["typhoon_season"]
TH = cfg["oni_threshold"]


def _figure_title(fig, title, *, tight=False, top=0.90):
    """添加整图标题，并为标题预留顶部空间。"""
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.subplots_adjust(top=top)
    fig.suptitle(title, y=0.965, fontsize=13, fontweight="bold")


def _sen(ax, x, y):
    """在 ax 上画序列并叠加 Theil–Sen 趋势与 95% 自助带。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        ax.plot(x, y, lw=1.4, color=fs.MONET[0])
        return
    slope, lo, hi = theil_sen_ci(x, y, nboot=cfg["statistics"]["bootstrap_samples"],
                                 block=cfg["statistics"]["bootstrap_block"],
                                 seed=cfg["statistics"]["random_seed"])
    fs.add_series(ax, x, y, slope=slope)
    ax.set_title(ax.get_title() + f"\n(Sen={slope*10:.3g}/dec)", fontsize=10)


def _bh_fdr(p):
    """Benjamini-Hochberg FDR values for a 1-D p-value array."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty_like(pv)
    qq[order] = np.clip(ranked, 0, 1)
    q[ok] = qq
    return q


def _p2_trend_stats(g, panels, fdr_family):
    """Calculate P2 annual trend diagnostics without changing source data."""
    rows = []
    for c, title, start in panels:
        sub = g.loc[g["season"] >= start, ["season", c]].dropna()
        x = sub["season"].to_numpy(float)
        y = sub[c].to_numpy(float)
        row = {
            "scope": "annual",
            "var": c,
            "label": title,
            "display_start": cfg["periods"]["record_start"],
            "start": start,
            "end": int(sub["season"].max()) if len(sub) else np.nan,
            "n": len(sub),
            "pearson_r": np.nan,
            "mk_method": "Hamed-Rao modified Mann-Kendall",
            "mk_p_raw": np.nan,
            "mk_trend": "",
            "sen_slope_per_year": np.nan,
            "sen_slope_per_decade": np.nan,
            "sen_ci_lo_per_year": np.nan,
            "sen_ci_hi_per_year": np.nan,
        }
        if len(sub) >= 5 and np.nanstd(y) > 0:
            row["pearson_r"] = stats.pearsonr(x, y).statistic
            mk_res = mk.hamed_rao_modification_test(y)
            row["mk_p_raw"] = mk_res.p
            row["mk_trend"] = mk_res.trend
            sl, lo, hi = theil_sen_ci(
                x, y, nboot=cfg["statistics"]["bootstrap_samples"],
                block=cfg["statistics"]["bootstrap_block"],
                seed=cfg["statistics"]["random_seed"])
            row["sen_slope_per_year"] = sl
            row["sen_slope_per_decade"] = sl * 10
            row["sen_ci_lo_per_year"] = lo
            row["sen_ci_hi_per_year"] = hi
        rows.append(row)
    out = pd.DataFrame(rows)
    out["mk_p_fdr_bh"] = _bh_fdr(out["mk_p_raw"].to_numpy(float))
    out["fdr_family"] = fdr_family
    out["note"] = "Series are displayed from record_start; pre-start values are excluded from Sen/MK/FDR."
    return out


def _p2_fig06_stats(g, panels):
    return _p2_trend_stats(
        g, panels, "Benjamini-Hochberg FDR across P2-6 four annual track-metric MK tests")


def _fmt_p(v):
    if not np.isfinite(v):
        return "NA"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _fmt_num(v, nd=2):
    return "NA" if not np.isfinite(v) else f"{v:.{nd}f}"


def _plot_p2_trend_panel(ax, g, c, title, start, stats_row, *, show_legend=False):
    """Display early records but fit and label trends only over the formal period."""
    display_start = cfg["periods"]["record_start"]
    display = g[g.season >= display_start].dropna(subset=[c])
    early = display[display.season < start]
    if len(early):
        ax.plot(early["season"], early[c], lw=1.15, ls=":",
                color=fs.MONET[0], alpha=0.42)
    sub = display[display.season >= start]
    _sen(ax, sub["season"], sub[c])
    star = "*" if np.isfinite(stats_row["mk_p_fdr_bh"]) and stats_row["mk_p_fdr_bh"] < 0.05 else ""
    ax.set_title(
        f"{title}\n"
        f"Sen={stats_row['sen_slope_per_decade']:.3g}/dec "
        f"({start}-{int(stats_row['end'])}); "
        f"r={_fmt_num(stats_row['pearson_r'])}; "
        f"MK p={_fmt_p(stats_row['mk_p_raw'])}; "
        f"FDR={_fmt_p(stats_row['mk_p_fdr_bh'])}{star}",
        fontsize=7.6)
    ax.set_xlabel("Year")
    ax.set_xlim(display_start - 2, g["season"].max() + 2)
    if show_legend:
        ax.plot([], [], lw=1.15, ls=":", color=fs.MONET[0], alpha=0.42,
                label="Early record (excluded)")
        ax.plot([], [], lw=1.4, color=fs.MONET[0], label="Trend-test period")
        ax.legend(loc="best", fontsize=5.8, handlelength=2.4)


# ---------------- P2-1 路径密度的年代际演变 ----------------
def fig01():
    files = sorted(glob.glob(f"{P}/density_dec_*.npz"))
    n = len(files)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    vmax = max(np.load(f)["dens"].max() for f in files)
    # Map axes preserve geographic aspect; a taller gridspec only creates
    # unusable blank bands.  Size rows to the map aspect instead.
    fig = plt.figure(figsize=(fs.COL2 * 1.18, fs.COL2 / ncol * nrow * 0.72))
    gs = fig.add_gridspec(nrow + 1, ncol,
                          height_ratios=[1] * nrow + [0.075],
                          wspace=0.18, hspace=0.18)
    axes = np.empty((nrow, ncol), dtype=object)
    ref_ax = None
    for i in range(nrow):
        for j in range(ncol):
            kwargs = {"sharex": ref_ax, "sharey": ref_ax} if ref_ax is not None else {}
            axes[i, j] = fig.add_subplot(gs[i, j], **kwargs)
            if ref_ax is None:
                ref_ax = axes[i, j]
    for idx, (ax, f) in enumerate(zip(axes.ravel(), files)):
        row, col = divmod(idx, ncol)
        d = np.load(f)
        fs.make_plain_map_ax(ax, left_labels=(col == 0), bottom_labels=(row == nrow - 1))
        if hasattr(ax, "_scale_ratio_text"):
            txt = ax._scale_ratio_text
            txt._compact_scale_ratio = True
            txt.set_position((0.975, 0.025))
            txt.set_fontsize(4.5)
            txt.set_alpha(0.75)
            txt.set_path_effects([pe.withStroke(linewidth=0.9, foreground="white")])
        pm = ax.pcolormesh(d["lon"], d["lat"], d["dens"], cmap=fs.SEQ,
                           vmin=0, vmax=vmax)
        # Decade and valid-year labels are moved to the caption; the panel is
        # identified only by its letter in the figure.
        fs.panel_letter(ax, "abcdefghijklmnopqrstuvwxyz"[idx])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    cax = fig.add_subplot(gs[nrow, :])
    fig.colorbar(pm, cax=cax, orientation="horizontal",
                 label="Points Grid$^{-1}$ yr$^{-1}$")
    _figure_title(fig, "Decadal Evolution of Tropical-Cyclone Track Density")
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.98,
                        wspace=0.18, hspace=0.12)
    fs.save(fig, "p2_fig01_density_decadal.png")


# ---------------- P2-2 路径密度的 ENSO 位相差异 ----------------
def fig02():
    order = ["El_Nino", "La_Nina", "Neutral"]
    phase_labels = {"El_Nino": "El Niño", "La_Nina": "La Niña", "Neutral": "Neutral"}
    dat = {ph: np.load(f"{P}/density_phase_{ph}.npz") for ph in order
           if os.path.exists(f"{P}/density_phase_{ph}.npz")}
    vmax = max(d["dens"].max() for d in dat.values())
    fig = plt.figure(figsize=(fs.COL2 * 1.12, fs.COL2 * 0.66))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.075],
                          wspace=0.18, hspace=0.16)
    axis_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axes = []
    ref_ax = None
    for row, col in axis_positions:
        kwargs = {"sharex": ref_ax, "sharey": ref_ax} if ref_ax is not None else {}
        ax = fig.add_subplot(gs[row, col], **kwargs)
        axes.append(ax)
        if ref_ax is None:
            ref_ax = ax
    axes = np.array(axes)
    for ax, ph, (row, col) in zip(axes, order, axis_positions):
        d = dat[ph]
        fs.make_plain_map_ax(ax, left_labels=(col == 0), bottom_labels=(row == 1))
        pm = ax.pcolormesh(d["lon"], d["lat"], d["dens"], cmap=fs.SEQ, vmin=0,
                           vmax=vmax)
        fs.panel_letter(ax, "abc"[len([p for p in axis_positions if p[0] < row or (p[0] == row and p[1] < col)])])
    cax_seq = fig.add_subplot(gs[2, 0])
    fig.colorbar(pm, cax=cax_seq, orientation="horizontal",
                 label="Points Grid$^{-1}$ yr$^{-1}$")
    # 第四面板：El Niño − La Niña 差值（发散色标）。
    diff = dat["El_Nino"]["dens"] - dat["La_Nina"]["dens"]
    lim = np.nanmax(np.abs(diff))
    fs.make_plain_map_ax(axes[3], left_labels=False, bottom_labels=True)
    pmd = axes[3].pcolormesh(dat["El_Nino"]["lon"], dat["El_Nino"]["lat"], diff,
                             cmap=fs.DIV, vmin=-lim, vmax=lim)
    fs.panel_letter(axes[3], "d")
    cax_diff = fig.add_subplot(gs[2, 1])
    fig.colorbar(pmd, cax=cax_diff, orientation="horizontal",
                 label="El Niño − La Niña (Points Grid$^{-1}$ yr$^{-1}$)")
    _figure_title(fig, "ENSO-Phase Differences in Tropical-Cyclone Track Density")
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.98,
                        wspace=0.18, hspace=0.16)
    fs.save(fig, "p2_fig02_density_enso.png")


# ---------------- P2-3 路径分型结果 ----------------
def fig03():
    cl = pd.read_csv(f"{P}/p2_clusters.csv")
    tr = pd.read_csv(f"{P}/tracks.csv", parse_dates=["iso_time"])
    tr = tr.merge(cl[["sid", "cluster"]], on="sid", how="inner")
    k = int(cl["cluster"].max()) + 1
    fig = plt.figure(figsize=(fs.COL2 * 1.18, fs.COL1 * 1.08))
    # 大量轨迹集合在部分 Cartopy/Shapely 版本组合下会破坏 gridliner 边界；
    # Plate Carrée 的普通坐标轴在此等价且渲染更稳定。
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    axm = fig.add_subplot(gs[0, 0])
    fs.make_plain_map_ax(axm)
    for c, g in tr.groupby("cluster"):
        col = fs.MONET[int(c) % len(fs.MONET)]
        # 每个簇只创建一个集合图元，避免数千条轨迹逐条触发 Cartopy 坐标转换。
        # 轨迹层在 PDF 中栅格化；文字、坐标轴和海岸线仍保持矢量。
        segments = [t[["lon", "lat"]].to_numpy()
                    for _, t in g.groupby("sid") if len(t) >= 2]
        lines = LineCollection(
            segments, linewidths=0.3, alpha=0.25, colors=col,
            rasterized=True, zorder=1)
        axm.add_collection(lines, autolim=False)
    axm.set_title(f"Track Clusters (k={k})")
    axb = fig.add_subplot(gs[0, 1])
    sizes = cl["cluster"].value_counts().sort_index()
    axb.bar(sizes.index, sizes.values,
            color=[fs.MONET[i % len(fs.MONET)] for i in sizes.index])
    axb.set_xlabel("Cluster")
    axb.set_ylabel("Storms")
    axb.yaxis.tick_right()
    axb.yaxis.set_label_position("right")
    axb.spines["right"].set_visible(True)
    axb.spines["left"].set_visible(False)
    fs.panel_letter(axm, "a")
    fs.panel_letter(axb, "b", dx=-0.08, dy=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.17, top=0.98, wspace=0.22)
    fs.save(fig, "p2_fig03_clusters.png")


# ---------------- P2-4 各型对应环流合成 ----------------
def fig04():
    cl = pd.read_csv(f"{P}/p2_clusters.csv")
    s = pd.read_csv(f"{P}/storms.csv", parse_dates=["genesis_time"]).merge(
        cl[["sid", "cluster"]], on="sid")
    st = xr.open_dataset(f"{I}/steering.nc")
    plev = xr.open_dataset(f"{I}/era5_wnp_dynamic_plev.nc")
    z500 = plev["z"].sel(level=500) / 9.80665 / 10.0           # dagpm
    # 三个场各解压一次；逐簇重复从压缩 NetCDF 读取会把相同数据重读 k 次。
    u_all = st["u_steer"].load()
    v_all = st["v_steer"].load()
    z_all = z500.load()
    st.close()
    plev.close()
    st_time = pd.to_datetime(u_all.time.values)
    z_time = pd.to_datetime(z_all.time.values)
    k = int(cl["cluster"].max()) + 1
    fig = plt.figure(figsize=(fs.COL2, fs.COL2 * 1.05))
    gs = fig.add_gridspec(4, 2, hspace=0.10, wspace=0.08)
    axes = [fig.add_subplot(gs[c // 2, c % 2]) for c in range(k)]
    empty_ax = fig.add_subplot(gs[3, 1])
    empty_ax.axis("off")
    for c in range(k):
        ax = axes[c]
        fs.make_plain_map_ax(
            ax,
            left_labels=(c % 2 == 0),
            bottom_labels=(c in (5, 6)))
        ts = pd.to_datetime(s[s.cluster == c]["genesis_time"])
        # 取成员 TC 生成所在的年-月，在月场上合成。
        keys = set(zip(ts.dt.year, ts.dt.month))
        sel = np.array([(t.year, t.month) in keys for t in st_time])
        u = u_all.isel(time=sel).mean("time")
        v = v_all.isel(time=sel).mean("time")
        selz = np.array([(t.year, t.month) in keys for t in z_time])
        z = z_all.isel(time=selz).mean("time")
        step = 6
        ax.quiver(u.longitude.values[::step], u.latitude.values[::step],
                  u.values[::step, ::step], v.values[::step, ::step],
                  scale=200, width=0.0035, color="#34707F", alpha=0.72)
        cs = ax.contour(z.longitude, z.latitude, z.values, levels=np.arange(580, 596, 2),
                        colors="#7C3E50", linewidths=0.65, zorder=5)
        cs588 = ax.contour(z.longitude, z.latitude, z.values, levels=[588],
                           colors="#7C3E50", linewidths=1.6, zorder=6)
        labels = ax.clabel(cs, fmt="%d", fontsize=6.0, inline=True, zorder=7)
        for txt in labels:
            txt.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
        ax.set_title(f"Cluster {c} (n={int((s.cluster==c).sum())})")
        fs.panel_letter(ax, chr(ord("a") + c), dx=0.02, dy=0.96)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.985,
                        hspace=0.10, wspace=0.08)
    fs.save(fig, "p2_fig04_cluster_circulation.png")


# ---------------- P2-5 副高四项指数年际变化与趋势 ----------------
def fig05():
    w = pd.read_csv(f"{P}/p2_wnpsh.csv")
    trend_start = cfg["periods"]["intensity_start"]
    # Keep the full variable meaning on the axes, but wrap long labels so the
    # title-free manuscript layout does not push them beyond the canvas.
    cols = [("wpsh_area", "WNPSH Area\n($10^6$ km²)"),
            ("wpsh_intensity", "WNPSH\nIntensity"),
            ("ridge_line", "Ridge Latitude\n(°N)"),
            ("west_ridge_point", "West Ridge Point\n(°E)")]
    fig, axes = plt.subplots(2, 2, figsize=(fs.COL2, fs.COL2 * 0.66))
    min_yr = w["season"].min()
    max_yr = w["season"].max()
    early = w[w["season"] < trend_start]
    modern = w[w["season"] >= trend_start]
    trend_rows = []

    for ax, (c, title) in zip(axes.ravel(), cols):
        ax.set_title(title)
        # Preserve the long ERA5 series as visual context, but restrict the
        # inferential trend to the TC intensity-homogeneous period (1982+).
        ax.axvspan(min_yr - 2, trend_start - 0.5, color="#B8B8B8",
                   alpha=0.08, lw=0, zorder=0)
        ax.plot(early["season"], early[c], color="#A9A9A9", lw=1.15,
                alpha=0.82, zorder=2)
        ax.plot(modern["season"], modern[c], color=fs.MONET[0], lw=1.55,
                zorder=3)
        ax.axvline(trend_start, color="#777777", ls=":", lw=0.9,
                   alpha=0.9, zorder=1)

        valid = modern[["season", c]].dropna()
        slope, lo, hi = theil_sen_ci(
            valid["season"], valid[c],
            nboot=cfg["statistics"]["bootstrap_samples"],
            block=cfg["statistics"]["bootstrap_block"],
            seed=cfg["statistics"]["random_seed"])
        intercept = valid[c].mean() - slope * valid["season"].mean()
        xs = np.array([valid["season"].min(), valid["season"].max()])
        ax.plot(xs, slope * xs + intercept, "--", lw=1.15,
                color="#7C3E50", zorder=4)
        ax.set_ylabel(title)
        ax.set_xlim(min_yr - 2, max_yr + 2)
        trend_rows.append({
            "variable": c,
            "start": int(valid["season"].min()),
            "end": int(valid["season"].max()),
            "n": int(len(valid)),
            "sen_slope_per_year": slope,
            "sen_slope_per_decade": slope * 10,
            "sen_ci_lo_per_decade": lo * 10,
            "sen_ci_hi_per_decade": hi * 10,
        })
    for i, ax in enumerate(axes.ravel()):
        fs.panel_letter(ax, "abcd"[i])
        if i % 2 == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color="#A9A9A9", lw=1.3,
               label=f"{min_yr}–{trend_start - 1} (context only)"),
        Line2D([0], [0], color=fs.MONET[0], lw=1.6,
               label=f"{trend_start}–{max_yr} (trend period)"),
        Line2D([0], [0], color="#7C3E50", lw=1.15, ls="--",
               label=f"Theil–Sen trend ({trend_start}–{max_yr})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=7.6,
               handlelength=2.8, columnspacing=1.7)
    fig.supxlabel("Year", y=0.070)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.19, top=0.98,
                        hspace=0.20, wspace=0.16)

    pd.DataFrame(trend_rows).to_csv(
        fs.FIGDIR / "Fig11_wnpsh_trend_stats.csv", index=False)
    fs.save(fig, "Fig11_wnpsh.png")
    for ext in ("png", "pdf"):
        shutil.copyfile(
            fs.FIGDIR / f"Fig11_wnpsh.{ext}",
            fs.FIGDIR / f"p2_fig05_wnpsh.{ext}")


# ---------------- P2-6 路径指标年际变化与趋势 ----------------
def fig06():
    m = pd.read_csv(f"{P}/p2_metrics.csv")
    g = m.groupby("season").agg(lmi_lat=("lmi_lat", "mean"),
                                recurv=("recurving", "mean"),
                                speed=("trans_speed", "mean"),
                                landfalls=("n_landfall", "sum")).reset_index()
    panels = [
        ("lmi_lat", "Mean LMI Latitude (°N)", cfg["periods"]["intensity_start"]),
        ("recurv", "Recurving Fraction", cfg["periods"]["freq_start"]),
        ("speed", "Mean Translation Speed (km/h)", cfg["periods"]["freq_start"]),
        ("landfalls", "Total Landfalls", cfg["periods"]["freq_start"]),
    ]
    trend_stats = _p2_fig06_stats(g, panels)
    trend_stats.to_csv(f"{P}/p2_fig06_trend_stats.csv", index=False)
    stats_by_var = trend_stats.set_index("var").to_dict("index")

    def fmt_p(v):
        if not np.isfinite(v):
            return "NA"
        return "<0.001" if v < 0.001 else f"{v:.3f}"

    def fmt_num(v, nd=2):
        return "NA" if not np.isfinite(v) else f"{v:.{nd}f}"

    fig, axes = plt.subplots(2, 2, figsize=(fs.COL2, fs.COL2 * 0.76))
    min_yr = cfg["periods"]["record_start"]
    max_yr = g["season"].max()
    for i, (ax, (c, title, start)) in enumerate(zip(axes.ravel(), panels)):
        display = g[g.season >= min_yr].dropna(subset=[c])
        early = display[display.season < start]
        ax.axvspan(min_yr - 2, start - 0.5, color="#B8B8B8",
                   alpha=0.08, lw=0, zorder=0)
        if len(early):
            ax.plot(early["season"], early[c], color="#A9A9A9",
                    lw=1.15, alpha=0.82, zorder=2)
        sub = display[display.season >= start]
        st = stats_by_var[c]
        ax.plot(sub["season"], sub[c], color=fs.MONET[0], lw=1.55,
                zorder=3)
        ax.axvline(start, color="#777777", ls=":", lw=0.9,
                   alpha=0.9, zorder=1)
        slope = st["sen_slope_per_year"]
        intercept = sub[c].mean() - slope * sub["season"].mean()
        xs = np.array([sub["season"].min(), sub["season"].max()])
        ax.plot(xs, slope * xs + intercept, "--", lw=1.15,
                color="#7C3E50", zorder=4)
        ax.set_ylabel(title)
        ax.set_xlim(min_yr - 2, max_yr + 2)
    for i, ax in enumerate(axes.ravel()):
        fs.panel_letter(ax, "abcd"[i])
        if i % 2 == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
    handles = [
        Line2D([0], [0], color="#A9A9A9", lw=1.3,
               label="Early record (context only)"),
        Line2D([0], [0], color=fs.MONET[0], lw=1.6,
               label="Trend analysis period"),
        Line2D([0], [0], color="#7C3E50", lw=1.15, ls="--",
               label="Theil–Sen trend"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, fontsize=7.6, frameon=False, handlelength=2.8,
               columnspacing=1.7)
    fig.supxlabel("Year", y=0.070)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.17, top=0.98,
                        wspace=0.20, hspace=0.48)
    fs.save(fig, "FigS13_metrics_trends.png")


# ---------------- P2-7 路径指标对环流指数的分布匹配回归 ----------------
def fig07():
    stats = pd.read_csv(f"{P}/p2_stats.csv")
    models = pd.read_csv(f"{P}/p2_models.csv")
    nonconst = models["term"] != "const"
    models["q_fdr"] = np.nan
    models.loc[nonconst, "q_fdr"] = _bh_fdr(models.loc[nonconst, "p"])
    d = stats[stats.scope == "annual"]
    panels = [("speed", "Translation Speed (km/h)"), ("lmi_lat", "LMI Latitude (°N)"),
              ("recurv_ratio", "Recurving Fraction"), ("landfalls", "Landfalls")]
    fig, axes = plt.subplots(2, 2, figsize=(fs.COL2, fs.COL2 * 0.60), sharex=True)
    for ax, (resp, title) in zip(axes.ravel(), panels):
        x = d["west_ridge_point"]
        y = d[resp]
        ax.scatter(x, y, s=10, color=fs.MONET[0], alpha=0.7)
        # 标注对应模型在 west_ridge_point 上的标准化系数（来自 p2_models）。
        mr = models[(models.scope == "annual") & (models.term == "west_ridge_point")]
        lbl = ""
        key = {"speed": "speed", "lmi_lat": "lmi_lat", "recurv_ratio": "recurving",
               "landfalls": "landfalls"}[resp]
        row = mr[mr.response == key]
        if len(row):
            lbl = f"β*={row.coef.iloc[0]:.2f}, p={row.p.iloc[0]:.3f}, q={row.q_fdr.iloc[0]:.3f}"
        ax.set_title(f"{title}\n{lbl}")
        ax.set_ylabel(title)
    for i, ax in enumerate(axes.ravel()):
        fs.panel_letter(ax, "abcd"[i])
        if i % 2 == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
    fig.supxlabel("West Ridge Point (°E)", y=0.025)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.15, top=0.98,
                        wspace=0.14, hspace=0.30)
    fs.save(fig, "p2_fig07_metrics_vs_wpsh.png")


# ---------------- P2-8 ENSO 位相环流合成差 + 场显著性 ----------------
def fig08(var="z500"):
    d = np.load(f"{P}/p2_fieldsig.npz", allow_pickle=True)
    lon, lat = d["lon"], d["lat"]
    obs = d[f"{var}_obs"]
    sig = d[f"{var}_sig"]
    fp = float(d[f"{var}_field_p"])
    fig = plt.figure(figsize=(fs.COL2 * 0.95, fs.COL2 * 0.64))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.045], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    fs.make_plain_map_ax(ax)
    lim = np.nanmax(np.abs(obs))
    pm = ax.contourf(lon, lat, obs, levels=13, cmap=fs.DIV, vmin=-lim, vmax=lim,
                     extend="both")
    fs.stipple(ax, lon, lat, sig)
    ax.set_title(f"{var.upper()} Composite (El Niño - La Niña), Field p={fp:.3f}")
    cax = fig.add_subplot(gs[0, 1])
    unit_labels = {
        "z500": "Z500 Difference (m; El Niño − La Niña)",
        "u": "Zonal-Wind Difference (m s$^{-1}$; El Niño − La Niña)",
        "v": "Meridional-Wind Difference (m s$^{-1}$; El Niño − La Niña)",
        "shear": "Vertical-Shear Difference (m s$^{-1}$; El Niño − La Niña)",
    }
    fig.colorbar(pm, cax=cax, label=unit_labels.get(var, f"{var.upper()} Difference"))
    fig.suptitle("ENSO Circulation Composite and Field Significance",
                 y=0.965, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.14, top=0.82)
    fs.save(fig, "p2_fig08_enso_circulation.png")


# ---------------- P2-9 分海岸登陆事件构成的十年际变化 ----------------
def fig09():
    lf = pd.read_csv(f"{P}/landfalls.csv", parse_dates=["time"])
    start_year, end_year = 1945, 2025
    lf = lf.loc[lf.time.dt.year.between(start_year, end_year)].copy()
    lf["decade"] = (lf.time.dt.year // 10) * 10
    decades = np.arange(1940, 2021, 10)
    piv = lf.pivot_table(index="decade", columns="coast", values="sid",
                         aggfunc="size", fill_value=0).reindex(
                             decades, fill_value=0)
    totals = piv.sum(axis=1)
    shares = piv.div(totals.replace(0, np.nan), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(fs.COL2, fs.COL1 * 0.82))
    bottom = np.zeros(len(shares))
    coast_labels = {
        "China_E": "East China Coast",
        "China_S": "South China Coast",
        "Taiwan": "Taiwan Island",
        "Japan": "Japanese Archipelago",
        "Korea": "Korean Peninsula",
        "Philippines": "Philippine Archipelago",
        "Vietnam": "Vietnam Coast",
    }
    coast_order = [
        "China_E", "China_S", "Taiwan", "Japan", "Korea",
        "Philippines", "Vietnam"
    ]
    # Match the coast-mask colors used in Fig01_study_domains_redrawn.
    # Keep "Other" neutral because it has no corresponding mask in Fig. 1.
    coast_colors = {
        "China_E": "#4E9BC0",
        "China_S": "#7DB8D0",
        "Taiwan": "#8D73B8",
        "Japan": "#79AD72",
        "Korea": "#CF7C7C",
        "Philippines": "#D6A824",
        "Vietnam": "#B862A9",
        "Other": "#8A93A6",
    }
    plot_order = [c for c in coast_order if c in shares.columns]
    plot_order += [
        c for c in shares.columns if c not in set(coast_order + ["Other"])
    ]
    if "Other" in shares.columns:
        plot_order.append("Other")
    for i, coast in enumerate(plot_order):
        ax.bar(np.arange(len(shares)), shares[coast], bottom=bottom,
               label=coast_labels.get(coast, coast),
               color=coast_colors.get(coast, fs.MONET[i % len(fs.MONET)]))
        bottom += shares[coast].to_numpy()
    decade_labels = [f"{y}s" for y in decades]
    ax.set_xticks(np.arange(len(decades)), decade_labels)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Share of landfall events (%)")
    ax.set_ylim(0, 106)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for x, total in enumerate(totals.to_numpy()):
        ax.text(x, 101.4, f"N={int(total)}", ha="center", va="bottom",
                fontsize=6.4, color="#4A4A4A")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncol=1, fontsize=6)
    fig.subplots_adjust(left=0.11, right=0.80, bottom=0.20, top=0.94)
    fs.save(fig, "p2_fig09_landfall_by_coast.png")
    for ext in ("png", "pdf"):
        shutil.copyfile(
            fs.FIGDIR / f"p2_fig09_landfall_by_coast.{ext}",
            fs.FIGDIR / f"FigS16_landfall_by_coast.{ext}")


# ---------------- P2-10 生成/路径/消亡 三联密度 ----------------
def fig10():
    d = np.load(f"{P}/p2_triptych_density.npz")
    panels = [("genesis", "Genesis"), ("track", "Track (Tropical TS)"),
              ("tropical_end", "Tropical End")]
    fig = plt.figure(figsize=(fs.COL2 * 1.24, fs.COL1 * 0.75))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.09],
                          hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    caxes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, (ax, cax, (key, title)) in enumerate(zip(axes, caxes, panels)):
        fs.make_plain_map_ax(ax, left_labels=(i == 0))
        if i == 0:
            ax.set_xticks([100, 120, 140, 160])
        elif i == 1:
            ax.set_xticks([120, 140, 160])
        else:
            ax.set_xticks([120, 140, 160, 180])
        field = d[key]
        pm = ax.pcolormesh(d["lon"], d["lat"], field, cmap=fs.SEQ,
                           vmin=0)
        ax.set_title(title)
        ax.tick_params(labelsize=6, pad=1)
        cb = fig.colorbar(pm, cax=cax, orientation="horizontal",
                          label=f"{title} Density")
        cb.ax.tick_params(labelsize=6, pad=1)
        fs.panel_letter(ax, "abc"[i], dx=0.02, dy=0.95)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.98,
                        wspace=0.12, hspace=0.12)
    fig.canvas.draw()
    for ax, cax in zip(axes, caxes):
        map_pos = ax.get_position()
        cax.set_position([map_pos.x0, map_pos.y0 - 0.095,
                          map_pos.width, 0.028])
    fs.save(fig, "p2_fig10_density_triptych.png")


# ---------------- P2-11 移速/路径长度/生命期 分布 ----------------
def fig11():
    d = pd.read_csv(f"{P}/p2_distributions.csv")
    panels = [("trans_speed", "Translation Speed (km/h)"),
              ("track_len_km", "Track Length (km)"), ("lifetime_h", "Lifetime (h)")]
    fig = plt.figure(figsize=(fs.COL2 * 0.86, fs.COL2 * 0.68))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.18)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
    ]
    empty_ax = fig.add_subplot(gs[1, 1])
    empty_ax.axis("off")
    for ax, (c, title) in zip(axes, panels):
        v = d[c].dropna()
        ax.hist(v, bins=30, color=fs.MONET[0], alpha=0.85)
        med = v.median()
        ax.axvline(med, color="#7C3E50", lw=1.2, ls="--")
        ax.set_title(f"{title}\nMedian = {med:.0f}")
        ax.set_xlabel(title)
        ax.set_ylabel("Storms")
    for i, ax in enumerate(axes):
        fs.panel_letter(ax, "abc"[i])
        if i == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.98,
                        hspace=0.38, wspace=0.18)
    fs.save(fig, "p2_fig11_distributions.png")


# ---------------- P2-12 转向点气候态与转向比例 ----------------
def fig12():
    dens = np.load(f"{P}/p2_recurve_density.npz")
    monthly = pd.read_csv(f"{P}/p2_recurve_monthly.csv")
    annual = pd.read_csv(f"{P}/p2_recurve_annual.csv")
    annual.columns = ["season", "recurv"]
    fig = plt.figure(figsize=(fs.COL2 * 0.90, fs.COL2 * 0.82))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.10, 1],
                          hspace=0.30, wspace=0.18)
    axm = fig.add_subplot(gs[0, 0])
    fs.make_plain_map_ax(axm)
    lonc = (dens["lon"][:-1] + dens["lon"][1:]) / 2
    latc = (dens["lat"][:-1] + dens["lat"][1:]) / 2
    pm = axm.pcolormesh(lonc, latc, dens["dens"], cmap=fs.SEQ, vmin=0)
    axm.set_title("Recurvature Points")
    cax = fig.add_subplot(gs[1, 0])
    fig.colorbar(pm, cax=cax, orientation="horizontal",
                 label="Recurvature-Point Density")
    axb = fig.add_subplot(gs[0, 1])
    axb.bar(monthly["month"], monthly["genesis_cohort_ratio"], color=fs.MONET[1])
    axb.set_xlabel("Genesis Month")
    axb.set_ylabel("Final Recurving Ratio")
    axb.set_xlim(0.5, 12.5)
    axb.set_xticks(range(1, 13))
    axb.yaxis.tick_right()
    axb.yaxis.set_label_position("right")
    axb.spines["right"].set_visible(True)
    axb.spines["left"].set_visible(False)
    axl = fig.add_subplot(gs[2, 0])
    empty_ax = fig.add_subplot(gs[2, 1])
    empty_ax.axis("off")
    _sen(axl, annual["season"], annual["recurv"])
    axl.set_title("Annual Recurving Ratio")
    axl.set_xlabel("Year")
    axl.set_ylabel("Annual Recurving Ratio")
    for ax, s in zip([axm, axb, axl], "abc"):
        fs.panel_letter(ax, s)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.11, top=0.98,
                        wspace=0.18, hspace=0.30)
    fs.save(fig, "p2_fig12_recurvature.png")


# ---------------- P2-13 登陆气候学（比例/强度/海岸） ----------------
def fig13():
    frac = pd.read_csv(f"{P}/p2_landfall_frac.csv")
    lf = pd.read_csv(f"{P}/landfalls.csv")
    bycoast = pd.read_csv(f"{P}/p2_landfall_by_coast.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(fs.COL2 * 1.18, fs.COL1 * 1.12),
        gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 0.32})
    _sen(axes[0], frac["season"], frac["landfall_frac"])
    axes[0].set_ylabel("Landfalling-Storm Fraction")
    axes[0].set_xlabel("Year")
    axes[1].hist(lf["wind"].dropna(), bins=25, color=fs.MONET[0], alpha=0.85)
    axes[1].set_ylabel("Landfall Intensity")
    axes[1].set_xlabel("Wind (kt)")
    cnt = bycoast.groupby("coast")["event_count"].sum().sort_values(ascending=False)
    inten = bycoast.groupby("coast")["wind_mean"].mean().reindex(cnt.index)
    coast_labels = {
        "China_E": "East China\nCoast",
        "China_S": "South China\nCoast",
        "Taiwan": "Taiwan\nIsland",
        "Japan": "Japanese\nArchipelago",
        "Korea": "Korean\nPeninsula",
        "Philippines": "Philippine\nArchipelago",
        "Vietnam": "Vietnam\nCoast",
    }
    ax2 = axes[2]
    ax2.bar(range(len(cnt)), cnt.values, color=fs.MONET[2])
    ax2.set_xticks(range(len(cnt)))
    tick_labels = ax2.set_xticklabels([coast_labels.get(c, c) for c in cnt.index],
                                      rotation=45, fontsize=6.5, ha="right")
    for label in tick_labels:
        label.set_x(label.get_position()[0] + 0.08)
    ax2.set_ylabel("Events")
    ax3 = ax2.twinx()
    mean_wind_color = "#7C3E50"
    ax3.plot(range(len(cnt)), inten.values, "o-", color=mean_wind_color, ms=3, lw=1)
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_color(mean_wind_color)
    ax3.yaxis.set_ticks_position("right")
    ax3.tick_params(axis="y", colors=mean_wind_color)
    ax3.set_ylabel("Mean Wind (kt)", color=mean_wind_color)
    for ax, s in zip(axes, "abc"):
        fs.panel_letter(ax, s)
    _figure_title(fig, "Landfall Climatology", top=0.88)
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.30, top=0.98, wspace=0.32)
    fs.save(fig, "p2_fig13_landfall_clim.png")


# ---------------- P2-14 各分型平均路径与季节/位相占比 ----------------
def fig14():
    mt = np.load(f"{P}/p2_cluster_meantracks.npz")
    monthly = pd.read_csv(f"{P}/p2_cluster_monthly.csv")
    phase = pd.read_csv(f"{P}/p2_cluster_phase.csv")
    mcols = [c for c in monthly.columns if c.startswith("membership_c")]
    k = len(mcols)
    fig = plt.figure(figsize=(fs.COL2 * 0.90, fs.COL2 * 0.72))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.20)
    axm = fig.add_subplot(gs[0, 0])
    fs.make_plain_map_ax(axm)
    scale_txt = getattr(axm, "_scale_ratio_text", None)
    if scale_txt is not None:
        scale_txt.set_position((0.98, 0.94))
        scale_txt.set_va("top")
    for j in range(k):
        if f"c{j}_lon" in mt:
            axm.plot(mt[f"c{j}_lon"], mt[f"c{j}_lat"], lw=2,
                     color=fs.MONET[j % len(fs.MONET)], label=f"c{j}")
    axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
               ncol=4, fontsize=6, columnspacing=1.0, handlelength=1.6)
    axm.set_title("Mean Cluster Tracks")
    axb = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(monthly))
    for j, c in enumerate(mcols):
        axb.bar(monthly["month"], monthly[c], bottom=bottom, width=0.72,
                color=fs.MONET[j % len(fs.MONET)], label=c)
        bottom += monthly[c].values
    axb.set_xlabel("Month")
    axb.set_ylabel("Composition")
    axb.yaxis.tick_right()
    axb.yaxis.set_label_position("right")
    axb.spines["right"].set_visible(True)
    axb.spines["left"].set_visible(False)
    # Reserve one empty month slot on both sides, keeping label b clear while
    # giving December the same breathing room as January.
    axb.set_xlim(-0.5, 13.5)
    axb.set_xticks(range(1, 13))
    axp = fig.add_subplot(gs[1, 0])
    empty_ax = fig.add_subplot(gs[1, 1])
    empty_ax.axis("off")
    pcol = phase.columns[0]
    bottom = np.zeros(len(phase))
    phase_labels = {
        "El_Nino": "El Niño",
        "La_Nina": "La Niña",
        "Neutral": "Neutral",
    }
    phase_x = [phase_labels.get(str(v), str(v).replace("_", " ")) for v in phase[pcol]]
    for j, c in enumerate(mcols):
        axp.bar(phase_x, phase[c], bottom=bottom, width=0.68,
                color=fs.MONET[j % len(fs.MONET)])
        bottom += phase[c].values
    axp.set_ylabel("Composition")
    axp.tick_params(axis="x", rotation=30)
    axp.set_xlim(-0.82, len(phase) - 0.5)
    fs.panel_letter(axm, "a", dx=0.02, dy=0.95)
    fs.panel_letter(axb, "b", dx=0.01, dy=0.96)
    fs.panel_letter(axp, "c", dx=0.01, dy=0.96)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.14, top=0.98,
                        hspace=0.45, wspace=0.20)
    fs.save(fig, "p2_fig14_cluster_profiles.png")


# ---------------- Supplementary S25 路径空间重分配与登陆构成 ----------------
def _revision_data_dir():
    return Path(P).parents[1] / "revision_outputs" / "data"


def _revision_summary_row(summary, analysis, weighting):
    selected = summary.loc[
        (summary["analysis"] == analysis)
        & (summary["weighting"] == weighting)
        & (summary["period_definition"] == "primary_equal_halves")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {analysis}/{weighting}, found {len(selected)}"
        )
    return selected.iloc[0]


def figS25():
    """正式补充图S25；无整图/小图标题，仅保留(a)-(d)。"""
    mod = importlib.import_module("paper2_dynamic.28_spatial_redistribution")
    data_dir = _revision_data_dir()
    arrays = np.load(data_dir / "p2_redistribution_annual.npz")
    summary = pd.read_csv(data_dir / "p2_redistribution_summary.csv")
    years = arrays["years"]
    early = years <= 1995
    late = years >= 1996

    point = arrays["point_relative_density"].reshape(len(years), -1)
    path_row = _revision_summary_row(summary, "path_density", "track_point")
    early_path = np.nanmean(point[early], axis=0)
    late_path = np.nanmean(point[late], axis=0)
    path_result = {
        "early_mean": early_path,
        "late_mean": late_path,
        "change": late_path - early_path,
        "tv": float(path_row["total_variation"]),
        "global_p": float(path_row["block_permutation_p"]),
    }

    coast = arrays["coast_event_share"]
    coast_row = _revision_summary_row(summary, "landfall_coast", "event")
    coast_result = {
        "early_mean": np.nanmean(coast[early], axis=0),
        "late_mean": np.nanmean(coast[late], axis=0),
        "tv": float(coast_row["total_variation"]),
        "global_p": float(coast_row["block_permutation_p"]),
    }
    north_row = _revision_summary_row(
        summary, "north_vs_south_named_coast", "event"
    )
    mod._diagnostic_figure(
        Path(fs.FIGDIR),
        years,
        arrays["lon_edges"],
        arrays["lat_edges"],
        path_result,
        arrays["coast_names"].tolist(),
        coast_result,
        arrays["north_named_event_share"],
        {"global_p": float(north_row["block_permutation_p"])},
        save_diagnostic=False,
        save_formal=True,
    )


# ---------------- Supplementary S26 多机构敏感性 ----------------
def figS26():
    """正式补充图S26；无整图/小图标题，仅保留(a)-(d)。"""
    mod = importlib.import_module("paper2_dynamic.29_multiagency_sensitivity")
    data_dir = _revision_data_dir()
    mod._figure(
        Path(fs.FIGDIR),
        pd.read_csv(data_dir / "p2_multiagency_annual.csv"),
        pd.read_csv(data_dir / "p2_multiagency_redistribution.csv"),
        save_diagnostic=False,
        save_formal=True,
    )


# ---------------- Supplementary S27 移速诊断性分解 ----------------
def figS27():
    """正式补充图S27；无整图/小图标题，仅保留(a)-(d)。"""
    mod = importlib.import_module("paper2_dynamic.30_speed_decomposition_trial")
    data_dir = _revision_data_dir()
    existing = pd.read_csv(Path(P) / "p2_stats.csv")
    existing = existing.loc[
        (existing["scope"] == "annual")
        & existing["season"].between(mod.PROJECT_START, mod.PROJECT_END),
        ["season", "speed"],
    ]
    mod._diagnostic_figure(
        Path(fs.FIGDIR),
        pd.read_csv(data_dir / "p2_speed_decomposition_annual.csv"),
        pd.read_csv(data_dir / "p2_speed_decomposition_trends.csv"),
        existing,
        save_diagnostic=False,
        save_formal=True,
    )


FIGURES = [fig01, fig02, fig03, fig04, fig05, fig06, fig07,
           fig08, fig09, fig10, fig11, fig12, fig13, fig14,
           figS25, figS26, figS27]


def main():
    os.makedirs(str(fs.FIGDIR), exist_ok=True)
    for fn in FIGURES:
        try:
            fn()
        except Exception as e:                       # 单图失败不影响其余
            print(f"[FAIL] {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
