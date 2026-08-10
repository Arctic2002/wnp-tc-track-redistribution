from __future__ import annotations

import shutil
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from JClimate_Mainline.src.figure_typography import scale_figure_typography

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent
LEGACY = WORK / "archive" / "previous_v2" / "legacy_generated"
MAIN = LEGACY / "figures" / "Main"
SUPP = LEGACY / "figures" / "Supplementary"
DATA = WORK / "data"

COLORS = {"PRIMARY": "#222222", "USA": "#4069A1", "TOKYO": "#7F9FC5", "CMA": "#C66D6D"}
DISPLAY = {"PRIMARY": "Primary", "USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}


def add_coast(ax, extent):
    shp = PROJECT / "data" / "raw" / "GSHHG" / "GSHHS_shp" / "i" / "GSHHS_i_L1.shp"
    if shp.exists():
        import geopandas as gpd
        land = gpd.read_file(shp, bbox=extent)
        land.plot(ax=ax, facecolor="#E6E4DF", edgecolor="#505050", linewidth=0.35, zorder=3)
    ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])


def format_geo_axes(ax):
    """Use compact directional tick labels without redundant axis titles."""
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}°E")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}°N")
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def panel_letters(axes):
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        ax.text(0.015, 0.985, label, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold", zorder=10)


def open_panel_axis(ax, *, right=False):
    """Remove non-data-bearing spines while retaining the requested y-axis side."""
    ax.spines["top"].set_visible(False)
    if right:
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    else:
        ax.spines["right"].set_visible(False)


def save(fig, name, *, scale=1.10):
    scale_figure_typography(fig, scale=scale)
    fig.savefig(MAIN / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(MAIN / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3():
    fields = np.load(DATA / "jcli_genesis_propagation_fields.npz")
    annual = pd.read_csv(DATA / "jcli_redistribution_index_annual.csv")
    robust = pd.read_csv(DATA / "jcli_robustness_matrix.csv")
    lon_edges = fields["PRIMARY_lon_edges"]; lat_edges = fields["PRIMARY_lat_edges"]
    change = fields["PRIMARY_total_field"].reshape(len(lat_edges)-1, len(lon_edges)-1) * 100
    vmax = np.nanmax(np.abs(change))
    fig = plt.figure(figsize=(12.2, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.05, 0.95], width_ratios=[0.055, 1.0, 1.0]
    )
    cax = fig.add_subplot(gs[0, 0])
    axa = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[0, 2])
    axc = fig.add_subplot(gs[1, :])
    m = axa.pcolormesh(lon_edges, lat_edges, change, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    add_coast(axa, (100, 0, 180, 40))
    format_geo_axes(axa)
    colorbar = fig.colorbar(m, cax=cax, orientation="vertical")
    colorbar.set_label(
        "Late minus early share\n(percentage points per cell)",
        labelpad=9,
    )
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    for agency in ["PRIMARY", "USA", "TOKYO", "CMA"]:
        d = annual[(annual.agency == agency) & (annual.weighting == "track_point")].sort_values("year")
        axb.plot(d.year, d.index_oos, lw=1.05, color=COLORS[agency], label=DISPLAY[agency])
    axb.axhline(0, color="0.45", lw=0.7); axb.axvline(1995.5, color="0.55", ls="--", lw=0.8)
    axb.set(xlabel="Year", ylabel="Leave-one-year-out\nredistribution index"); axb.legend(frameon=False, ncol=2)
    # Keep panels a and b physically identical even though their data
    # coordinate ranges have different aspect ratios.
    axa.set_box_aspect(0.72)
    axb.set_box_aspect(0.72)
    axa.set_xticks([100, 120, 140, 160, 180])
    selected = []
    for label, query in [
        ("1° grid", "analysis == 'grid_primary' and catalog == 'PRIMARY' and weighting == 'track_point' and grid_deg == 1.0"),
        ("2.5° grid", "analysis == 'grid_primary' and catalog == 'PRIMARY' and weighting == 'track_point' and grid_deg == 2.5"),
        ("5° grid", "analysis == 'grid_primary' and catalog == 'PRIMARY' and weighting == 'track_point' and grid_deg == 5.0"),
        ("End 2024", "analysis == 'end_2024_drop_1995' and catalog == 'PRIMARY' and weighting == 'track_point'"),
        ("Start 1967", "analysis == 'start_1967_drop_1996' and catalog == 'PRIMARY' and weighting == 'track_point'"),
        ("Exclude 2020–2025", "analysis == 'exclude_2020_2025' and catalog == 'PRIMARY' and weighting == 'track_point'"),
        ("TY: point", "analysis == 'typhoon_threshold' and catalog == 'PRIMARY_TY' and weighting == 'track_point'"),
        ("TY: storm-equal", "analysis == 'typhoon_threshold' and catalog == 'PRIMARY_TY' and weighting == 'storm_equal'"),
    ]:
        row = robust.query(query).iloc[0]
        selected.append((label, row.block_permutation_p))
    labels, vals = zip(*selected)
    x = np.arange(len(labels)); bars = axc.bar(x, vals, color=["#8DA0CB" if p >= 0.05 else "#4C72B0" for p in vals])
    axc.axhline(0.05, color="#B44A4A", ls="--", lw=1.0)
    axc.set_xticks(x, labels, rotation=20, ha="right"); axc.set_ylabel("Block-permutation p")
    for bar, p in zip(bars, vals): axc.text(bar.get_x()+bar.get_width()/2, p+0.006, f"{p:.3f}", ha="center", fontsize=8)
    open_panel_axis(axb, right=True); open_panel_axis(axc)
    panel_letters([axa, axb, axc]); save(fig, "Fig03_track_redistribution_index_robustness", scale=1.20)


def fig4():
    fields = np.load(DATA / "jcli_genesis_propagation_fields.npz")
    summary = pd.read_csv(DATA / "jcli_genesis_propagation_summary.csv")
    lon_edges = fields["PRIMARY_lon_edges"]; lat_edges = fields["PRIMARY_lat_edges"]
    maps = [fields[f"PRIMARY_{k}_field"].reshape(len(lat_edges)-1, len(lon_edges)-1)*100 for k in ("total", "genesis", "propagation")]
    vmax = max(np.max(np.abs(x)) for x in maps)
    fig = plt.figure(figsize=(12, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.055])
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    cax = fig.add_subplot(gs[2, 0])
    for ax, arr in zip(axes.flat[:3], maps):
        m = ax.pcolormesh(lon_edges, lat_edges, arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        add_coast(ax, (100, 0, 180, 40))
        format_geo_axes(ax)
    open_panel_axis(axes[0, 1], right=True)
    # Panel b is a map: retain a complete geographic frame while its y-axis
    # labels remain on the outer (right) side of the two-column layout.
    axes[0, 1].spines["top"].set_visible(True)
    axes[0, 1].spines["left"].set_visible(True)
    fig.colorbar(m, cax=cax, orientation="horizontal", label="Contribution to late minus early share (percentage points per cell)")
    ax = axes[1, 1]
    s = summary[summary.catalog.isin(["PRIMARY", "USA", "TOKYO", "CMA"])].set_index("catalog").loc[["PRIMARY", "USA", "TOKYO", "CMA"]]
    x = np.arange(len(s)); width = 0.34
    g = s.genesis_projection_fraction*100; p = s.propagation_projection_fraction*100
    ax.bar(x-width/2, g, width, color="#4C78A8", label="Genesis distribution")
    ax.bar(x+width/2, p, width, color="#E07B39", label="Post-genesis propagation")
    lo = p - s.propagation_projection_fraction_boot_lo*100; hi = s.propagation_projection_fraction_boot_hi*100 - p
    ax.errorbar(x+width/2, p, yerr=[lo, hi], fmt="none", ecolor="0.25", capsize=3, lw=0.9)
    ax.set_xticks(x, [DISPLAY[name] for name in s.index]); ax.set_ylabel("Projection contribution (%)"); ax.set_ylim(0, 105)
    # Keep the key out of the bars and their uncertainty intervals.
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for panel in axes.flat:
        panel.set_box_aspect(0.58)
    open_panel_axis(ax, right=True)
    fig.canvas.draw()
    cax_pos = cax.get_position()
    cax.set_position(
        [cax_pos.x0, cax_pos.y0 - 0.004, cax_pos.width, cax_pos.height]
    )
    panel_letters(axes); save(fig, "Fig04_genesis_propagation_decomposition")


def fig5():
    fields = np.load(DATA / "jcli_circulation_fields.npz")
    coefs = pd.read_csv(DATA / "jcli_regression_models.csv")
    lon = fields["longitude"]; lat = fields["latitude"]
    vmax = max(np.max(np.abs(fields["raw_z_eddy_beta"])), np.max(np.abs(fields["detrended_z_eddy_beta"])))
    fig = plt.figure(figsize=(9.15, 7.1))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.105,
        right=0.985,
        bottom=0.085,
        top=0.975,
        width_ratios=[0.045, 1.0, 1.0],
        height_ratios=[1.08, 0.92],
        hspace=0.10,
        wspace=0.10,
    )
    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, :]),
    ]
    cax = fig.add_subplot(gs[0, 0])
    for ax, scale in zip(axes[:2], ["raw", "detrended"]):
        m = ax.pcolormesh(lon, lat, fields[f"{scale}_z_eddy_beta"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        step=2
        ax.quiver(lon[::step], lat[::step], fields[f"{scale}_u_beta"][::step,::step], fields[f"{scale}_v_beta"][::step,::step], color="0.15", scale=18, width=0.0022)
        mask = fields[f"{scale}_u_q"] < 0.05
        yy, xx = np.where(mask)
        if len(xx): ax.scatter(lon[xx], lat[yy], s=3, color="black", alpha=0.45)
        add_coast(ax, (80, 0, 180, 65))
        format_geo_axes(ax)
    axes[0].set_xticks([80, 100, 120, 140, 160])
    axes[1].set_xticks([100, 120, 140, 160, 180])
    axes[1].tick_params(axis="y", labelleft=False)
    colorbar = fig.colorbar(m, cax=cax, orientation="vertical")
    colorbar.set_label(
        "Regional-background-\nadjusted Z500\nregression\n(m per index SD)",
        labelpad=7,
    )
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    fig.canvas.draw()
    cax_pos = cax.get_position()
    cax.set_position(
        [cax_pos.x0 - 0.045, cax_pos.y0, cax_pos.width, cax_pos.height]
    )
    keep = coefs[(coefs.term != "const") & coefs.model.isin(["eddy_wnpsh_oni_pdo", "steering_oni_pdo"])].copy()
    keep = keep[keep.term.isin(["eddy_wnpsh_mean_m", "corridor_u_steer_ms", "jas_oni", "pdo"])]
    order = [
        ("raw", "eddy_wnpsh_mean_m", "Regional-background-adjusted\nZ500 index"), ("detrended", "eddy_wnpsh_mean_m", "Regional-background-adjusted\nZ500 index, detrended"),
        ("raw", "corridor_u_steer_ms", "Zonal steering flow"), ("detrended", "corridor_u_steer_ms", "Zonal steering flow, detrended"),
        ("raw", "pdo", "PDO"), ("detrended", "pdo", "PDO, detrended"),
    ]
    rows=[]
    for scale, term, label in order:
        r = keep[(keep.timescale==scale)&(keep.term==term)].iloc[0]
        rows.append((label, r.coefficient_standardized, r.ci_low, r.ci_high, r.q_bh_within_timescale))
    y=np.arange(len(rows)); beta=np.array([r[1] for r in rows]); lo=beta-np.array([r[2] for r in rows]); hi=np.array([r[3] for r in rows])-beta
    axes[2].errorbar(beta, y, xerr=[lo,hi], fmt="o", color="#365E8D", ecolor="0.25", capsize=3)
    axes[2].axvline(0,color="0.45",lw=0.8); axes[2].set_yticks(y,[r[0] for r in rows]); axes[2].invert_yaxis(); axes[2].set_xlabel("Standardized coefficient (HAC 95% CI)")
    for yi,r in enumerate(rows): axes[2].text(r[3]+0.02, yi, f"q={r[4]:.3f}", va="center", fontsize=8)
    open_panel_axis(axes[2])
    panel_letters(axes); save(fig, "Fig05_circulation_context")


def fig6():
    events = pd.read_csv(DATA / "jcli_landfall_unique_events.csv")
    summary = pd.read_csv(DATA / "jcli_landfall_unique_summary.csv")
    fig = plt.figure(figsize=(12, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes=[fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]
    s = summary[((summary.assignment_rule=="first_named") & summary.agency.isin(["PRIMARY","USA","TOKYO","CMA"])) | ((summary.agency=="PRIMARY") & (summary.assignment_rule=="strongest"))].copy()
    s["label"] = s.agency + "\n" + s.assignment_rule.str.replace("_", " ")
    x=np.arange(len(s)); bars=axes[0].bar(x,s.north_share_change_percentage_points,color="#7A9E7E")
    axes[0].set_xticks(x,s.label); axes[0].set_ylabel("North-share change (percentage points)")
    for bar,p in zip(bars,s.north_share_block_p): axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,f"p={p:.3f}",ha="center",fontsize=8)
    d=events[(events.agency=="PRIMARY")&(events.assignment_rule=="first_named")]
    coasts=["China_E","China_S","Japan","Korea","Other","Philippines","Taiwan","Vietnam"]
    arr=[]
    for period,a,b in [("1966–1995",1966,1995),("1996–2025",1996,2025)]:
        c=d[d.season.between(a,b)].coast.value_counts().reindex(coasts,fill_value=0); c=c/c.sum()*100
        arr.append(c)
    xx=np.arange(len(coasts)); w=.38
    axes[1].bar(xx-w/2,arr[0],w,label="1966–1995",color="#4C72B0"); axes[1].bar(xx+w/2,arr[1],w,label="1996–2025",color="#86A5CC")
    axes[1].set_xticks(xx,[x.replace("China_E","East China").replace("China_S","South China") for x in coasts],rotation=30,ha="right"); axes[1].set_ylabel("Share of uniquely assigned storms (%)"); axes[1].legend(frameon=False)
    s2=summary[summary.assignment_rule.isin(["first_any","first_named","strongest"]) & (summary.agency=="PRIMARY")]
    xx=np.arange(len(s2)); axes[2].bar(xx,s2.eight_category_tv,color="#B98563")
    axes[2].set_xticks(xx,s2.assignment_rule.str.replace("_"," ")); axes[2].set_ylabel("Eight-category total-variation distance")
    for i,p in enumerate(s2.eight_category_block_p): axes[2].text(i,s2.eight_category_tv.iloc[i]+0.003,f"p={p:.3f}",ha="center",fontsize=8)
    panel_letters(axes); save(fig,"Fig06_landfall_unique_sensitivity")


def copy_existing():
    pairs = [
        (PROJECT / "figures" / "Fig01_study_domains_redrawn.png", MAIN / "Fig01_study_region.png"),
        (PROJECT / "figures" / "FigS26_multiagency_sensitivity.png", MAIN / "Fig02_cross_agency_sensitivity.png"),
    ]
    for src,dst in pairs: shutil.copy2(src,dst)
    supplemental = [
        "Fig02_timeseries_trends.png", "Fig03_lmi_location.png", "Fig07_pi_clim_trend.png",
        "Fig08_gpi_decomp.png", "Fig09_enso_composite.png", "Fig10_cluster_circulation.png",
        "Fig11_wnpsh.png", "FigS25_spatial_redistribution.png", "FigS27_speed_decomposition.png",
    ]
    for name in supplemental:
        src=PROJECT/"figures"/name
        if src.exists(): shutil.copy2(src,SUPP/name)
    for name in ["jcli_diagnostic_evidence_overview.png","jcli_diagnostic_circulation_link.png"]:
        src=WORK/"figures"/name
        if src.exists(): shutil.copy2(src,SUPP/name)


def main():
    MAIN.mkdir(parents=True,exist_ok=True); SUPP.mkdir(parents=True,exist_ok=True)
    copy_existing(); fig3(); fig4(); fig5(); fig6()
    print("main figures",len(list(MAIN.glob('*.png'))),"supplementary",len(list(SUPP.glob('*.png'))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures", nargs="*", choices=["3", "4", "5"])
    args = parser.parse_args()
    if args.output_dir is not None:
        MAIN = args.output_dir
        MAIN.mkdir(parents=True, exist_ok=True)
    selected = args.figures or ["3", "4", "5"]
    for number, func in {"3": fig3, "4": fig4, "5": fig5}.items():
        if number in selected:
            func()
