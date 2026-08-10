from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import WORK, load_config
from .figure_typography import scale_figure_typography
from .stats import block_order, block_permutation_difference


def field_test(x, n_early, *, block, nperm, seed):
    x = np.asarray(x, float)
    early = np.arange(n_early)
    late = np.arange(n_early, len(x))
    change = x[late].mean(axis=0) - x[early].mean(axis=0)
    observed = 0.5 * np.abs(change).sum()
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = block_order(len(x), block, rng)
        trial = x[order[late]].mean(axis=0) - x[order[early]].mean(axis=0)
        exceed += 0.5 * np.abs(trial).sum() >= observed
    return observed, (exceed + 1) / (nperm + 1)


def scalar_test(years, values, early_years, late_years, cfg):
    table = pd.Series(np.asarray(values, float), index=np.asarray(years, int))
    selected = table.reindex(np.r_[early_years, late_years]).to_numpy(float)
    early = np.arange(len(early_years))
    late = np.arange(len(early_years), len(selected))
    return block_permutation_difference(selected, early, late, block=3,
                                        nperm=cfg["n_permutations"], seed=cfg["random_seed"])


def landfall_annual(events, years):
    events["time"] = pd.to_datetime(events["time"])
    first = events.sort_values("time").drop_duplicates(["agency", "sid"], keep="first")
    north = {"China_E", "Taiwan", "Korea", "Japan"}
    south = {"China_S", "Vietnam", "Philippines"}
    rows = []
    for agency in sorted(events["agency"].unique()):
        for year in years:
            all_group = events.loc[events["agency"].eq(agency) & events["season"].eq(year)]
            first_group = first.loc[first["agency"].eq(agency) & first["season"].eq(year)]
            named = all_group.loc[all_group["coast"].isin(north | south)]
            n = int(named["coast"].isin(north).sum())
            s = int(named["coast"].isin(south).sum())
            rows.append({"agency": agency, "season": year,
                         "first_landfall_mean_latitude": first_group["lat"].mean(),
                         "north_share_all_events": n / (n + s) if n + s else np.nan})
    return pd.DataFrame(rows)


def run():
    cfg = load_config()
    out = WORK / "analysis" / "02_cutpoint_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(WORK / "analysis" / "03_common_storms" / "annual_path_composition_2p5deg.npz")
    years = z["years"].astype(int)
    core = pd.read_csv(WORK / "analysis" / "03_common_storms" / "core_crossagency_annual.csv")
    events = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_events_exact.csv")
    landing = landfall_annual(events, years)
    landing.to_csv(out / "cutpoint_landfall_annual_inputs.csv", index=False)

    rows = []
    for scheme in ["full_record", "adjacent_25yr"]:
        for cutpoint in range(1991, 2002):
            if scheme == "full_record":
                early_years = np.arange(1966, cutpoint)
                late_years = np.arange(cutpoint, 2026)
            else:
                early_years = np.arange(cutpoint - 25, cutpoint)
                late_years = np.arange(cutpoint, cutpoint + 25)
            selected_years = np.r_[early_years, late_years]
            selected_idx = np.array([np.flatnonzero(years == year)[0] for year in selected_years])
            for agency in ["USA", "TOKYO", "CMA"]:
                tv, p = field_test(z[agency][selected_idx], len(early_years), block=3,
                                   nperm=cfg["n_permutations"], seed=cfg["random_seed"])
                rows.append({"scheme": scheme, "cutpoint": cutpoint, "agency": agency,
                             "metric": "path_total_variation", "effect": tv, "p_value": p,
                             "early_start": early_years[0], "early_end": early_years[-1],
                             "late_start": late_years[0], "late_end": late_years[-1]})
                lf = landing.loc[landing["agency"].eq(agency)]
                for column, metric in [("first_landfall_mean_latitude", "first_landfall_latitude_difference"),
                                       ("north_share_all_events", "north_share_difference")]:
                    effect, p = scalar_test(lf["season"], lf[column], early_years, late_years, cfg)
                    rows.append({"scheme": scheme, "cutpoint": cutpoint, "agency": agency,
                                 "metric": metric, "effect": effect, "p_value": p,
                                 "early_start": early_years[0], "early_end": early_years[-1],
                                 "late_start": late_years[0], "late_end": late_years[-1]})
                # LMI begins in 1982; use all available years within each requested window.
                lmi = core.loc[core["agency"].eq(agency) & core["mean_lmi_lat_full"].notna()]
                lmi_early = early_years[early_years >= 1982]
                lmi_late = late_years[late_years >= 1982]
                effect, p = scalar_test(lmi["season"], lmi["mean_lmi_lat_full"], lmi_early, lmi_late, cfg)
                rows.append({"scheme": scheme, "cutpoint": cutpoint, "agency": agency,
                             "metric": "lmi_latitude_difference", "effect": effect, "p_value": p,
                             "early_start": lmi_early[0], "early_end": lmi_early[-1],
                             "late_start": lmi_late[0], "late_end": lmi_late[-1]})
    table = pd.DataFrame(rows)
    table.to_csv(out / "cutpoint_sensitivity.csv", index=False)

    metrics = ["path_total_variation", "first_landfall_latitude_difference",
               "north_share_difference", "lmi_latitude_difference"]
    labels = ["Path TVD", "First-landfall latitude (°)", "North-share change", "LMI latitude (°)"]
    colors = {"USA": "#4C72B0", "TOKYO": "#7E9CC4", "CMA": "#C76D6D"}
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, constrained_layout=True)
    show = table.loc[table["scheme"].eq("full_record")]
    for j, (metric, label) in enumerate(zip(metrics, labels)):
        for agency, color in colors.items():
            part = show.loc[show["metric"].eq(metric) & show["agency"].eq(agency)]
            axes[0, j].plot(part["cutpoint"], part["effect"], marker="o", ms=3, color=color,
                            label="JMA" if agency == "TOKYO" else agency)
            axes[1, j].plot(part["cutpoint"], part["p_value"], marker="o", ms=3, color=color)
        axes[0, j].axvline(1996, color="0.35", lw=1, ls="--")
        axes[1, j].axvline(1996, color="0.35", lw=1, ls="--")
        axes[1, j].axhline(0.05, color="0.5", lw=1, ls=":")
        axes[0, j].set_title(label)
        axes[1, j].set_xlabel("First year of late period")
        axes[1, j].set_ylim(0, min(1, max(0.12, show.loc[show["metric"].eq(metric), "p_value"].max() * 1.1)))
    axes[0, 0].set_ylabel("Effect size")
    axes[1, 0].set_ylabel("Block-permutation p")
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.86))
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for label, ax in zip("abcdefgh", axes.flat):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold")
    scale_figure_typography(fig, scale=1.22)
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"cutpoint_sensitivity.{suffix}", dpi=300)
    plt.close(fig)

    method = f"""# 分期切点敏感性

- 主切点仍为1996年；1991—2001年仅用于敏感性，不按显著性重新选择切点。
- 全样本方案：早期为1966年至切点前一年，晚期为切点年至2025年。
- 等长窗口方案：切点前、后各25年，沿时间轴滚动。
- 路径场复用三机构2.5°年度归一化轨迹点构成，以总变差距离和3年分块置换检验。
- 登陆指标仅使用交点前机构原生时次已达到热带风暴或以上的精确交点，分别计算首次合格登陆的年度平均纬度及全部合格登陆事件中的北部命名海岸份额；海岸归属使用互斥岸段。
- LMI始于1982年，因此全样本方案的LMI早期窗口截自1982年；表中明确记录各指标实际起止年。
- 置换{cfg['n_permutations']}次，随机种子{cfg['random_seed']}。图中1996虚线是预设主切点，p=0.05虚线只用于观察稳定性。
"""
    (out / "method.md").write_text(method, encoding="utf-8")
    print(table.groupby(["scheme", "metric"])["effect"].agg(["min", "max"]).to_dict("index"))


if __name__ == "__main__":
    run()
