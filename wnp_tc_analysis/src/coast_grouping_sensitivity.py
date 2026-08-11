from __future__ import annotations

import numpy as np
import pandas as pd

from .common import WORK, load_config
from .stats import bh_fdr, block_permutation_difference


YEARS = np.arange(1966, 2026)
BASE_NORTH = {"China_E", "Taiwan", "Korea", "Japan"}
BASE_SOUTH = {"China_S", "Vietnam", "Philippines"}


def select_events(events, rule):
    if rule == "all_events":
        return events.copy()
    groups = []
    for (_, _), group in events.groupby(["agency", "sid"], sort=False):
        group = group.sort_values("time")
        if rule == "first_any":
            row = group.iloc[0]
        elif rule == "first_named":
            named = group.loc[group["coast"].ne("Other")]
            row = named.iloc[0] if len(named) else group.iloc[0]
        elif rule == "strongest_landfall":
            valid = group.loc[group["wind"].notna()]
            if valid.empty:
                continue
            row = valid.loc[valid["wind"].idxmax()]
        else:
            raise ValueError(rule)
        groups.append(row)
    return pd.DataFrame(groups)


def annual_share(frame, agency, north, south, denominator):
    values = []
    for year in YEARS:
        group = frame.loc[frame["agency"].eq(agency) & frame["season"].eq(year)]
        n = int(group["coast"].isin(north).sum())
        s = int(group["coast"].isin(south).sum())
        other = int(group["coast"].eq("Other").sum())
        denom = n + s if denominator == "named_only" else n + s + other
        values.append(n / denom if denom else np.nan)
    return np.asarray(values, float)


def summarize(events, cfg, threshold_km):
    classified = events.copy()
    classified["coast"] = np.where(
        classified["nearest_distance_km"].astype(float) <= threshold_km,
        classified["nearest_segment"],
        "Other",
    )
    selected = {
        rule: select_events(classified, rule)
        for rule in ["all_events", "first_any", "first_named"]
    }
    rows = []
    early, late = np.arange(30), np.arange(30, 60)
    taiwan_options = {
        "taiwan_north": (BASE_NORTH, BASE_SOUTH),
        "taiwan_south": (BASE_NORTH - {"Taiwan"}, BASE_SOUTH | {"Taiwan"}),
        "taiwan_excluded": (BASE_NORTH - {"Taiwan"}, BASE_SOUTH),
    }
    for rule, frame in selected.items():
        for agency in ["USA", "TOKYO", "CMA"]:
            for option, (north, south) in taiwan_options.items():
                for denominator in ["named_only", "include_other"]:
                    share = annual_share(frame, agency, north, south, denominator)
                    effect, p = block_permutation_difference(
                        share,
                        early,
                        late,
                        block=3,
                        nperm=cfg["n_permutations"],
                        seed=cfg["random_seed"],
                    )
                    rows.append(
                        {
                            "agency": agency,
                            "event_rule": rule,
                            "taiwan_rule": option,
                            "denominator": denominator,
                            "threshold_km": float(threshold_km),
                            "early_share": np.nanmean(share[early]),
                            "late_share": np.nanmean(share[late]),
                            "change_percentage_points": effect * 100,
                            "block_permutation_p": p,
                            "block_years": 3,
                            "n_permutations": cfg["n_permutations"],
                        }
                    )
    result = pd.DataFrame(rows)
    result["fdr_family"] = (
        "coast_share_"
        + result["event_rule"]
        + "_"
        + result["taiwan_rule"]
        + "_"
        + result["denominator"]
        + "_"
        + result["threshold_km"].map(lambda value: f"{value:g}km")
    )
    result["q_bh"] = np.nan
    for _, index in result.groupby("fdr_family").groups.items():
        result.loc[index, "q_bh"] = bh_fdr(
            result.loc[index, "block_permutation_p"]
        )
    return result


def run():
    cfg = load_config()
    out = WORK / "analysis" / "06_landfall_grouping"
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_events_exact.csv")
    events["time"] = pd.to_datetime(events["time"])
    events = events.loc[events["season"].between(1966, 2025)].copy()
    threshold_summary = pd.concat(
        [summarize(events, cfg, threshold) for threshold in [25, 50, 75, 100]],
        ignore_index=True,
    )
    threshold_summary.to_csv(
        out / "coast_grouping_threshold_sensitivity.csv", index=False
    )
    summary = threshold_summary.loc[threshold_summary["threshold_km"].eq(50)].copy()
    summary.to_csv(out / "coast_grouping_sensitivity.csv", index=False)

    crossing_rows = []
    for agency, group in events.groupby("agency"):
        by_storm = group.groupby("sid")["coast"].agg(list)
        cross = by_storm.apply(lambda values: bool(set(values) & BASE_NORTH) and bool(set(values) & BASE_SOUTH))
        crossing_rows.append({"agency": agency, "n_landfalling_storms": len(by_storm),
                              "n_crossing_north_and_south": int(cross.sum()),
                              "fraction_crossing_north_and_south": float(cross.mean())})
    pd.DataFrame(crossing_rows).to_csv(out / "cross_group_storm_counts.csv", index=False)

    method = f"""# 海岸分组与计数敏感性

- 预设北部组为华东、台湾、朝鲜半岛和日本；南部组为华南、越南和菲律宾。该划分沿用研究设计中的东亚海岸段和西行/转向路径通道，不依据分析结果调整。
- 输入仅含海岸交点前机构原生时次已达到热带风暴或以上的精确登陆事件；海岸类别统一由互斥GSHHG岸段及最近距离确定。
- 事件口径包括全部登陆、每风暴首次任意海岸和每风暴首次命名海岸；后两者均为唯一海岸归属。
- 分母分别采用仅七类命名海岸，以及命名海岸加Other。全部事件口径允许同一风暴多次登陆；唯一归属口径用于消除跨组重复计数。
- 台湾依次归入北部、归入南部和从二分类中排除。“单列”在二分类份额计算中等价于不进入北/南分母，以`taiwan_excluded`表示。
- 菲律宾未进一步人为南北切分：当前海岸归属只有Philippines整体类别，且没有预设、文献支持的岛群分界；事后选择纬度线会引入结果导向分组。
- 效应量为年度北部份额后期均值减前期均值（百分点）；25、50、75和100 km阈值均独立输出。3年分块置换{cfg['n_permutations']}次，随机种子{cfg['random_seed']}；BH-FDR按事件口径、台湾规则、分母和阈值分别成族，每族含三个机构。
"""
    (out / "method.md").write_text(method, encoding="utf-8")
    main = summary.query("event_rule == 'first_any' and taiwan_rule == 'taiwan_north' and denominator == 'named_only'")
    print(main[["agency", "change_percentage_points", "block_permutation_p"]].to_dict("records"))


if __name__ == "__main__":
    run()
