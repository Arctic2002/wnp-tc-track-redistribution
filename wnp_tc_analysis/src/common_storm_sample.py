from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .common import PROJECT, WORK, load_config

from paper2_dynamic.agency_data import AGENCIES, build_agency_catalog, read_ibtracs_agencies
from paper2_dynamic.revision_stats import compositional_change_test

from .core_crossagency_recheck import annual_path_composition, safe_corr
from .stats import bh_fdr, block_bootstrap_many, block_permutation_many


NORTH = {"China_E", "Taiwan", "Korea", "Japan"}
SOUTH = {"China_S", "Vietnam", "Philippines"}


def period_membership(catalogs, common, start, end):
    rows = []
    sets = {
        agency: {sid for sid in catalog["eligible_sids"] if start <= int(sid[:4]) <= end}
        for agency, catalog in catalogs.items()
    }
    triple = set.intersection(*sets.values())
    for agency, values in sets.items():
        rows.append({"period": f"{start}-{end}", "sample": agency, "n_storms": len(values),
                     "n_triple_common": len(triple), "triple_fraction_of_agency": len(triple) / len(values)})
    for left, right in combinations(AGENCIES, 2):
        inter = sets[left] & sets[right]
        rows.append({"period": f"{start}-{end}", "sample": f"{left}&{right}", "n_storms": len(inter),
                     "n_triple_common": len(triple), "triple_fraction_of_agency": np.nan})
    return rows


def annual_landfall_metrics(events, years, common):
    filtered = events.loc[events["sid"].isin(common)].copy()
    filtered["time"] = pd.to_datetime(filtered["time"])
    first = filtered.sort_values("time").drop_duplicates(["agency", "sid"], keep="first")
    rows = []
    for agency in AGENCIES:
        for definition, frame in [("all_events", filtered), ("first_landfall", first)]:
            part = frame.loc[frame["agency"].eq(agency)]
            for year in years:
                group = part.loc[part["season"].eq(year)]
                named = group.loc[group["coast"].isin(NORTH | SOUTH)]
                north = int(named["coast"].isin(NORTH).sum())
                south = int(named["coast"].isin(SOUTH).sum())
                rows.append({"agency": agency, "definition": definition, "season": year,
                             "mean_latitude": group["lat"].mean(), "median_latitude": group["lat"].median(),
                             "n_events": len(group), "north_share": north / (north + south) if north + south else np.nan,
                             "north_events": north, "south_events": south})
    return pd.DataFrame(rows)


def summarize_period_change(annual, years, cfg):
    rows = []
    early = np.flatnonzero(years <= 1995)
    late = np.flatnonzero(years >= 1996)
    for (agency, definition), group in annual.groupby(["agency", "definition"]):
        group = group.set_index("season").reindex(years)
        matrix = group[["mean_latitude", "north_share"]].to_numpy(float)
        changes, pvalues = block_permutation_many(matrix, early, late, block=3,
                                                   nperm=cfg["n_permutations"], seed=cfg["random_seed"])
        lo, hi = block_bootstrap_many(matrix[early], matrix[late], block=3,
                                      nboot=cfg["n_bootstrap"], seed=cfg["random_seed"])
        for i, metric in enumerate(["mean_landfall_latitude", "north_named_coast_share"]):
            rows.append({"agency": agency, "definition": definition, "metric": metric,
                         "early_mean": np.nanmean(matrix[early, i]), "late_mean": np.nanmean(matrix[late, i]),
                         "late_minus_early": changes[i], "ci_low": lo[i], "ci_high": hi[i],
                         "block_permutation_p": pvalues[i], "block_years": 3,
                         "n_permutations": cfg["n_permutations"], "n_bootstrap": cfg["n_bootstrap"]})
    summary = pd.DataFrame(rows)
    summary["fdr_family"] = (
        "common_storm_landfall_" + summary["definition"] + "_" + summary["metric"]
    )
    summary["q_bh"] = np.nan
    for _, idx in summary.groupby("fdr_family").groups.items():
        summary.loc[idx, "q_bh"] = bh_fdr(summary.loc[idx, "block_permutation_p"])
    return summary


def run():
    cfg = load_config()
    out = WORK / "analysis" / "03_common_storms"
    years = np.arange(1966, 2026)
    source = read_ibtracs_agencies(PROJECT / "data" / "raw" / "IBTrACS.WP.v04r01.csv", start=1945, end=2025)
    catalogs = {agency: build_agency_catalog(source, agency) for agency in AGENCIES}
    period_sids = {
        agency: set(catalog["ts_points"].loc[catalog["ts_points"]["season"].between(1966, 2025), "sid"])
        for agency, catalog in catalogs.items()
    }
    common = set.intersection(*period_sids.values())

    membership = []
    for start, end in [(1966, 1995), (1996, 2025), (1966, 2025)]:
        membership.extend(period_membership(catalogs, common, start, end))
    pd.DataFrame(membership).to_csv(out / "common_storm_membership.csv", index=False)

    lon_edges = np.arange(100, 182.5, 2.5)
    lat_edges = np.arange(0, 42.5, 2.5)
    compositions = {}
    results = {}
    for agency, catalog in catalogs.items():
        points = catalog["ts_points"].loc[catalog["ts_points"]["sid"].isin(common)]
        compositions[agency] = annual_path_composition(points, years, lon_edges, lat_edges)
        results[agency] = compositional_change_test(
            compositions[agency], np.arange(30), np.arange(30, 60),
            nperm=cfg["n_permutations"], block=3, seed=cfg["random_seed"],
        )
    path = pd.DataFrame([{"agency": agency, "sample": "triple_common_native_tracks",
                          "n_common_storms": len(common), "total_variation": result["tv"],
                          "block_permutation_p": result["global_p"], "block_years": 3,
                          "n_permutations": cfg["n_permutations"]}
                         for agency, result in results.items()])
    path["fdr_family"] = "common_storm_path_redistribution"
    path["q_bh"] = bh_fdr(path["block_permutation_p"])
    path.to_csv(out / "common_storm_path_redistribution.csv", index=False)
    correlations = pd.DataFrame([{"agency_left": left, "agency_right": right,
                                  "metric": "common_storm_path_change_map",
                                  "correlation": safe_corr(results[left]["change"], results[right]["change"])}
                                 for left, right in combinations(AGENCIES, 2)])
    correlations.to_csv(out / "common_storm_path_agreement.csv", index=False)
    np.savez_compressed(out / "common_storm_annual_path_composition_2p5deg.npz", years=years,
                        lon_edges=lon_edges, lat_edges=lat_edges, **compositions)

    events = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_events_exact.csv")
    annual = annual_landfall_metrics(events, years, common)
    annual.to_csv(out / "common_storm_landfall_annual.csv", index=False)
    landfall = summarize_period_change(annual, years, cfg)
    landfall.to_csv(out / "common_storm_landfall_summary.csv", index=False)

    method = f"""# 三机构共同气旋样本分析

- 共同样本以IBTrACS SID定义；三机构原生热带风暴及以上入选集合取交集，共{len(common)}个风暴（1966—2025）。
- 两两交集和三者交集比例按1966—1995、1996—2025及全时段分别输出；这些比例用于区分样本选择差异与同一风暴记录差异。
- 路径分析在共同SID内保留各机构自身轨迹坐标，不把某一机构位置复制给其他机构；2.5°年度归一化构成、总变差距离及3年分块置换与主分析一致。
- 登陆分析复用高分辨率海岸线精确交点，只保留交点前机构原生时次已达到热带风暴或以上的事件，再筛选共同SID；分别报告全部合格登陆事件和每个机构记录下的首次合格登陆。
- 路径场、全部事件登陆纬度、首次登陆纬度及各海岸份额口径分别成族，每族包含USA、JMA和CMA三个检验，并采用BH-FDR校正。
- 北/南份额的分母只含预设七类命名海岸，Other不进入分母；效应量为年度份额的后期均值减前期均值。
- 随机种子{cfg['random_seed']}；置换{cfg['n_permutations']}次，分块自助{cfg['n_bootstrap']}次。

三套机构共享观测基础，故共同样本结果用于评估记录差异，不能解释为三次独立实验。
"""
    (out / "common_storm_method.md").write_text(method, encoding="utf-8")
    print({"common_sids": len(common), "path": path[["agency", "total_variation", "block_permutation_p", "q_bh"]].to_dict("records")})


if __name__ == "__main__":
    run()
