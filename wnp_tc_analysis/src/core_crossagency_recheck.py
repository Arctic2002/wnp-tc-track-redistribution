from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PROJECT, REPORTS, WORK, load_config

from core.utils import load_config as load_project_config
from paper2_dynamic.agency_data import AGENCIES, build_agency_catalog, read_ibtracs_agencies
from paper2_dynamic.revision_stats import add_family_fdr, compositional_change_test, trend_row


def annual_path_composition(points, years, lon_edges, lat_edges):
    points = points.loc[
        points["lon"].between(lon_edges[0], lon_edges[-1])
        & points["lat"].between(lat_edges[0], lat_edges[-1])
    ]
    rows = []
    for year in years:
        group = points.loc[points["season"].eq(year)]
        field = np.histogram2d(group["lon"], group["lat"], bins=[lon_edges, lat_edges])[0].T
        if field.sum() == 0:
            raise ValueError(f"no agency path points in {year}")
        rows.append((field / field.sum()).ravel())
    return np.asarray(rows)


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def numeric_comparison(new, old, keys, columns, source):
    merged = new.merge(old, on=keys, suffixes=("_new", "_frozen"), how="outer", indicator=True)
    rows = []
    for _, row in merged.iterrows():
        base = {key: row[key] for key in keys}
        for column in columns:
            left = row.get(f"{column}_new", np.nan)
            right = row.get(f"{column}_frozen", np.nan)
            match = bool(
                row["_merge"] == "both"
                and ((pd.isna(left) and pd.isna(right)) or np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True))
            )
            rows.append({**base, "source": source, "field": column, "new": left, "frozen": right, "match": match})
    return rows


def update_registry(comparison):
    path = REPORTS / "results_registry.csv"
    registry = pd.read_csv(path, dtype=str).fillna("")
    bad = comparison.loc[~comparison["match"]]
    status = "recalculated_unchanged" if bad.empty else "recalculated_changed"
    core = registry["claim_id"].str.startswith(("trend_frequency_", "trend_lmi_", "redistribution_path_density_", "agreement_"))
    registry.loc[core, "status"] = status
    registry.loc[core, "source_script"] = "src/core_crossagency_recheck.py"
    registry.loc[core, "notes"] = registry.loc[core, "notes"].str.rstrip("; ") + "; independent v2 rerun comparison recorded in analysis/03_common_storms/comparison_with_frozen.csv"
    registry.to_csv(path, index=False)


def run():
    spec = load_config()
    project_cfg = load_project_config()
    out = WORK / "analysis" / "03_common_storms"
    out.mkdir(parents=True, exist_ok=True)
    years = np.arange(1966, 2026)
    lmi_years = np.arange(1982, 2026)
    raw = PROJECT / "data" / "raw" / "IBTrACS.WP.v04r01.csv"
    source = read_ibtracs_agencies(raw, start=1945, end=2025)
    catalogs = {agency: build_agency_catalog(source, agency) for agency in AGENCIES}
    common_lmi = set.intersection(*(set(catalogs[a]["lmi"]["sid"]) for a in AGENCIES))

    annual_rows = []
    trend_rows = []
    for agency, catalog in catalogs.items():
        frequency = catalog["frequency"].set_index("season")["n_tc"].reindex(years, fill_value=0)
        lmi = catalog["lmi"].loc[catalog["lmi"]["season"].between(1982, 2025)]
        full = lmi.groupby("season")["lmi_lat"].mean().reindex(lmi_years)
        common = lmi.loc[lmi["sid"].isin(common_lmi)].groupby("season")["lmi_lat"].mean().reindex(lmi_years)
        for year in years:
            annual_rows.append({
                "agency": agency, "season": year, "n_tc": int(frequency.loc[year]),
                "mean_lmi_lat_full": full.loc[year] if year >= 1982 else np.nan,
                "mean_lmi_lat_common": common.loc[year] if year >= 1982 else np.nan,
            })
        for end in (2024, 2025):
            for variable, start, metric, values in [
                ("n_tc", 1966, "frequency", frequency.loc[1966:end]),
                ("mean_lmi_lat_full", 1982, "lmi_full_catalog", full.loc[1982:end]),
                ("mean_lmi_lat_common", 1982, "lmi_common_storms", common.loc[1982:end]),
            ]:
                trend_rows.append(trend_row(
                    values.index, values.values, label=agency,
                    family=f"multiagency_{metric}_{start}_{end}", cfg=project_cfg,
                    extra={"analysis": metric, "agency": agency, "variable": variable},
                ))
    annual = pd.DataFrame(annual_rows)
    trends = add_family_fdr(trend_rows)
    annual.to_csv(out / "core_crossagency_annual.csv", index=False)
    trends.to_csv(out / "core_crossagency_recheck_trends.csv", index=False)

    lon_edges = np.arange(100, 180 + 2.5, 2.5)
    lat_edges = np.arange(0, 40 + 2.5, 2.5)
    compositions = {a: annual_path_composition(c["ts_points"], years, lon_edges, lat_edges) for a, c in catalogs.items()}
    results = {
        a: compositional_change_test(x, np.arange(30), np.arange(30, 60),
                                     nperm=spec["n_permutations"], block=3, seed=spec["random_seed"])
        for a, x in compositions.items()
    }
    redistribution = pd.DataFrame([{
        "agency": a, "analysis": "path_density", "early_start": 1966, "early_end": 1995,
        "late_start": 1996, "late_end": 2025, "total_variation": r["tv"],
        "block_permutation_p": r["global_p"], "n_permutations": r["nperm"], "block_years": r["block"],
    } for a, r in results.items()])
    redistribution.to_csv(out / "core_crossagency_recheck_redistribution.csv", index=False)

    agreement = pd.DataFrame([{
        "agency_left": left, "agency_right": right, "metric": "path_change_map",
        "correlation": safe_corr(results[left]["change"], results[right]["change"]),
    } for left, right in combinations(AGENCIES, 2)])
    agreement.to_csv(out / "core_crossagency_recheck_agreement.csv", index=False)
    np.savez_compressed(out / "annual_path_composition_2p5deg.npz", years=years, lon_edges=lon_edges,
                        lat_edges=lat_edges, **compositions)

    frozen_dir = WORK / "data" / "upstream_revision"
    old_trends = pd.read_csv(frozen_dir / "p2_multiagency_trends.csv")
    old_trends = old_trends.loc[old_trends["analysis"].isin(["frequency", "lmi_full_catalog", "lmi_common_storms"])]
    new_trends = trends.loc[trends["end"].isin([2024, 2025])]
    keys_t = ["agency", "analysis", "start", "end"]
    cols_t = ["sen_slope_per_decade", "sen_ci_lo_per_decade", "sen_ci_hi_per_decade", "mk_p_raw", "mk_p_fdr_bh"]
    comparison_rows = numeric_comparison(new_trends, old_trends, keys_t, cols_t, "trends")
    old_red = pd.read_csv(frozen_dir / "p2_multiagency_redistribution.csv").query("analysis == 'path_density'")
    comparison_rows += numeric_comparison(redistribution, old_red, ["agency", "analysis"], ["total_variation", "block_permutation_p"], "redistribution")
    old_agreement = pd.read_csv(frozen_dir / "p2_multiagency_agreement.csv").query("metric == 'path_change_map'")
    comparison_rows += numeric_comparison(agreement, old_agreement, ["agency_left", "agency_right", "metric"], ["correlation"], "agreement")
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "comparison_with_frozen.csv", index=False)
    update_registry(comparison)

    method = f"""# 三机构核心结果独立复算

- 输入：`data/raw/IBTrACS.WP.v04r01.csv`，USA、JMA（IBTrACS字段名TOKYO）和CMA原生记录。
- 频数：各机构原生热带风暴判据；1966—2025年，并另算截至2024年。
- LMI纬度：各机构自身最大风速时刻；1982—2025年；同时报告各机构全样本和三机构共同SID样本（共同样本数：{len(common_lmi)}）。
- 趋势：Theil–Sen斜率、Hamed–Rao修正Mann–Kendall检验、预设族内BH-FDR；置信区间沿用项目配置中的分块自助法。
- 路径空间构成：100°—180°E、0°—40°N，2.5°网格；每年轨迹点网格频数归一为构成向量；1966—1995年与1996—2025年以总变差距离比较，3年分块置换{spec['n_permutations']}次。
- 一致性：机构间前后期差值图Pearson相关。
- 随机种子：{spec['random_seed']}。中间年度构成保存为压缩NPZ，供共同样本和切点敏感性复用。

复算与参照输出逐字段比较见`comparison_with_frozen.csv`。`match=true`要求绝对误差不超过1e-12或相对误差不超过1e-10。
"""
    (out / "method.md").write_text(method, encoding="utf-8")
    summary = {
        "common_lmi_sids": len(common_lmi), "comparison_fields": len(comparison),
        "mismatched_fields": int((~comparison["match"]).sum()),
    }
    pd.Series(summary).to_json(WORK / "outputs" / "logs" / "03_core_crossagency_recheck.json", force_ascii=False, indent=2)
    print(summary)


if __name__ == "__main__":
    run()
