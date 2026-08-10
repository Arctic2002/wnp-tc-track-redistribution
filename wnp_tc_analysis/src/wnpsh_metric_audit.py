from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend

from .common import WORK
from .stats import bh_fdr, block_order

from core.utils import load_config as load_project_config
from paper2_dynamic.revision_stats import add_family_fdr, trend_row


def correlation_permutation(x, y, *, block=3, nperm=9999, seed=202406):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, len(x)
    raw = float(np.corrcoef(x, y)[0, 1])
    x_test = detrend(x, type="linear")
    y_test = detrend(y, type="linear")
    if np.std(x_test) == 0 or np.std(y_test) == 0:
        return raw, np.nan, np.nan, len(x)
    observed = float(np.corrcoef(x_test, y_test)[0, 1])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = block_order(len(x), block, rng)
        trial = float(np.corrcoef(x_test[order], y_test)[0, 1])
        exceed += abs(trial) >= abs(observed)
    return raw, observed, (exceed + 1) / (nperm + 1), len(x)


def run():
    out = WORK / "analysis" / "07_wnpsh_dynamic_metric"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_project_config()
    annual = pd.read_csv(WORK / "data" / "wnp_tc_eddy_wnpsh_annual.csv")
    variables = ["eddy_wnpsh_mean_m", "wpsh_area", "wpsh_intensity", "west_ridge_point", "ridge_line"]
    trend_rows = []
    for start in [1966, 1982]:
        subset = annual.loc[annual["year"].between(start, 2025)]
        for variable in variables:
            trend_rows.append(trend_row(
                subset["year"], subset[variable], label=variable,
                family=f"wnpsh_metrics_{start}_2025", cfg=cfg,
                extra={"variable": variable, "metric_type": "domain_adjusted_proxy" if variable == "eddy_wnpsh_mean_m" else "fixed_588_geometry"},
            ))
    trends = add_family_fdr(trend_rows)
    trends.to_csv(out / "wnpsh_metric_trends.csv", index=False)

    index = pd.read_csv(WORK / "data" / "wnp_tc_redistribution_index_annual.csv")
    index = index.loc[index["weighting"].eq("track_point"), ["agency", "year", "index_oos"]]
    lmi = pd.read_csv(WORK / "analysis" / "03_common_storms" / "core_crossagency_annual.csv")
    landing = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_latitude_annual.csv")
    landing = landing.loc[landing["definition"].eq("first_landfall"), ["agency", "year", "mean_lat"]]
    rows = []
    for agency in ["USA", "TOKYO", "CMA"]:
        targets = index.loc[index["agency"].eq(agency)].merge(
            lmi.loc[lmi["agency"].eq(agency), ["season", "mean_lmi_lat_full"]].rename(columns={"season": "year"}),
            on="year", how="left").merge(landing.loc[landing["agency"].eq(agency)], on="year", how="left")
        merged = annual.merge(targets, on="year", how="left")
        for predictor in variables:
            for response in ["index_oos", "mean_lat", "mean_lmi_lat_full"]:
                raw_r, detrended_r, p, n = correlation_permutation(merged[predictor], merged[response])
                rows.append({"agency": agency, "wnpsh_metric": predictor, "response": response,
                             "pearson_r": raw_r, "pearson_r_detrended": detrended_r,
                             "block_permutation_p": p, "inference_series": "linear_detrended", "n": n})
    assoc = pd.DataFrame(rows)
    assoc["q_bh_within_response"] = np.nan
    for _, idx in assoc.groupby("response").groups.items():
        assoc.loc[idx, "q_bh_within_response"] = bh_fdr(assoc.loc[idx, "block_permutation_p"])
    assoc.to_csv(out / "wnpsh_metric_associations.csv", index=False)

    method = """# WNPSH指标证据审计

- 固定588 dagpm指标复用`data/processed/p2_wnpsh.csv`中的面积、强度、西伸脊点和脊线；它们只表示固定阈值几何量。
- 现有`eddy_wnpsh_mean_m`由80°—180°E局地域内逐纬度去经向平均后的500 hPa高度，在110°—160°E、15°—35°N取均值。它可作为局地域背景场调整后的代理量，但不等同于He等人要求的全球0°—360°经度带状均值定义。
- 本地ERA5文件最多覆盖80°—180°E，无法从现有输入严格重算He等人的全经度涡动位势高度。因此本轮不把该代理量更名为严格的He指标，也不据此声称副高动力增强。
- 对1966—2025和1982—2025分别计算Theil–Sen趋势、分块自助置信区间、Hamed–Rao MK及族内BH-FDR。
- 与三机构路径重分配指数、精确首次登陆纬度和LMI纬度的年际关系同时报告原序列Pearson相关和线性去趋势相关；3年分块置换及按响应变量的BH-FDR均以去趋势序列为推断对象，避免共同长期趋势造成伪相关。
"""
    (out / "method.md").write_text(method, encoding="utf-8")
    print(trends[["variable", "start", "sen_slope_per_decade", "mk_p_raw", "mk_p_fdr_bh"]].to_dict("records"))


if __name__ == "__main__":
    run()
