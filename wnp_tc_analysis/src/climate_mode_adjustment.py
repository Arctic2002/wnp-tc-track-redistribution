from __future__ import annotations

import numpy as np
import pandas as pd

from .common import PROJECT, WORK, load_config
from .stats import block_order


def fit_statistic(y, z, period):
    reduced_fit = z @ np.linalg.lstsq(z, y, rcond=None)[0]
    residual = y - reduced_fit
    period_residual = period - z @ np.linalg.lstsq(z, period, rcond=None)[0]
    denom = float((period_residual.T @ period_residual).item())
    beta = (period_residual.T @ residual) / denom
    period_fit = period_residual @ beta
    ss_period = float(np.sum(period_fit ** 2))
    full_residual = residual - period_fit
    ss_error = float(np.sum(full_residual ** 2))
    df_error = y.shape[0] - z.shape[1] - 1
    f_value = (ss_period / 1) / (ss_error / df_error)
    partial_r2 = ss_period / (ss_period + ss_error)
    return f_value, partial_r2, reduced_fit, residual


def freedman_lane(y, z, period, *, block, nperm, seed):
    observed, partial_r2, fitted, residual = fit_statistic(y, z, period)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = block_order(len(y), block, rng)
        trial = fitted + residual[order]
        statistic = fit_statistic(trial, z, period)[0]
        exceed += statistic >= observed
    return observed, partial_r2, (exceed + 1) / (nperm + 1)


def stratified_test(composition, mask, early, *, nperm, seed):
    x = composition[mask]
    label = early[mask]
    if label.sum() < 5 or (~label).sum() < 5:
        return np.nan, np.nan, int(label.sum()), int((~label).sum())
    observed_change = x[~label].mean(axis=0) - x[label].mean(axis=0)
    observed = float(0.5 * np.abs(observed_change).sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        shuffled = rng.permutation(label)
        change = x[~shuffled].mean(axis=0) - x[shuffled].mean(axis=0)
        exceed += 0.5 * np.abs(change).sum() >= observed
    return observed, (exceed + 1) / (nperm + 1), int(label.sum()), int((~label).sum())


def run():
    cfg = load_config()
    out = WORK / "analysis" / "05_climate_mode_adjustment"
    out.mkdir(parents=True, exist_ok=True)
    zfile = np.load(WORK / "analysis" / "03_common_storms" / "annual_path_composition_2p5deg.npz")
    years = zfile["years"].astype(int)
    climate = pd.DataFrame({"season": years})
    climate = climate.merge(pd.read_csv(PROJECT / "data" / "raw" / "indices" / "oni.csv"), on="season", how="left")
    climate = climate.merge(pd.read_csv(PROJECT / "data" / "raw" / "indices" / "pdo.csv"), on="season", how="left")
    climate["late_period"] = (climate["season"] >= 1996).astype(int)
    climate["enso_phase"] = np.select([climate["jas_oni"] >= 0.5, climate["jas_oni"] <= -0.5],
                                       ["El_Nino", "La_Nina"], default="Neutral")
    climate["pdo_phase"] = np.where(climate["pdo"] >= 0, "PDO_positive", "PDO_negative")
    climate.to_csv(out / "climate_covariates_annual.csv", index=False)

    covariates = climate[["jas_oni", "pdo"]].to_numpy(float)
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0, ddof=1)
    reduced = np.column_stack([np.ones(len(years)), covariates])
    period = climate[["late_period"]].to_numpy(float)
    main_rows, sensitivity_rows = [], []
    for agency in ["USA", "TOKYO", "CMA"]:
        composition = np.asarray(zfile[agency], float)
        hellinger = np.sqrt(composition)
        f_value, partial_r2, p = freedman_lane(
            hellinger, reduced, period, block=3,
            nperm=cfg["n_permutations"], seed=cfg["random_seed"],
        )
        main_rows.append({"agency": agency, "response": "hellinger_annual_path_composition",
                          "tested_term": "late_period_after_JAS_ONI_and_PDO", "pseudo_f": f_value,
                          "partial_r2": partial_r2, "block_permutation_p": p,
                          "block_years": 3, "n_permutations": cfg["n_permutations"]})
        early = climate["late_period"].eq(0).to_numpy()
        for variable, levels in [("enso_phase", ["El_Nino", "Neutral", "La_Nina"]),
                                 ("pdo_phase", ["PDO_positive", "PDO_negative"])]:
            for level in levels:
                mask = climate[variable].eq(level).to_numpy()
                tv, p_strat, n_early, n_late = stratified_test(
                    composition, mask, early, nperm=cfg["n_permutations"], seed=cfg["random_seed"])
                sensitivity_rows.append({"agency": agency, "stratum_variable": variable,
                                         "stratum": level, "n_early": n_early, "n_late": n_late,
                                         "total_variation": tv, "unrestricted_permutation_p": p_strat,
                                         "n_permutations": cfg["n_permutations"]})
    main = pd.DataFrame(main_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    main.to_csv(out / "hellinger_period_effect_adjusted.csv", index=False)
    sensitivity.to_csv(out / "enso_pdo_stratified_sensitivity.csv", index=False)

    method = f"""# ENSO/PDO调整后的时期效应

- 响应为三机构各自2.5°年度归一化路径构成的Hellinger变换（逐格平方根）。
- 约简模型含截距、标准化JAS ONI和年平均PDO；完整模型增加1996—2025时期指示变量。
- 以时期项的多变量增量平方和构造伪F和partial R²；在约简模型残差上执行Freedman–Lane置换，按相邻年份3年分块重排，置换{cfg['n_permutations']}次，随机种子{cfg['random_seed']}。
- 敏感性分析按JAS ONI阈值±0.5划分El Niño、Neutral和La Niña，并按PDO正负位相分层；每层保持前后期样本数，以无约束标签置换比较总变差距离。由于位相筛选后的年份并不连续，分层p值不承担主推断，只用于检查效应方向和量级。
- 该分析检验控制气候模态样本构成后的统计时期效应，不等同于ENSO/PDO因果归因。
"""
    (out / "method.md").write_text(method, encoding="utf-8")
    print(main.to_dict("records"))


if __name__ == "__main__":
    run()
