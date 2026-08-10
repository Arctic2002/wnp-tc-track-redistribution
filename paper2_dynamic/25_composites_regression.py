"""年度路径指标 × 环流指数的分布匹配回归。

产出 processed/p2_stats.csv（年度底表）、p2_models.csv（系数/CI/诊断）、p2_vif.csv。
详见 Docs/02 §4.6。
移速/LMI 用样本量加权线性模型(HAC)；转向用事件级二项 GLM(按 season 聚类稳健)；
登陆事件用带 TC 数暴露量(offset=log n_tc)的 Poisson，过度离散改 NB2
（alpha 由 discrete.NegativeBinomial 的 MLE 估计后代回 GLM-NB 取 HAC）。

运行环境提示：依赖 statsmodels 的 GLM(cov_type="HAC"/"cluster") 与 discrete.NegativeBinomial；
落地前确认版本支持这些稳健协方差，NB2 偶有不收敛已设 maxiter=200。
"""
import pandas as pd, numpy as np, statsmodels.api as sm
from core.utils import load_config
from core.stats_utils import vif_table


def annualize(m, scope, intensity_start):
    """把逐 TC 指标聚合为年度响应，并保留每个平均值/比例的分母。"""
    m = m.copy()
    # where 在 1982 年前把 LMI 纬度设为 NaN，但不影响同期路径指标。
    m["lmi_lat_valid"] = m["lmi_lat"].where(m.season >= intensity_start)
    out = m.groupby("season").agg(
        n_tc=("sid", "size"), lmi_lat=("lmi_lat_valid", "mean"),
        n_lmi=("lmi_lat_valid", "count"), recurv_n=("recurving", "sum"),
        recurv_ratio=("recurving", "mean"), speed=("trans_speed", "mean"),
        n_speed=("trans_speed", "count"), landfalls=("n_landfall", "sum")).reset_index()
    out["scope"] = scope
    return out


def main():
    cfg = load_config()
    m = pd.read_csv(f"{cfg['paths']['processed']}/p2_metrics.csv", parse_dates=["genesis_time"])
    wn = pd.read_csv(f"{cfg['paths']['processed']}/p2_wnpsh.csv")
    mt = pd.read_csv(f"{cfg['paths']['processed']}/p2_monsoon.csv")
    st = pd.read_csv(f"{cfg['paths']['processed']}/p2_steering_annual.csv")
    oni = pd.read_csv(f"{cfg['paths']['raw']}/indices/oni.csv")
    pdo = pd.read_csv(f"{cfg['paths']['raw']}/indices/pdo.csv")
    # peak_m 是 6–10 月生成的气旋队列；annual 和 peak 并行分析。
    peak_m = m[m.genesis_time.dt.month.isin(cfg["typhoon_season"])]
    yr = pd.concat([annualize(m, "annual", cfg["periods"]["intensity_start"]),
                    annualize(peak_m, "peak", cfg["periods"]["intensity_start"])],
                   ignore_index=True)
    # 按 season 把年度响应和四类环境/气候指数连接起来。
    df = (yr.merge(wn, on="season").merge(mt, on="season").merge(st, on="season")
            .merge(oni, on="season").merge(pdo, on="season"))
    df["trend"] = df.groupby("scope").season.transform(lambda x: x - x.mean())
    df.to_csv(f"{cfg['paths']['processed']}/p2_stats.csv", index=False)

    predictors = ["west_ridge_point", "mt_westerly_east", "jas_oni", "pdo", "trend"]
    rows = []
    vif_rows = []

    def collect(scope, name, res, n):
        """把 statsmodels 结果整理成长表；每行对应一个模型系数。"""
        ci = res.conf_int()
        for term in res.params.index:
            rows.append({"scope": scope, "response": name, "term": term,
                         "coef": res.params[term], "se": res.bse[term],
                         "ci_lo": ci.loc[term, 0], "ci_hi": ci.loc[term, 1],
                         "p": res.pvalues[term], "n": n})

    for scope, base in [("annual", m), ("peak", peak_m)]:
        d = df[df.scope == scope].copy()
        # VIF 用于发现预测因子之间的线性共线性；它不代表变量显著性。
        vf = vif_table(d[predictors].dropna())
        vif_rows.extend({"scope": scope, "term": k, "vif": v} for k, v in vf.items())
        for y, weight, start in [("speed", "n_speed", cfg["periods"]["freq_start"]),
                                 ("lmi_lat", "n_lmi", cfg["periods"]["intensity_start"])]:
            dat = d[d.season >= start].dropna(subset=[y, weight] + predictors)
            X0 = (dat[predictors] - dat[predictors].mean()) / dat[predictors].std()
            # WLS 按年度样本数赋权；样本更多的年度均值通常更稳定。
            res = sm.WLS(dat[y], sm.add_constant(X0), weights=dat[weight]).fit(
                cov_type="HAC", cov_kwds={"maxlags": 1})
            collect(scope, y, res, len(dat))

        event = base.merge(d[["season"] + predictors], on="season").dropna(
            subset=predictors + ["recurving"])
        X0 = (event[predictors] - event[predictors].mean()) / event[predictors].std()
        # recurving 只有 0/1 两种结果，因此使用二项分布而非普通 OLS。
        rec = sm.GLM(event.recurving, sm.add_constant(X0), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": event.season})
        collect(scope, "recurving", rec, len(event))

        dat = d.dropna(subset=predictors + ["landfalls", "n_tc"])
        X0 = (dat[predictors] - dat[predictors].mean()) / dat[predictors].std()
        # 登陆数是计数；offset=log(n_tc) 把每年可登陆的 TC 总数作为暴露量。
        land = sm.GLM(dat.landfalls, sm.add_constant(X0), family=sm.families.Poisson(),
                      offset=np.log(dat.n_tc)).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        collect(scope, "landfalls", land, len(dat))
        # 过度离散时改用负二项 NB2；offset=log(n_tc) 继续作为暴露量。
        dispersion = land.pearson_chi2 / land.df_resid
        if dispersion > 1.5:
            # alpha 由 discrete.NegativeBinomial 的 MLE 估计（支持 offset），
            # 再把 MLE alpha 代回 GLM-NB 以保留 HAC 时间相关稳健 SE。
            nb_mle = sm.NegativeBinomial(dat.landfalls, sm.add_constant(X0),
                                         offset=np.log(dat.n_tc)).fit(disp=0, maxiter=200)
            alpha_hat = float(nb_mle.params.iloc[-1])           # MLE 离散参数
            land_nb = sm.GLM(dat.landfalls, sm.add_constant(X0),
                             family=sm.families.NegativeBinomial(alpha=max(alpha_hat, 1e-6)),
                             offset=np.log(dat.n_tc)).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
            collect(scope, "landfalls_NB", land_nb, len(dat))
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['processed']}/p2_models.csv", index=False)
    pd.DataFrame(vif_rows).to_csv(f"{cfg['paths']['processed']}/p2_vif.csv", index=False)
    print(f"p2_stats.csv / p2_models.csv / p2_vif.csv written; "
          f"{len(rows)} coef rows, {len(vif_rows)} vif rows")


if __name__ == "__main__":
    main()
