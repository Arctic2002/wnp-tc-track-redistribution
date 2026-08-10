from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import detrend
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import DATA, PROJECT, ensure_dirs


MODELS = {
    "oni": ["jas_oni"],
    "oni_pdo": ["jas_oni", "pdo"],
    "eddy_wnpsh_oni_pdo": ["eddy_wnpsh_mean_m", "jas_oni", "pdo"],
    "steering_oni_pdo": ["corridor_u_steer_ms", "corridor_v_steer_ms", "jas_oni", "pdo"],
}


def zscore(frame):
    return (frame - frame.mean()) / frame.std(ddof=1)


def fit_one(data, predictors, timescale, model_name):
    cols = ["redistribution_index_oos_z", *predictors]
    d = data[cols].dropna().copy()
    if timescale == "detrended":
        for col in cols:
            d[col] = detrend(d[col].to_numpy(float))
    d[cols] = zscore(d[cols])
    X = sm.add_constant(d[predictors])
    fit = sm.OLS(d["redistribution_index_oos_z"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    coef = []
    for term in fit.params.index:
        coef.append({
            "timescale": timescale,
            "model": model_name,
            "term": term,
            "coefficient_standardized": float(fit.params[term]),
            "se_hac": float(fit.bse[term]),
            "p_hac": float(fit.pvalues[term]),
            "ci_low": float(fit.conf_int().loc[term, 0]),
            "ci_high": float(fit.conf_int().loc[term, 1]),
            "n": int(fit.nobs),
        })
    summary = {
        "timescale": timescale,
        "model": model_name,
        "n": int(fit.nobs),
        "r2": float(fit.rsquared),
        "adjusted_r2": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }
    vif = []
    if len(predictors) > 1:
        arr = d[predictors].to_numpy(float)
        for i, term in enumerate(predictors):
            vif.append({"timescale": timescale, "model": model_name, "term": term, "vif": float(variance_inflation_factor(arr, i))})
    return coef, summary, vif


def run() -> None:
    ensure_dirs()
    d = pd.read_csv(DATA / "jcli_eddy_wnpsh_annual.csv")
    oni = pd.read_csv(PROJECT / "data" / "raw" / "indices" / "oni.csv").rename(columns={"season": "year"})
    pdo = pd.read_csv(PROJECT / "data" / "raw" / "indices" / "pdo.csv").rename(columns={"season": "year"})
    d = d.merge(oni, on="year", how="left").merge(pdo, on="year", how="left")
    d.to_csv(DATA / "jcli_regression_input_annual.csv", index=False)
    coef_rows, summary_rows, vif_rows = [], [], []
    for timescale in ("raw", "detrended"):
        for name, predictors in MODELS.items():
            coef, summary, vif = fit_one(d, predictors, timescale, name)
            coef_rows.extend(coef); summary_rows.append(summary); vif_rows.extend(vif)
    coefs = pd.DataFrame(coef_rows)
    from paper2_dynamic.revision_stats import bh_fdr
    coefs["q_bh_within_timescale"] = np.nan
    for _, idx in coefs.loc[coefs.term != "const"].groupby("timescale").groups.items():
        coefs.loc[idx, "q_bh_within_timescale"] = bh_fdr(coefs.loc[idx, "p_hac"])
    coefs.to_csv(DATA / "jcli_regression_models.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(DATA / "jcli_regression_diagnostics.csv", index=False)
    pd.DataFrame(vif_rows).to_csv(DATA / "jcli_regression_vif.csv", index=False)
    print(coefs.loc[coefs.term != "const"].to_string(index=False))
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    run()
