"""转向气候学：转向点密度、事件发生月频率、按生成月队列的最终转向比例、年度比例。

产出 processed/p2_recurve_density.npz、p2_recurve_monthly.csv、p2_recurve_annual.csv。
详见 Docs/02 §4.11。直接读取 p2_metrics 的转向时空点。
"""
import pandas as pd, numpy as np
from core.utils import load_config


def main():
    cfg = load_config()
    r = cfg["regions"]["tc"]
    b = cfg["grids"]["density_bin"]
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    m = pd.read_csv(f"{cfg['paths']['processed']}/p2_metrics.csv", parse_dates=["recurve_time"])
    # 只保留已判定转向且拥有有效转向位置的气旋。
    rec = m[m.recurving == 1].dropna(subset=["recurve_lon", "recurve_lat"])
    pts = rec[["recurve_lon", "recurve_lat"]].to_numpy()
    glon = np.arange(r["lon_min"], r["lon_max"] + b, b)
    glat = np.arange(r["lat_min"], r["lat_max"] + b, b)
    # 条件表达式保证没有转向事件时仍输出形状正确的全零数组。
    H = (np.histogram2d(pts[:, 0], pts[:, 1], bins=[glon, glat])[0].T
         if len(pts) else np.zeros((len(glat) - 1, len(glon) - 1)))
    np.savez(f"{cfg['paths']['processed']}/p2_recurve_density.npz", lon=glon, lat=glat, dens=H)

    s = pd.read_csv(f"{cfg['paths']['processed']}/storms.csv", parse_dates=["genesis_time"]) \
        .merge(m[["sid", "recurving"]], on="sid", how="left")
    s["genesis_month"] = s["genesis_time"].dt.month
    nyr = cfg["periods"]["end"] - cfg["periods"]["freq_start"] + 1
    # 第一列按生成月计算「最终会转向的比例」；第二列按实际转向月计算年均事件数。
    monthly = pd.concat([
        s.groupby("genesis_month")["recurving"].mean().rename("genesis_cohort_ratio"),
        (rec.assign(event_month=rec.recurve_time.dt.month).groupby("event_month").size() / nyr)
        .rename("events_per_year")], axis=1).reindex(range(1, 13))
    # 某月没有任何转向事件时，事件频率应为 0，而不是缺失值。
    monthly["events_per_year"] = monthly["events_per_year"].fillna(0.0)
    monthly.to_csv(f"{cfg['paths']['processed']}/p2_recurve_monthly.csv", index_label="month")
    # 0/1 变量的年度平均值就是该年气旋转向比例。
    s.groupby("season")["recurving"].mean().to_csv(
        f"{cfg['paths']['processed']}/p2_recurve_annual.csv")
    print(f"p2_recurve_*: {len(rec)} recurving events, nyr={nyr}")


if __name__ == "__main__":
    main()
