"""分型代表路径与季节/ENSO 位相剖面（全模糊隶属度加权）。

产出 processed/p2_cluster_meantracks.npz、p2_cluster_monthly.csv、
p2_cluster_phase.csv、p2_cluster_phase_rate.csv。详见 Docs/02 §4.13。
代表路径和类别频率均以完整隶属度加权；月份输出构成比例，ENSO 输出按位相年数
归一的频率；经度在求均值前展开。重采样与经度处理与聚类模块一致。
"""
import pandas as pd, numpy as np
from core.utils import load_config

N = 20


def resample(g):
    """使用与聚类模块一致的方法，把轨迹连续插值为 20 点。"""
    g = g.sort_values("iso_time")
    h = (g.iso_time - g.iso_time.iloc[0]).dt.total_seconds().to_numpy() / 3600
    q = np.linspace(0, h[-1], N)
    lon = np.rad2deg(np.unwrap(np.deg2rad(g.lon)))
    return np.interp(q, h, lon), np.interp(q, h, g.lat)


def main():
    cfg = load_config()
    cl = pd.read_csv(f"{cfg['paths']['processed']}/p2_clusters.csv")
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    s = pd.read_csv(f"{cfg['paths']['processed']}/storms.csv", parse_dates=["genesis_time"])
    oni = pd.read_csv(f"{cfg['paths']['raw']}/indices/oni.csv")
    th = cfg["oni_threshold"]
    oni["phase"] = np.select([oni.jas_oni >= th, oni.jas_oni <= -th],
                             ["El_Nino", "La_Nina"], default="Neutral")
    # 自动寻找 membership_c0、membership_c1... 列，因此不需要提前知道最终簇数。
    mcols = [c for c in cl.columns if c.startswith("membership_c")]
    s = s.merge(cl[["sid"] + mcols], on="sid").merge(oni[["season", "phase"]], on="season", how="left")
    s["month"] = s["genesis_time"].dt.month

    means = {}
    # 每个簇分别计算隶属度加权的代表路径。
    for j, mc in enumerate(mcols):
        lons, lats, weights = [], [], []
        for _, row in cl.iterrows():
            sid = row.sid
            t = tr[tr["sid"] == sid]
            if len(t) >= 4:
                lo, la = resample(t)
                lons.append(lo)
                lats.append(la)
                weights.append(row[mc])
        if lons:
            # np.average 的 weights 参数让高隶属度成员对代表路径贡献更大。
            means[f"c{j}_lon"] = np.average(lons, axis=0, weights=weights) % 360
            means[f"c{j}_lat"] = np.average(lats, axis=0, weights=weights)
    np.savez(f"{cfg['paths']['processed']}/p2_cluster_meantracks.npz", **means)
    # 对每月各簇隶属度求和，再除以该月总和，得到构成比例。
    monthly = s.groupby("month")[mcols].sum()
    monthly = monthly.div(monthly.sum(axis=1), axis=0)
    monthly.to_csv(f"{cfg['paths']['processed']}/p2_cluster_monthly.csv")
    phase = s.groupby("phase")[mcols].sum()
    # 除以每个位相的年份数，避免 El Niño/La Niña 样本年数不同导致总量不可比。
    ny = oni.groupby("phase").size()
    rate = phase.div(ny, axis=0)
    rate.to_csv(f"{cfg['paths']['processed']}/p2_cluster_phase_rate.csv")
    phase.div(phase.sum(axis=1), axis=0).to_csv(
        f"{cfg['paths']['processed']}/p2_cluster_phase.csv")
    print(f"p2_cluster_*: {len(mcols)} clusters profiled")


if __name__ == "__main__":
    main()
