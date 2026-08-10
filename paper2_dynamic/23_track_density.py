"""路径点密度场（年代际与 ENSO 位相），单位为「每年每格 6 小时轨迹点数」。

产出 processed/density_dec_{decade}.npz、density_phase_{phase}.npz。详见 Docs/02 §4.4。
先逐年计算 2.5° 网格密度，再在年代/显式 ENSO 位相内对年度密度取平均，
避免不完整十年或位相样本数主导结果。
"""
import pandas as pd, numpy as np
from core.utils import load_config


def density(g, glon, glat):
    """统计一个轨迹子集在经纬度箱中的点数，返回 lat×lon 数组。"""
    H, _, _ = np.histogram2d(g["lon"], g["lat"], bins=[glon, glat])
    return H.T                                            # (lat, lon)


def main():
    cfg = load_config()
    r = cfg["regions"]["tc"]
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv")
    tr = tr[tr.wind >= cfg["ts_threshold_kt"]].copy()
    if "nature" in tr:
        tr = tr[(tr.nature == "TS") | tr.nature.isna()]
    b = cfg["grids"]["density_bin"]
    # edges 是网格边界；绘图坐标使用相邻边界的平均值，即网格中心。
    lon_edges = np.arange(r["lon_min"], r["lon_max"] + b, b)
    lat_edges = np.arange(r["lat_min"], r["lat_max"] + b, b)
    lon = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat = (lat_edges[:-1] + lat_edges[1:]) / 2
    tr = tr[tr.lon.between(r["lon_min"], r["lon_max"]) &
            tr.lat.between(r["lat_min"], r["lat_max"])]
    # 字典推导式生成 year→二维年度密度场 的映射。
    annual = {int(y): density(g, lon_edges, lat_edges) for y, g in tr.groupby("season")}

    ndec = 0
    for dec in sorted({(y // 10) * 10 for y in annual}):
        # 找出属于当前十年的所有实际年份，不假定每个十年都有 10 年资料。
        years = [y for y in annual if (y // 10) * 10 == dec]
        np.savez(f"{cfg['paths']['processed']}/density_dec_{dec}.npz",
                 # 先逐年统计再取平均，避免不完整十年因年份少而总量偏低。
                 lon=lon, lat=lat, dens=np.mean([annual[y] for y in years], axis=0),
                 years=years, n_years=len(years), unit="points_per_year")
        ndec += 1

    oni = pd.read_csv(f"{cfg['paths']['raw']}/indices/oni.csv")
    th = cfg["oni_threshold"]
    # 使用显式 >= 和 <= 保证恰好 ±0.5 的年份正确归类。
    oni["phase"] = np.select([oni.jas_oni >= th, oni.jas_oni <= -th],
                             ["El_Nino", "La_Nina"], default="Neutral")
    nph = 0
    for ph, tab in oni.groupby("phase"):
        years = [int(y) for y in tab.season if int(y) in annual]
        if not years:
            continue
        np.savez(f"{cfg['paths']['processed']}/density_phase_{ph}.npz",
                 lon=lon, lat=lat, dens=np.mean([annual[y] for y in years], axis=0),
                 years=years, n_years=len(years), unit="points_per_year")
        nph += 1
    print(f"density: {ndec} decade files, {nph} phase files, {len(annual)} annual years")


if __name__ == "__main__":
    main()
