"""生成 / 热带路径 / 热带阶段终止 三联密度气候态。

产出 processed/p2_triptych_density.npz。详见 Docs/02 §4.9。
生成点取首次 34 kt；路径取热带 TS 阶段；终止点定义为最后一个热带 TS 点
并命名 tropical_end（不把 IBTrACS 最后记录点称为物理消亡）。三类均按有效年数归一。
"""
import pandas as pd, numpy as np
from core.utils import load_config


def hist(x, y, glon, glat):
    """把经纬度点放入二维网格，返回适合地图绘制的 lat×lon 数组。"""
    H, _, _ = np.histogram2d(x, y, bins=[glon, glat])
    return H.T


def main():
    cfg = load_config()
    r = cfg["regions"]["tc"]
    b = cfg["grids"]["density_bin"]
    glon = np.arange(r["lon_min"], r["lon_max"] + b, b)
    glat = np.arange(r["lat_min"], r["lat_max"] + b, b)
    s = pd.read_csv(f"{cfg['paths']['processed']}/storms.csv")
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    # stage 只保留达到 TS 阈值且仍为热带性质的轨迹点。
    stage = tr[tr.wind >= cfg["ts_threshold_kt"]].copy()
    if "nature" in stage:
        stage = stage[(stage.nature == "TS") | stage.nature.isna()]
    # groupby 后 tail(1) 取得每个气旋最后一个热带 TS 点。
    lys = stage.sort_values("iso_time").groupby("sid").tail(1)
    nyr = cfg["periods"]["end"] - cfg["periods"]["freq_start"] + 1
    # 文件中保存网格中心供绘图；glon/glat 本身是直方图边界。
    lon = (glon[:-1] + glon[1:]) / 2
    lat = (glat[:-1] + glat[1:]) / 2
    np.savez(f"{cfg['paths']['processed']}/p2_triptych_density.npz", lon=lon, lat=lat,
             genesis=hist(s["genesis_lon"], s["genesis_lat"], glon, glat) / nyr,
             track=hist(stage["lon"], stage["lat"], glon, glat) / nyr,
             tropical_end=hist(lys["lon"], lys["lat"], glon, glat) / nyr, n_years=nyr)
    print(f"p2_triptych_density.npz written; n_years={nyr}, storms={len(s)}")


if __name__ == "__main__":
    main()
