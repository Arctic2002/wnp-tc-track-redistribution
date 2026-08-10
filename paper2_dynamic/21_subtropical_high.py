"""西北太平洋副高四项 ERA5 对应指数（面积/强度/西伸脊点/脊线）。

产出 processed/p2_wnpsh.csv。详见 Docs/02 §4.2。
读取 dynamic 框 ERA5（era5_wnp_dynamic_plev.nc），其覆盖 80–180°E、0–65°N，
完整包含副高定义域（thermo 框只到 40°N，不足以诊断脊线/西伸至 60°N）。
"""
import xarray as xr, numpy as np, pandas as pd
from core.utils import load_config


def cell_area_km2(field):
    """规则经纬网近似格点面积；面积指数因此不随网格分辨率改变。"""
    lat = field["latitude"]
    # np.diff 求相邻坐标间隔；median 降低浮点微小误差影响；deg2rad 转弧度。
    dlat = np.deg2rad(abs(float(np.median(np.diff(lat)))))
    dlon = np.deg2rad(abs(float(np.median(np.diff(field["longitude"])))))
    # 单纬度面积是一维数组，broadcast_like 把它扩展成与 field 相同的二维网格。
    return (6371.0 ** 2 * dlat * dlon * np.cos(np.deg2rad(lat))).broadcast_like(field)


def ridge_latitude(uu):
    """寻找各经度上由东风(u<0)向北过渡到西风(u>=0)的零线纬度。"""
    ridge = []
    ub = uu.sel(latitude=slice(60, 10), longitude=slice(110, 150)).sortby("latitude")
    for lo in ub.longitude.values:
        lat = ub.latitude.values
        col = ub.sel(longitude=lo).values
        # np.where 返回满足符号转换条件的位置索引。
        idx = np.where((col[:-1] < 0) & (col[1:] >= 0))[0]  # 向北由东风转西风
        if len(idx):
            i = idx[0]
            # 用相邻两个格点的 u 值做线性插值，估计 u 恰好等于 0 的纬度。
            ridge.append(lat[i] - col[i] * (lat[i + 1] - lat[i]) / (col[i + 1] - col[i]))
    return float(np.mean(ridge)) if ridge else np.nan


def main():
    cfg = load_config()
    plev = xr.open_dataset(f"{cfg['paths']['interim']}/era5_wnp_dynamic_plev.nc")
    # ERA5 z 是位势(m²/s²)：除重力加速度得到位势高度 m，再除 10 得到 dagpm。
    H = plev["z"].sel(level=500) / 9.80665 / 10.0          # → dagpm
    u = plev["u"].sel(level=500)
    mon = cfg["typhoon_season"]
    # 先筛 6–10 月，再按年份分组并对 time 维取平均；groupby 用字符串形式在
    # 已筛选对象上解析，避免按整段 time 分组导致长度不匹配。
    Hy = H.sel(time=H["time"].dt.month.isin(mon)).groupby("time.year").mean("time")
    uy = u.sel(time=u["time"].dt.month.isin(mon)).groupby("time.year").mean("time")

    recs = []
    for yr in Hy["year"].values:
        h = Hy.sel(year=yr)
        uu = uy.sel(year=yr)
        reg = h.sel(latitude=slice(60, 10), longitude=slice(110, 180))
        # mask 是布尔二维数组，True 表示该格点位于 588 dagpm 范围内。
        mask = reg >= 588
        akm2 = cell_area_km2(reg)
        area = float(akm2.where(mask).sum() / 1e6)
        inten = float(((reg - 587).where(mask) * akm2).sum() / 1e6)
        regw = h.sel(latitude=slice(60, 10), longitude=slice(90, 180)) >= 588
        # any("latitude") 判断每条经线上是否至少有一个 588 区格点。
        lons = regw["longitude"].where(regw.any("latitude"), drop=True)
        west = float(lons.min()) if lons.size else np.nan
        ridge_lat = ridge_latitude(uu)
        recs.append({"season": int(yr), "wpsh_area": area, "wpsh_intensity": inten,
                     "west_ridge_point": west, "ridge_line": ridge_lat})
    out = pd.DataFrame(recs)
    out.to_csv(f"{cfg['paths']['processed']}/p2_wnpsh.csv", index=False)
    print(f"p2_wnpsh.csv: {len(out)} years; mean area={out.wpsh_area.mean():.2f} Mkm^2, "
          f"mean west_ridge={out.west_ridge_point.mean():.1f}E, "
          f"mean ridge_lat={out.ridge_line.mean():.1f}N")


if __name__ == "__main__":
    main()
