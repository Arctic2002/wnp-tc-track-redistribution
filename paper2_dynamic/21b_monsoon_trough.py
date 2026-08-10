"""季风槽 ERA5 代理指标：850 hPa 面积平均相对涡度、槽轴纬度、低层西风东伸位置。

产出 processed/p2_monsoon.csv。详见 Docs/02 §4.2b。
明确标注为「代理指标」；读取 dynamic 框 ERA5（含 u/vo @850）。
"""
import numpy as np, pandas as pd, xarray as xr
from core.utils import load_config


def main():
    cfg = load_config()
    ds = xr.open_dataset(f"{cfg['paths']['interim']}/era5_wnp_dynamic_plev.nc")
    mon = cfg["typhoon_season"]
    # sel(level=850) 选 850 hPa；后面的 sel 筛月份，groupby 再得到逐年台风季平均。
    u = ds["u"].sel(level=850, time=ds.time.dt.month.isin(mon)).groupby("time.year").mean()
    vo = ds["vo"].sel(level=850, time=ds.time.dt.month.isin(mon)).groupby("time.year").mean()
    rows = []
    for year in u.year.values:
        uy, vy = u.sel(year=year), vo.sel(year=year)
        # ERA5 纬度为降序，因此 slice 写成北界 20 到南界 5。
        box = vy.sel(latitude=slice(20, 5), longitude=slice(110, 160))
        w = np.cos(np.deg2rad(box.latitude))
        # weighted(...).mean 对经纬度同时做面积加权平均。
        vort_mean = float(box.weighted(w).mean(("latitude", "longitude")))
        # idxmax("latitude") 返回每条经线上最大涡度所在的纬度坐标。
        axis = box.sortby("latitude").idxmax("latitude").where(box.max("latitude") > 0)
        axis_lat = float(axis.mean("longitude"))
        ulow = uy.sel(latitude=slice(15, 5), longitude=slice(110, 160)).mean("latitude")
        vlow = vy.sel(latitude=slice(15, 5), longitude=slice(110, 160)).mean("latitude")
        # 低层西风(u>0)且相对涡度为正的经度才视为活跃季风槽区域。
        active = ulow.longitude.where((ulow > 0) & (vlow > 0), drop=True)
        east = float(active.max()) if active.size else np.nan
        rows.append({"season": int(year), "mt_vort850": vort_mean,
                     "mt_axis_lat": axis_lat, "mt_westerly_east": east})
    out = pd.DataFrame(rows)
    out.to_csv(f"{cfg['paths']['processed']}/p2_monsoon.csv", index=False)
    print(f"p2_monsoon.csv: {len(out)} years; mean vort={out.mt_vort850.mean():.2e}, "
          f"mean axis_lat={out.mt_axis_lat.mean():.1f}N, "
          f"mean westerly_east={out.mt_westerly_east.mean():.1f}E")


if __name__ == "__main__":
    main()
