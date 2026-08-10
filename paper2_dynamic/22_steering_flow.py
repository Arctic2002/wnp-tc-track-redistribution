"""引导气流（850–300 hPa 深层质量平均风）与垂直切变（200–850 hPa 矢量差）。

产出 interim/steering.nc（月场 u_steer/v_steer/shear）与
processed/p2_steering_annual.csv（固定区域台风季年度引导指数）。详见 Docs/02 §4.3。
读取 dynamic 框 ERA5；其层次含 850/700/600/500/400/300/200 hPa。

实现说明：深层质量平均严格按 `∫u dp/Δp`（dp 权重，非气压值归一）。为在小内存环境
（dynamic 框单变量约 3 GB）下稳定运行，按年份分批 **eager** 载入薄层切片逐年计算，
并把结果**逐年增量写入** netCDF（netCDF4 直写），峰值内存仅约单年薄层，
避免一次性持有/拼接整段场而 OOM。

注意：月平均区域风未去除气旋本体涡旋与 beta 漂移，只用于气候态环流解释，
不能替代逐 TC 日尺度引导诊断。
"""
import xarray as xr, numpy as np, pandas as pd, netCDF4
from core.utils import load_config

TUNITS = "hours since 1940-01-01 00:00:00"
TCAL = "standard"


def main():
    cfg = load_config()
    spath = f"{cfg['paths']['interim']}/steering.nc"
    plev = xr.open_dataset(f"{cfg['paths']['interim']}/era5_wnp_dynamic_plev.nc")
    p = plev["level"]
    # 引导层 300–850 hPa，升序后用于沿气压积分；总厚度 Δp=550 hPa。
    levs = sorted(float(l) for l in p.values if 300 <= l <= 850)
    depth = max(levs) - min(levs)
    lat = plev["latitude"].values
    lon = plev["longitude"].values
    times = pd.to_datetime(plev["time"].values)
    years = np.unique(plev["time"].dt.year.values)

    # 预创建 netCDF 结构，随后逐年把切片写入，限制峰值内存。
    nc = netCDF4.Dataset(spath, "w")
    nc.createDimension("time", len(times))
    nc.createDimension("latitude", len(lat))
    nc.createDimension("longitude", len(lon))
    tv = nc.createVariable("time", "f8", ("time",))
    tv.units = TUNITS
    tv.calendar = TCAL
    tv[:] = netCDF4.date2num(times.to_pydatetime(), TUNITS, TCAL)
    nc.createVariable("latitude", "f4", ("latitude",))[:] = lat
    nc.createVariable("longitude", "f4", ("longitude",))[:] = lon
    vars3 = {n: nc.createVariable(n, "f4", ("time", "latitude", "longitude"),
                                  zlib=True, complevel=4)
             for n in ("u_steer", "v_steer", "shear")}

    pos = 0
    for yr in years:
        sub = plev.sel(time=plev["time"].dt.year == yr)
        n = sub.sizes["time"]
        # eager 载入该年薄层切片（约几十 MB），避免 dask/惰性持有整段场。
        ul = sub["u"].sel(level=levs).sortby("level").load()
        vl = sub["v"].sel(level=levs).sortby("level").load()
        # 压力坐标质量平均：沿 level 梯形积分再除以总厚度。
        us = (ul.integrate("level") / depth).astype("float32").values
        vs = (vl.integrate("level") / depth).astype("float32").values
        # 切变对 u、v 分别相减再求矢量模，不能直接相减风速大小。
        du = sub["u"].sel(level=200).load() - sub["u"].sel(level=850).load()
        dv = sub["v"].sel(level=200).load() - sub["v"].sel(level=850).load()
        sh = ((du ** 2 + dv ** 2) ** 0.5).astype("float32").values
        vars3["u_steer"][pos:pos + n] = us
        vars3["v_steer"][pos:pos + n] = vs
        vars3["shear"][pos:pos + n] = sh
        pos += n
    nc.close()
    plev.close()

    # 固定区域年度指数用于回归；它不随每年气旋位置变化。从已落盘文件读回（小体积）。
    sds = xr.open_dataset(spath)
    box = sds.sel(latitude=slice(30, 10), longitude=slice(110, 160))
    w = np.cos(np.deg2rad(box.latitude))
    idx = box.weighted(w).mean(("latitude", "longitude"))
    idx = idx.sel(time=idx.time.dt.month.isin(cfg["typhoon_season"]))
    annual = idx.groupby("time.year").mean().rename(year="season").to_dataframe()
    annual = annual.drop(columns=[c for c in ("number", "expver") if c in annual.columns])
    annual.to_csv(f"{cfg['paths']['processed']}/p2_steering_annual.csv")
    fin = float(np.isfinite(sds["u_steer"].isel(time=8)).mean())
    print(f"steering.nc + p2_steering_annual.csv: {len(annual)} years; "
          f"finite u_steer frac={fin:.3f}; mean u_steer={annual.u_steer.mean():.2f} m/s, "
          f"mean shear={annual.shear.mean():.2f} m/s")


if __name__ == "__main__":
    main()
