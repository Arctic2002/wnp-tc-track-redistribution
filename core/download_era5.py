"""按年下载 ERA5 月平均单层与气压层数据，便于校验、重试和增量更新。

产出：data/raw/era5/single_{year}.nc 与 plev_{year}.nc（每文件含一个完整年份）。
CDS 请求字段以数据页“Show API request”当时生成内容为准。
"""
from pathlib import Path
# cdsapi 与 Copernicus 服务通信；xarray 只用于检查下载文件能否正常打开。
import cdsapi
import xarray as xr
from core.utils import load_config


def readable(path, expected_months=12):
    """判断年度文件是否可读且月份完整；返回 True/False。"""
    if not path.exists():
        return False
    try:
        # with 语句退出后自动关闭 NetCDF，避免批量检查时占用大量文件句柄。
        with xr.open_dataset(path) as ds:
            # 不同 CDS 版本可能把时间坐标命名为 time 或 valid_time。
            tname = "time" if "time" in ds.coords else "valid_time"
            return ds.sizes.get(tname, 0) == expected_months
    except Exception:
        # 文件损坏、格式错误等情况都视为不可读，随后重新下载。
        return False


def main():
    """按年下载 ERA5 单层和气压层月平均资料。"""
    cfg = load_config()
    c = cdsapi.Client()
    # 生成 "01".."12"；f 字符串中的 :02d 表示用两位数字并在前面补 0。
    mon = [f"{m:02d}" for m in range(1, 13)]
    r = cfg["regions"]["dynamic"]
    # CDS 的 area 顺序固定为 [北, 西, 南, 东]，不能写成常见的西南东北顺序。
    area = [r["lat_max"], r["lon_min"], r["lat_min"], r["lon_max"]]  # N,W,S,E
    outdir = Path(cfg["paths"]["raw"]) / "era5"
    outdir.mkdir(parents=True, exist_ok=True)

    # range 的右端不包含在内，因此 end 需要加 1。
    for year in range(cfg["periods"]["env_start"], cfg["periods"]["end"] + 1):
        single = outdir / f"single_{year}.nc"
        plev = outdir / f"plev_{year}.nc"
        # 两类请求共有的字段集中写在 common，减少重复和口径不一致。
        common = {
            "product_type": ["monthly_averaged_reanalysis"],
            "year": [str(year)], "month": mon, "time": ["00:00"],
            "area": area, "data_format": "netcdf", "download_format": "unarchived",
        }
        # **common 把 common 字典的键值展开到新的请求字典中。
        if not readable(single):
            c.retrieve("reanalysis-era5-single-levels-monthly-means", {
                **common,
                "variable": ["sea_surface_temperature", "mean_sea_level_pressure"],
            }, str(single))
        if not readable(plev):
            # 气压层请求额外指定变量和气压层列表。
            c.retrieve("reanalysis-era5-pressure-levels-monthly-means", {
                **common,
                "variable": ["temperature", "specific_humidity", "relative_humidity",
                             "u_component_of_wind", "v_component_of_wind",
                             "geopotential", "vorticity"],
                "pressure_level": [str(l) for l in cfg["era5_levels"]],
            }, str(plev))


if __name__ == "__main__":
    main()
