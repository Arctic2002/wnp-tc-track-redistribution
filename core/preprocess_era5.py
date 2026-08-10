"""合并/规范化 ERA5 原始环境场，裁到各自空间域，压缩写出 interim 产物。

本仓库实际下载按用途拆成两套空间域（详见 README 与 Docs/00 §4.4），因此产出三件：

| interim 产物                  | 空间域            | 变量              | 用途      |
|------------------------------|------------------|-------------------|-----------|
| era5_wnp_single.nc           | thermo 100-180/0-40 | sst, msl       | Paper I   |
| era5_wnp_plev.nc             | thermo 100-180/0-40 | t,q,r,u,v,z,vo | Paper I   |
| era5_wnp_dynamic_plev.nc     | dynamic 80-180/0-65 | u,v,z,vo       | Paper II  |

每个产物的源文件按候选顺序探测：先找文档式逐年分块 data/raw/era5/{tag}_*.nc，
再回退到本仓库的合并文件 data/raw/era5_{...}.nc / data/interim/era5_dynamic_plev.nc。
保留断点续跑（合并文件不比任一源文件旧则跳过）与临时文件原子改名。
"""
from pathlib import Path
import numpy as np
import xarray as xr
from core.utils import load_config


def normalize(ds):
    """统一一个 ERA5 Dataset 的坐标名称、方向和时间顺序。"""
    # aliases 左侧是可能遇到的旧/新名称，右侧是项目统一采用的名称。
    aliases = {"valid_time": "time", "pressure_level": "level",
               "lat": "latitude", "lon": "longitude"}
    # 字典推导式只保留数据中实际存在的别名，避免 rename 不存在的坐标时报错。
    ds = ds.rename({k: v for k, v in aliases.items() if k in ds.dims or k in ds.coords})

    # % 360 把负经度转换到 0-360，例如 -170 转为 190。
    lon = ds["longitude"] % 360
    ds = ds.assign_coords(longitude=lon).sortby("longitude")
    # 项目约定纬度北->南（降序），时间则由早到晚（升序）。
    ds = ds.sortby("latitude", ascending=False).sortby("time")
    # 某些分块文件边界可能重复月份，先去重再合并。
    _, keep = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=sorted(keep))


def subset(ds, region):
    """按配置字典裁剪经纬度；ERA5 纬度降序，所以 slice 先北后南。"""
    return ds.sel(longitude=slice(region["lon_min"], region["lon_max"]),
                  latitude=slice(region["lat_max"], region["lat_min"]))


def _is_fresh(out_path, src_files):
    """合并文件存在、可打开且 mtime 不早于任一源文件，则视为“新鲜”可跳过。"""
    out_path = Path(out_path)
    if not out_path.exists():
        return False
    try:
        with xr.open_dataset(out_path):
            pass
    except Exception:
        return False
    newest = max(Path(f).stat().st_mtime for f in src_files)
    return out_path.stat().st_mtime >= newest


def _find_sources(cfg, candidates):
    """按候选顺序返回第一组存在的源文件列表（glob 或单文件）。"""
    raw = Path(cfg["paths"]["raw"])
    interim = Path(cfg["paths"]["interim"])
    base = {"raw": raw, "interim": interim}
    for root, pattern in candidates:
        d = base[root]
        if "*" in pattern:
            files = sorted(d.glob(pattern))
            if files:
                return files
        else:
            p = d / pattern
            if p.exists():
                return [p]
    return []


# 每个产物：(输出名, 空间域键, [源候选(root, 模式)...])
# 源候选按优先级排列：先文档式逐年分块，再回退到本仓库现有的合并文件。
JOBS = {
    "single": ("era5_wnp_single.nc", "thermo",
               [("raw", "era5/single_*.nc"), ("raw", "era5_single.nc")]),
    "plev": ("era5_wnp_plev.nc", "thermo",
             [("raw", "era5/plev_*.nc"), ("raw", "era5_plev.nc")]),
    "dynamic_plev": ("era5_wnp_dynamic_plev.nc", "dynamic",
                     [("raw", "era5/dynamic_plev_*.nc"), ("raw", "era5_dynamic_plev.nc"),
                      ("interim", "era5_dynamic_plev.nc")]),
}


def process(cfg, tag):
    """处理单个产物 tag（single / plev / dynamic_plev）。"""
    out_name, region_key, candidates = JOBS[tag]
    out = f"{cfg['paths']['interim']}/{out_name}"
    files = _find_sources(cfg, candidates)
    if not files:
        # 找不到源，但目标已存在：视为外部已就绪，跳过；否则报错。
        if Path(out).exists():
            print(f"skip {tag}: 未找到源文件，但 {out_name} 已存在")
            return
        raise FileNotFoundError(f"{tag}: 未找到任何源文件，候选={candidates}")
    if _is_fresh(out, files):
        print(f"skip {tag}: {out_name} 已是最新")
        return
    ds = normalize(xr.open_mfdataset(files, combine="by_coords"))
    ds = subset(ds, cfg["regions"][region_key])
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    tmp = Path(out).with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(out)
    print(f"preprocessed {tag} -> {out_name}  dims={dict(ds.sizes)}")


def main(tags=None):
    """tags 为 None 时处理全部产物；可传子集如 ['single'] 单独处理。"""
    cfg = load_config()
    for tag in (tags or list(JOBS)):
        process(cfg, tag)


if __name__ == "__main__":
    main()
