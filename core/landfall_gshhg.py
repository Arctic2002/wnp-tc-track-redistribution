"""用 GSHHG 高精度海岸线做登陆判定，并回填每个 TC 的登陆次数。

产出：data/processed/landfalls.csv（每行一次登陆事件），
      并把 n_landfall 回填进 data/processed/storms.csv。
登陆口径：对相邻 6 小时轨迹段加密后，首次出现海->陆跳变即记登陆。
"""
import numpy as np
import pandas as pd
from core.utils import load_config

# geopandas/rasterio/affine 仅在构建海陆掩膜时需要，故采用惰性导入：
# 这样纯几何函数 densify/is_land 在没有 GIS 依赖的环境下也可被导入和单元测试。

# 每个元组格式为 (名称, (西界, 东界, 南界, 北界))。
# 列表有先后顺序，因此先放面积较小、容易与其他框重叠的地区。
COASTS = [  # 小区域优先，避免台湾/越南被较大的中国盒子抢先匹配
    ("Taiwan", (119, 122.5, 21.5, 25.5)),
    ("Philippines", (117, 127, 5, 19)),
    ("Vietnam", (102, 110, 8, 23)),
    ("Korea", (124, 130, 33, 39)),
    ("Japan", (128, 146, 30, 46)),
    ("China_E", (118, 123, 26, 35)),
    ("China_S", (105, 120, 18, 26)),
]


def attribute_coast(lat, lon):
    """用粗略经纬度框给登陆点赋海岸名称；匹配不到时返回 Other。"""
    for name, (w, e, s, n) in COASTS:
        if w <= lon <= e and s <= lat <= n:
            return name
    return "Other"


def build_land_mask(cfg):
    """读取 GSHHG 多边形并生成海陆掩膜及其定位参数。"""
    # rasterize 把多边形转换为规则海陆栅格；GIS 依赖在此惰性导入。
    import geopandas as gpd
    from rasterio.features import rasterize
    from affine import Affine
    land = gpd.read_file(cfg["gshhg_path"])
    r = cfg["regions"]["dynamic"]
    res = cfg["land_mask_res_deg"]
    # nx/ny 分别是东西、南北方向格点数。
    nx = int((r["lon_max"] - r["lon_min"]) / res)
    ny = int((r["lat_max"] - r["lat_min"]) / res)
    # 栅格原点放在左上角；纬度向下递减，所以南北尺度为 -res。
    transform = Affine.translation(r["lon_min"], r["lat_max"]) * Affine.scale(res, -res)
    mask = rasterize(((g, 1) for g in land.geometry), out_shape=(ny, nx),
                     transform=transform, fill=0, dtype="uint8")
    return mask, r["lon_min"], r["lat_max"], res


def is_land(mask, lon0, lat1, res, lat, lon):
    """把经纬度换算为掩膜行列号；返回 True/False，域外返回 None。"""
    # j 是列号（经度方向），i 是行号（纬度方向）。
    j = int((lon - lon0) / res)
    i = int((lat1 - lat) / res)
    if not (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]):
        return None  # 域外不是海洋；避免从域外进入陆地时产生假登陆
    return bool(mask[i, j] == 1)


def densify(a, b, max_step):
    """在两个轨迹点之间插值。

    返回插值比例 f 及对应纬度、经度；f=0 是点 a，f=1 是点 b。
    a、b 需具备 .lat / .lon 属性（如 DataFrame 的一行）。
    """
    # 取经纬度最大变化量决定插值段数，保证每小段不超过 max_step。
    n = max(1, int(np.ceil(max(abs(b.lat - a.lat), abs(b.lon - a.lon)) / max_step)))
    f = np.linspace(0, 1, n + 1)
    return f, a.lat + f * (b.lat - a.lat), a.lon + f * (b.lon - a.lon)


def main():
    cfg = load_config()
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    mask, lon0, lat1, res = build_land_mask(cfg)
    ev = []
    # 每个 SID 独立检测，防止把两个气旋的相邻记录错误连接。
    for sid, g in tr.sort_values("iso_time").groupby("sid"):
        g = g.reset_index(drop=True)
        last_event = None
        # k 和 k+1 组成一条 6 小时轨迹段。
        for k in range(len(g) - 1):
            a, b = g.iloc[k], g.iloc[k + 1]
            f, lat, lon = densify(a, b, res / 2)
            # zip 把同位置的纬度和经度配成一对；列表记录每个插值点的海陆状态。
            state = [is_land(mask, lon0, lat1, res, la, lo) for la, lo in zip(lat, lon)]
            for j in range(1, len(state)):
                if state[j] is True and state[j - 1] is False:
                    # 用同一比例 f 在线性时间轴和风速上插值登陆值。
                    tt = a.iso_time + (b.iso_time - a.iso_time) * float(f[j])
                    if last_event is not None and (tt - last_event).total_seconds() / 3600 < cfg["landfall_min_separation_h"]:
                        continue
                    wind = a.wind + float(f[j]) * (b.wind - a.wind) if pd.notna(a.wind) and pd.notna(b.wind) else np.nan
                    ev.append({"sid": sid, "time": tt, "lat": lat[j], "lon": lon[j],
                               "wind": wind, "coast": attribute_coast(lat[j], lon[j])})
                    last_event = tt
                    break
    # 即使 ev 为空，也显式给列名，确保下游读取空 CSV 时不会报“无列”错误。
    columns = ["sid", "time", "lat", "lon", "wind", "coast"]
    lf = pd.DataFrame(ev, columns=columns)
    lf.to_csv(f"{cfg['paths']['processed']}/landfalls.csv", index=False)

    s = pd.read_csv(f"{cfg['paths']['processed']}/storms.csv")
    # size 统计每个 SID 的登陆事件行数，再左连接回全部气旋。
    cnt = (lf.groupby("sid").size().rename("n_landfall")
           if len(lf) else pd.Series(dtype=int, name="n_landfall"))
    s = s.merge(cnt, on="sid", how="left")
    s["n_landfall"] = s["n_landfall"].fillna(0).astype(int)
    s.to_csv(f"{cfg['paths']['processed']}/storms.csv", index=False)
    print("landfalls:", len(lf), " storms updated")


if __name__ == "__main__":
    main()
