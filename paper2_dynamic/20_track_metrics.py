"""逐 TC 路径指标：时间加权移速 + 持续转向判定，合并主表。

产出 processed/p2_metrics.csv。详见 Docs/02 §4.1。
移速只用生成后热带 TS、首次登陆前、0–12 小时有效路段，以总距离/总时间计算；
转向先 unwrap 经度并平滑，要求 20°N 以北且随后至少 24 小时持续净东移和北移。
"""
import pandas as pd, numpy as np
from pandas.errors import EmptyDataError
from core.utils import load_config, haversine


def tropical_stage(g, ts):
    """从单个气旋轨迹 g 中筛出达到 TS 阈值且仍属热带性质的点。"""
    out = g[g["wind"] >= ts].copy()
    # 若 nature 列有实际信息，就只保留 TS 或暂时缺失的点。
    if "nature" in out and out["nature"].notna().any():
        out = out[(out["nature"] == "TS") | out["nature"].isna()]
    return out.sort_values("iso_time")


def translation_speed(g, ts, t_landfall=None):
    """计算首次登陆前热带 TS 阶段的时间加权平均移速，单位 km/h。"""
    g = tropical_stage(g, ts)
    g = g.sort_values("iso_time")
    if t_landfall is not None:
        g = g[g["iso_time"] <= t_landfall]          # 只取首次登陆前路段
    if len(g) < 2:
        return np.nan
    # [:-1] 和 [1:] 分别表示每条路段的起点数组与终点数组。
    d = haversine(g["lat"].values[:-1], g["lon"].values[:-1],
                  g["lat"].values[1:],  g["lon"].values[1:])
    dt = np.diff(g["iso_time"].values) / np.timedelta64(1, "h")
    # 只接受有限距离、正时间差且不超过 12 小时的路段。
    ok = np.isfinite(d) & (dt > 0) & (dt <= 12)
    # 总距离/总时间可正确处理 6 小时、12 小时等不同长度路段。
    return float(d[ok].sum() / dt[ok].sum()) if ok.any() else np.nan


def recurvature(g, ts):
    """判断是否持续转向，并返回标志、时间、纬度、经度。"""
    g = tropical_stage(g, ts).reset_index(drop=True)
    if len(g) < 6:
        return 0, pd.NaT, np.nan, np.nan
    # unwrap 把 179°→181° 视为跨日期变更线的小变化，而不是 179°→-179° 的大跳跃。
    lon = np.rad2deg(np.unwrap(np.deg2rad(g["lon"].to_numpy())))
    # 三点滚动平均抑制单个定位点的小幅摆动；center=True 让窗口以当前点为中心。
    lat = g["lat"].rolling(3, center=True, min_periods=1).mean().to_numpy()
    lon = pd.Series(lon).rolling(3, center=True, min_periods=1).mean().to_numpy()
    for i in range(1, len(g) - 4):
        hours = (g.iso_time.iloc[i + 4] - g.iso_time.iloc[i]).total_seconds() / 3600
        # 当前经度接近截至该时刻的最西位置，才把它作为候选转向点。
        at_west_edge = lon[i] <= np.nanmin(lon[:i + 1]) + 0.5
        if hours >= 24 and lat[i] >= 20 and at_west_edge and \
           lon[i + 4] - lon[i] >= 2 and lat[i + 4] - lat[i] >= 2:
            return 1, g.iso_time.iloc[i], lat[i], lon[i] % 360
    return 0, pd.NaT, np.nan, np.nan


def main():
    cfg = load_config()
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    s = pd.read_csv(f"{cfg['paths']['processed']}/storms.csv", parse_dates=["genesis_time"])
    try:
        lf = pd.read_csv(f"{cfg['paths']['processed']}/landfalls.csv", parse_dates=["time"])
        # to_dict 生成 sid→首次登陆时间 的快速查找表。
        first_lf = lf.groupby("sid")["time"].min().to_dict()
    except (FileNotFoundError, EmptyDataError):
        first_lf = {}

    ts = cfg["ts_threshold_kt"]
    rows = []
    for sid, g in tr.groupby("sid"):
        # first_lf.get(sid) 在无登陆记录时返回 None。
        rec, rt, rlat, rlon = recurvature(g, ts)
        rows.append({"sid": sid,
                     "trans_speed": translation_speed(g, ts, first_lf.get(sid)),
                     "recurving": rec, "recurve_time": rt,
                     "recurve_lat": rlat, "recurve_lon": rlon})
    m = pd.DataFrame(rows).merge(s, on="sid")          # s 含 n_landfall
    m.to_csv(f"{cfg['paths']['processed']}/p2_metrics.csv", index=False)
    print(f"p2_metrics.csv: {len(m)} storms, recurving={int(m.recurving.sum())}, "
          f"median speed={m.trans_speed.median():.1f} km/h")


if __name__ == "__main__":
    main()
