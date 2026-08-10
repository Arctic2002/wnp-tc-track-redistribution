"""逐 TC 路径长度、生命期、移动速度分布。

产出 processed/p2_distributions.csv。详见 Docs/02 §4.10。
仅在热带 TS 阶段计算路径长度和生命期；相邻路段要求 0–12 小时；移速读取同口径
的 p2_metrics。
"""
import pandas as pd, numpy as np
from core.utils import load_config, haversine


def main():
    cfg = load_config()
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    m = pd.read_csv(f"{cfg['paths']['processed']}/p2_metrics.csv")
    rows = []
    for sid, g in tr.sort_values("iso_time").groupby("sid"):
        # 对每个 SID 重复与主路径指标相同的热带 TS 筛选。
        g = g[g.wind >= cfg["ts_threshold_kt"]].copy()
        if "nature" in g:
            g = g[(g.nature == "TS") | g.nature.isna()]
        if len(g) < 2:
            continue
        la = g["lat"].values
        lo = g["lon"].values
        # np.diff 计算相邻时间差；除以 1 小时后得到浮点小时数。
        dt = np.diff(g.iso_time.values) / np.timedelta64(1, "h")
        seg = haversine(la[:-1], lo[:-1], la[1:], lo[1:])
        ok = (dt > 0) & (dt <= 12) & np.isfinite(seg)
        length = float(seg[ok].sum())
        # 生命期是最后与最早有效 TS 时刻之差。
        life = (g["iso_time"].max() - g["iso_time"].min()) / np.timedelta64(1, "h")
        rows.append({"sid": sid, "track_len_km": length, "lifetime_h": life})
    out = pd.DataFrame(rows).merge(m[["sid", "trans_speed"]], on="sid", how="left")
    out.to_csv(f"{cfg['paths']['processed']}/p2_distributions.csv", index=False)
    print(f"p2_distributions.csv: {len(out)} storms; median len={out.track_len_km.median():.0f} km, "
          f"median life={out.lifetime_h.median():.0f} h")


if __name__ == "__main__":
    main()
