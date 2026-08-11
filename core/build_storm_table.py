"""把原始最佳路径整理成两张分析就绪表（两篇论文共同的“主表脊梁”）。

产出：data/processed/tracks.csv（点级，每行一个主时次观测点）、
      data/processed/storms.csv（风暴级，每行一个 TC）。
登陆信息由 landfall_gshhg 回填，本模块保持单一职责。
"""
import pandas as pd
import numpy as np
from core.utils import load_config


def main():
    cfg = load_config()
    # 根据配置选择同一机构的一套风压资料，避免 1 分钟与 10 分钟风速混用。
    wcol = "USA_WIND" if cfg["wind_source"] == "USA" else "WMO_WIND"
    pcol = "USA_PRES" if cfg["wind_source"] == "USA" else "WMO_PRES"

    # low_memory=False 让 pandas 先完整判断列类型，减少分块推断产生的混合类型警告。
    df = pd.read_csv(f"{cfg['paths']['raw']}/ibtracs_wp.csv", low_memory=False)
    # 只取下游分析需要的列，并统一为小写项目字段名。
    cols = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "NATURE", wcol, pcol]
    df = df[cols].copy()
    df.columns = ["sid", "season", "name", "iso_time", "lat", "lon", "nature", "wind", "pres"]
    # errors="coerce" 会把无法解析的内容变成 NaT/NaN，便于统一清理。
    df["iso_time"] = pd.to_datetime(df["iso_time"], errors="coerce")
    for c in ["lat", "lon", "wind", "pres"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 时间和位置是所有分析的基础，因此缺少任一项的记录不能保留。
    df = df.dropna(subset=["iso_time", "lat", "lon"])
    df["lon"] = df["lon"] % 360
    # 同一气旋同一时刻若有重复记录，保留文件中最后一条。
    df = df.sort_values(["sid", "iso_time"]).drop_duplicates(["sid", "iso_time"], keep="last")

    # ACE 等标准指标只使用 00/06/12/18 UTC 主时次。
    df = df[df["iso_time"].dt.hour.isin([0, 6, 12, 18])]          # 主同步时次
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    # 主表纳入下限用 record_start（1945，飞机侦察期开端）；缺省回退 freq_start。
    rec_start = cfg["periods"].get("record_start", cfg["periods"]["freq_start"])
    df = df[(df["season"] >= rec_start) &
            (df["season"] <= cfg["periods"]["end"])]

    ts = cfg["ts_threshold_kt"]
    # NATURE="TS" 表示热带气旋阶段；缺失 nature 时暂时保留并在 QC 中说明。
    tropical = df[(df["nature"] == "TS") | df["nature"].isna()]
    # groupby 后取每个 SID 最大风，仅保留生命期至少达到 TS 阈值的系统。
    keep = tropical.groupby("sid")["wind"].max()
    keep = keep[keep >= ts].index                                # 只选 TC
    df = df[df["sid"].isin(keep)]
    df.to_csv(f"{cfg['paths']['processed']}/tracks.csv", index=False)

    rows = []
    # 逐个气旋汇总。sid 是分组键，g 是该气旋的全部点级记录。
    for sid, g in df.sort_values("iso_time").groupby("sid"):
        g = g.reset_index(drop=True)
        ga = g[(g["nature"] == "TS") | g["nature"].isna()].reset_index(drop=True)
        # 布尔条件筛选出达到 TS 的点，iloc[0] 取得第一次达到阈值的记录。
        gen = ga[ga["wind"] >= ts].iloc[0]                       # 热带阶段首次达 TS
        # idxmax 返回最大风速所在的行号，即 LMI 时刻。
        lmi_idx = ga["wind"].idxmax()
        # ACE 只累加达到 TS 阈值的 6 小时风速平方；1e-4 是 ACE 惯用缩放因子。
        ace = np.nansum((ga.loc[ga["wind"] >= ts, "wind"]) ** 2) * 1e-4
        # diff 计算相邻时次差；转成小时后可发现缺测造成的长时间间隔。
        gaps = ga["iso_time"].diff().dt.total_seconds().div(3600)
        rows.append({
            "sid": sid, "season": g["season"].iloc[0], "name": g["name"].iloc[0],
            "genesis_time": gen["iso_time"],
            "genesis_lat": gen["lat"], "genesis_lon": gen["lon"],
            "lmi_kt": ga["wind"].max(),
            "lmi_time": ga["iso_time"].iloc[lmi_idx],
            "lmi_lat": ga["lat"].iloc[lmi_idx], "lmi_lon": ga["lon"].iloc[lmi_idx],
            "ace": ace,
            "is_super": int(ga["wind"].max() >= cfg["super_typhoon_kt"]),
            # 分时段标志：卫星期(1966+，可信频数) 与均一强度期(1982+)。
            "is_satellite_era": int(g["season"].iloc[0] >= cfg["periods"]["freq_start"]),
            "is_intensity_period": int(g["season"].iloc[0] >= cfg["periods"]["intensity_start"]),
            "n_track_points": len(g),
            "wind_valid_frac": float(g["wind"].notna().mean()),
            "max_gap_h": float(gaps.max()) if gaps.notna().any() else np.nan,
        })
    # rows 是“字典列表”，DataFrame 会把每个字典转成一行、键转成列名。
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['processed']}/storms.csv", index=False)
    print("storms:", len(rows), " tracks:", df.shape)


if __name__ == "__main__":
    main()
