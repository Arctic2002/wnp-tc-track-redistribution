"""登陆气候学：逐年登陆气旋比例、年份×海岸事件/气旋计数与登陆强度。

产出 processed/p2_landfall_frac.csv、p2_landfall_by_coast.csv。详见 Docs/02 §4.12。
频数可用全期；平均/中位登陆强度只用 1982 年后（资料均一期），且仅基于源数据有可用
风速的事件。1982 年后源 landfalls.csv 仍可能缺风速（个别"年份×海岸"组可能全部缺测），
相应强度统计记为缺失（NaN），不以 0 或插值伪造——验收标准是"输出与源可用风速一致"。
"""
import pandas as pd
from core.utils import load_config


def main():
    cfg = load_config()
    s = pd.read_csv(f"{cfg['paths']['processed']}/storms.csv")
    lf = pd.read_csv(f"{cfg['paths']['processed']}/landfalls.csv", parse_dates=["time"])
    # 布尔值 True/False 转成 1/0 后，年度均值就是登陆气旋比例。
    s["made_landfall"] = (s["n_landfall"] > 0).astype(int)
    s.groupby("season")["made_landfall"].agg(
        n_tc="size", landfall_storms="sum", landfall_frac="mean") \
        .to_csv(f"{cfg['paths']['processed']}/p2_landfall_frac.csv")
    # 登陆事件表本身没有 season 时，从主表按 SID 补入。
    lf = lf.merge(s[["sid", "season"]], on="sid", how="left")
    # size 计算事件数；nunique 计算不同气旋数，二者不能混为同一指标。
    counts = lf.groupby(["season", "coast"]).agg(
        event_count=("sid", "size"), storm_count=("sid", "nunique"))
    post = lf[lf.season >= cfg["periods"]["intensity_start"]]
    # mean/median 自动跳过缺测风速；源缺风速→该组强度记 NaN，不伪造。
    intensity = post.groupby(["season", "coast"])["wind"].agg(wind_mean="mean", wind_median="median")
    # join 按相同的「season, coast」多层索引连接频数和强度统计。
    counts.join(intensity).reset_index().to_csv(
        f"{cfg['paths']['processed']}/p2_landfall_by_coast.csv", index=False)
    # 透明报告：1982 年后源数据缺风速的事件数与"全缺"的年份×海岸组数（属正常，非计算错误）。
    n_missing = int(post["wind"].isna().sum())
    n_groups_allmissing = int((post.groupby(["season", "coast"])["wind"]
                               .apply(lambda x: x.notna().sum() == 0)).sum())
    print(f"p2_landfall_*: {len(lf)} landfall events, "
          f"{lf.coast.nunique()} coasts, {s.made_landfall.sum()} landfalling storms; "
          f"post-{cfg['periods']['intensity_start']} missing wind: {n_missing} events, "
          f"{n_groups_allmissing} all-missing season×coast groups")


if __name__ == "__main__":
    main()
