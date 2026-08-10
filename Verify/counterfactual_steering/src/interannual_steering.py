"""年际引导流分布反事实试验（事后探索，非预注册）。

⚠ 本分析在看到主试验结果之后设计，**不属于 `DESIGN.md` 冻结的预设判据体系**。
其结论只能作为探索性线索，不得当作预注册检验的结果引用。主试验结论不受本文件影响。

动机
----
主试验用两期**气候态**引导流，把年际变率平均掉了，结果显示两期均值只差 0.14σ、
不足以驱动路径重组。剩下的可能性是重组来自引导流**分布（频率）**的变化而非均值位移。
本试验在不引入任何新数据的前提下，把气候态换成**逐年月场**，向该方向推进一格。

设计
----
- 参照组 A：P1 生成点，各风暴使用**自己那一年**的月场
- 反事实 B：P1 生成点，各风暴改用**随机抽取的某个 P2 年**的同日历月场
- d_ia = B − A，隔离引导流年际分布变化的效应
- 随机配对重复 `N_REAL` 次取平均，抑制抽样噪声
- 模型参数沿用主试验全期标定值，不重新标定

局限
----
月场仍无法表达天气尺度（数日）的引导形势差异。本试验只推进到**年际**一层，
不等价于 6 小时分辨率的天气尺度检验。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

spec = importlib.util.spec_from_file_location("cfs", HERE / "counterfactual_steering.py")
cfs = importlib.util.module_from_spec(spec)
sys.modules["cfs"] = cfs
spec.loader.exec_module(cfs)

N_REAL = 20   # 稳定性复核：提高集合数，检验 r 是否随之漂移
SEED = 20260804


def _interp2(field: np.ndarray, lat0: float, dlat: float, nlat: int,
             lon0: float, dlon: float, nlon: int, la: float, lo: float) -> float:
    """标量双线性插值。

    与 `cfs.bilinear` 数学等价，但不创建临时数组——原实现每步为单个点分配
    多个 numpy 数组，在 21 次积分、约 35 万个点的规模下开销占绝对主导。
    """
    fy = (la - lat0) / dlat
    fx = (lo - lon0) / dlon
    j = int(fy); i = int(fx)
    if j < 0 or j >= nlat - 1 or i < 0 or i >= nlon - 1:
        return float("nan")
    wy = fy - j; wx = fx - i
    return (field[j, i] * (1 - wy) * (1 - wx)
            + field[j, i + 1] * (1 - wy) * wx
            + field[j + 1, i] * wy * (1 - wx)
            + field[j + 1, i + 1] * wy * wx)


def integrate_yearly(seeds: pd.DataFrame, year_map: dict, cache: dict,
                     par: dict, region: dict) -> pd.DataFrame:
    """逐风暴使用其指定年份的月场做平流。"""
    rows = []
    a, ub, vb = par["alpha"], par["u_beta"], par["v_beta"]
    lat, lon = cache["lat"], cache["lon"]
    lat0, dlat, nlat = lat[0], lat[1] - lat[0], len(lat)
    lon0, dlon, nlon = lon[0], lon[1] - lon[0], len(lon)
    u_st, v_st = cache["u"], cache["v"]
    # (year, month) -> 月场行号，避免每个风暴重复线性搜索
    idx_of = {(y, m): k for k, (y, m) in enumerate(zip(cache["yrs"], cache["mons"]))}
    lo_min, lo_max = region["lon_min"], region["lon_max"]
    la_min, la_max = region["lat_min"], region["lat_max"]
    for r in seeds.itertuples():
        k = idx_of.get((year_map[r.sid], r.month))
        if k is None:
            continue
        u_f = u_st[k]; v_f = v_st[k]
        la, lo = r.lat, r.lon
        for _ in range(int(r.steps)):
            if not (lo_min <= lo <= lo_max and la_min <= la <= la_max):
                break
            rows.append((r.sid, r.season, lo, la))
            su = _interp2(u_f, lat0, dlat, nlat, lon0, dlon, nlon, la, lo)
            sv = _interp2(v_f, lat0, dlat, nlat, lon0, dlon, nlon, la, lo)
            if not (su == su and sv == sv):   # NaN 检查
                break
            lo += (a * su + ub) * cfs.DT / (cfs.EARTH_R * np.cos(la * cfs.RAD)) / cfs.RAD
            la += (a * sv + vb) * cfs.DT / cfs.EARTH_R / cfs.RAD
    return pd.DataFrame(rows, columns=["sid", "season", "lon", "lat"])


def main() -> None:
    cfg = cfs.load_config()
    tracks, region = cfs.load_tracks(cfg)
    lon_e, lat_e = cfs.grid_edges(cfg, region)
    p1y = np.arange(cfs.START, cfs.SPLIT + 1)
    p2y = np.arange(cfs.SPLIT + 1, cfs.END + 1)

    f1 = cfs.annual_relative_field(tracks[tracks.season <= cfs.SPLIT], p1y, lon_e, lat_e)
    f2 = cfs.annual_relative_field(tracks[tracks.season > cfs.SPLIT], p2y, lon_e, lat_e)
    d_obs = f2.mean(0) - f1.mean(0)

    cache = cfs._load_steering_once()
    # 直接读主试验存档的标定值：既省去重复标定的开销，也保证参数与主试验完全一致
    prev = json.loads((EXP / "results" / "criteria.json").read_text(encoding="utf-8"))
    par = prev["calibration"]
    print(f"沿用主试验标定 α={par['alpha']:.3f} beta=({par['u_beta']:+.2f},{par['v_beta']:+.2f})"
          f"（读自 results/criteria.json，未重新标定）")

    g = tracks.groupby("sid")
    seeds = g.first().reset_index()[["sid", "season", "lat", "lon", "iso_time"]]
    seeds["month"] = seeds["iso_time"].dt.month
    seeds["steps"] = g.size().values
    s1 = seeds[seeds.season <= cfs.SPLIT].reset_index(drop=True)

    # 对称设计：A、B 两组都用随机抽年并取 N_REAL 次平均。
    # 若 A 只用"各风暴自己那一年"（单次实现）而 B 取多次平均，两组平滑程度不同，
    # 差值会混入 A 的抽样噪声，使相关系数被压向 0。必须对称。
    rng = np.random.default_rng(SEED)

    def ensemble(years: np.ndarray, tag: str) -> tuple[np.ndarray, float]:
        acc = None
        reals = []
        for k in range(N_REAL):
            m = {r.sid: int(rng.choice(years)) for r in s1.itertuples()}
            syn = integrate_yearly(s1, m, cache, par, region)
            f = cfs.annual_relative_field(syn, p1y, lon_e, lat_e).mean(0)
            reals.append(f)
            acc = f if acc is None else acc + f
        stack = np.asarray(reals)
        # 实现间标准差的场均值：衡量该组自身的抽样噪声水平
        noise = float(stack.std(0).mean())
        print(f"  {tag} 完成 {N_REAL} 次实现，实现间噪声 {noise:.3e}")
        return acc / N_REAL, noise

    fld_a, noise_a = ensemble(p1y, "参照组A(随机P1年)")
    fld_b, noise_b = ensemble(p2y, "反事实B(随机P2年)")

    d_ia = fld_b - fld_a
    r_ia = float(np.corrcoef(d_ia.ravel(), d_obs.ravel())[0, 1])
    ratio = cfs.tvd(d_ia) / cfs.tvd(d_obs)

    print("\n== 年际分布反事实（事后探索，非预注册）==")
    print(f"  r(d_ia, d_obs)          = {r_ia:+.3f}")
    print(f"  TVD(d_ia)/TVD(d_obs)    = {ratio:.3f}")
    print(f"  对照：气候态版 r        = {prev['r_steer_obs']:+.3f}，TVD比 {prev['tvd_ratio_steer']:.3f}")
    gain = "是" if abs(r_ia) > abs(prev["r_steer_obs"]) + 0.05 else "否"
    print(f"  换成年际分布后是否实质改善：{gain}")

    out = {
        "note": "事后探索，非预注册；不得作为预设判据结果引用",
        "n_realizations": N_REAL,
        "seed": SEED,
        "r_ia_obs": r_ia,
        "noise_a": noise_a,
        "noise_b": noise_b,
        "signal_to_noise": float(np.abs(d_ia).mean() / max(noise_a, noise_b)),
        "tvd_ratio_ia": ratio,
        "ref_r_steer_clim": prev["r_steer_obs"],
        "ref_tvd_ratio_clim": prev["tvd_ratio_steer"],
    }
    (EXP / "results" / "interannual.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(EXP / "results" / "interannual_fields.npz", d_ia=d_ia, d_obs=d_obs)
    print(f"\n结果写入 {EXP/'results'/'interannual.json'}")


if __name__ == "__main__":
    main()
