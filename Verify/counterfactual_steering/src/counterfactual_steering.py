"""反事实引导流轨迹试验。

设计与预设判据见 `../docs/DESIGN.md`（判据在跑数之前已冻结）。

把生成点与引导流分别替换为前后两期取值，检验观测到的路径场差异能否由引导流变化产生。
模型参数（α、beta 漂移）在全期标定一次后固定，各情景之间唯一的差别是引导流气候态与
生成点集合，不存在事后调参空间。

用法：
    python Verify/counterfactual_steering/src/counterfactual_steering.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent.parent

START, END, SPLIT = 1966, 2025, 1995
EARTH_R = 6371.0e3
DT = 6 * 3600.0  # 6 小时，秒
RAD = np.pi / 180.0


# ------------------------------------------------------------------ 观测侧

def load_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


def load_tracks(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """严格复用 28_spatial_redistribution.py 的筛选口径。"""
    t = pd.read_csv(ROOT / "data" / "processed" / "tracks.csv")
    t = t.loc[
        t["season"].between(START, END)
        & (t["wind"] >= cfg["ts_threshold_kt"])
        & ((t["nature"] == "TS") | t["nature"].isna())
    ].copy()
    r = cfg["regions"]["tc"]
    t = t.loc[
        t["lon"].between(r["lon_min"], r["lon_max"])
        & t["lat"].between(r["lat_min"], r["lat_max"])
    ].copy()
    t["iso_time"] = pd.to_datetime(t["iso_time"])
    return t.sort_values(["sid", "iso_time"]).reset_index(drop=True), r


def grid_edges(cfg: dict, region: dict) -> tuple[np.ndarray, np.ndarray]:
    b = cfg["grids"]["density_bin"]
    return (
        np.arange(region["lon_min"], region["lon_max"] + b, b),
        np.arange(region["lat_min"], region["lat_max"] + b, b),
    )


def annual_relative_field(df: pd.DataFrame, years, lon_e, lat_e) -> np.ndarray:
    """逐年归一化的点权重相对密度，等价于 _annual_path_fields 的 point 分支。"""
    out = []
    for y in years:
        a = df.loc[df["season"] == y]
        if len(a) == 0:
            continue
        h = np.histogram2d(a["lon"], a["lat"], bins=[lon_e, lat_e])[0].T
        if h.sum() == 0:
            continue
        out.append(h / h.sum())
    return np.asarray(out)


# ------------------------------------------------------------------ 引导流

_STEER_CACHE: dict | None = None


def _load_steering_once() -> dict:
    """整场一次性读入内存。

    挂载盘上对 867MB 文件做多次随机索引读极慢（实测CPU占用为0，全部阻塞在I/O）。
    优先使用本地副本，并只读一次，之后各期气候态在内存中切片计算。
    """
    global _STEER_CACHE
    if _STEER_CACHE is not None:
        return _STEER_CACHE
    local = Path("/tmp/steering.nc")
    src = local if local.is_file() else ROOT / "data" / "interim" / "steering.nc"
    ds = Dataset(src)
    lat = np.asarray(ds.variables["latitude"][:], dtype=float)
    lon = np.asarray(ds.variables["longitude"][:], dtype=float)
    u = np.asarray(ds.variables["u_steer"][:], dtype=np.float32)
    v = np.asarray(ds.variables["v_steer"][:], dtype=np.float32)
    ds.close()
    n = u.shape[0]
    _STEER_CACHE = {
        "u": u, "v": v, "lat": lat, "lon": lon,
        "yrs": 1940 + np.arange(n) // 12,
        "mons": 1 + np.arange(n) % 12,
    }
    return _STEER_CACHE


def steering_climatology(period_years: range | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (u[12,ny,nx], v[12,ny,nx], lat, lon) 的月气候态。"""
    c = _load_steering_once()
    want = np.isin(c["yrs"], np.asarray(period_years))
    ny, nx = len(c["lat"]), len(c["lon"])
    u = np.empty((12, ny, nx)); v = np.empty((12, ny, nx))
    for m in range(1, 13):
        idx = np.flatnonzero(want & (c["mons"] == m))
        u[m - 1] = c["u"][idx].mean(axis=0)
        v[m - 1] = c["v"][idx].mean(axis=0)
    return u, v, c["lat"], c["lon"]


def bilinear(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             plat: np.ndarray, plon: np.ndarray) -> np.ndarray:
    """在规则网格上做双线性插值；域外返回 NaN。"""
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    fy = (plat - lat[0]) / dlat
    fx = (plon - lon[0]) / dlon
    j = np.floor(fy).astype(int)
    i = np.floor(fx).astype(int)
    ok = (j >= 0) & (j < len(lat) - 1) & (i >= 0) & (i < len(lon) - 1)
    out = np.full(plat.shape, np.nan)
    if not ok.any():
        return out
    jj, ii = j[ok], i[ok]
    wy = (fy[ok] - jj)[:, None].ravel()
    wx = (fx[ok] - ii)[:, None].ravel()
    f00 = field[jj, ii]
    f01 = field[jj, ii + 1]
    f10 = field[jj + 1, ii]
    f11 = field[jj + 1, ii + 1]
    out[ok] = (
        f00 * (1 - wy) * (1 - wx)
        + f01 * (1 - wy) * wx
        + f10 * wy * (1 - wx)
        + f11 * wy * wx
    )
    return out


# ------------------------------------------------------------------ 标定

def calibrate(tracks: pd.DataFrame, u_all, v_all, lat, lon) -> dict:
    """在全期观测位移上最小二乘标定 α 与 beta 漂移，之后固定。

    c_obs = α · V_steer + V_beta
    对 u、v 分量分别回归，斜率共享（α 单一标量）以保持 BAM 的物理形式。
    """
    seg_u, seg_v, st_u, st_v = [], [], [], []
    for _, g in tracks.groupby("sid"):
        if len(g) < 2:
            continue
        la = g["lat"].to_numpy()
        lo = g["lon"].to_numpy()
        mo = g["iso_time"].dt.month.to_numpy()
        # 观测位移速度（m/s）
        du = (lo[1:] - lo[:-1]) * RAD * EARTH_R * np.cos(0.5 * (la[1:] + la[:-1]) * RAD) / DT
        dv = (la[1:] - la[:-1]) * RAD * EARTH_R / DT
        su = np.empty(len(du)); sv = np.empty(len(dv))
        for k in range(len(du)):
            m = mo[k] - 1
            su[k] = bilinear(u_all[m], lat, lon, np.array([la[k]]), np.array([lo[k]]))[0]
            sv[k] = bilinear(v_all[m], lat, lon, np.array([la[k]]), np.array([lo[k]]))[0]
        good = np.isfinite(su) & np.isfinite(sv) & np.isfinite(du) & np.isfinite(dv)
        seg_u.append(du[good]); seg_v.append(dv[good])
        st_u.append(su[good]); st_v.append(sv[good])
    du = np.concatenate(seg_u); dv = np.concatenate(seg_v)
    su = np.concatenate(st_u); sv = np.concatenate(st_v)
    # 联合最小二乘: [su;sv]·α + [1 0;0 1]·beta = [du;dv]
    n = len(du)
    A = np.zeros((2 * n, 3))
    A[:n, 0] = su; A[:n, 1] = 1.0
    A[n:, 0] = sv; A[n:, 2] = 1.0
    b = np.concatenate([du, dv])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    alpha, ub, vb = coef
    pred_u = alpha * su + ub
    pred_v = alpha * sv + vb
    return {
        "alpha": float(alpha),
        "u_beta": float(ub),
        "v_beta": float(vb),
        "n_segments": int(n),
        "r_u": float(np.corrcoef(pred_u, du)[0, 1]),
        "r_v": float(np.corrcoef(pred_v, dv)[0, 1]),
        "rmse_u": float(np.sqrt(np.mean((pred_u - du) ** 2))),
        "rmse_v": float(np.sqrt(np.mean((pred_v - dv) ** 2))),
    }


# ------------------------------------------------------------------ 积分

def integrate(seeds: pd.DataFrame, u_c, v_c, lat, lon, par: dict, region: dict) -> pd.DataFrame:
    """对每个种子按其观测寿命做 beta 平流，返回合成轨迹点。"""
    rows = []
    a, ub, vb = par["alpha"], par["u_beta"], par["v_beta"]
    for r in seeds.itertuples():
        la, lo = r.lat, r.lon
        m = r.month - 1
        for _ in range(int(r.steps)):
            if not (region["lon_min"] <= lo <= region["lon_max"]
                    and region["lat_min"] <= la <= region["lat_max"]):
                break
            rows.append((r.sid, r.season, lo, la))
            su = bilinear(u_c[m], lat, lon, np.array([la]), np.array([lo]))[0]
            sv = bilinear(v_c[m], lat, lon, np.array([la]), np.array([lo]))[0]
            if not (np.isfinite(su) and np.isfinite(sv)):
                break
            cu = a * su + ub
            cv = a * sv + vb
            lo += cu * DT / (EARTH_R * np.cos(la * RAD)) / RAD
            la += cv * DT / EARTH_R / RAD
    return pd.DataFrame(rows, columns=["sid", "season", "lon", "lat"])


# ------------------------------------------------------------------ 判据

def tvd(d: np.ndarray) -> float:
    return 0.5 * np.abs(d).sum()


def main() -> None:
    cfg = load_config()
    tracks, region = load_tracks(cfg)
    lon_e, lat_e = grid_edges(cfg, region)
    p1_years = np.arange(START, SPLIT + 1)
    p2_years = np.arange(SPLIT + 1, END + 1)

    print("== 观测侧 ==")
    f1 = annual_relative_field(tracks[tracks.season <= SPLIT], p1_years, lon_e, lat_e)
    f2 = annual_relative_field(tracks[tracks.season > SPLIT], p2_years, lon_e, lat_e)
    d_obs = f2.mean(0) - f1.mean(0)
    print(f"   观测 d_obs: TVD={tvd(d_obs):.4f}")

    print("== 引导流气候态 ==")
    u1, v1, slat, slon = steering_climatology(p1_years)
    u2, v2, _, _ = steering_climatology(p2_years)
    u_all, v_all, _, _ = steering_climatology(np.arange(START, END + 1))
    print(f"   P1/P2 月气候态就绪，网格 {len(slat)}×{len(slon)}")

    print("== 标定（全期，之后固定）==")
    par = calibrate(tracks, u_all, v_all, slat, slon)
    print(f"   α={par['alpha']:.3f}  beta=({par['u_beta']:+.2f},{par['v_beta']:+.2f}) m/s"
          f"  r_u={par['r_u']:.3f} r_v={par['r_v']:.3f}  n={par['n_segments']}")

    # 种子：观测生成点 + 观测域内寿命
    g = tracks.groupby("sid")
    seeds = g.first().reset_index()[["sid", "season", "lat", "lon", "iso_time"]]
    seeds["month"] = seeds["iso_time"].dt.month
    seeds["steps"] = g.size().values
    s1 = seeds[seeds.season <= SPLIT]
    s2 = seeds[seeds.season > SPLIT]
    print(f"   种子 P1={len(s1)}  P2={len(s2)}")

    # 寿命是生成"后"的属性。若直接用 P2 种子做生成情景，会把寿命变化混进生成项。
    # 用分位映射把一组种子的寿命替换为另一组的寿命分布，从而把位置与寿命拆开。
    def remap_steps(target: pd.DataFrame, donor: pd.DataFrame) -> pd.DataFrame:
        out = target.copy().reset_index(drop=True)
        d = np.sort(donor["steps"].to_numpy())
        q = (out["steps"].rank(method="first").to_numpy() - 0.5) / len(out)
        out["steps"] = np.quantile(d, q).round().astype(int).clip(min=1)
        return out

    s2_life1 = remap_steps(s2, s1)   # P2 位置 + P1 寿命
    s1_life2 = remap_steps(s1, s2)   # P1 位置 + P2 寿命
    print(f"   域内寿命均值 P1={s1['steps'].mean():.1f} P2={s2['steps'].mean():.1f} 步"
          f"（寿命属生成后属性，已拆出）")

    print("== 情景 ==")
    scen = {
        "BASE_P1": (s1, u1, v1, p1_years),
        "BASE_P2": (s2, u2, v2, p2_years),
        "STEER":   (s1, u2, v2, p1_years),        # 仅换平均引导流   → 生成后
        "LOC":     (s2_life1, u1, v1, p2_years),  # 仅换生成位置     → 生成
        "LIFE":    (s1_life2, u1, v1, p1_years),  # 仅换域内寿命     → 生成后
    }
    fields = {}
    for name, (sd, uc, vc, yrs) in scen.items():
        syn = integrate(sd, uc, vc, slat, slon, par, region)
        fld = annual_relative_field(syn, yrs, lon_e, lat_e)
        fields[name] = fld.mean(0)
        print(f"   {name:8s} 合成点 {len(syn):6d}")

    d_full = fields["BASE_P2"] - fields["BASE_P1"]
    d_steer = fields["STEER"] - fields["BASE_P1"]
    d_loc = fields["LOC"] - fields["BASE_P1"]
    d_life = fields["LIFE"] - fields["BASE_P1"]

    def r(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    n_steer = np.abs(d_steer).sum()
    n_loc = np.abs(d_loc).sum()
    n_life = np.abs(d_life).sum()
    n_tot = n_steer + n_loc + n_life
    res = {
        "calibration": par,
        "r_full_obs": r(d_full, d_obs),
        "r_steer_obs": r(d_steer, d_obs),
        "r_loc_obs": r(d_loc, d_obs),
        "r_life_obs": r(d_life, d_obs),
        "tvd_obs": tvd(d_obs),
        "tvd_full": tvd(d_full),
        "tvd_steer": tvd(d_steer),
        "tvd_loc": tvd(d_loc),
        "tvd_life": tvd(d_life),
        "tvd_ratio_steer": tvd(d_steer) / tvd(d_obs),
        "share_steer": float(n_steer / n_tot),
        "share_loc": float(n_loc / n_tot),
        "share_life": float(n_life / n_tot),
        # 生成后 = 平均引导流 + 域内寿命；生成 = 生成位置
        "share_postgenesis": float((n_steer + n_life) / n_tot),
    }

    print("\n== 预设判据核对 ==")
    print(f"  判据0 前置有效性  r(d_full,d_obs) = {res['r_full_obs']:+.3f}  阈值≥0.30  "
          f"→ {'通过' if res['r_full_obs'] >= 0.30 else '未通过：模型失败，不解读d_steer'}")
    if res["r_full_obs"] >= 0.30:
        rs = res["r_steer_obs"]
        verdict = "实质复现" if rs >= 0.40 else ("部分复现" if rs >= 0.20 else "不足以解释")
        print(f"  判据1 主判据      r(d_steer,d_obs) = {rs:+.3f}  → {verdict}")
        print(f"  判据2 量级        TVD(d_steer)/TVD(d_obs) = {res['tvd_ratio_steer']:.3f}")
        pg = res["share_postgenesis"]
        print(f"  判据3 生成后份额  {pg*100:.1f}%  = 平均引导流 {res['share_steer']*100:.1f}%"
              f" + 域内寿命 {res['share_life']*100:.1f}%；生成位置 {res['share_loc']*100:.1f}%")
        print(f"        （Shapley 生成后 73.7–80.0%，印证区间 60–90%）"
              f" → {'印证' if 0.60 <= pg <= 0.90 else '分歧，须在讨论中说明'}")
    else:
        print("  判据1–3 不适用：判据0未过，本试验对机制问题不给结论。")

    (EXP / "results").mkdir(exist_ok=True)
    np.savez(EXP / "results" / "fields.npz", d_obs=d_obs, d_full=d_full,
             d_steer=d_steer, d_loc=d_loc, d_life=d_life, lon_edges=lon_e, lat_edges=lat_e,
             **{f"f_{k}": v for k, v in fields.items()})
    (EXP / "results" / "criteria.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {EXP / 'results'}")


if __name__ == "__main__":
    main()
