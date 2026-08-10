"""纯函数单元测试：只用合成数据与解析解对照，不依赖真实大数据，秒级完成。

运行：pytest -q core/tests/test_core_units.py
"""
import numpy as np
import pandas as pd
import pytest


def test_haversine_known_distance():
    """haversine：赤道上经度相差 1 度约 111.19km；与解析值对照（容差 1km）。"""
    from core.utils import haversine
    d = haversine(0.0, 0.0, 0.0, 1.0)                 # (lat1,lon1,lat2,lon2)
    assert abs(d - 111.19) < 1.0


def test_kt_ms_roundtrip():
    """kt<->m/s 往返换算误差应接近机器精度。"""
    from core.utils import kt_to_ms, ms_to_kt
    for v in [0.0, 34.0, 64.0, 130.0]:
        assert abs(ms_to_kt(kt_to_ms(v)) - v) < 1e-9


def test_landfall_crossing_synthetic():
    """登陆穿越合成测试：一条由海(经度小)向陆(经度大)的直线轨迹，
    在已知陆地边界(lon>=130)处应恰好检出一次海->陆穿越。"""
    from core.landfall_gshhg import densify, is_land
    # 构造一个简单矩形“陆地”掩膜：经度 >= 130 视为陆地。
    res = 0.02
    lon0, lat1 = 120.0, 20.0          # 掩膜左上角
    nx, ny = int((140 - 120) / res), int((20 - 10) / res)
    mask = np.zeros((ny, nx), dtype="uint8")
    lon_idx_130 = int((130 - lon0) / res)
    mask[:, lon_idx_130:] = 1         # 右半部分是陆地

    track = pd.DataFrame({"lon": [125., 128., 131., 134.], "lat": [15., 15., 15., 15.],
                          "iso_time": pd.date_range("2000-08-01", periods=4, freq="6h")})
    crossings = []
    for k in range(len(track) - 1):
        a, b = track.iloc[k], track.iloc[k + 1]
        f, lat, lon = densify(a, b, res / 2)
        state = [is_land(mask, lon0, lat1, res, la, lo) for la, lo in zip(lat, lon)]
        for j in range(1, len(state)):
            if state[j] is True and state[j - 1] is False:
                crossings.append(lon[j])
                break
    assert len(crossings) == 1 and 129.5 <= crossings[0] <= 131.5


def test_neff_reduces_with_autocorrelation():
    """Bretherton 有效样本量：强正自相关序列的有效样本量应明显小于名义 N。"""
    from core.stats_utils import effective_n
    rng = np.random.default_rng(1)
    n = 200
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):                               # AR(1)，phi=0.8
        x[i] = 0.8 * x[i - 1] + e[i]
    assert effective_n(x, x) < 0.6 * n                  # 强自相关 -> 有效样本量大幅缩水


def test_theil_sen_reproducible_and_recovers_trend():
    """Theil-Sen + 残差块自助：固定 seed 可复现，且能恢复已知斜率。"""
    from core.stats_utils import theil_sen_ci
    rng = np.random.default_rng(0)
    yrs = np.arange(1980, 2025)
    series = 0.5 * (yrs - 1980) + rng.normal(0, 1, len(yrs))
    s1 = theil_sen_ci(yrs, series, nboot=300, seed=42)
    s2 = theil_sen_ci(yrs, series, nboot=300, seed=42)
    assert s1 == s2                                     # 固定 seed 完全可复现
    slope, lo, hi = s1
    assert 0.3 < slope < 0.7 and lo < slope < hi       # 恢复 ~0.5 斜率且区间包含


def test_gpi_logsum_closes_with_common_mask():
    """GPI 对数分解闭合：四项随机正场(含部分缺测)，施加共同有效掩膜后，
    四项异常之和应严格等于总异常（仅余浮点误差）。"""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(0)
    dims = ("time", "y", "x")
    shp = (24, 4, 5)
    terms = {k: xr.DataArray(np.exp(rng.normal(size=shp)), dims=dims) for k in "ABCD"}
    terms["A"] = terms["A"].where(rng.random(shp) > 0.1)
    terms["C"] = terms["C"].where(rng.random(shp) > 0.1)
    logs = {k: np.log(v.where(np.isfinite(v) & (v > 0))) for k, v in terms.items()}
    valid = None
    for lv in logs.values():
        m = np.isfinite(lv)
        valid = m if valid is None else (valid & m)
    logs = {k: lv.where(valid) for k, lv in logs.items()}
    t = pd.date_range("2000-01-01", periods=shp[0], freq="MS")
    logs = {k: lv.assign_coords(time=("time", t)) for k, lv in logs.items()}
    contrib = {k: (lv.groupby("time.month") - lv.groupby("time.month").mean("time"))
               for k, lv in logs.items()}
    lg = sum(logs.values())
    total = lg.groupby("time.month") - lg.groupby("time.month").mean("time")
    resid = total - sum(contrib.values())
    assert float(np.abs(resid).max(skipna=True)) < 1e-10


def test_wavelet_recovers_period():
    """4 单位周期正弦的全局小波谱峰值周期应 ~4。"""
    from core.wavelet import cwt_morlet, global_spectrum
    n = 512
    t = np.arange(n)
    sig = np.sin(2 * np.pi * t / 4.0)
    power, period, scale, coi = cwt_morlet(sig, dt=1.0)
    peak = period[int(np.argmax(global_spectrum(power)))]
    assert abs(peak - 4.0) < 1.0
