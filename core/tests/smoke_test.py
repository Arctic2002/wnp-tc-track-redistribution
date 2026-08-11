"""端到端冒烟：用极小合成数据跑通 Core 主链路，断言产物存在与基本不变量。

运行：python -m core.tests.smoke_test  （或 pytest core/tests/smoke_test.py）

说明：Paper I 的 PI 事件级采样在 paper1_thermo 实现；该层目前为占位。
若 paper1_thermo.collocate_env 不存在，则相应断言自动跳过，只验证 Core 链路。
"""
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


def build_toy_dataset(root: Path):
    """造一个几格点 x 几个月 x 几条轨迹的玩具数据集，仅供链路连通性测试。"""
    import xarray as xr
    root.mkdir(parents=True, exist_ok=True)
    # 玩具 PI 场：3x4 格点、12 个月，全部填合理量级，状态码 ifl=1。
    t = pd.date_range("2000-01-01", periods=12, freq="MS")
    lat = np.array([20., 15., 10.])
    lon = np.array([120., 125., 130., 135.])
    vpot = xr.DataArray(np.full((12, 3, 4), 75.0), dims=("time", "latitude", "longitude"),
                        coords=dict(time=t, latitude=lat, longitude=lon))
    ifl = xr.ones_like(vpot)
    xr.Dataset({"vpot": vpot, "ifl": ifl}).to_netcdf(root / "pi.nc")
    # 玩具 storms：两条轨迹，落在格点范围内。
    storms = pd.DataFrame({"sid": ["A", "B"], "season": [2000, 2000],
                           "genesis_time": pd.to_datetime(["2000-08-01", "2000-09-01"]),
                           "genesis_lat": [15., 12.], "genesis_lon": [125., 130.],
                           "lmi_time": pd.to_datetime(["2000-08-03", "2000-09-03"]),
                           "lmi_lat": [16., 13.], "lmi_lon": [126., 131.]})
    storms.to_csv(root / "storms.csv", index=False)
    return root


def test_smoke_core_chain():
    """Core 主链路冒烟：玩具轨迹上的登陆几何与统计工具应正常工作、量级合理。"""
    from core.landfall_gshhg import densify, is_land
    from core.stats_utils import effective_n, theil_sen_ci
    from core.wavelet import cwt_morlet, global_spectrum

    # 1) 登陆几何：海->陆直线恰一次穿越。
    res = 0.02
    lon0, lat1 = 120.0, 20.0
    nx, ny = int((140 - 120) / res), int((20 - 10) / res)
    mask = np.zeros((ny, nx), dtype="uint8")
    mask[:, int((130 - lon0) / res):] = 1
    track = pd.DataFrame({"lon": [125., 131.], "lat": [15., 15.]})
    f, lat, lon = densify(track.iloc[0], track.iloc[1], res / 2)
    state = [is_land(mask, lon0, lat1, res, la, lo) for la, lo in zip(lat, lon)]
    crossings = sum(1 for j in range(1, len(state)) if state[j] and not state[j - 1])
    assert crossings == 1

    # 2) 统计工具在合成年序列上可运行且产出合理。
    yrs = np.arange(1980, 2025)
    series = 0.3 * (yrs - 1980) + np.sin(np.arange(len(yrs)))
    assert effective_n(series, series) >= 3
    slope, lo, hi = theil_sen_ci(yrs, series, nboot=200, seed=0)
    assert lo <= slope <= hi

    # 3) 小波在合成 4 周期信号上识别出 ~4 的峰值周期。
    sig = np.sin(2 * np.pi * np.arange(256) / 4.0)
    power, period, scale, coi = cwt_morlet(sig, dt=1.0)
    assert abs(period[int(np.argmax(global_spectrum(power)))] - 4.0) < 1.0


def test_smoke_pi_sampling_if_paper1_present():
    """若 paper1_thermo 的采样函数已实现，则在玩具 PI 场上验证采样与域外 NaN。"""
    import pytest
    sample_points = None
    try:
        from paper1_thermo.collocate_env import sample_points  # noqa
    except Exception:
        pytest.skip("paper1_thermo 不属于本分析包的 Core 层测试范围，跳过 PI 采样冒烟")

    import xarray as xr
    with tempfile.TemporaryDirectory() as d:
        root = build_toy_dataset(Path(d))
        pi = xr.open_dataset(root / "pi.nc")["vpot"].load()
        s = pd.read_csv(root / "storms.csv", parse_dates=["genesis_time", "lmi_time"])
        s["pi_gen"] = sample_points(pi, s.genesis_time, s.genesis_lat, s.genesis_lon)
        assert s["pi_gen"].notna().all() and s["pi_gen"].between(50, 100).all()
        oob = sample_points(pi, s.genesis_time.iloc[:1], [99.0], [999.0])
        assert np.isnan(oob[0])


if __name__ == "__main__":
    test_smoke_core_chain()
    print("core smoke test passed")
