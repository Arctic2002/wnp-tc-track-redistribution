"""无依赖的 Morlet 连续小波变换（Torrence & Compo, 1998）。

被两篇论文导入做周期分析（ENSO/PDO 频带、年际-年代际信号）。
纯 numpy 实现，不依赖 pycwt 等需编译的库。
分析前对输入去线性趋势并标准化；长期周期只有在 COI 外有效周期数充足时才解释。

主要接口
--------
- cwt_morlet(signal, dt, dj, s0, j1, w0=6) -> (power, period, scale, coi)
- ar1(x) -> 红噪声滞后 1 自相关参数
- significance_ratio(power, signal, dt, scale, period, w0=6) -> 相对 AR(1) 谱的显著性比
- global_spectrum(power) -> 全局（时间平均）小波谱
"""
import numpy as np


def ar1(x):
    """估计序列滞后 1 自相关（红噪声参数）。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    c0 = np.dot(x, x) / len(x)
    c1 = np.dot(x[:-1], x[1:]) / (len(x) - 1)
    if c0 == 0:
        return 0.0
    return float(np.clip(c1 / c0, -0.999, 0.999))


def _detrend_standardize(signal):
    """去线性趋势并标准化为均值 0、方差 1。"""
    s = np.asarray(signal, float)
    n = len(s)
    t = np.arange(n)
    # 最小二乘拟合一条直线并扣除，避免趋势污染低频小波功率。
    coef = np.polyfit(t, s, 1)
    s = s - np.polyval(coef, t)
    std = s.std(ddof=1)
    if std == 0:
        return s
    return s / std


def cwt_morlet(signal, dt, dj=0.125, s0=None, j1=None, w0=6.0):
    """Morlet 连续小波变换。

    参数
    ----
    signal : 1D array，等间隔时间序列（建议先去趋势）。
    dt     : 采样间隔（如年序列为 1）。
    dj     : 尺度分辨率（每倍频程的子尺度数，0.125 较常用）。
    s0     : 最小尺度，默认 2*dt。
    j1     : 尺度数减一，默认覆盖到序列长度对应的最大尺度。
    w0     : Morlet 无量纲频率，默认 6（满足容许条件）。

    返回
    ----
    power  : |W|^2 功率谱，形状 (尺度, 时间)。
    period : 各尺度对应的傅里叶周期（与 dt 同单位）。
    scale  : 各尺度。
    coi    : 各时间点的影响锥周期。
    """
    x = _detrend_standardize(signal)
    n = len(x)
    if s0 is None:
        s0 = 2 * dt
    if j1 is None:
        j1 = int(np.round(np.log2(n * dt / s0) / dj))
    # 尺度按 2 的幂等比排列。
    scale = s0 * 2.0 ** (np.arange(j1 + 1) * dj)

    # 角频率向量（FFT 频率）。
    k = np.arange(1, n // 2 + 1)
    omega_pos = 2 * np.pi * k / (n * dt)
    # Even-length FFTs contain a Nyquist term that should not be mirrored;
    # odd-length FFTs do not, so the full positive-frequency tail is mirrored.
    neg_start = 1 if n % 2 == 0 else 0
    omega = np.concatenate([[0.0], omega_pos, -omega_pos[::-1][neg_start:]])
    omega = omega[:n]

    xf = np.fft.fft(x)
    power = np.empty((len(scale), n))
    wave = np.empty((len(scale), n), dtype=complex)
    # 归一化常数，使不同尺度功率可比（Torrence & Compo 公式 6）。
    for i, sc in enumerate(scale):
        expnt = -((sc * omega - w0) ** 2) / 2.0 * (omega > 0)
        norm = np.sqrt(2 * np.pi * sc / dt) * (np.pi ** -0.25)
        daughter = norm * np.exp(expnt) * (omega > 0)
        wave[i] = np.fft.ifft(xf * daughter)
    power = np.abs(wave) ** 2

    # 尺度->傅里叶周期换算（Morlet）。
    flambda = 4 * np.pi / (w0 + np.sqrt(2 + w0 ** 2))
    period = scale * flambda

    # 影响锥：边界处 e 折时间对应的周期。
    coi_factor = flambda / np.sqrt(2)
    tt = np.arange((n + 1) // 2)
    coi_half = coi_factor * dt * np.concatenate([[1e-5], tt[1:]])
    coi = np.concatenate([coi_half, coi_half[::-1][(n % 2):]])[:n]
    return power, period, scale, coi


def significance_ratio(power, signal, dt, scale, period, w0=6.0, siglvl=0.95):
    """返回 power 相对 AR(1) 红噪声背景谱的比值；>1 表示超过给定显著性水平。"""
    from scipy.stats import chi2
    x = _detrend_standardize(signal)
    n = len(x)
    a = ar1(x)
    # AR(1) 理论功率谱（归一化方差为 1）。
    freq = dt / period
    pk = (1 - a ** 2) / (1 - 2 * a * np.cos(2 * np.pi * freq) + a ** 2)
    dof = 2
    fac = chi2.ppf(siglvl, dof) / dof
    sig = pk * fac                      # 各尺度的显著性阈值（背景谱 x 卡方因子）
    # 广播：power(尺度,时间) / sig(尺度,1)。
    return power / sig[:, None]


def global_spectrum(power):
    """全局小波谱：对时间维取平均，得到每个尺度的平均功率。"""
    return np.asarray(power).mean(axis=1)


if __name__ == "__main__":
    # 自检：合成 4 单位周期正弦，全局谱峰值周期应 ~ 4。
    n = 512
    dt = 1.0
    t = np.arange(n) * dt
    sig = np.sin(2 * np.pi * t / 4.0)
    power, period, scale, coi = cwt_morlet(sig, dt)
    gs = global_spectrum(power)
    peak = period[int(np.argmax(gs))]
    print(f"peak period = {peak:.3f} (expected ~4)")
    assert abs(peak - 4.0) < 4.0 * 0.125 * 2, "峰值周期偏离过大"
    print("wavelet self-check passed")
