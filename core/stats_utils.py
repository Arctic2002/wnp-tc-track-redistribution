"""共享统计工具：应对气候年序列的趋势、自相关、缺测与共线性。

被两篇论文导入；本模块不读写数据文件。
"""
import numpy as np
import pandas as pd
from scipy import stats

# 从 utils 重新导出 haversine，便于按 core.stats_utils 统一导入空间/统计工具。
from core.utils import haversine  # noqa: F401


def ar1(x):
    """估计序列与其前一时刻之间的相关系数（滞后 1 自相关）。"""
    x = np.asarray(x, float)
    # x[:-1] 是除最后一项外的序列，x[1:] 是除第一项外的序列。
    ok = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    a, b = x[:-1][ok], x[1:][ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def effective_n(x, y):
    """Bretherton et al.(1999) 有效样本量。

    若只传一个序列，可用 effective_n(x, x) 评估其自身的有效样本量。
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    r1, r2 = ar1(x), ar1(y)
    # max(...,1e-12) 防止分母非常接近 0 时发生除零错误。
    raw = n * (1 - r1 * r2) / max(1 + r1 * r2, 1e-12)
    # 有效样本量不允许超过原样本量，也至少保留 3 以便计算相关检验。
    return float(np.clip(raw, 3, n))


def corr_with_effn(x, y):
    """相关系数 + 用有效自由度校正的 p 值。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, float(len(x))
    # corrcoef 返回 2x2 相关矩阵，[0,1] 是 x 与 y 的相关系数。
    r = np.corrcoef(x, y)[0, 1]
    neff = effective_n(x, y)
    # 把相关系数转换成 t 统计量，再使用有效自由度而非原始样本量。
    t = r * np.sqrt((neff - 2) / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), df=max(neff - 2, 1))
    return r, p, neff


def theil_sen_ci(years, series, nboot=2000, block=3, seed=0):
    """返回 Theil-Sen 趋势斜率及残差移动块自助法 95% 区间。"""
    rng = np.random.default_rng(seed)
    yrs, s = np.asarray(years, float), np.asarray(series, float)
    ok = np.isfinite(yrs) & np.isfinite(s)
    yrs, s = yrs[ok], s[ok]
    # theilslopes 返回多个量，第 0 项是稳健趋势斜率。
    slope = stats.theilslopes(s, yrs)[0]
    intercept = np.median(s - slope * yrs)
    fitted = intercept + slope * yrs
    # 残差=观测值-趋势线；自助法只重采样残差，年份顺序保持不变。
    resid = s - fitted
    n = len(s)
    block = min(block, n)
    nb = int(np.ceil(n / block))
    boots = []
    for _ in range(nboot):
        # 随机选择若干连续块的起点，再拼接到原序列长度。
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(i, i + block) for i in starts])[:n]
        # 年份轴保持不变，只重采样去趋势残差。
        boots.append(stats.theilslopes(fitted + resid[idx], yrs)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, lo, hi


def vif_table(X):
    """X: 预测因子 DataFrame；返回各列 VIF。"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # 标准化把不同单位的变量变成均值 0、标准差 1。
    Xs = (X - X.mean()) / X.std()
    return pd.Series({c: variance_inflation_factor(Xs.values, i)
                      for i, c in enumerate(Xs.columns)})


def ridge_std(X, y):
    """标准化岭回归；时间顺序 CV 定 lambda，返回标准化系数与最优 lambda。"""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    # 先把预测因子和响应按行合并，再一次性删除任一变量缺失的年份。
    dat = pd.concat([X, pd.Series(y, name="__y")], axis=1).dropna()
    Xs = (dat[X.columns] - dat[X.columns].mean()) / dat[X.columns].std()
    ys = (dat["__y"] - dat["__y"].mean()) / dat["__y"].std()
    # TimeSeriesSplit 始终用较早资料训练、较晚资料验证，避免未来信息泄漏。
    cv = TimeSeriesSplit(n_splits=min(5, max(2, len(dat) // 8)))
    m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=cv).fit(Xs, ys)
    return pd.Series(m.coef_, index=X.columns), m.alpha_
