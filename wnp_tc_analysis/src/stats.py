from __future__ import annotations

import numpy as np
import pymannkendall as mk
from scipy.stats import theilslopes


def bh_fdr(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return q
    order = np.argsort(pv)
    adjusted = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(pv)
    restored[order] = np.minimum(adjusted, 1.0)
    q[valid] = restored
    return q


def block_order(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def block_permutation_difference(values, early, late, *, block=3, nperm=9999, seed=202406):
    x = np.asarray(values, dtype=float)
    early = np.asarray(early, dtype=int)
    late = np.asarray(late, dtype=int)
    observed = float(np.nanmean(x[late]) - np.nanmean(x[early]))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        order = block_order(len(x), block, rng)
        diff = float(np.nanmean(x[order[late]]) - np.nanmean(x[order[early]]))
        exceed += abs(diff) >= abs(observed)
    return observed, (exceed + 1) / (nperm + 1)


def moving_block_sample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    starts = rng.integers(0, len(x), size=int(np.ceil(len(x) / block)))
    chunks = [x[(start + np.arange(block)) % len(x)] for start in starts]
    return np.concatenate(chunks)[: len(x)]


def block_bootstrap_difference_ci(early, late, *, block=3, nboot=4999, seed=202406):
    early = np.asarray(early, dtype=float)
    late = np.asarray(late, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(nboot)
    for i in range(nboot):
        a = moving_block_sample(early, block, rng)
        b = moving_block_sample(late, block, rng)
        samples[i] = np.nanmean(b) - np.nanmean(a)
    return tuple(np.nanpercentile(samples, [2.5, 97.5]))


def block_permutation_many(matrix, early, late, *, block=3, nperm=9999, seed=202406, batch=200):
    """Vectorized block permutation for a years x metrics matrix."""
    x = np.asarray(matrix, dtype=float)
    early = np.asarray(early, dtype=int)
    late = np.asarray(late, dtype=int)
    observed = np.nanmean(x[late], axis=0) - np.nanmean(x[early], axis=0)
    exceed = np.zeros(x.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < nperm:
        size = min(batch, nperm - completed)
        orders = np.stack([block_order(len(x), block, rng) for _ in range(size)])
        diff = np.nanmean(x[orders[:, late], :], axis=1) - np.nanmean(x[orders[:, early], :], axis=1)
        exceed += np.sum(np.abs(diff) >= np.abs(observed), axis=0)
        completed += size
    return observed, (exceed + 1) / (nperm + 1)


def block_bootstrap_many(early_matrix, late_matrix, *, block=3, nboot=4999, seed=202406, batch=200):
    """Vectorized moving-block confidence intervals for multiple metrics."""
    early = np.asarray(early_matrix, dtype=float)
    late = np.asarray(late_matrix, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty((nboot, early.shape[1]), dtype=float)
    completed = 0
    while completed < nboot:
        size = min(batch, nboot - completed)
        early_idx = np.stack([
            np.concatenate([(start + np.arange(block)) % len(early) for start in rng.integers(0, len(early), size=int(np.ceil(len(early) / block)))])[: len(early)]
            for _ in range(size)
        ])
        late_idx = np.stack([
            np.concatenate([(start + np.arange(block)) % len(late) for start in rng.integers(0, len(late), size=int(np.ceil(len(late) / block)))])[: len(late)]
            for _ in range(size)
        ])
        samples[completed:completed + size] = np.nanmean(late[late_idx, :], axis=1) - np.nanmean(early[early_idx, :], axis=1)
        completed += size
    return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


def trend_summary(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(years) & np.isfinite(values)
    years = years[valid]
    values = values[valid]
    result = {
        "n_years": len(values), "sen_slope_per_decade": np.nan,
        "sen_ci_low_per_decade": np.nan, "sen_ci_high_per_decade": np.nan,
        "mk_p": np.nan, "mk_trend": "",
    }
    if len(values) >= 5 and np.nanstd(values) > 0:
        slope, intercept, low, high = theilslopes(values, years, alpha=0.95)
        hr = mk.hamed_rao_modification_test(values)
        result.update({
            "sen_slope_per_decade": float(slope * 10),
            "sen_ci_low_per_decade": float(low * 10),
            "sen_ci_high_per_decade": float(high * 10),
            "mk_p": float(hr.p), "mk_trend": hr.trend,
        })
    return result
