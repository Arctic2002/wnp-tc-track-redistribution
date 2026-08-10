"""Small statistical helpers used by the supervisor-revision analyses.

These functions deliberately live outside the original Paper II pipeline.  The
revision analyses can therefore be rerun and audited without silently changing
the statistics behind the already published manuscript figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymannkendall as mk

from core.stats_utils import theil_sen_ci


def bh_fdr(pvalues):
    """Return Benjamini-Hochberg adjusted p-values, preserving missing values."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / (np.arange(pv.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(pv)
    restored[order] = np.clip(ranked, 0.0, 1.0)
    q[valid] = restored
    return q


def trend_row(years, values, *, label, family, cfg, extra=None):
    """Hamed-Rao MK plus Theil-Sen trend and residual-block bootstrap CI."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    row = {
        "label": label,
        "fdr_family": family,
        "start": int(x.min()) if x.size else np.nan,
        "end": int(x.max()) if x.size else np.nan,
        "n": int(x.size),
        "mean": float(np.mean(y)) if y.size else np.nan,
        "sen_slope_per_year": np.nan,
        "sen_slope_per_decade": np.nan,
        "sen_ci_lo_per_decade": np.nan,
        "sen_ci_hi_per_decade": np.nan,
        "mk_p_raw": np.nan,
        "mk_trend": "",
    }
    if extra:
        row.update(extra)
    if x.size >= 5 and np.nanstd(y) > 0:
        result = mk.hamed_rao_modification_test(y)
        slope, lo, hi = theil_sen_ci(
            x,
            y,
            nboot=cfg["statistics"]["bootstrap_samples"],
            block=cfg["statistics"]["bootstrap_block"],
            seed=cfg["statistics"]["random_seed"],
        )
        row.update(
            {
                "sen_slope_per_year": slope,
                "sen_slope_per_decade": slope * 10,
                "sen_ci_lo_per_decade": lo * 10,
                "sen_ci_hi_per_decade": hi * 10,
                "mk_p_raw": result.p,
                "mk_trend": result.trend,
            }
        )
    return row


def add_family_fdr(rows):
    """Apply BH-FDR separately within each explicitly named inferential family."""
    out = pd.DataFrame(rows)
    if out.empty:
        out["mk_p_fdr_bh"] = []
        return out
    out["mk_p_fdr_bh"] = np.nan
    for _, idx in out.groupby("fdr_family", dropna=False).groups.items():
        out.loc[idx, "mk_p_fdr_bh"] = bh_fdr(out.loc[idx, "mk_p_raw"])
    return out


def _block_permutation_indices(n_years, block, rng):
    """Permute ordered year blocks while retaining the order within each block."""
    blocks = [np.arange(i, min(i + block, n_years)) for i in range(0, n_years, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def compositional_change_test(
    annual_composition,
    early_index,
    late_index,
    *,
    nperm=9999,
    block=3,
    seed=202406,
):
    """Test a multivariate early-late redistribution using total variation.

    Rows are years and columns are mutually exclusive spatial categories whose
    row sums equal one.  Ordered year blocks are shuffled to retain short-range
    persistence.  The global statistic is total-variation distance.  Cellwise
    two-sided p-values, BH-FDR q-values and max-|change| adjusted p-values are
    returned for diagnosis; the global test remains the primary inference.
    """
    x = np.asarray(annual_composition, dtype=float)
    early = np.asarray(early_index, dtype=int)
    late = np.asarray(late_index, dtype=int)
    if x.ndim != 2:
        raise ValueError("annual_composition must be a years x categories matrix")
    if not np.allclose(x.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("each annual composition must sum to one")

    observed = x[late].mean(axis=0) - x[early].mean(axis=0)
    observed_abs = np.abs(observed)
    observed_tv = float(0.5 * observed_abs.sum())

    rng = np.random.default_rng(seed)
    exceed_cell = np.zeros(x.shape[1], dtype=np.int64)
    exceed_tv = 0
    null_max = np.empty(nperm, dtype=float)
    for i in range(nperm):
        order = _block_permutation_indices(len(x), block, rng)
        change = x[order[late]].mean(axis=0) - x[order[early]].mean(axis=0)
        change_abs = np.abs(change)
        exceed_cell += change_abs >= observed_abs
        tv = 0.5 * change_abs.sum()
        exceed_tv += tv >= observed_tv
        null_max[i] = change_abs.max()

    p_cell = (exceed_cell + 1) / (nperm + 1)
    p_max = ((null_max[:, None] >= observed_abs[None, :]).sum(axis=0) + 1) / (nperm + 1)
    return {
        "early_mean": x[early].mean(axis=0),
        "late_mean": x[late].mean(axis=0),
        "change": observed,
        "tv": observed_tv,
        "global_p": (exceed_tv + 1) / (nperm + 1),
        "cell_p": p_cell,
        "cell_q_bh": bh_fdr(p_cell),
        "cell_p_max": p_max,
        "nperm": nperm,
        "block": block,
    }

