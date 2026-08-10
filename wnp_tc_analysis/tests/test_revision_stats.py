"""Unit tests for the revision-only statistical helpers."""

import numpy as np

from paper2_dynamic.revision_stats import bh_fdr, compositional_change_test


def test_bh_fdr_is_monotone_in_rank_and_preserves_nan():
    q = bh_fdr([0.01, 0.04, np.nan, 0.03])
    assert np.isnan(q[2])
    assert q[0] <= q[3] <= q[1]
    assert np.all((q[np.isfinite(q)] >= 0) & (q[np.isfinite(q)] <= 1))


def test_composition_test_is_reproducible_and_detects_large_shift():
    rng = np.random.default_rng(7)
    early = rng.dirichlet([30, 5, 5], size=15)
    late = rng.dirichlet([5, 30, 5], size=15)
    annual = np.vstack([early, late])
    kwargs = dict(
        early_index=np.arange(15),
        late_index=np.arange(15, 30),
        nperm=499,
        block=3,
        seed=42,
    )
    first = compositional_change_test(annual, **kwargs)
    second = compositional_change_test(annual, **kwargs)
    assert first["tv"] == second["tv"]
    assert first["global_p"] == second["global_p"]
    assert 0 <= first["tv"] <= 1
    assert first["global_p"] < 0.05

