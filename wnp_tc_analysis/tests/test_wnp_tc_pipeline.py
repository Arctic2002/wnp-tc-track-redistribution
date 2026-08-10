from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from common import block_permutation_projection, projection_scores, temporal_blocks
from wnp_tc_analysis.src.wnpsh_metric_audit import correlation_permutation


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_projection_scores_have_finite_oos_values():
    years = np.arange(1966, 2026)
    rng = np.random.default_rng(4)
    fields = rng.dirichlet(np.ones(6), size=len(years))
    full, oos, pattern = projection_scores(fields, years)
    assert np.isfinite(full).all()
    assert np.isfinite(oos).all()
    assert abs(pattern.sum()) < 1e-12


def test_projection_target_year_excluded_from_its_pattern():
    years = np.arange(1966, 2026)
    rng = np.random.default_rng(7)
    fields = rng.dirichlet(np.ones(5), size=len(years))
    _, oos, _ = projection_scores(fields, years)
    i = 10
    e = np.arange(30); l = np.arange(30, 60)
    ee = e[e != i]; ll = l[l != i]
    em = fields[ee].mean(0); lm = fields[ll].mean(0); d = lm - em
    expected = 2 * ((fields[i] - 0.5 * (lm + em)) @ d) / (d @ d)
    assert np.isclose(oos[i], expected)


def test_projection_permutation_refits_the_oos_pattern():
    years = np.arange(1966, 2026)
    rng = np.random.default_rng(11)
    fields = rng.dirichlet(np.ones(4), size=len(years))
    observed, p = block_permutation_projection(fields, years, block=3, nperm=19, seed=23)
    _, oos, _ = projection_scores(fields, years)
    assert np.isclose(observed, oos[30:].mean() - oos[:30].mean())

    blocks = temporal_blocks(len(years), 3, years)
    trial_rng = np.random.default_rng(23)
    exceed = 0
    for _ in range(19):
        order = np.concatenate([blocks[i] for i in trial_rng.permutation(len(blocks))])
        _, trial_oos, _ = projection_scores(fields[order], years)
        trial = trial_oos[30:].mean() - trial_oos[:30].mean()
        exceed += abs(trial) >= abs(observed)
    assert p == (exceed + 1) / 20


def test_temporal_blocks_do_not_bridge_removed_years():
    years = np.r_[np.arange(1966, 1971), np.arange(1980, 1985)]
    blocks = temporal_blocks(len(years), 3, years)
    assert all(np.all(np.diff(years[one]) == 1) for one in blocks if len(one) > 1)


def test_genesis_propagation_closure_is_an_algebraic_invariant():
    mod = load_script("gp", "33_genesis_propagation_decomposition.py")
    rng = np.random.default_rng(3)
    years = np.repeat(np.arange(1966, 2026), 6)
    meta = pd.DataFrame({"year": years, "gen_bin": np.tile([0, 0, 0, 1, 1, 1], 60)})
    q = rng.dirichlet(np.ones(8), size=len(meta))
    result = mod.decompose(meta, q, np.arange(1966, 1996), np.arange(1996, 2026), min_storms=1)
    assert result["closure_max_abs"] < 1e-12
    assert np.isclose(result["genesis_projection_fraction"] + result["propagation_projection_fraction"], 1.0)


def test_wnpsh_association_inference_uses_detrended_series():
    t = np.arange(60, dtype=float)
    x = 0.4 * t + np.sin(t / 3)
    y = 0.7 * t + np.cos(t / 5)
    raw, adjusted, p, n = correlation_permutation(x, y, nperm=19, seed=7)
    assert np.isclose(raw, np.corrcoef(x, y)[0, 1])
    assert np.isclose(adjusted, np.corrcoef(detrend(x), detrend(y))[0, 1])
    assert 0 < p <= 1
    assert n == len(t)


def test_geopotential_height_conversion_uses_cf_metadata():
    mod = load_script("circulation", "34_circulation_link.py")
    geopotential = np.array([54146.0, 55000.0])
    height = mod.geopotential_to_height(
        geopotential,
        {"standard_name": "geopotential", "units": "m**2 s**-2"},
    )
    assert np.allclose(height, geopotential / 9.80665)
    already_height = mod.geopotential_to_height(
        height,
        {"standard_name": "geopotential_height", "units": "m"},
    )
    assert np.array_equal(already_height, height)
    with pytest.raises(ValueError, match="cannot determine"):
        mod.geopotential_to_height(geopotential, {"units": "unknown"})


def test_raw_circulation_fields_are_descriptive_only():
    mod = load_script("circulation_raw", "34_circulation_link.py")
    rng = np.random.default_rng(41)
    x = np.arange(60, dtype=float) + rng.normal(size=60)
    y = rng.normal(size=(60, 2, 3)) + np.arange(60)[:, None, None]
    _, r, p, q, global_r2, global_p = mod.field_test(x, y, inferential=False)
    assert np.isfinite(r).all() and np.isfinite(global_r2)
    assert np.isnan(p).all() and np.isnan(q).all() and np.isnan(global_p)


def test_unique_assignment_is_one_row_per_storm():
    mod = load_script("land", "36_landfall_unique_assignment.py")
    events = pd.DataFrame({
        "sid": ["a", "a", "b"],
        "time": ["2000-01-02", "2000-01-01", "2001-01-01"],
        "coast": ["Japan", "Other", "China_S"],
        "wind": [50.0, 40.0, 35.0],
    })
    out = mod.unique_table(events, ["sid"], "first_any")
    assert len(out) == out["sid"].nunique() == 2
    assert out.loc[out.sid == "a", "coast"].iloc[0] == "Other"


def test_legacy_landfall_summary_rejects_zero_denominators():
    mod = load_script("land_empty", "36_landfall_unique_assignment.py")
    mod.YEARS = np.array([2000])
    empty = pd.DataFrame({"season": [], "coast": []})
    with pytest.raises(ValueError, match="no assigned landfall events"):
        mod.annual_composition(empty)
