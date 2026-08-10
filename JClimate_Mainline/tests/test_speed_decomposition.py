"""Algebra and segment-level checks for the speed-decomposition trial."""

import numpy as np
import pandas as pd
from importlib import import_module

decompose_segments = import_module(
    "paper2_dynamic.30_speed_decomposition_trial"
).decompose_segments


def test_three_term_decomposition_closure_is_an_algebraic_invariant():
    rows = []
    for year in range(2000, 2010):
        for band, speed, count in [(5.5, 10 + (year - 2000), 3), (25.5, 30, 2 + year % 2)]:
            for _ in range(count):
                rows.append(
                    {
                        "season": year,
                        "lat": band,
                        "lon": 130.0,
                        "speed_total": speed,
                    }
                )
    segments = pd.DataFrame(rows)
    annual, _ = decompose_segments(
        segments, np.arange(2000, 2010), lat_max=40, speed_column="speed_total"
    )
    assert np.max(np.abs(annual["closure_error"])) < 1e-12
    assert np.max(np.abs(annual["direct_speed"] - annual["basin_speed"])) < 1e-12
