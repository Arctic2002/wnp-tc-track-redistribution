"""Synthetic known-answer tests for the exposure decomposition."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from decompose_ohc_exposure import (  # noqa: E402
    analysis_view,
    annual_weight_matrix,
    build_annual_cross_products,
    circular_block_indices,
    cross_product_components,
    state_table,
    symmetric_decomposition,
)


class TestSymmetricDecomposition(unittest.TestCase):
    def test_known_two_state_answer(self) -> None:
        weights1 = np.array([[1.0, 0.0]])
        weights2 = np.array([[0.0, 1.0]])
        fields1 = np.array([[10.0, 20.0]])
        fields2 = np.array([[12.0, 22.0]])
        summary, _ = symmetric_decomposition(
            weights1, fields1, weights2, fields2, 1e-12, 1e-12
        )
        self.assertAlmostEqual(summary["delta_exposure"], 12.0)
        self.assertAlmostEqual(summary["redistribution_component"], 10.0)
        self.assertAlmostEqual(summary["ocean_component"], 2.0)
        self.assertAlmostEqual(summary["covariability_component"], 0.0)
        self.assertAlmostEqual(summary["observed_closure_residual"], 0.0)

    def test_equal_weights_have_zero_redistribution(self) -> None:
        weights = np.array([[0.25, 0.75], [0.25, 0.75]])
        fields1 = np.array([[10.0, 20.0], [12.0, 18.0]])
        fields2 = fields1 + 3.0
        summary, _ = symmetric_decomposition(
            weights, fields1, weights, fields2, 1e-12, 1e-12
        )
        self.assertAlmostEqual(summary["redistribution_component"], 0.0)
        self.assertAlmostEqual(summary["ocean_component"], 3.0)

    def test_equal_fields_have_zero_ocean_component(self) -> None:
        fields = np.array([[10.0, 20.0], [10.0, 20.0]])
        weights1 = np.array([[1.0, 0.0], [1.0, 0.0]])
        weights2 = np.array([[0.0, 1.0], [0.0, 1.0]])
        summary, _ = symmetric_decomposition(
            weights1, fields, weights2, fields, 1e-12, 1e-12
        )
        self.assertAlmostEqual(summary["ocean_component"], 0.0)
        self.assertAlmostEqual(summary["redistribution_component"], 10.0)

    def test_block_indices_have_fixed_length(self) -> None:
        rng = np.random.default_rng(7)
        sample = circular_block_indices(30, 3, rng)
        self.assertEqual(len(sample), 30)
        self.assertTrue(((sample >= 0) & (sample < 30)).all())

    def test_cross_product_shortcut_matches_explicit_resample(self) -> None:
        rng = np.random.default_rng(19)
        weights1 = rng.dirichlet(np.ones(9), size=8)
        weights2 = rng.dirichlet(np.ones(9), size=8)
        fields1 = rng.normal(2.0e10, 2.0e8, size=(8, 9))
        fields2 = rng.normal(2.1e10, 2.0e8, size=(8, 9))
        index1 = np.array([0, 1, 2, 4, 5, 6, 6, 7])
        index2 = np.array([1, 2, 3, 3, 4, 5, 6, 7])
        count1 = np.bincount(index1, minlength=8).astype(float) / len(index1)
        count2 = np.bincount(index2, minlength=8).astype(float) / len(index2)
        products = build_annual_cross_products(weights1, fields1, weights2, fields2)
        shortcut = cross_product_components(products, count1, count2)
        explicit, _ = symmetric_decomposition(
            weights1[index1], fields1[index1],
            weights2[index2], fields2[index2],
            1e-4, 1e-12,
        )
        absolute_tolerance = np.finfo(float).eps * np.max(np.abs([fields1, fields2])) * 10000.0
        for name in [
            "delta_exposure",
            "redistribution_component",
            "ocean_component",
            "covariability_component",
        ]:
            self.assertAlmostEqual(shortcut[name], explicit[name], delta=absolute_tolerance)

    def test_calendar_year_and_season_aligned_views_differ_only_at_cross_year_points(self) -> None:
        frame = pd.DataFrame({
            "agency": ["PRIMARY", "PRIMARY"],
            "definition": ["ts_only", "ts_only"],
            "sid": ["A", "A"],
            "season": [2000, 2000],
            "ohc_year": [2000, 2001],
            "point_is_land": [False, False],
            "key_id": [0, 1],
        })
        calendar = analysis_view(frame, "calendar_year", "ocean_points_only", 2000, 2001)
        aligned = analysis_view(frame, "season_year_aligned_only", "ocean_points_only", 2000, 2001)
        self.assertEqual(calendar["analysis_year"].tolist(), [2000, 2001])
        self.assertEqual(aligned["analysis_year"].tolist(), [2000])

    def test_cross_year_storm_fragment_is_not_counted_as_a_full_storm_twice(self) -> None:
        frame = pd.DataFrame({
            "sid": ["A", "A", "A", "B"],
            "analysis_year": [2000, 2000, 2001, 2000],
            "key_id": [0, 0, 1, 2],
        })
        weights, storms, points, effective_storms = annual_weight_matrix(
            frame, [2000, 2001], 3, "storm_normalized_equal_year"
        )
        self.assertTrue(np.allclose(weights[0], [0.4, 0.0, 0.6]))
        self.assertTrue(np.allclose(weights[1], [0.0, 1.0, 0.0]))
        self.assertEqual(storms.tolist(), [2, 1])
        self.assertTrue(np.allclose(effective_storms, [5.0 / 3.0, 1.0 / 3.0]))
        self.assertEqual(points.tolist(), [3, 1])

    def test_land_treatment_changes_only_source_land_points(self) -> None:
        frame = pd.DataFrame({
            "agency": ["PRIMARY", "PRIMARY"],
            "definition": ["ts_only", "ts_only"],
            "sid": ["A", "A"],
            "season": [2000, 2000],
            "ohc_year": [2000, 2000],
            "point_is_land": [False, True],
            "key_id": [0, 1],
        })
        ocean = analysis_view(frame, "calendar_year", "ocean_points_only", 2000, 2000)
        all_points = analysis_view(frame, "calendar_year", "nearest_ocean_all_points", 2000, 2000)
        self.assertEqual(len(ocean), 1)
        self.assertEqual(len(all_points), 2)

    def test_state_table_labels_sensitivity_only_land_source_keys(self) -> None:
        frame = pd.DataFrame({
            "match_status": ["matched", "matched"],
            "ohc_month": [8, 8],
            "grid_y": [10, 11],
            "grid_x": [20, 21],
            "grid_lat": [15.0, 16.0],
            "grid_lon": [120.0, 121.0],
            "point_is_land": [False, True],
        })
        keys = state_table(frame).set_index(["ohc_month", "grid_y", "grid_x"])
        ocean_key = keys.loc[(8, 10, 20)]
        land_key = keys.loc[(8, 11, 21)]
        self.assertTrue(bool(ocean_key["used_ocean_points_only"]))
        self.assertFalse(bool(ocean_key["sensitivity_only_land_source_key"]))
        self.assertFalse(bool(land_key["used_ocean_points_only"]))
        self.assertTrue(bool(land_key["used_nearest_ocean_all_points"]))
        self.assertTrue(bool(land_key["sensitivity_only_land_source_key"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
