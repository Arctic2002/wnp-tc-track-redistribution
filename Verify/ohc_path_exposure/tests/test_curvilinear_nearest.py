"""Synthetic spherical nearest-ocean-cell tests; not yet executed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from match_tracks_to_ohc import nearest_ocean_cells  # noqa: E402


class TestCurvilinearNearest(unittest.TestCase):
    def test_nearest_valid_ocean_cell(self) -> None:
        latitude = np.array([[10.0, 10.2], [11.0, 11.2]])
        longitude = np.array([[130.0, 131.0], [130.1, 131.1]])
        values = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = np.ones((2, 2), dtype=bool)
        result = nearest_ocean_cells(
            np.array([10.15]), np.array([130.95]),
            latitude, longitude, values, mask, 6371.0088,
        )
        self.assertEqual(int(result["grid_y"][0]), 0)
        self.assertEqual(int(result["grid_x"][0]), 0)
        self.assertEqual(float(result["ohc300"][0]), 1.0)

    def test_longitude_wrap_is_spherical(self) -> None:
        latitude = np.array([[0.0, 0.0]])
        longitude = np.array([[359.8, 10.0]])
        values = np.array([[2.0, 3.0]])
        mask = np.ones((1, 2), dtype=bool)
        result = nearest_ocean_cells(
            np.array([0.0]), np.array([-0.1]),
            latitude, longitude, values, mask, 6371.0088,
        )
        self.assertEqual(int(result["grid_x"][0]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
