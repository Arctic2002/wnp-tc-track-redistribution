"""Synthetic GSHHG-raster lookup tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from land_mask import classify_points  # noqa: E402


class TestLandMaskLookup(unittest.TestCase):
    def setUp(self) -> None:
        self.info = {
            "mask": np.array([[1, 0], [0, 1]], dtype=np.uint8),
            "west": 100.0,
            "east": 102.0,
            "south": 0.0,
            "north": 2.0,
            "resolution": 1.0,
        }

    def test_land_ocean_and_outside(self) -> None:
        result = classify_points(
            np.array([1.5, 1.5, 3.0]),
            np.array([100.5, 101.5, 100.5]),
            self.info,
        )
        self.assertEqual(result.tolist(), [1, 0, -1])

    def test_east_and_south_edges_are_clipped_inside(self) -> None:
        result = classify_points(np.array([0.0]), np.array([102.0]), self.info)
        self.assertEqual(result.tolist(), [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
