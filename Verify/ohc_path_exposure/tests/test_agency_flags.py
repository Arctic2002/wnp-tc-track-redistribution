"""Synthetic IFLAG and native TS classification tests; not yet executed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from extract_agency_tracks import agency_flag, native_ts_mask  # noqa: E402


class TestAgencyFlags(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "IFLAG": ["OOO", "VVV", "PPP", "III"],
            "USA_LAT": [10.0] * 4,
            "USA_LON": [130.0] * 4,
            "USA_WIND": [40.0] * 4,
            "USA_STATUS": ["TS"] * 4,
            "TOKYO_LAT": [10.0] * 4,
            "TOKYO_LON": [130.0] * 4,
            "TOKYO_WIND": [np.nan] * 4,
            "TOKYO_GRADE": [3] * 4,
            "CMA_LAT": [10.0] * 4,
            "CMA_LON": [130.0] * 4,
            "CMA_WIND": [np.nan] * 4,
            "CMA_CAT": [2] * 4,
        })

    def test_position_allows_interpolated_agency_report(self) -> None:
        self.assertEqual(agency_flag(self.frame, "USA", intensity=False).tolist(), [True, True, False, True])

    def test_intensity_rejects_i_and_p(self) -> None:
        self.assertEqual(agency_flag(self.frame, "USA", intensity=True).tolist(), [True, True, False, False])

    def test_native_ts_uses_original_or_verified_only(self) -> None:
        self.assertEqual(native_ts_mask(self.frame, "JMA").tolist(), [True, True, False, False])
        self.assertEqual(native_ts_mask(self.frame, "CMA").tolist(), [True, True, False, False])

    def test_usa_explicit_non_tropical_status_is_excluded(self) -> None:
        frame = self.frame.iloc[[0]].copy()
        frame["USA_STATUS"] = "EX"
        self.assertFalse(bool(native_ts_mask(frame, "USA").iloc[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
