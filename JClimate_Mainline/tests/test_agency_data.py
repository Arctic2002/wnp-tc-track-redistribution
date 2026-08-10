"""Focused tests for agency-native classification and interpolation flags."""

import numpy as np
import pandas as pd

from paper2_dynamic.agency_data import native_ts_mask


def _base():
    return pd.DataFrame(
        {
            "IFLAG": ["OVO____________", "OOO____________", "PPP____________"],
            "USA_LAT": [10, 10, 10],
            "USA_LON": [130, 130, 130],
            "USA_WIND": [40, 20, 80],
            "USA_STATUS": [" ", "TS", "TY"],
            "TOKYO_LAT": [10, 10, 10],
            "TOKYO_LON": [130, 130, 130],
            "TOKYO_WIND": [35, 35, 80],
            "TOKYO_GRADE": [3, 2, 5],
            "CMA_LAT": [10, 10, 10],
            "CMA_LON": [130, 130, 130],
            "CMA_WIND": [35, 25, 80],
            "CMA_CAT": [2, 1, 6],
        }
    )


def test_native_classification_excludes_interpolated_reports():
    frame = _base()
    assert native_ts_mask(frame, "USA").tolist() == [True, False, False]
    assert native_ts_mask(frame, "TOKYO").tolist() == [True, False, False]
    assert native_ts_mask(frame, "CMA").tolist() == [True, False, False]

