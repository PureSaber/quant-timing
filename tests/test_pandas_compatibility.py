import warnings

import numpy as np
import pandas as pd

from quant_timing.overlay import _available
from quant_timing.style import style_internal_weights


def test_amount_benchmark_keeps_numeric_dtype_and_active_weight_limits():
    dates = pd.bdate_range("2026-01-02", periods=8)
    prices = pd.DataFrame({"A": range(10, 18), "B": range(20, 28)}, index=dates)
    amounts = pd.DataFrame(
        {"A": [0.0, 0.0] + [100.0] * 6, "B": [0.0, 0.0] + [300.0] * 6}, index=dates
    )
    style = {
        "return_window": 1,
        "tilt": 0.7,
        "threshold": 0.0,
        "groups": [
            {
                "name": "size",
                "members": ["A", "B"],
                "group_weight": 1.0,
                "benchmark": "amount_share",
                "amount_window": 2,
                "max_active_deviation": 0.1,
            }
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        weights = style_internal_weights(prices, style, amounts=amounts)
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in weights.dtypes)
    assert weights.iloc[0].isna().all()
    np.testing.assert_allclose(weights.iloc[1:].sum(axis=1), 1.0)
    assert weights.loc[dates[3] :, "A"].between(0.15 - 1e-10, 0.35 + 1e-10).all()
    assert weights.loc[dates[3] :, "B"].between(0.65 - 1e-10, 0.85 + 1e-10).all()


def test_missing_contract_availability_stays_false_without_implicit_downcast():
    dates = pd.bdate_range("2026-01-02", periods=3)
    available = pd.DataFrame({"IF": [True, False]}, index=dates[:2])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        actual = _available(dates, available)
    expected = pd.DataFrame({"IF": [True, False, False], "IC": False, "IM": False}, index=dates)
    pd.testing.assert_frame_equal(actual, expected)
