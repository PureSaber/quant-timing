from __future__ import annotations

import pandas as pd
import pytest

from quant_timing.style import style_internal_weights


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "market": [100.0, 100.0, 100.0],
            "small": [100.0, 110.0, 110.0],
            "large": [100.0, 100.0, 110.0],
        },
        index=index,
    )


def _style(**updates: float) -> dict:
    style = {
        "return_window": 1,
        "tilt": 0.8,
        "threshold": 0.0,
        "pairs": [{"name": "size", "left": "small", "right": "large", "group_weight": 1.0}],
    }
    style.update(updates)
    return style


def _expect(weights: pd.Series, expected: dict[str, float]) -> None:
    assert list(weights.index) == list(expected)
    for name, value in expected.items():
        assert weights[name] == pytest.approx(value)


def test_pair_tilt_uses_only_the_trailing_window() -> None:
    weights = style_internal_weights(_prices(), _style())
    assert weights.iloc[0].isna().all()
    _expect(weights.iloc[1], {"small": 0.8, "large": 0.2})
    _expect(weights.iloc[2], {"small": 0.2, "large": 0.8})


def test_a_later_price_does_not_change_today_and_an_earlier_price_does() -> None:
    prices = _prices()
    original = style_internal_weights(prices, _style())
    revised = prices.copy()
    revised.iloc[-1, revised.columns.get_loc("small")] = 1000.0
    assert style_internal_weights(revised, _style()).iloc[1].equals(original.iloc[1])

    earlier = prices.copy()
    earlier.iloc[0, earlier.columns.get_loc("small")] = 200.0
    flipped = style_internal_weights(earlier, _style())
    _expect(flipped.iloc[1], {"small": 0.2, "large": 0.8})


def test_threshold_keeps_the_pair_balanced() -> None:
    prices = _prices()
    prices["large"] = prices["small"]
    weights = style_internal_weights(prices, _style(threshold=0.0))
    _expect(weights.iloc[1], {"small": 0.5, "large": 0.5})
