from __future__ import annotations

import pandas as pd
import pytest

from quant_timing.book import simulate
from quant_timing.constraints import apply_earnings_yield_cap, constrain_scale
from quant_timing.macro_history import apply_macro_history
from quant_timing.overlay import apply_futures_overlay
from quant_timing.position_models import build_position
from quant_timing.style import style_internal_weights


def test_vol_target_is_target_over_realized_vol() -> None:
    index = pd.bdate_range("2020-01-01", periods=40)
    steps = pd.Series(0.01, index=index)
    steps.iloc[20:] = [0.02 * ((-1) ** i) for i in range(20)]
    close = 100 * (1 + steps).cumprod()
    history = build_position(
        close,
        {"model": "vol_target", "target_vol": 0.2, "vol_window": 10, "floor": 0.0, "cap": 1.0},
    )
    mature = history["regime"].eq("vol_target")
    assert mature.any()
    assert history.loc[mature, "position_scale"].between(0, 1).all()


def test_tsmom_and_moving_average_ignore_future_prices() -> None:
    index = pd.bdate_range("2020-01-01", periods=30)
    close = pd.Series(range(100, 130), index=index, dtype=float)
    rules = {"model": "tsmom", "return_window": 5, "floor": 0.0, "cap": 1.0}
    original = build_position(close, rules)
    revised = close.copy()
    revised.iloc[-1] = 1.0
    assert build_position(revised, rules)["position_scale"].iloc[-2] == original["position_scale"].iloc[-2]
    trend = build_position(
        close,
        {"model": "moving_average", "fast_window": 3, "slow_window": 8, "floor": 0.0, "cap": 1.0},
    )
    assert set(trend["regime"]).issubset({"warmup", "moving_average"})


def test_speed_limit_slows_changes_after_the_initial_allocation() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    scale = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)
    equity = pd.Series(0.0, index=index)
    limited = constrain_scale(
        scale,
        equity,
        equity,
        max_daily_change=0.25,
        drawdown_line=None,
        drawdown_floor=0.0,
    )
    assert limited.iloc[1] == pytest.approx(0.25)
    assert limited.iloc[2] == pytest.approx(0.5)
    assert limited.iloc[3] == pytest.approx(0.25)


def test_drawdown_line_cuts_exposure_after_the_loss_is_known() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    scale = pd.Series(1.0, index=index)
    equity = pd.Series([0.0, -0.1, -0.1, -0.1, 0.0, 0.0], index=index)
    limited = constrain_scale(
        scale,
        equity,
        pd.Series(0.0, index=index),
        max_daily_change=None,
        drawdown_line=0.15,
        drawdown_floor=0.2,
    )
    assert limited.iloc[1] == pytest.approx(1.0)
    assert limited.iloc[-1] == pytest.approx(0.2)


def test_futures_overlay_does_not_create_cash_from_a_short() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"market": [100.0, 110.0]}, index=index)
    weights = pd.DataFrame({"market": [1.0, 1.0], "CASH": [0.0, 0.0]}, index=index)
    scale = pd.Series([0.0, 0.0], index=index)
    hedged = apply_futures_overlay(weights, scale, margin_rate=0.0)
    funded = [column for column in hedged.columns if column not in {"IF", "IC", "IM"}]
    assert hedged[funded].sum(axis=1).iloc[-1] == pytest.approx(1.0)
    assert hedged["IF"].iloc[-1] == pytest.approx(-1.0)
    assert hedged["CASH"].iloc[-1] == pytest.approx(0.0)
    realized = simulate(
        hedged,
        hedged,
        prices,
        "market",
        0.0,
        extra_returns={"IF": prices["market"].pct_change()},
    )
    assert realized["gross_return"].iloc[-1] == pytest.approx(0.0)


def test_small_cap_sleeve_is_hedged_with_im_when_it_is_listed() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    weights = pd.DataFrame({"csi1000": [1.0, 1.0], "CASH": [0.0, 0.0]}, index=index)
    scale = pd.Series([0.0, 0.0], index=index)
    listed = pd.DataFrame({"IF": True, "IC": True, "IM": True}, index=index)
    hedged = apply_futures_overlay(weights, scale, margin_rate=0.0, available=listed)
    assert hedged["IM"].iloc[-1] == pytest.approx(-1.0)
    assert hedged["IF"].iloc[-1] == pytest.approx(0.0)
    missing = pd.DataFrame({"IF": True, "IC": False, "IM": False}, index=index)
    fallback = apply_futures_overlay(weights, scale, margin_rate=0.0, available=missing)
    assert fallback["IF"].iloc[-1] == pytest.approx(-1.0)
    assert fallback["IM"].iloc[-1] == pytest.approx(0.0)


def test_macro_history_does_not_use_a_later_release() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    scale = pd.Series([1.0, 1.0], index=index)
    history = pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "series": ["northbound"],
            "value": [1.0],
            "available_at": ["2024-01-03"],
        }
    )
    adjusted = apply_macro_history(
        scale,
        history,
        {
            "incomplete_policy": "baseline",
            "baseline_scale": 0.9,
            "rules": [{"series": "northbound", "compare": "above", "level": 0.0, "scale_cap": 0.2}],
        },
    )
    assert adjusted.iloc[0] == pytest.approx(0.9)
    assert adjusted.iloc[1] == pytest.approx(0.2)


def test_style_signals_and_active_deviation_cap() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {"small": [100.0, 100.0], "large": [100.0, 100.0], "bank": [10.0, 10.0], "pharma": [10.0, 11.0]},
        index=index,
    )
    signals = pd.DataFrame({"size": [pd.NA, 0.2], "industry.bank": [1.0, 1.0], "industry.pharma": [1.0, 3.0]}, index=index)
    weights = style_internal_weights(
        prices,
        {
            "return_window": 1,
            "tilt": 0.8,
            "threshold": 0.0,
            "pairs": [
                {
                    "name": "size",
                    "left": "small",
                    "right": "large",
                    "group_weight": 0.5,
                    "signal": "crowding",
                    "signal_column": "size",
                    "max_active_deviation": 0.05,
                }
            ],
            "groups": [
                {
                    "name": "industry",
                    "members": ["bank", "pharma"],
                    "group_weight": 0.5,
                    "signal": "momentum",
                    "max_active_deviation": 0.05,
                }
            ],
        },
        signals,
    )
    assert weights.loc[index[1], "small"] == pytest.approx(0.2)
    assert abs(weights.loc[index[1], "bank"] - 0.25) <= 0.05 + 1e-8
    assert abs(weights.loc[index[1], "pharma"] - 0.25) <= 0.05 + 1e-8


def test_impact_cost_grows_faster_than_turnover() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"market": [100.0, 100.0, 100.0]}, index=index)
    small = pd.DataFrame({"market": [1.0, 0.5, 0.5], "CASH": [0.0, 0.5, 0.5]}, index=index)
    large = pd.DataFrame({"market": [1.0, 0.0, 0.0], "CASH": [0.0, 1.0, 1.0]}, index=index)
    small_cost = simulate(small, small, prices, "market", 0.0, impact_coef=0.2)["cost"].iloc[2]
    large_cost = simulate(large, large, prices, "market", 0.0, impact_coef=0.2)["cost"].iloc[2]
    assert large_cost / small_cost > 2.0


def test_revision_signal_overweights_the_positive_side() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"value": [10.0, 10.0], "growth": [10.0, 10.0]}, index=index)
    signals = pd.DataFrame({"revision": [pd.NA, 0.05]}, index=index)
    weights = style_internal_weights(
        prices,
        {
            "return_window": 1,
            "tilt": 0.8,
            "threshold": 0.0,
            "pairs": [
                {
                    "name": "revision",
                    "left": "value",
                    "right": "growth",
                    "group_weight": 1.0,
                    "signal": "revision",
                    "signal_column": "revision",
                }
            ],
        },
        signals,
    )
    assert weights.loc[index[1], "value"] == pytest.approx(0.8)


def test_industry_valuation_spread_prefers_the_cheaper_sleeve() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"bank": [10.0, 10.0], "pharma": [10.0, 10.0]}, index=index)
    signals = pd.DataFrame(
        {"industry.bank": [10.0, 10.0], "industry.pharma": [30.0, 30.0]},
        index=index,
    )
    weights = style_internal_weights(
        prices,
        {
            "return_window": 1,
            "tilt": 0.8,
            "threshold": 0.0,
            "groups": [
                {
                    "name": "industry",
                    "members": ["bank", "pharma"],
                    "group_weight": 1.0,
                    "signal": "valuation_spread",
                }
            ],
        },
        signals,
    )
    assert weights.loc[index[1], "bank"] > weights.loc[index[1], "pharma"]


def test_amount_share_benchmark_caps_active_weight() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"bank": 10.0, "pharma": 10.0}, index=index)
    signals = pd.DataFrame({"industry.bank": 10.0, "industry.pharma": 10.0}, index=index)
    amounts = pd.DataFrame({"bank": 90.0, "pharma": 10.0}, index=index)
    weights = style_internal_weights(
        prices,
        {
            "return_window": 1,
            "tilt": 0.8,
            "threshold": 0.0,
            "groups": [
                {
                    "name": "industry",
                    "members": ["bank", "pharma"],
                    "group_weight": 1.0,
                    "signal": "valuation_spread",
                    "benchmark": "amount_share",
                    "amount_window": 3,
                    "max_active_deviation": 0.02,
                }
            ],
        },
        signals,
        amounts,
    )
    assert abs(weights["bank"].iloc[-1] - 0.9) <= 0.02 + 1e-8


def test_adv_impact_uses_participation_and_reports_a_different_cost() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"market": [100.0, 100.0, 100.0]}, index=index)
    weights = pd.DataFrame({"market": [1.0, 0.5, 0.5], "CASH": [0.0, 0.5, 0.5]}, index=index)
    adv = pd.Series(100.0, index=index)
    cost = simulate(weights, weights, prices, "market", 0.0, impact_coef=0.2, adv=adv, capital=50.0)
    assert cost["cost"].iloc[2] == pytest.approx(0.05)


def test_earnings_yield_cap_uses_only_past_valuations() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    peg = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 40.0], index=index)
    scale = pd.Series(1.0, index=index)
    capped = apply_earnings_yield_cap(
        scale, peg, lookback=5, expensive_percentile=0.3, scale_cap=0.6
    )
    assert capped.iloc[-2] == pytest.approx(1.0)
    assert capped.iloc[-1] == pytest.approx(0.6)
