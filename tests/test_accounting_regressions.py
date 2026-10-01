from types import SimpleNamespace

import pandas as pd
import pytest

from quant_timing.book import simulate
from quant_timing.constraints import full_equity_return
from quant_timing.export import _standard_frames
from quant_timing.study import run_study


def test_style_return_uses_previous_mix_including_first_ready_day():
    dates = pd.bdate_range("2026-09-01", periods=4)
    prices = pd.DataFrame(
        {"market": [100, 90, 90, 90], "A": [100, 100, 80, 80], "B": 100}, index=dates
    )
    internal = pd.DataFrame(
        {"A": [float("nan"), 0.7, 0.3, 0.3], "B": [float("nan"), 0.3, 0.7, 0.7]}, index=dates
    )
    result = full_equity_return(prices, "market", internal)
    assert result.iloc[1:].tolist() == pytest.approx([-0.1, -0.14, 0.0])


def test_style_switch_cuts_exposure_on_realized_drawdown():
    dates = pd.bdate_range("2026-09-01", periods=12)
    prices = pd.DataFrame(
        {"market": range(100, 112), "A": [100.0] * 3 + [80.0] * 9, "B": 100.0}, index=dates
    )
    signals = pd.DataFrame({"rotation": [1.0] * 3 + [-1.0] * 9}, index=dates)
    config = {
        "market": "market",
        "position": {"model": "tsmom", "return_window": 1, "floor": 1.0, "cap": 1.0},
        "style": {
            "return_window": 1,
            "tilt": 0.7,
            "threshold": 0.0,
            "pairs": [
                {
                    "name": "rotation",
                    "left": "A",
                    "right": "B",
                    "group_weight": 1.0,
                    "signal": "revision",
                }
            ],
        },
        "constraints": {"drawdown_line": 0.1, "drawdown_floor": 0.2},
        "costs": {"bps": 0.0},
        "validation": {"train_size": 2, "test_size": 2, "step_size": 2, "embargo": 0},
    }
    result = run_study(prices, config, signal_panel=signals, audit=True)
    assert result.realized["net_return"].iloc[3] == pytest.approx(-0.14)
    assert result.position["position_scale"].iloc[3] == pytest.approx(0.2)
    assert result.weights["CASH"].iloc[3] == pytest.approx(0.8)
    assert result.leakage["passed"]
    assert result.summary["status"] == "complete"


def test_drift_cost_orders_and_fee_funded_cash_share_one_trade_path():
    dates = pd.bdate_range("2026-09-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 110.0, 110.0], "B": 100.0}, index=dates)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5, "CASH": 0.0}, index=dates)
    ledger = simulate(weights, weights, prices, "A", cost_bps=100.0)
    assert prices.index.name is None
    assert weights.index.name is None
    assert ledger.index.name == "date"
    traded = 0.5 * (abs(0.5 - 0.55 / 1.05) + abs(0.5 - 0.5 / 1.05))
    assert ledger["turnover"].iloc[2] == pytest.approx(traded)
    assert ledger["cost"].iloc[2] == pytest.approx(traded * 0.01)
    assert ledger["matched_cost"].iloc[2] == pytest.approx(traded * 0.01)
    # Fees leave a cash debit; the following rebalance funds it by selling assets.
    cost = traded * 0.01
    assert ledger["turnover"].iloc[3] == pytest.approx(cost / (1 - cost))
    frames = _standard_frames(
        SimpleNamespace(
            weights=weights,
            realized=ledger,
            position=pd.DataFrame({"position_scale": 1.0}, index=dates),
        ),
        prices,
    )
    orders = frames["orders"]
    day = orders[orders["timestamp"] == "2026-09-02"].set_index("symbol")
    assert day.loc["A", "side"] == "sell"
    assert day.loc["B", "side"] == "buy"
    assert day.loc["A", "quantity"] * 110 == pytest.approx(0.025)
    assert day.loc["B", "quantity"] * 100 == pytest.approx(0.025)


def test_futures_pnl_settles_to_cash_before_rebalancing():
    dates = pd.bdate_range("2026-09-01", periods=3)
    prices = pd.DataFrame({"A": 100.0}, index=dates)
    weights = pd.DataFrame({"A": 0.9, "IF": -0.5, "CASH": 0.05, "MARGIN": 0.05}, index=dates)
    ledger = simulate(
        weights,
        weights,
        prices,
        "A",
        0.0,
        extra_returns={"IF": pd.Series([float("nan"), 0.1, 0.0], index=dates)},
    )
    # NAV falls to .95; futures settle -.05 into cash, margin remains .05.
    before = [0.9 / 0.95, -0.55 / 0.95, 0.0, 0.05 / 0.95]
    expected = sum(abs(a - b) for a, b in zip(weights.iloc[1], before)) / 2
    assert ledger["turnover"].iloc[2] == pytest.approx(expected)


def test_cash_yield_drift_and_impact_use_decision_day_activity():
    dates = pd.bdate_range("2026-09-01", periods=3)
    prices = pd.DataFrame({"A": 100.0}, index=dates)
    weights = pd.DataFrame({"A": 0.5, "CASH": 0.5}, index=dates)
    ledger = simulate(
        weights,
        weights,
        prices,
        "A",
        0.0,
        cash_returns=pd.Series([0.0, 0.1, 0.0], index=dates),
        impact_coef=0.2,
        adv=pd.Series([100.0, 100.0, 10000.0], index=dates),
        capital=50.0,
    )
    traded = 0.5 - 0.5 / 1.05
    assert ledger["turnover"].iloc[2] == pytest.approx(traded)
    assert ledger["cost"].iloc[2] == pytest.approx(0.2 * traded * (traded * 50 / 100) ** 0.5)
