from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_timing.book import simulate
from quant_timing.export import _standard_frames


def test_weight_earns_the_next_bar_and_turnover_is_charged_there() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"market": [100.0, 110.0, 121.0]}, index=index)
    weights = pd.DataFrame(
        {"market": [1.0, 0.0, 1.0], "CASH": [0.0, 1.0, 0.0]},
        index=index,
    )
    realized = simulate(weights, weights.copy(), prices, "market", cost_bps=100)

    assert pd.isna(realized["decision_date"].iloc[0])
    assert realized.loc[index[1], "decision_date"] == index[0]
    assert realized.loc[index[1], "gross_return"] == pytest.approx(0.1)
    assert realized.loc[index[1], "net_return"] == pytest.approx(0.1)
    assert realized.loc[index[1], "nav"] == pytest.approx(1.1)

    # Switching the whole book between market and cash is one unit of one-way turnover.
    assert realized.loc[index[2], "decision_date"] == index[1]
    assert realized.loc[index[2], "gross_return"] == pytest.approx(0.0)
    assert realized.loc[index[2], "turnover"] == pytest.approx(1.0)
    assert realized.loc[index[2], "cost"] == pytest.approx(0.01)
    assert realized.loc[index[2], "net_return"] == pytest.approx(-0.01)
    assert realized.loc[index[2], "benchmark_return"] == pytest.approx(0.1)
    assert realized.loc[index[2], "nav"] == pytest.approx(1.089)
    assert (realized["decision_date"].dropna() < realized.index[1:]).all()


def test_standard_positions_explain_the_same_day_return() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"market": [100.0, 110.0, 121.0]}, index=index)
    weights = pd.DataFrame(
        {"market": [1.0, 0.0, 1.0], "CASH": [0.0, 1.0, 0.0]},
        index=index,
    )
    realized = simulate(weights, weights.copy(), prices, "market", cost_bps=100)
    position = pd.DataFrame({"position_scale": weights["market"]}, index=index)
    frames = _standard_frames(
        SimpleNamespace(weights=weights, realized=realized, position=position),
        prices,
    )
    held = frames["positions"]
    earned = held[(held["date"] == "2024-01-03") & (held["symbol"] == "market")].iloc[0]
    assert earned["weight"] == pytest.approx(1.0)
    later = held[(held["date"] == "2024-01-04") & (held["symbol"] == "market")].iloc[0]
    assert later["weight"] == pytest.approx(0.0)
