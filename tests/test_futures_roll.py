from __future__ import annotations

import pandas as pd
import pytest

from quant_timing.futures import (
    dominant_schedule,
    price_index_from_returns,
    stitch_returns,
    third_friday,
)


def test_roll_uses_the_new_contract_return_instead_of_the_gap() -> None:
    index = pd.bdate_range("2024-03-01", periods=15)
    expiry = third_friday(2024, 3)
    contracts = [("IF2403", expiry), ("IF2406", third_friday(2024, 6))]
    sessions = pd.bdate_range("2024-01-01", "2024-06-30")
    schedule = dominant_schedule(index, contracts, roll_sessions=2, trading_sessions=sessions)
    old = pd.Series(100.0, index=index)
    new = pd.Series(110.0 + pd.RangeIndex(len(index)), index=index, dtype=float)
    stitched = stitch_returns({"IF2403": old, "IF2406": new}, schedule)
    roll_day = schedule.index[schedule.eq("IF2406")][0]
    previous = schedule.index[schedule.index.get_loc(roll_day) - 1]
    expected = new.loc[roll_day] / new.loc[previous] - 1.0
    assert stitched.loc[roll_day] == expected
    assert abs(stitched.loc[roll_day]) < 0.02


def test_panel_truncation_cannot_move_the_roll_date():
    sessions = pd.bdate_range("2026-01-01", "2027-06-30")
    contracts = [("IF2612", third_friday(2026, 12)), ("IF2703", third_friday(2027, 3))]
    dates = sessions[(sessions >= "2026-09-01") & (sessions <= "2026-12-31")]
    full = dominant_schedule(dates, contracts, trading_sessions=sessions)
    for end in ("2026-09-28", "2026-12-10", "2026-12-15"):
        partial = dominant_schedule(dates[dates <= end], contracts, trading_sessions=sessions)
        pd.testing.assert_series_equal(partial, full.loc[:end])
    assert full.loc["2026-09-28"] == "IF2612"
    assert full.loc["2026-12-11"] == "IF2703"
    with pytest.raises(ValueError, match="cover the roll window"):
        dominant_schedule(dates[:10], contracts, trading_sessions=dates[:10])


def test_roll_counts_exchange_sessions_including_holiday_expiry():
    # A holiday Friday expires on Monday; count actual sessions back from that Monday.
    sessions = pd.bdate_range("2024-03-01", "2024-06-30").difference(
        pd.DatetimeIndex(["2024-03-15"])
    )
    dates = sessions[(sessions >= "2024-03-07") & (sessions <= "2024-03-14")]
    schedule = dominant_schedule(
        dates,
        [("IF2403", third_friday(2024, 3)), ("IF2406", third_friday(2024, 6))],
        trading_sessions=sessions,
    )
    assert schedule.loc["2024-03-07"] == "IF2403"
    assert schedule.loc["2024-03-08"] == "IF2406"


def test_price_index_preserves_prelisting_and_internal_gaps():
    returns = pd.Series([float("nan"), float("nan"), 0.1, 0.2, float("nan"), 0.1])
    index = price_index_from_returns(returns)
    assert index.iloc[:2].isna().all()
    assert pd.isna(index.iloc[4])
    assert index.iloc[2] == pytest.approx(1100.0)
    assert index.pct_change(fill_method=None).iloc[3] == pytest.approx(0.2)


def test_prelisting_hedge_uses_if_and_missing_held_returns_fail():
    from quant_timing.book import simulate
    from quant_timing.overlay import apply_futures_overlay
    from quant_timing.study import _hedge_inputs

    dates = pd.bdate_range("2022-07-01", periods=4)
    prices = pd.DataFrame({"csi1000": [100.0, 101, 102, 103]}, index=dates)
    futures = pd.DataFrame(
        {"IF": [100.0, 101, 102, 103], "IM": [float("nan"), float("nan"), 100, 101]}, index=dates
    )
    funded = pd.DataFrame({"csi1000": 0.5, "CASH": 0.5}, index=dates)
    betas, fallback, available = _hedge_inputs(prices, futures, list(funded), 2)
    actual = apply_futures_overlay(
        funded,
        pd.Series(0.5, index=dates),
        margin_rate=0.12,
        betas=betas,
        fallback_betas=fallback,
        available=available,
    )
    assert actual["IM"].iloc[:2].eq(0.0).all()
    assert actual["IF"].iloc[:2].lt(0.0).all()
    broken = futures["IF"].pct_change(fill_method=None)
    broken.iloc[1] = float("nan")
    with pytest.raises(ValueError, match="missing futures return while holding IF"):
        simulate(
            actual,
            actual,
            prices,
            "csi1000",
            0.0,
            extra_returns={"IF": broken, "IM": futures["IM"].pct_change(fill_method=None)},
        )
