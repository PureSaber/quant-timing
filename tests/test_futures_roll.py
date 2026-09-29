from __future__ import annotations

import pandas as pd

from quant_timing.futures import dominant_schedule, stitch_returns, third_friday


def test_roll_uses_the_new_contract_return_instead_of_the_gap() -> None:
    index = pd.bdate_range("2024-03-01", periods=15)
    expiry = third_friday(2024, 3)
    contracts = [("IF2403", expiry), ("IF2406", third_friday(2024, 6))]
    schedule = dominant_schedule(index, contracts, roll_sessions=2)
    old = pd.Series(100.0, index=index)
    new = pd.Series(110.0 + pd.RangeIndex(len(index)), index=index, dtype=float)
    stitched = stitch_returns({"IF2403": old, "IF2406": new}, schedule)
    roll_day = schedule.index[schedule.eq("IF2406")][0]
    previous = schedule.index[schedule.index.get_loc(roll_day) - 1]
    expected = new.loc[roll_day] / new.loc[previous] - 1.0
    assert stitched.loc[roll_day] == expected
    assert abs(stitched.loc[roll_day]) < 0.02
