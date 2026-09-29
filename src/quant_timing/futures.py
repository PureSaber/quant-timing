from __future__ import annotations

import calendar

import pandas as pd


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (calendar.FRIDAY - first.weekday()) % 7
    return first + pd.Timedelta(days=int(offset) + 14)


def contract_symbol(prefix: str, year: int, month: int) -> str:
    return f"{prefix}{year % 100:02d}{month:02d}"


def quarter_contracts(
    prefix: str, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[str, pd.Timestamp]]:
    contracts = []
    year = int(start.year) - 1
    while year <= int(end.year) + 1:
        for month in (3, 6, 9, 12):
            expiry = third_friday(year, month)
            if expiry < start - pd.Timedelta(days=120) or expiry > end + pd.Timedelta(days=120):
                continue
            contracts.append((contract_symbol(prefix, year, month), expiry))
        year += 1
    return contracts


def dominant_schedule(
    calendar_index: pd.DatetimeIndex,
    contracts: list[tuple[str, pd.Timestamp]],
    *,
    roll_sessions: int = 5,
    trading_sessions: pd.DatetimeIndex,
) -> pd.Series:
    """Roll against a complete, independent exchange calendar, never the price panel.

    Expiries on holidays advance to the next exchange session. The calendar must
    cover the expiry of the front contract on every decision date.
    """
    ordered = sorted(contracts, key=lambda item: item[1])
    calendar = pd.DatetimeIndex(calendar_index).sort_values()
    sessions = pd.DatetimeIndex(trading_sessions)
    if (
        not ordered
        or roll_sessions < 0
        or not sessions.is_monotonic_increasing
        or sessions.has_duplicates
    ):
        raise ValueError("contracts and a sorted, unique trading calendar are required")
    if not calendar.isin(sessions).all():
        raise ValueError("decision dates are missing from the trading calendar")
    schedule = []
    for day in calendar:
        upcoming = [(symbol, expiry) for symbol, expiry in ordered if expiry >= day]
        if not upcoming:
            schedule.append(None)
            continue
        symbol, expiry = upcoming[0]
        expiry_loc = sessions.searchsorted(expiry, side="left")
        if expiry_loc == len(sessions) or expiry_loc < roll_sessions:
            raise ValueError(f"trading calendar does not cover the roll window for {symbol}")
        roll_day = sessions[int(expiry_loc) - roll_sessions]
        if day >= roll_day and len(upcoming) > 1:
            symbol = upcoming[1][0]
        schedule.append(symbol)
    return pd.Series(schedule, index=calendar, name="contract")


def stitch_returns(prices: dict[str, pd.Series], schedule: pd.Series) -> pd.Series:
    """Return of the dominant contract, using that contract's own previous close.

    A roll therefore does not book the gap between the old contract and the new one.
    """
    returns = []
    previous_day = None
    for day, contract in schedule.items():
        series = prices.get(str(contract))
        if (
            previous_day is None
            or series is None
            or day not in series.index
            or previous_day not in series.index
        ):
            returns.append(float("nan"))
        else:
            previous = float(series.loc[previous_day])
            current = float(series.loc[day])
            returns.append(current / previous - 1.0 if previous > 0 else float("nan"))
        previous_day = day
    return pd.Series(returns, index=schedule.index, name="return")


def price_index_from_returns(returns: pd.Series, start: float = 1000.0) -> pd.Series:
    """Retain unavailable dates, including all dates before the first observed return."""
    growth = (1.0 + returns).cumprod()
    return growth * float(start)
