from __future__ import annotations

import pandas as pd


def iter_folds(
    index: pd.Index,
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    embargo: int,
) -> list[dict]:
    """Expanding origin, then a gap, then a test window of decision dates.

    Rules are not refit inside the test window. The train span only withholds those dates from
    the scored sample.
    """
    folds = []
    origin = train_size
    while True:
        test_start = origin + embargo
        test_end = test_start + test_size
        if origin < 1 or test_end > len(index):
            break
        decision_dates = index[test_start:test_end]
        folds.append(
            {
                "train_end": index[origin - 1],
                "test_start": decision_dates[0],
                "test_end": decision_dates[-1],
                "decision_dates": decision_dates,
            }
        )
        origin += step_size
    return folds


def score_folds(realized: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    rows = []
    for number, fold in enumerate(folds, start=1):
        part = realized.loc[realized["decision_date"].isin(fold["decision_dates"])].dropna(
            subset=["net_return"]
        )
        net = _compound(part["net_return"])
        benchmark = _compound(part["benchmark_return"])
        matched = _compound(part["matched_net_return"])
        rows.append(
            {
                "fold": number,
                "train_end": _day(fold["train_end"]),
                "test_start": _day(fold["test_start"]),
                "test_end": _day(fold["test_end"]),
                "n_decisions": len(part),
                "net_return": net,
                "benchmark_return": benchmark,
                "matched_net_return": matched,
                "excess_return": _gap(net, benchmark),
                "matched_excess_return": _gap(net, matched),
                "avg_turnover": None if part.empty else float(part["turnover"].mean()),
                "max_drawdown": _drawdown(part["net_return"]),
            }
        )
    return pd.DataFrame(rows, columns=_FOLD_COLUMNS)


_FOLD_COLUMNS = [
    "fold",
    "train_end",
    "test_start",
    "test_end",
    "n_decisions",
    "net_return",
    "benchmark_return",
    "matched_net_return",
    "excess_return",
    "matched_excess_return",
    "avg_turnover",
    "max_drawdown",
]


def _day(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _compound(series: pd.Series) -> float | None:
    if series.empty or series.isna().any():
        return None
    return float((1.0 + series).prod() - 1.0)


def _gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _drawdown(series: pd.Series) -> float | None:
    if series.empty or series.isna().any():
        return None
    nav = (1.0 + series).cumprod()
    return float((nav / nav.cummax() - 1.0).min())
