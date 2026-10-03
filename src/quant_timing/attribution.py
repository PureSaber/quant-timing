"""Additive, NAV-linked attribution of the native strategy and matched return paths.

This is an accounting decomposition, not a causal allocation experiment. Each scored
fold is linked independently from unit capital on its existing decision-date mask.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_timing.book import FUTURES_COLUMNS
from quant_timing.contract import file_sha256

SCHEMA = "quant-timing.return-attribution/v1"
DAILY_PATH = "attribution/daily.csv"
DAILY_COLUMNS = [
    "date",
    "decision_date",
    "book",
    "component",
    "kind",
    "return_weight",
    "asset_return",
    "contribution",
    "opening_nav",
    "pnl",
]
BOOKS = {"strategy": "", "matched": "matched_"}
SEMANTICS = {
    "method": "opening_net_nav_times_daily_return_component",
    "scope": "descriptive_full_sample_and_individually_rebased_scored_folds",
    "unit": "return_fraction_of_each_period_initial_capital",
    "cost_semantics": {
        "base_cost": "native_turnover_times_configured_bps_proxy",
        "impact_cost": "native_configured_nonlinear_impact_model",
        "observed_execution_slippage": False,
    },
    "folds_are_independent_attribution_windows": True,
    "folds_are_separately_executed_accounts": False,
    "overlapping_folds_are_not_aggregated": True,
}


def _kind(symbol: str) -> str:
    if symbol in FUTURES_COLUMNS:
        return "futures"
    return {"CASH": "cash", "MARGIN": "margin"}.get(symbol, "asset")


def _opening_nav(net: pd.Series) -> np.ndarray:
    return np.r_[1.0, (1.0 + net.to_numpy(dtype=float)).cumprod()[:-1]]


def write_attribution(result, run_dir: Path, returns: pd.DataFrame) -> dict:
    realized = result.realized.dropna(subset=["net_return"])
    rows = []
    symbols = list(result.weights.columns)
    for book, prefix in BOOKS.items():
        weights = result.weights if book == "strategy" else result.matched_weights
        held = weights.shift(1).loc[realized.index]
        opening = _opening_nav(realized[f"{prefix}net_return"])
        for offset, (stamp, row) in enumerate(realized.iterrows()):
            base = {
                "date": str(stamp.date()),
                "decision_date": str(pd.Timestamp(row["decision_date"]).date()),
                "book": book,
                "opening_nav": float(opening[offset]),
            }
            for symbol in symbols:
                contribution = float(row[f"{prefix}contribution:{symbol}"])
                rows.append(
                    {
                        **base,
                        "component": symbol,
                        "kind": _kind(symbol),
                        "return_weight": float(held.loc[stamp, symbol]),
                        "asset_return": row[f"asset_return:{symbol}"],
                        "contribution": contribution,
                        "pnl": opening[offset] * contribution,
                    }
                )
            for cost in ("base_cost", "impact_cost"):
                contribution = -float(row[f"{prefix}{cost}"])
                rows.append(
                    {
                        **base,
                        "component": cost,
                        "kind": "modeled_cost",
                        "return_weight": None,
                        "asset_return": None,
                        "contribution": contribution,
                        "pnl": opening[offset] * contribution,
                    }
                )
    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    path = Path(run_dir) / DAILY_PATH
    path.parent.mkdir()
    daily.to_csv(path, index=False)
    payload = {
        "schema": SCHEMA,
        **SEMANTICS,
        "assets": symbols,
        "daily": {"path": DAILY_PATH, "sha256": file_sha256(path), "rows": len(daily)},
        "periods": _periods(daily, returns, result.folds.to_dict(orient="records")),
    }
    return payload


def _periods(daily: pd.DataFrame, returns: pd.DataFrame, folds: list[dict]) -> list[dict]:
    periods = [_period("descriptive", returns, daily)]
    for fold in folds:
        selected = returns.loc[
            returns["decision_date"].between(fold["test_start"], fold["test_end"])
        ]
        if len(selected) != fold["n_decisions"]:
            raise ValueError("attribution fold decision count mismatch")
        period = _period(f"fold-{fold['fold']}", selected, daily)
        period["test_start"] = fold["test_start"]
        period["test_end"] = fold["test_end"]
        for key in (
            "net_return",
            "matched_net_return",
            "benchmark_return",
            "excess_return",
            "matched_excess_return",
        ):
            _same(period[key], fold[key], f"fold {fold['fold']} {key}")
        periods.append(period)
    return periods


def _period(name: str, returns: pd.DataFrame, daily: pd.DataFrame) -> dict:
    result = {
        "id": name,
        "n_decisions": len(returns),
        "first_return_date": None if returns.empty else returns["date"].iloc[0],
        "last_return_date": None if returns.empty else returns["date"].iloc[-1],
        "components": [],
    }
    for key in ("net_return", "matched_net_return", "benchmark_return"):
        result[key] = None if returns.empty else float((1.0 + returns[key]).prod() - 1.0)
    result["excess_return"] = _gap(result["net_return"], result["benchmark_return"])
    result["matched_excess_return"] = _gap(result["net_return"], result["matched_net_return"])
    result["strategy_residual"] = None
    result["matched_residual"] = None
    if returns.empty:
        return result
    linked = {}
    for book, prefix in BOOKS.items():
        part = daily.loc[(daily["book"] == book) & daily["date"].isin(returns["date"])]
        pivot = part.pivot(index="date", columns=["kind", "component"], values="contribution")
        pivot = pivot.reindex(returns["date"])
        if pivot.empty or pivot.isna().any().any():
            raise ValueError("attribution period has incomplete components")
        linked[book] = pivot.mul(_opening_nav(returns[f"{prefix}net_return"]), axis=0).sum()
        residual = result[f"{prefix}net_return"] - float(linked[book].sum())
        _same(residual, 0.0, f"{name} {book} linked reconciliation")
        result[f"{book}_residual"] = residual
    if not linked["strategy"].index.equals(linked["matched"].index):
        raise ValueError("attribution comparison components mismatch")
    for (kind, component), value in linked["strategy"].items():
        matched = float(linked["matched"].loc[(kind, component)])
        result["components"].append(
            {
                "kind": kind,
                "component": component,
                "strategy": float(value),
                "matched": matched,
                "difference": float(value) - matched,
            }
        )
    _same(
        sum(row["difference"] for row in result["components"]),
        result["matched_excess_return"],
        f"{name} matched difference",
    )
    return result


def _gap(left, right):
    return None if left is None or right is None else left - right


def _same(actual, expected, label: str) -> None:
    if pd.isna(actual) or pd.isna(expected):
        if pd.isna(actual) and pd.isna(expected):
            return
        raise ValueError(f"attribution {label} missing")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), float(expected), abs_tol=1e-10, rel_tol=1e-10
    ):
        raise ValueError(f"attribution {label} mismatch")


def _numeric_equal(actual, expected, label: str) -> None:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"attribution {label} invalid")
    if not np.allclose(left, right, atol=1e-10, rtol=1e-10):
        raise ValueError(f"attribution {label} mismatch")


def _compare(actual, expected, label="periods") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"attribution {label} fields mismatch")
        for key in expected:
            _compare(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"attribution {label} count mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare(left, right, f"{label}.{index}")
    elif isinstance(expected, (int, float)):
        if actual is None or isinstance(actual, bool):
            raise ValueError(f"attribution {label} numeric value missing")
        _same(actual, expected, label)
    elif actual != expected:
        raise ValueError(f"attribution {label} mismatch")


def validate_attribution(run_dir: Path, metrics: dict) -> dict:
    """Verify bound daily evidence and recompute all periods against native standard CSVs."""
    payload = metrics["return_attribution"]
    if payload.get("schema") != SCHEMA or payload["daily"]["path"] != DAILY_PATH:
        raise ValueError("unsupported timing attribution schema or path")
    if any(payload.get(key) != value for key, value in SEMANTICS.items()):
        raise ValueError("attribution measurement semantics mismatch")
    root = Path(run_dir).resolve()
    path = root / DAILY_PATH
    if not path.resolve().is_relative_to(root):
        raise ValueError("attribution evidence escaped run")
    if not path.is_file() or file_sha256(path) != payload["daily"]["sha256"]:
        raise ValueError("attribution evidence missing or mutated")
    daily = pd.read_csv(path, dtype={"component": str, "date": str, "decision_date": str})
    if list(daily.columns) != DAILY_COLUMNS or len(daily) != payload["daily"]["rows"]:
        raise ValueError("attribution daily schema or row count mismatch")
    standard = root / "standard"
    returns = pd.read_csv(standard / "returns.csv", dtype={"date": str, "decision_date": str})
    costs = pd.read_csv(standard / "costs.csv", dtype={"date": str}).set_index("date")
    positions = pd.read_csv(standard / "positions.csv", dtype={"date": str, "symbol": str})
    _validate_daily(daily, returns, costs, positions, payload["assets"])
    fold_path = root / "validation/fold_metrics.csv"
    if (
        not fold_path.resolve().is_relative_to(root)
        or not fold_path.is_file()
        or file_sha256(fold_path) != metrics["fold_metrics_sha256"]
    ):
        raise ValueError("attribution fold evidence missing or mutated")
    fold_frame = pd.read_csv(fold_path)
    fold_rows = [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in fold_frame.to_dict(orient="records")
    ]
    _compare(metrics["walk_forward_folds"], fold_rows, "native folds")
    periods = _periods(daily, returns, metrics["walk_forward_folds"])
    _compare(payload["periods"], periods)
    _same(periods[0]["net_return"], metrics["descriptive_full_sample_net"], "full return")
    _same(
        periods[0]["benchmark_return"],
        metrics["descriptive_full_sample_benchmark"],
        "full benchmark",
    )
    return payload


def _validate_daily(daily, returns, costs, positions, symbols):
    dates = returns["date"].tolist()
    if (
        not dates
        or len(set(dates)) != len(dates)
        or dates != sorted(dates)
        or returns["strategy"].unique().tolist() != ["timing"]
    ):
        raise ValueError("attribution requires unique ordered timing dates")
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or set(daily["book"]) != set(BOOKS)
        or daily.duplicated(["date", "book", "kind", "component"]).any()
    ):
        raise ValueError("attribution components missing or duplicated")
    expected = {(_kind(symbol), symbol) for symbol in symbols}
    expected.update(("modeled_cost", name) for name in ("base_cost", "impact_cost"))
    for book, prefix in BOOKS.items():
        rows = daily.loc[daily["book"] == book]
        if set(zip(rows["kind"], rows["component"])) != expected:
            raise ValueError("attribution component set mismatch")
        if len(rows) != len(dates) * len(expected):
            raise ValueError("attribution daily components incomplete")
        opening = _opening_nav(returns[f"{prefix}net_return"])
        pieces = {}
        funded = np.zeros(len(dates))
        for kind, component in sorted(expected):
            part = rows.loc[(rows["kind"] == kind) & (rows["component"] == component)]
            if set(part["date"]) != set(dates):
                raise ValueError("attribution date coverage mismatch")
            part = part.set_index("date").loc[dates]
            if part["decision_date"].tolist() != returns["decision_date"].tolist():
                raise ValueError("attribution decision date mismatch")
            _numeric_equal(part["opening_nav"], opening, "opening NAV")
            _numeric_equal(part["pnl"], part["contribution"] * opening, "component PnL")
            values = part["contribution"].to_numpy(dtype=float)
            pieces[(kind, component)] = values
            if kind == "modeled_cost":
                if (
                    part[["return_weight", "asset_return"]].notna().any().any()
                    or (values > 0).any()
                ):
                    raise ValueError("attribution modeled cost semantics invalid")
                _numeric_equal(-values, returns[f"{prefix}{component}"], "cost component")
                continue
            weight = part["return_weight"].to_numpy(dtype=float)
            asset_return = part["asset_return"].to_numpy(dtype=float)
            contribution = weight * asset_return
            # The native futures path permits unavailable returns only for an idle leg.
            idle_missing = (np.abs(weight) <= 1e-12) & np.isnan(asset_return)
            if kind == "futures":
                contribution[idle_missing] = 0.0
            _numeric_equal(values, contribution, "lagged weight times asset return")
            if kind != "futures":
                funded += weight
            if book == "strategy":
                pos = positions.loc[positions["symbol"] == component]
                if pos["date"].duplicated().any() or set(pos["date"]) != set(dates):
                    raise ValueError("attribution position coverage mismatch")
                _numeric_equal(
                    weight,
                    pos.set_index("date").loc[dates, "return_weight"],
                    "native position return weight",
                )
        _numeric_equal(funded, np.ones(len(dates)), "funded weights")
        total = np.sum(list(pieces.values()), axis=0)
        gross = np.sum(
            [value for (kind, _), value in pieces.items() if kind != "modeled_cost"], axis=0
        )
        _numeric_equal(total, returns[f"{prefix}net_return"], "daily net return")
        _numeric_equal(gross, returns[f"{prefix}gross_return"], "daily gross return")
        if book == "strategy":
            _numeric_equal(
                opening * (1 + returns["net_return"].to_numpy()), returns["nav"], "native NAV"
            )
            if costs.index.has_duplicates or set(costs.index) != set(dates):
                raise ValueError("attribution native cost dates mismatch")
            for component, field in (("base_cost", "slippage"), ("impact_cost", "market_impact")):
                _numeric_equal(
                    -pieces[("modeled_cost", component)] * opening,
                    costs.loc[dates, field],
                    "native currency cost",
                )
            _numeric_equal(
                (gross - total) * opening,
                costs.loc[dates, "total_cost"],
                "native total currency cost",
            )
    asset_rows = daily.loc[daily["kind"] != "modeled_cost"]
    for symbol in symbols:
        observed = asset_rows.loc[asset_rows["component"] == symbol].pivot(
            index="date", columns="book", values="asset_return"
        )
        if not np.allclose(
            observed["strategy"], observed["matched"], equal_nan=True, atol=0, rtol=0
        ):
            raise ValueError("attribution books use different asset returns")
