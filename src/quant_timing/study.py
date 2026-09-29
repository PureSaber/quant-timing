from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_timing.book import assemble_weights, simulate
from quant_timing.config import resolve_config
from quant_timing.constraints import (
    apply_earnings_yield_cap,
    constrain_scale,
    full_equity_return,
)
from quant_timing.macro_history import apply_macro_history
from quant_timing.macro_policy import apply_macro, normalize_macro
from quant_timing.overlay import DEFAULT_HEDGE, apply_futures_overlay, cash_returns_from_yield
from quant_timing.position_models import build_position
from quant_timing.style import style_internal_weights, style_sleeves
from quant_timing.walkforward import iter_folds, score_folds

DISCLAIMER = "固定规则的样本外对照，不是收益承诺。参数来自配置，验证段不搜索参数。"


@dataclass
class StudyResult:
    config: dict[str, Any]
    raw_config: dict[str, Any]
    position: pd.DataFrame
    weights: pd.DataFrame
    matched_weights: pd.DataFrame
    realized: pd.DataFrame
    folds: pd.DataFrame
    summary: dict[str, Any]
    leakage: dict[str, Any]
    latest: dict[str, Any]


def run_study(
    prices: pd.DataFrame,
    raw_config: dict[str, Any],
    *,
    regime_history: pd.Series | None = None,
    regime_snapshot: float | None = None,
    macro: dict | None = None,
    signal_panel: pd.DataFrame | None = None,
    macro_history: pd.DataFrame | None = None,
    cash_yield: pd.Series | None = None,
    futures_price: pd.Series | None = None,
    futures_prices: pd.DataFrame | None = None,
    amounts: pd.DataFrame | None = None,
    peg: pd.Series | None = None,
    audit: bool = True,
) -> StudyResult:
    config = resolve_config(raw_config)
    _require_columns(prices, config)
    external, snapshot = _external_scales(config, regime_history, regime_snapshot)
    context = normalize_macro(macro)
    position = build_position(prices[config["market"]], config["position"], external)
    position["position_scale"] = apply_macro_history(
        position["position_scale"], macro_history, config["macro"]
    )
    if config["valuation"] is not None:
        if peg is None:
            raise ValueError("valuation peg series was not loaded")
        rule = config["valuation"]
        position["position_scale"] = apply_earnings_yield_cap(
            position["position_scale"],
            peg,
            lookback=int(rule["lookback"]),
            expensive_percentile=float(rule["expensive_percentile"]),
            scale_cap=float(rule["scale_cap"]),
        )
    style = config["style"]
    internal = style_internal_weights(prices, style, signal_panel, amounts) if style else None
    cash_returns = _cash_returns(prices.index, config, cash_yield)
    constraints = config["constraints"]
    if constraints["max_daily_change"] is not None or constraints["drawdown_line"] is not None:
        position["position_scale"] = constrain_scale(
            position["position_scale"],
            full_equity_return(prices, config["market"], internal),
            cash_returns.fillna(0.0),
            max_daily_change=constraints["max_daily_change"],
            drawdown_line=constraints["drawdown_line"],
            drawdown_floor=float(constraints["drawdown_floor"]),
        )
    weights, matched = assemble_weights(prices, position, style, internal, config["market"])
    extra_returns = None
    overlay = config["overlay"]
    if overlay["mode"] == "futures":
        futures = _futures_frame(futures_prices, futures_price, prices.index, overlay["contracts"])
        betas, fallback, available = _hedge_inputs(
            prices, futures, list(weights.columns), int(overlay["beta_window"])
        )
        margin = float(overlay["margin_rate"])
        weights = apply_futures_overlay(
            weights,
            position["position_scale"],
            margin_rate=margin,
            betas=betas,
            fallback_betas=fallback,
            available=available,
        )
        matched = apply_futures_overlay(
            matched,
            position["position_scale"],
            margin_rate=margin,
            betas=betas,
            fallback_betas=fallback,
            available=available,
        )
        extra_returns = {
            name: futures[name].pct_change(fill_method=None) for name in futures.columns
        }
    adv = (
        _book_activity(amounts, weights)
        if amounts is not None and config["activity_unit"] == "CNY"
        else None
    )
    realized = simulate(
        weights,
        matched,
        prices,
        config["market"],
        float(config["costs_bps"]),
        None if config["cash"]["mode"] == "zero" else cash_returns,
        float(config["impact_coef"]),
        extra_returns,
        adv,
        config["capital"],
    )
    folds = score_folds(
        realized,
        iter_folds(prices.index, **config["validation"]),
    )
    leakage = (
        _audit(
            prices,
            raw_config,
            regime_history,
            snapshot,
            position,
            weights,
            signal_panel=signal_panel,
            macro_history=macro_history,
            cash_yield=cash_yield,
            futures_price=futures_price,
            futures_prices=futures_prices,
            amounts=amounts,
            peg=peg,
        )
        if audit
        else {"passed": False, "reason": "not_run", "checks": []}
    )
    summary = _summary(folds, realized, leakage)
    _add_capacity(summary, realized, adv, config)
    latest = _latest(position.iloc[-1], config, context, snapshot)
    return StudyResult(
        config=config,
        raw_config=raw_config,
        position=position,
        weights=weights,
        matched_weights=matched,
        realized=realized,
        folds=folds,
        summary=summary,
        leakage=leakage,
        latest=latest,
    )


def _require_columns(prices: pd.DataFrame, config: dict[str, Any]) -> None:
    required = {config["market"]}
    if config["style"]:
        required.update(style_sleeves(config["style"]))
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price panel is missing {sorted(missing)}")


def _external_scales(
    config: dict[str, Any],
    regime_history: pd.Series | None,
    regime_snapshot: float | None,
) -> tuple[pd.Series | None, float | None]:
    combine = config["regime"]["combine"]
    provided = (regime_history is not None, regime_snapshot is not None)
    if combine == "model":
        if any(provided):
            raise ValueError("external position_scale requires regime.combine cap")
        return None, None
    if provided == (False, False) or provided == (True, True):
        raise ValueError("cap requires exactly one external position_scale source")
    return regime_history, regime_snapshot


def _latest(
    row: pd.Series,
    config: dict[str, Any],
    macro: dict | None,
    snapshot: float | None,
) -> dict[str, Any]:
    model_scale = float(row["model_scale"])
    scale = float(row["position_scale"])
    signals: dict[str, Any] = {
        "vol_percentile": _optional_float(row["vol_percentile"]),
        "window_return": _optional_float(row["window_return"]),
        "model_scale": model_scale,
        "external_scale": _optional_float(row["external_scale"]),
    }
    latest: dict[str, Any] = {
        "as_of": pd.Timestamp(row.name).strftime("%Y-%m-%d"),
        "regime": str(row["regime"]),
        "model_position_scale": model_scale,
        "position_scale": scale,
        "action": "use",
        "research_only": True,
        "signals": signals,
    }
    if latest["regime"] == "warmup":
        latest["action"] = "blocked"
        latest["position_scale"] = None
        signals["macro_policy"] = "not_applied"
        signals["block_reason"] = "warmup"
        return latest
    adjusted, policy = apply_macro(scale, macro, config["macro"])
    if adjusted is not None and snapshot is not None:
        adjusted = min(adjusted, snapshot)
        policy = f"{policy}+snapshot_cap"
    signals["macro_policy"] = policy
    if adjusted is None:
        latest["action"] = "hold_previous"
        latest["position_scale"] = None
    else:
        latest["position_scale"] = float(adjusted)
    return latest


def _summary(
    folds: pd.DataFrame, realized: pd.DataFrame, leakage: dict[str, Any]
) -> dict[str, Any]:
    scored = 0 if folds.empty else int((folds["n_decisions"] > 0).sum())
    sample = realized.dropna(subset=["net_return"])
    if leakage["reason"] == "not_run":
        status = "audit_not_run"
    elif not leakage["passed"]:
        status = "leakage_failed"
    elif scored < 1:
        status = "insufficient_history"
    else:
        status = "complete"
    return {
        "status": status,
        "leakage_passed": bool(leakage["passed"]),
        "fold_count": len(folds),
        "scored_folds": scored,
        "mean_excess_return": _mean(folds["excess_return"]) if not folds.empty else None,
        "mean_matched_excess_return": _mean(folds["matched_excess_return"])
        if not folds.empty
        else None,
        "mean_turnover": _mean(folds["avg_turnover"]) if not folds.empty else None,
        "descriptive_full_sample_net": _compound(sample["net_return"]),
        "descriptive_full_sample_benchmark": _compound(sample["benchmark_return"]),
        "acceptance": "walk_forward_folds",
        "disclaimer": DISCLAIMER,
    }


def _cash_returns(
    index: pd.Index, config: dict[str, Any], cash_yield: pd.Series | None
) -> pd.Series:
    mode = config["cash"]["mode"]
    if mode == "zero":
        return pd.Series(0.0, index=index)
    if cash_yield is None:
        raise ValueError("cash yield series was not loaded")
    if mode == "yield":
        return cash_returns_from_yield(index, cash_yield, int(config["cash"]["daycount"]))
    return cash_yield.reindex(index).fillna(0.0)


def _audit(
    prices: pd.DataFrame,
    raw_config: dict[str, Any],
    regime_history: pd.Series | None,
    regime_snapshot: float | None,
    position: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    signal_panel: pd.DataFrame | None,
    macro_history: pd.DataFrame | None,
    cash_yield: pd.Series | None,
    futures_price: pd.Series | None,
    futures_prices: pd.DataFrame | None,
    amounts: pd.DataFrame | None,
    peg: pd.Series | None,
) -> dict[str, Any]:
    candidates = list(prices.index[:-1])
    if len(candidates) < 2:
        return {"passed": False, "reason": "no_audit_dates", "checks": []}
    checks = []
    for quantile in (0.4, 0.6, 0.8):
        stamp = candidates[min(len(candidates) - 1, int(len(candidates) * quantile))]
        partial = run_study(
            prices.loc[:stamp].copy(),
            raw_config,
            regime_history=regime_history,
            regime_snapshot=regime_snapshot,
            signal_panel=signal_panel,
            macro_history=macro_history,
            cash_yield=cash_yield,
            futures_price=futures_price,
            futures_prices=futures_prices,
            amounts=amounts,
            peg=peg,
            audit=False,
        )
        scale_diff = abs(
            float(partial.position.loc[stamp, "position_scale"])
            - float(position.loc[stamp, "position_scale"])
        )
        weight_diff = (
            partial.weights.loc[stamp].astype(float) - weights.loc[stamp].astype(float)
        ).abs()
        regime_match = str(partial.position.loc[stamp, "regime"]) == str(
            position.loc[stamp, "regime"]
        )
        checks.append(
            {
                "as_of": pd.Timestamp(stamp).strftime("%Y-%m-%d"),
                "max_abs_weight_diff": float(weight_diff.max()),
                "scale_diff": float(scale_diff),
                "regime_match": bool(regime_match),
            }
        )
    passed = all(
        item["regime_match"]
        and item["scale_diff"] <= 1e-10
        and item["max_abs_weight_diff"] <= 1e-10
        for item in checks
    )
    return {
        "passed": passed,
        "reason": "ok" if passed else "mismatch",
        "checks": checks,
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _mean(series: pd.Series) -> float | None:
    values = []
    for value in series:
        number = _optional_float(value)
        if number is not None:
            values.append(number)
    if not values:
        return None
    return float(sum(values) / len(values))


def _compound(series: pd.Series) -> float | None:
    if series.empty or series.isna().any():
        return None
    return float((1.0 + series).prod() - 1.0)


def _futures_frame(
    frame: pd.DataFrame | None,
    series: pd.Series | None,
    index: pd.Index,
    contracts: list[str],
) -> pd.DataFrame:
    if frame is None and series is None:
        raise ValueError("futures prices were not loaded")
    prices = pd.DataFrame({"IF": series}) if frame is None else frame
    missing = [name for name in contracts if name not in prices.columns]
    if missing:
        raise ValueError(f"futures prices missing {missing}")
    return prices.loc[:, list(contracts)].reindex(index).astype(float)


def _hedge_inputs(
    prices: pd.DataFrame,
    futures: pd.DataFrame,
    columns: list[str],
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocked = {"CASH", "MARGIN", "IF", "IC", "IM", "FUTURES"}
    sleeves = [column for column in columns if column not in blocked and column in prices.columns]
    betas = pd.DataFrame(index=prices.index, columns=sleeves, dtype=float)
    fallback = pd.DataFrame(index=prices.index, columns=sleeves, dtype=float)
    market = (
        futures["IF"].pct_change(fill_method=None)
        if "IF" in futures.columns
        else prices.iloc[:, 0].pct_change()
    )
    for sleeve in sleeves:
        sleeve_return = prices[sleeve].pct_change()
        contract = DEFAULT_HEDGE.get(sleeve, "IF")
        hedge = (
            futures[contract].pct_change(fill_method=None)
            if contract in futures.columns
            else market
        )
        betas[sleeve] = _trailing_beta(sleeve_return, hedge, window)
        fallback[sleeve] = _trailing_beta(sleeve_return, market, window)
    available = pd.DataFrame({name: futures[name].notna() for name in futures.columns})
    return betas, fallback, available


def _trailing_beta(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    covariance = left.rolling(window).cov(right)
    variance = right.rolling(window).var().replace(0, pd.NA)
    return (covariance / variance).clip(lower=0.2, upper=3.0).fillna(1.0)


def _book_activity(amounts: pd.DataFrame, weights: pd.DataFrame) -> pd.Series | None:
    equity = [column for column in weights.columns if column in amounts.columns]
    if not equity:
        return None
    held = weights[equity].clip(lower=0.0)
    total = held.sum(axis=1).replace(0, pd.NA)
    share = held.div(total, axis=0).fillna(0.0)
    return (share * amounts.reindex(index=weights.index, columns=equity).fillna(0.0)).sum(axis=1)


def _add_capacity(
    summary: dict[str, Any],
    realized: pd.DataFrame,
    adv: pd.Series | None,
    config: dict[str, Any],
) -> None:
    capital = config["capital"]
    if adv is None or capital is None:
        summary["median_participation"] = None
        summary["capacity"] = None
        summary["capacity_basis"] = "unavailable_without_monetary_turnover_and_capital"
        return
    summary["capacity_basis"] = "CNY_book_activity_proxy"
    activity = adv.reindex(realized.index).shift(1).replace(0, pd.NA)
    participation = realized["turnover"] * float(capital) / activity
    finite = participation.replace([float("inf"), float("-inf")], pd.NA).dropna()
    active = finite[finite.gt(1e-8)]
    summary["median_participation"] = None if active.empty else float(active.median())
    cap = float(config["participation_cap"])
    summary["capacity"] = None if active.empty else float((float(capital) * cap / active).median())
