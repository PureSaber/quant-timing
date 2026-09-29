from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_timing.book import assemble_weights, simulate
from quant_timing.config import resolve_config
from quant_timing.macro_policy import apply_macro, normalize_macro
from quant_timing.position import position_history
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
    audit: bool = True,
) -> StudyResult:
    config = resolve_config(raw_config)
    _require_columns(prices, config)
    external, snapshot = _external_scales(config, regime_history, regime_snapshot)
    context = normalize_macro(macro)
    position = position_history(prices[config["market"]], config["position"], external)
    style = config["style"]
    internal = style_internal_weights(prices, style) if style else None
    weights, matched = assemble_weights(prices, position, style, internal, config["market"])
    realized = simulate(weights, matched, prices, config["market"], float(config["costs_bps"]))
    folds = score_folds(
        realized,
        iter_folds(prices.index, **config["validation"]),
    )
    leakage = (
        _audit(prices, raw_config, regime_history, snapshot, position, weights)
        if audit
        else {"passed": False, "reason": "not_run", "checks": []}
    )
    summary = _summary(folds, realized, leakage)
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


def _summary(folds: pd.DataFrame, realized: pd.DataFrame, leakage: dict[str, Any]) -> dict[str, Any]:
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
        "mean_matched_excess_return": _mean(folds["matched_excess_return"]) if not folds.empty else None,
        "mean_turnover": _mean(folds["avg_turnover"]) if not folds.empty else None,
        "descriptive_full_sample_net": _compound(sample["net_return"]),
        "descriptive_full_sample_benchmark": _compound(sample["benchmark_return"]),
        "acceptance": "walk_forward_folds",
        "disclaimer": DISCLAIMER,
    }


def _audit(
    prices: pd.DataFrame,
    raw_config: dict[str, Any],
    regime_history: pd.Series | None,
    regime_snapshot: float | None,
    position: pd.DataFrame,
    weights: pd.DataFrame,
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
            audit=False,
        )
        scale_diff = abs(
            float(partial.position.loc[stamp, "position_scale"])
            - float(position.loc[stamp, "position_scale"])
        )
        weight_diff = (
            partial.weights.loc[stamp].astype(float) - weights.loc[stamp].astype(float)
        ).abs()
        regime_match = str(partial.position.loc[stamp, "regime"]) == str(position.loc[stamp, "regime"])
        checks.append(
            {
                "as_of": pd.Timestamp(stamp).strftime("%Y-%m-%d"),
                "max_abs_weight_diff": float(weight_diff.max()),
                "scale_diff": float(scale_diff),
                "regime_match": bool(regime_match),
            }
        )
    passed = all(
        item["regime_match"] and item["scale_diff"] <= 1e-10 and item["max_abs_weight_diff"] <= 1e-10
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
