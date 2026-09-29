from __future__ import annotations

import copy

import pandas as pd

from quant_timing.study import run_study


def run_horse_race(
    prices: pd.DataFrame,
    raw_config: dict,
    *,
    regime_history: pd.Series | None = None,
    regime_snapshot: float | None = None,
    macro: dict | None = None,
    signal_panel: pd.DataFrame | None = None,
    macro_history: pd.DataFrame | None = None,
    cash_yield: pd.Series | None = None,
    futures_price: pd.Series | None = None,
    audit: bool = True,
) -> pd.DataFrame:
    """Score each configured position model on the same costs, folds, and style book."""
    specs = raw_config.get("models") or {}
    names = list((raw_config.get("comparison") or {}).get("models") or specs or ["rules"])
    rows = []
    for name in names:
        config = copy.deepcopy(raw_config)
        config["position"] = {"model": name, **dict(specs.get(name) or {})}
        config["run_id"] = str(name)
        result = run_study(
            prices,
            config,
            regime_history=regime_history,
            regime_snapshot=regime_snapshot,
            macro=macro,
            signal_panel=signal_panel,
            macro_history=macro_history,
            cash_yield=cash_yield,
            futures_price=futures_price,
            audit=audit,
        )
        realized = result.realized.dropna(subset=["nav"])
        drawdown = None
        if not realized.empty:
            drawdown = float((realized["nav"] / realized["nav"].cummax() - 1.0).min())
        rows.append(
            {
                "model": name,
                "status": result.summary["status"],
                "scored_folds": result.summary["scored_folds"],
                "mean_excess_return": result.summary["mean_excess_return"],
                "mean_matched_excess_return": result.summary["mean_matched_excess_return"],
                "mean_turnover": result.summary["mean_turnover"],
                "full_sample_drawdown": drawdown,
                "descriptive_full_sample_net": result.summary["descriptive_full_sample_net"],
                "descriptive_full_sample_benchmark": result.summary[
                    "descriptive_full_sample_benchmark"
                ],
                "leakage_passed": result.summary["leakage_passed"],
            }
        )
    return pd.DataFrame(rows)
