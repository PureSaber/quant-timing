from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import yaml

from quant_timing import __version__
from quant_timing.attribution import SCHEMA as ATTRIBUTION_SCHEMA, write_attribution
from quant_timing.contract import file_sha256, validate_standard_run, write_standard_run
from quant_timing.overlay import FUTURES_COLUMNS
from quant_timing.study import StudyResult

STRATEGY = "timing"


def gate_decision(latest: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Hold back a consumable scale unless the causal audit and a scored fold both passed."""
    decision = {
        "as_of": latest["as_of"],
        "regime": latest["regime"],
        "model_position_scale": latest["model_position_scale"],
        "position_scale": latest["position_scale"],
        "action": latest["action"],
        "research_only": True,
        "signals": dict(latest["signals"]),
    }
    if not summary["leakage_passed"]:
        decision["action"] = "blocked"
        decision["position_scale"] = None
        decision["signals"]["block_reason"] = "leakage"
        return decision
    if summary["scored_folds"] < 1 and decision["action"] == "use":
        decision["action"] = "blocked"
        decision["position_scale"] = None
        decision["signals"]["block_reason"] = "insufficient_history"
    return decision


def exit_code(summary: dict[str, Any], decision: dict[str, Any]) -> int:
    if not summary["leakage_passed"]:
        return 1
    if summary["scored_folds"] < 1 or decision["action"] == "blocked":
        return 2
    return 0


def write_run(
    result: StudyResult,
    prices: pd.DataFrame,
    out_dir: Path,
    *,
    input_context: dict | None = None,
    allow_position_publication: bool = True,
) -> dict[str, Any]:
    """Validate a complete staged run, then publish it with one directory rename."""
    out_dir = Path(out_dir)
    _require_empty_destination(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as directory:
        staged = Path(directory)
        decision = _write_staged_run(
            result,
            prices,
            staged,
            input_context=input_context,
            allow_position_publication=allow_position_publication,
        )
        if input_context is not None:
            _write_json(staged / "run_context.json", input_context)
        _require_empty_destination(out_dir)
        if out_dir.exists():
            out_dir.rmdir()  # Removes only an empty destination; never existing run contents.
        staged.rename(out_dir)
    return decision


def _require_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite a nonempty run destination: {path}")


def _write_staged_run(
    result: StudyResult,
    prices: pd.DataFrame,
    out_dir: Path,
    *,
    input_context: dict | None = None,
    allow_position_publication: bool = True,
) -> dict[str, Any]:
    decision = gate_decision(result.latest, result.summary)
    if not allow_position_publication:
        decision["action"] = "blocked"
        decision["position_scale"] = None
        decision["signals"]["publication_policy"] = "research_comparison_only"
        decision["signals"].setdefault("block_reason", "research_comparison_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = out_dir / "validation"
    validation.mkdir(exist_ok=True)
    result.folds.to_csv(validation / "fold_metrics.csv", index=False)
    _write_json(validation / "summary.json", result.summary)
    _write_json(validation / "leakage_audit.json", result.leakage)
    _write_frame(out_dir / "position_history.csv", result.position)
    _write_frame(out_dir / "style_weights.csv", result.weights)
    metrics = _metrics(result, decision)
    metrics["fold_metrics_sha256"] = file_sha256(validation / "fold_metrics.csv")
    frames = _standard_frames(result, result.valuation_prices)
    metrics["return_attribution"] = write_attribution(result, out_dir, frames["returns"])
    if input_context is not None:
        metrics["input_context"] = input_context
    write_standard_run(
        out_dir,
        project="quant-timing",
        run_id=str(result.config["run_id"]),
        strategy="position_and_style" if result.config["style"] else "position_only",
        frames=frames,
        metrics=metrics,
        config=result.raw_config,
        code_version=__version__,
        dataset_snapshots={
            "prices": _prices_sha256(prices),
            "valuation_prices": _prices_sha256(result.valuation_prices),
        },
        tags={
            "orders": "research_target_not_routed",
            "asset_class": "index_style",
            "acceptance": "walk_forward_folds",
            "positions": "close_pretrade_after_costs",
            "position_return_weight": "previous_decision_weight_for_return_attribution",
            "cost_unit": "currency",
            "futures_market_value": "signed_notional_excluded_from_funded_nav",
            "return_attribution": ATTRIBUTION_SCHEMA,
            "slippage_semantics": "configured_bps_proxy_not_observed_execution_slippage",
            "market_impact_semantics": "configured_nonlinear_model",
        },
    )
    validate_standard_run(out_dir)
    _write_json(out_dir / "decision.json", decision)
    if decision["action"] == "use":
        _write_overlay(out_dir / "portfolio_overlay.yaml", decision)
        _write_json(out_dir / "position_scale.json", _paper_sim_payload(decision))
    return decision


def _paper_sim_payload(decision: dict[str, Any]) -> dict[str, Any]:
    scale = decision["position_scale"]
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError("paper-sim export requires a numeric position_scale")
    return {
        "as_of": decision["as_of"],
        "regime": decision["regime"],
        "position_scale": float(scale),
        "research_only": True,
        "signals": decision["signals"],
    }


def _write_overlay(path: Path, decision: dict[str, Any]) -> None:
    payload = {
        "as_of": decision["as_of"],
        "position_scale": decision["position_scale"],
        "usage": "把 position_scale 抄到已有 quant-portfolio 策略条目上，不修改组合库接口。",
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _metrics(result: StudyResult, decision: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result.summary)
    payload["export_action"] = decision["action"]
    payload["export_position_scale"] = decision["position_scale"]
    payload["research_view_schema"] = "quant-timing.research-view/v1"
    payload["publication"] = decision
    payload["leakage_audit"] = result.leakage
    payload["walk_forward_folds"] = [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in result.folds.to_dict(orient="records")
    ]
    sample = result.realized.dropna(subset=["net_return"])
    payload["measurement_basis"] = {
        "period_start": _day(sample.index[0]) if not sample.empty else None,
        "period_end": _day(sample.index[-1]) if not sample.empty else None,
        "sample": "descriptive_full_sample_not_walk_forward_acceptance",
        "return_basis": "描述性全区间收益；样本外验收以独立测试折为准",
        "acceptance": result.summary["acceptance"],
        "source": result.raw_config.get("source"),
    }
    payload["backtest_stats"] = [
        {
            "portfolio": label,
            "total_return": result.summary[key],
            "ann_return": None,
            "sharpe": None,
            "max_drawdown": None,
        }
        for label, key in (
            ("策略：描述性全区间（非样本外验收指标）", "descriptive_full_sample_net"),
            ("基准：描述性全区间（非样本外验收指标）", "descriptive_full_sample_benchmark"),
        )
    ]
    return payload


def _standard_frames(result: StudyResult, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nav = result.realized["nav"]
    returns_rows = []
    position_rows = []
    order_rows = []
    cost_rows = []
    exposure_rows = []
    previous_nav = nav.shift(1)
    held = result.weights.shift(1)
    trades = result.realized[[f"trade:{s}" for s in result.weights.columns]].copy()
    trades.columns = result.weights.columns
    before = result.weights - trades
    equity_columns = [
        symbol
        for symbol in result.weights.columns
        if symbol not in FUTURES_COLUMNS | {"CASH", "MARGIN"}
    ]
    for stamp, weight_row in before.iloc[1:].iterrows():
        day = _day(stamp)
        nav_value = float(nav.loc[stamp])
        for symbol, weight in weight_row.items():
            weight_value = float(weight)
            market_value = weight_value * nav_value
            quantity = (
                market_value / _valuation_price(prices, stamp, symbol)
                if abs(weight_value) > 1e-12
                else 0.0
            )
            position_rows.append(
                {
                    "date": day,
                    "strategy": STRATEGY,
                    "symbol": symbol,
                    "quantity": quantity,
                    "market_value": market_value,
                    "weight": weight_value,
                    "side": "long"
                    if weight_value > 1e-12
                    else "short"
                    if weight_value < -1e-12
                    else "flat",
                    "return_weight": float(held.loc[stamp, symbol]),
                }
            )
            exposure_rows.append(
                {
                    "date": day,
                    "strategy": STRATEGY,
                    "exposure_type": "allocation",
                    "name": symbol,
                    "value": weight_value,
                }
            )
        exposure_rows.append(
            {
                "date": day,
                "strategy": STRATEGY,
                "exposure_type": "gross",
                "name": "equity",
                "value": float(weight_row[equity_columns].abs().sum()),
            }
        )
        realized = result.realized.loc[stamp]
        if pd.notna(realized["net_return"]):
            returns_rows.append(
                {
                    "date": day,
                    "strategy": STRATEGY,
                    "gross_return": float(realized["gross_return"]),
                    "net_return": float(realized["net_return"]),
                    "nav": nav_value,
                    "benchmark_return": float(realized["benchmark_return"]),
                    "decision_date": _day(realized["decision_date"]),
                    **{
                        key: float(realized[key])
                        for key in (
                            "base_cost",
                            "impact_cost",
                            "matched_gross_return",
                            "matched_net_return",
                            "matched_base_cost",
                            "matched_impact_cost",
                        )
                    },
                }
            )
        start_nav = (
            float(previous_nav.loc[stamp]) if pd.notna(previous_nav.loc[stamp]) else nav_value
        )
        fraction = 0.0 if pd.isna(realized["cost"]) else float(realized["cost"])
        money = fraction * start_nav
        cost_rows.append(
            {
                "date": day,
                "strategy": STRATEGY,
                "symbol": "BOOK",
                "commission": 0.0,
                "slippage": float(realized["base_cost"]) * start_nav,
                "market_impact": float(realized["impact_cost"]) * start_nav,
                "borrow_cost": 0.0,
                "total_cost": money,
            }
        )
    for loc in range(1, len(result.weights)):
        stamp = result.weights.index[loc]
        delta = trades.loc[stamp]
        nav_value = float(nav.loc[stamp])
        for symbol, change in delta.items():
            change_value = float(change)
            if abs(change_value) <= 1e-12:
                continue
            price = _valuation_price(prices, stamp, symbol)
            order_rows.append(
                {
                    "timestamp": _day(stamp),
                    "strategy": STRATEGY,
                    "symbol": symbol,
                    "side": "buy" if change_value > 0 else "sell",
                    "quantity": abs(change_value) * nav_value / price,
                    "target_weight": float(result.weights.iloc[loc][symbol]),
                    "order_type": "market",
                    "status": "target",
                }
            )
    return {
        "returns": pd.DataFrame(returns_rows),
        "positions": pd.DataFrame(position_rows),
        "orders": pd.DataFrame(order_rows),
        "costs": pd.DataFrame(cost_rows),
        "exposures": pd.DataFrame(exposure_rows),
    }


def _valuation_price(prices: pd.DataFrame, stamp: object, symbol: str) -> float:
    # CASH and MARGIN are denominated in account currency, not traded price indices.
    if symbol in {"CASH", "MARGIN"}:
        return 1.0
    if symbol not in prices.columns:
        raise ValueError(f"missing valuation price for {symbol}")
    price = float(prices.loc[stamp, symbol])
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"invalid valuation price for {symbol} at {stamp}")
    return price


def _prices_sha256(prices: pd.DataFrame) -> str:
    payload = prices.to_csv(date_format="%Y-%m-%d").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.insert(0, "date", [_day(value) for value in output.index])
    output.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _day(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
