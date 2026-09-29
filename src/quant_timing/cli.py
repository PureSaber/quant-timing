from __future__ import annotations

import argparse
from pathlib import Path

from quant_timing.comparison import run_horse_race
from quant_timing.config import load_yaml, resolve_config
from quant_timing.export import exit_code, write_run
from quant_timing.io import (
    load_macro_history,
    load_prices,
    load_scale_history,
    load_signal_panel,
    read_macro_context,
    read_position_scale,
    resolve_path,
)
from quant_timing.study import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="指数仓位择时与风格择时研究")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="跑一条可复现的择时研究")
    run.add_argument("--config", required=True, help="YAML 配置")
    run.add_argument("--out", required=True, help="输出目录")
    compare = sub.add_parser("compare", help="在同一套费用和走步折下对照仓位模型")
    compare.add_argument("--config", required=True)
    compare.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare_command(Path(args.config), Path(args.out))
    return run_command(Path(args.config), Path(args.out))


def run_command(config_path: Path, out_dir: Path) -> int:
    raw = load_yaml(config_path)
    resolved = resolve_config(raw)
    prices, extras = _load_market(raw, resolved, config_path)
    result = run_study(prices, raw, **extras)
    decision = write_run(result, prices, out_dir)
    code = exit_code(result.summary, decision)
    scale = decision["position_scale"]
    print(
        f"wrote {out_dir} status={result.summary['status']} action={decision['action']} "
        f"scale={scale} folds={result.summary['scored_folds']} exit={code}"
    )
    return code


def compare_command(config_path: Path, out_dir: Path) -> int:
    raw = load_yaml(config_path)
    resolved = resolve_config({**raw, "position": _first_model(raw)})
    prices, extras = _load_market(raw, resolved, config_path)
    table = run_horse_race(prices, raw, **extras)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"wrote {out_dir / 'model_comparison.csv'}")
    if table.empty or not bool(table["leakage_passed"].all()):
        return 1
    if int(table["scored_folds"].min()) < 1:
        return 2
    return 0


def _load_prices(raw: dict, config_path: Path):
    input_cfg = raw.get("input")
    if not isinstance(input_cfg, dict) or not isinstance(input_cfg.get("path"), str):
        raise ValueError("input.path is required")
    prices = load_prices(
        resolve_path(str(input_cfg["path"]), config_path),
        date_col=str(input_cfg.get("date_col", "date")),
        fmt=str(input_cfg.get("format", "wide")),
    )
    if input_cfg.get("start"):
        prices = prices.loc[str(input_cfg["start"]) :]
    if input_cfg.get("end"):
        prices = prices.loc[: str(input_cfg["end"])]
    if prices.empty:
        raise ValueError("input date range contains no prices")
    return prices


def _load_market(raw: dict, resolved: dict, config_path: Path) -> tuple:
    prices = _load_prices(raw, config_path)
    extras: dict = {
        "regime_history": None,
        "regime_snapshot": None,
        "macro": None,
        "signal_panel": None,
        "macro_history": None,
        "cash_yield": None,
        "futures_price": None,
        "futures_prices": None,
        "amounts": None,
        "peg": None,
    }
    if resolved["regime"]["history"]:
        extras["regime_history"] = load_scale_history(
            resolve_path(resolved["regime"]["history"], config_path)
        )
    if resolved["regime"]["snapshot"]:
        extras["regime_snapshot"] = read_position_scale(
            resolve_path(resolved["regime"]["snapshot"], config_path)
        )
    if resolved["macro"]["context"]:
        extras["macro"] = read_macro_context(
            resolve_path(resolved["macro"]["context"], config_path)
        )
    if resolved["macro"].get("history"):
        extras["macro_history"] = load_macro_history(
            resolve_path(resolved["macro"]["history"], config_path)
        )
    signal_path = (raw.get("signals") or {}).get("path")
    if signal_path:
        extras["signal_panel"] = load_signal_panel(resolve_path(str(signal_path), config_path))
    if resolved["valuation"] is not None:
        column = str(resolved["valuation"]["column"])
        panel = extras["signal_panel"]
        if panel is None or column not in panel.columns:
            raise ValueError(f"valuation column {column} is missing from the signal panel")
        extras["peg"] = panel[column]
    activity_path = (raw.get("activity") or {}).get("path")
    if activity_path:
        extras["amounts"] = load_signal_panel(resolve_path(str(activity_path), config_path))
    if resolved["cash"]["mode"] == "yield":
        column = str(resolved["cash"]["yield_column"])
        if column not in prices.columns:
            raise ValueError(f"cash yield column {column} is missing from prices")
        extras["cash_yield"] = prices[column]
        prices = prices.drop(columns=[column])
    elif resolved["cash"]["mode"] == "price":
        column = str(resolved["cash"]["yield_column"])
        if column not in prices.columns:
            raise ValueError(f"cash price column {column} is missing from prices")
        extras["cash_yield"] = prices[column].astype(float).ffill().pct_change()
        prices = prices.drop(columns=[column])
    if resolved["overlay"]["mode"] == "futures":
        futures_path = (raw.get("futures") or {}).get("path")
        if futures_path:
            extras["futures_prices"] = load_signal_panel(
                resolve_path(str(futures_path), config_path)
            )
        else:
            columns = list(resolved["overlay"]["contracts"])
            missing = [column for column in columns if column not in prices.columns]
            if missing:
                raise ValueError(f"futures price columns {missing} are missing from prices")
            extras["futures_prices"] = prices[columns]
            prices = prices.drop(columns=columns)
    return prices, extras


def _first_model(raw: dict) -> dict:
    models = (raw.get("comparison") or {}).get("models") or []
    specs = raw.get("models") or {}
    if models:
        name = models[0]
        return {"model": name, **dict(specs.get(name) or {})}
    if isinstance(raw.get("position"), dict):
        return raw["position"]
    raise ValueError("comparison.models or position is required")


if __name__ == "__main__":
    raise SystemExit(main())
