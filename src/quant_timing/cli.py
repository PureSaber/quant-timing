from __future__ import annotations

import argparse
from pathlib import Path

from quant_timing.config import load_yaml, resolve_config
from quant_timing.export import exit_code, write_run
from quant_timing.io import (
    load_prices,
    load_scale_history,
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
    args = parser.parse_args(argv)
    return run_command(Path(args.config), Path(args.out))


def run_command(config_path: Path, out_dir: Path) -> int:
    raw = load_yaml(config_path)
    resolved = resolve_config(raw)
    prices = _load_prices(raw, config_path)
    history = None
    snapshot = None
    macro = None
    if resolved["regime"]["history"]:
        history = load_scale_history(resolve_path(resolved["regime"]["history"], config_path))
    if resolved["regime"]["snapshot"]:
        snapshot = read_position_scale(resolve_path(resolved["regime"]["snapshot"], config_path))
    if resolved["macro"]["context"]:
        macro = read_macro_context(resolve_path(resolved["macro"]["context"], config_path))
    result = run_study(
        prices,
        raw,
        regime_history=history,
        regime_snapshot=snapshot,
        macro=macro,
    )
    decision = write_run(result, prices, out_dir)
    code = exit_code(result.summary, decision)
    scale = decision["position_scale"]
    print(
        f"wrote {out_dir} status={result.summary['status']} action={decision['action']} "
        f"scale={scale} folds={result.summary['scored_folds']} exit={code}"
    )
    return code


def _load_prices(raw: dict, config_path: Path):
    input_cfg = raw.get("input")
    if not isinstance(input_cfg, dict) or not isinstance(input_cfg.get("path"), str):
        raise ValueError("input.path is required")
    return load_prices(
        resolve_path(str(input_cfg["path"]), config_path),
        date_col=str(input_cfg.get("date_col", "date")),
        fmt=str(input_cfg.get("format", "wide")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
