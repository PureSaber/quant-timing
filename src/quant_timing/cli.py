from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from quant_timing.comparison import run_horse_race
from quant_timing.config import load_yaml, resolve_config
from quant_timing.export import exit_code, write_run
from quant_timing.inputs import load_market
from quant_timing.preflight import prepare_inputs
from quant_timing.study import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="指数仓位择时与风格择时研究")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="只读检查配置、来源与滚动窗口，不运行研究")
    preflight.add_argument("--config", required=True, help="已有YAML研究配置")
    preflight.add_argument("--overrides", type=Path, help="独立成本系数配置，不修改原研究文件")
    run = sub.add_parser("run", help="跑一条可复现的择时研究")
    run.add_argument("--config", required=True, help="YAML 配置")
    run.add_argument("--out", required=True, help="输出目录")
    run.add_argument("--overrides", type=Path, help="独立成本系数配置，不修改原研究文件")
    compare = sub.add_parser("compare", help="在同一套费用和走步折下对照仓位模型")
    compare.add_argument("--config", required=True)
    compare.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare_command(Path(args.config), Path(args.out))
    try:
        if args.command == "preflight":
            payload = prepare_inputs(Path(args.config), args.overrides).preflight()
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
            return 0
        return run_command(Path(args.config), Path(args.out), args.overrides)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Timing input or research error: {exc}", file=sys.stderr)
        return 2


def run_command(config_path: Path, out_dir: Path, overrides_path: Path | None = None) -> int:
    prepared = prepare_inputs(config_path, overrides_path)
    destination = out_dir.resolve()
    if any(
        path == destination or destination in path.parents for path in prepared.sources.values()
    ):
        raise ValueError("output directory contains research source files")
    result = run_study(prepared.prices, prepared.raw, **prepared.extras)
    prepared.verify()
    decision = write_run(
        result,
        prepared.prices,
        out_dir,
        input_context={
            "evidence_kind": prepared.evidence_kind,
            "input_checks": prepared.describe(),
        },
    )
    code = exit_code(result.summary, decision)
    scale = decision["position_scale"]
    print(
        f"wrote {out_dir} status={result.summary['status']} action={decision['action']} "
        f"scale={scale} folds={result.summary['scored_folds']} exit={code}"
    )
    if code:
        reason = decision["signals"].get("block_reason", result.summary["status"])
        print(f"Timing research publication blocked: {reason}", file=sys.stderr)
    return code


def compare_command(config_path: Path, out_dir: Path) -> int:
    raw = load_yaml(config_path)
    resolved = resolve_config({**raw, "position": _first_model(raw)})
    prices, extras = load_market(raw, resolved, config_path)
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
