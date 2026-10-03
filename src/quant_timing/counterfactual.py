"""Fixed, source-bound position/style interventions on the native accounting path."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_timing.config import resolve_config
from quant_timing.contract import file_sha256, json_sha256, validate_standard_run
from quant_timing.export import write_run
from quant_timing.preflight import prepare_inputs
from quant_timing.study import run_study

SCHEMA = "quant-timing.counterfactual/v1"
LABELS = {
    "base": "原策略",
    "position_neutral": "取消仓位模型降档",
    "style_neutral": "取消风格信号倾斜",
    "joint": "联合干预",
}
SEMANTICS = {
    "effect": "intervention_net_return_minus_base",
    "interaction": "joint_minus_position_minus_style_plus_base",
    "scope": "fixed_input_model_replays_without_new_forward_observations",
    "position": "unit_model_levels_with_original_warmup_and_all_downstream_constraints",
    "style": "tilt_half_with_original_groups_benchmarks_and_deviation_bounds",
    "folds": "native_decision_masks_no_account_reset_no_overlap_aggregation",
    "publication": "disabled_for_every_candidate",
    "costs": "original_model_parameters_not_observed_execution_slippage",
    "new_independent_forward_dates": 0,
}


def intervention_configs(raw: dict) -> dict[str, dict]:
    """Only model levels and signal tilt may change; preserve every other field."""
    resolved = resolve_config(raw)
    variants = {"base": deepcopy(raw), "position_neutral": deepcopy(raw)}
    model = resolved["position"]["model"]
    levels = (
        ("high_vol_scale", "risk_off_scale", "risk_on_scale")
        if model == "rules"
        else ("floor", "cap")
    )
    for key in levels:
        variants["position_neutral"]["position"][key] = 1.0
    if resolved["style"] is not None:
        variants["style_neutral"] = deepcopy(raw)
        variants["style_neutral"]["style"]["tilt"] = 0.5
        variants["joint"] = deepcopy(variants["position_neutral"])
        variants["joint"]["style"]["tilt"] = 0.5
    for name, config in variants.items():
        if name != "base":
            config["run_id"] = f"{raw.get('run_id', 'timing')}--{name}"
    return variants


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"counterfactual document must be an object: {path.name}")
    return payload


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_counterfactual(config_path: Path, out_dir: Path) -> dict:
    prepared = prepare_inputs(config_path)
    prepared.preflight()
    out_dir = out_dir.resolve()
    if any(path == out_dir or out_dir in path.parents for path in prepared.sources.values()):
        raise ValueError("counterfactual output contains research source files")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(
            "counterfactual destination must be empty; prior attempts are retained"
        )
    variants = intervention_configs(prepared.raw)
    protocol = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "semantics": SEMANTICS,
        "inputs": prepared.describe(),
        "variants": variants,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the fixed family before executing any candidate, including the base.
    _write_json(out_dir / "protocol.json", protocol)
    protocol_hash = file_sha256(out_dir / "protocol.json")
    records = {}
    for name, config in variants.items():
        destination = out_dir / "runs" / name
        try:
            prepared.verify()
            result = run_study(prepared.prices, config, **prepared.extras)
            prepared.verify()
            context = {
                "evidence_kind": prepared.evidence_kind,
                "counterfactual": {
                    "protocol_sha256": protocol_hash,
                    "variant": name,
                    "effective_config": config,
                    "input_files": prepared.identities,
                },
            }
            write_run(
                result,
                prepared.prices,
                destination,
                input_context=context,
                allow_position_publication=False,
            )
            if result.summary["status"] != "complete":
                raise ValueError(f"native research incomplete: {result.summary['status']}")
            records[name] = {
                "status": "complete",
                "native_manifest_sha256": file_sha256(destination / "standard/run_manifest.json"),
                "position_history_sha256": file_sha256(destination / "position_history.csv"),
                "run_context_sha256": file_sha256(destination / "run_context.json"),
            }
        except (OSError, ValueError) as exc:
            failure = {"variant": name, "error_type": type(exc).__name__, "message": str(exc)}
            failure_path = out_dir / "failures" / f"{name}.json"
            _write_json(failure_path, failure)
            records[name] = {
                "status": "failed",
                "failure_sha256": file_sha256(failure_path),
                "reason": str(exc),
            }
    source_error = None
    try:
        prepared.verify()
    except (OSError, ValueError) as exc:
        source_error = str(exc)
    if file_sha256(out_dir / "protocol.json") != protocol_hash:
        raise ValueError("counterfactual protocol changed during execution")
    comparison = _compare_runs(out_dir, protocol, protocol_hash, records, source_error)
    receipt = {
        "schema": SCHEMA,
        "protocol_sha256": protocol_hash,
        "candidates": records,
        "source_error": source_error,
        **comparison,
    }
    artifacts = _render_artifacts(receipt)
    receipt["artifacts"] = {name: _text_hash(text) for name, text in artifacts.items()}
    for name, text in artifacts.items():
        (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    _write_json(out_dir / "result.json", receipt)
    return validate_counterfactual(out_dir)


def _safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"counterfactual artifact escaped run: {relative}")
    return path


def _verify_file(root: Path, relative: str, digest: str) -> Path:
    path = _safe_path(root, relative)
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"counterfactual artifact missing or mutated: {relative}")
    return path


def _candidate_metrics(root: Path, protocol: dict, digest: str, name: str, record: dict) -> dict:
    directory = _safe_path(root, f"runs/{name}")
    for filename, field in (
        ("standard/run_manifest.json", "native_manifest_sha256"),
        ("position_history.csv", "position_history_sha256"),
        ("run_context.json", "run_context_sha256"),
    ):
        _verify_file(root, f"runs/{name}/{filename}", record[field])
    manifest = validate_standard_run(directory)
    if datetime.fromisoformat(manifest.created_at) < datetime.fromisoformat(protocol["created_at"]):
        raise ValueError("candidate predates the frozen protocol")
    config = protocol["variants"][name]
    if manifest.config_sha256 != json_sha256(config):
        raise ValueError(f"candidate config differs from frozen protocol: {name}")
    metrics = _read(directory / "standard/metrics.json")
    context = _read(directory / "run_context.json")
    expected_context = {
        "evidence_kind": protocol["inputs"]["evidence_kind"],
        "counterfactual": {
            "protocol_sha256": digest,
            "variant": name,
            "effective_config": config,
            "input_files": protocol["inputs"]["input_files"],
        },
    }
    if context != expected_context or metrics.get("input_context") != expected_context:
        raise ValueError(f"candidate input/config binding differs: {name}")
    if metrics.get("status") != "complete" or not metrics.get("leakage_passed"):
        raise ValueError(f"candidate research not complete: {name}")
    publication = metrics.get("publication", {})
    if (
        publication.get("action") != "blocked"
        or publication.get("position_scale") is not None
        or publication.get("signals", {}).get("publication_policy") != "research_comparison_only"
        or _read(directory / "decision.json") != publication
        or any(
            (directory / path).exists()
            for path in ("position_scale.json", "portfolio_overlay.yaml")
        )
    ):
        raise ValueError(f"candidate must not publish downstream positions: {name}")
    metrics["dataset_snapshots"] = manifest.dataset_snapshots
    returns = pd.read_csv(directory / "standard/returns.csv")
    metrics["return_clock"] = returns[["date", "decision_date", "benchmark_return"]].to_dict(
        orient="records"
    )
    return metrics


def _compare_runs(root: Path, protocol: dict, digest: str, records: dict, source_error) -> dict:
    if list(records) != list(protocol["variants"]):
        raise ValueError("candidate family is missing, reordered or extended")
    all_metrics = {}
    for name, record in records.items():
        if record.get("status") == "complete":
            all_metrics[name] = _candidate_metrics(root, protocol, digest, name, record)
        elif record.get("status") == "failed":
            failure = _read(_verify_file(root, f"failures/{name}.json", record["failure_sha256"]))
            if (
                failure.get("variant") != name
                or not failure.get("message")
                or failure["message"] != record.get("reason")
            ):
                raise ValueError("invalid failure evidence")
        else:
            raise ValueError("unknown candidate status")
    style_applicable = "style_neutral" in records
    common = {
        "style_applicable": style_applicable,
        "evidence_kind": protocol["inputs"]["evidence_kind"],
    }
    if len(all_metrics) != len(records) or source_error is not None:
        return {"status": "incomplete", **common, "periods": []}
    base_metrics = all_metrics["base"]
    base_periods = base_metrics["return_attribution"]["periods"]
    for metrics in all_metrics.values():
        if metrics["dataset_snapshots"] != base_metrics["dataset_snapshots"]:
            raise ValueError("candidate price inputs differ")
        if metrics["return_clock"] != base_metrics["return_clock"]:
            raise ValueError("candidate realized dates or benchmark returns differ")
        fold_keys = ("fold", "train_end", "test_start", "test_end", "n_decisions")
        if [tuple(fold[key] for key in fold_keys) for fold in metrics["walk_forward_folds"]] != [
            tuple(fold[key] for key in fold_keys) for fold in base_metrics["walk_forward_folds"]
        ]:
            raise ValueError("candidate fold decision windows differ")
        periods = metrics["return_attribution"]["periods"]
        if [_period_identity(p) for p in periods] != [_period_identity(p) for p in base_periods]:
            raise ValueError("candidate scored dates or fold coverage differ")
    rows = []
    for index, base in enumerate(base_periods):
        baseline = _finite(base["net_return"])
        base_parts = {item["component"]: item for item in base["components"]}
        effects = {}
        for name, metrics in all_metrics.items():
            candidate = metrics["return_attribution"]["periods"][index]
            parts = {item["component"]: item for item in candidate["components"]}
            if set(parts) != set(base_parts):
                raise ValueError("candidate attribution component sets differ")
            differences = {}
            for component, item in parts.items():
                if item["kind"] != base_parts[component]["kind"]:
                    raise ValueError("candidate attribution component kinds differ")
                differences[component] = _finite(item["strategy"]) - _finite(
                    base_parts[component]["strategy"]
                )
            net = _finite(candidate["net_return"])
            effect = net - baseline
            residual = sum(differences.values()) - effect
            if abs(residual) > 1e-10:
                raise ValueError("intervention component differences do not reconcile")
            effects[name] = {
                "net_return": net,
                "effect": effect,
                "component_effects": differences,
                "reconciliation_residual": residual,
            }
        interaction = None
        if style_applicable:
            interaction = (
                effects["joint"]["effect"]
                - effects["position_neutral"]["effect"]
                - effects["style_neutral"]["effect"]
            )
        rows.append(
            {
                "id": base["id"],
                "n_decisions": base["n_decisions"],
                "first_return_date": base["first_return_date"],
                "last_return_date": base["last_return_date"],
                "effects": effects,
                "interaction": interaction,
            }
        )
    return {"status": "complete", **common, "periods": rows}


def _finite(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("counterfactual metrics must be finite numbers")
    return float(value)


def _period_identity(period: dict) -> tuple:
    return tuple(
        period[key] for key in ("id", "n_decisions", "first_return_date", "last_return_date")
    )


def validate_counterfactual(root: Path) -> dict:
    try:
        return _validate_counterfactual(Path(root))
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed counterfactual evidence") from exc


def _validate_counterfactual(root: Path) -> dict:
    root = Path(root)
    receipt = _read(root / "result.json")
    protocol = _read(_verify_file(root, "protocol.json", receipt["protocol_sha256"]))
    if receipt.get("schema") != SCHEMA or protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported counterfactual schema")
    if protocol.get("semantics") != SEMANTICS:
        raise ValueError("counterfactual semantics changed")
    variants = protocol["variants"]
    if variants != intervention_configs(variants["base"]):
        raise ValueError("counterfactual config changes exceed the fixed intervention family")
    inputs = protocol["inputs"]
    if inputs["effective_config"] != variants["base"] or inputs[
        "effective_config_sha256"
    ] != json_sha256(variants["base"]):
        raise ValueError("base config does not match the frozen input context")
    computed = _compare_runs(
        root, protocol, receipt["protocol_sha256"], receipt["candidates"], receipt["source_error"]
    )
    if any(receipt.get(key) != value for key, value in computed.items()):
        raise ValueError("counterfactual summary differs from native evidence")
    artifacts = _render_artifacts(receipt)
    if receipt.get("artifacts") != {name: _text_hash(text) for name, text in artifacts.items()}:
        raise ValueError("counterfactual report differs from recomputed evidence")
    for name, digest in receipt["artifacts"].items():
        _verify_file(root, name, digest)
    return receipt


def _render_artifacts(receipt: dict) -> dict[str, str]:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "period",
            "first_return_date",
            "last_return_date",
            "n_decisions",
            "variant",
            "net_return",
            "effect",
            "joint_interaction",
        ]
    )
    sections = []
    for period in receipt["periods"]:
        rows = []
        for name, effect in period["effects"].items():
            writer.writerow(
                [
                    period["id"],
                    period["first_return_date"],
                    period["last_return_date"],
                    period["n_decisions"],
                    name,
                    effect["net_return"],
                    effect["effect"],
                    period["interaction"],
                ]
            )
            parts = "；".join(
                f"{html.escape(key)}：{value * 100:+.4f}"
                for key, value in effect["component_effects"].items()
            )
            rows.append(
                f"<tr><td>{LABELS[name]}</td><td>{effect['net_return'] * 100:+.4f}%</td>"
                f"<td>{effect['effect'] * 100:+.4f}</td><td><details><summary>分量差</summary>"
                f"{parts}<p>对账残差：{effect['reconciliation_residual']:.3g}</p></details></td></tr>"
            )
        label = "描述性全区间" if period["id"] == "descriptive" else html.escape(period["id"])
        interaction = (
            "不适用：原配置没有风格模块"
            if period["interaction"] is None
            else f"{period['interaction'] * 100:+.4f}个百分点"
        )
        sections.append(
            f"<details {'open' if period['id'] == 'descriptive' else ''}><summary>{label}</summary>"
            f"<p>{period['first_return_date']}至{period['last_return_date']} · "
            f"{period['n_decisions']}个收益观测</p><div class='table'><table><thead><tr>"
            "<th>候选</th><th>净收益</th><th>相对原策略/百分点</th><th>账本核对/百分点</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p>联合交互：{interaction}</p></details>"
        )
    states = "".join(
        f"<li>{LABELS[name]}："
        f"{'完成' if record['status'] == 'complete' else html.escape(record['reason'])}</li>"
        for name, record in receipt["candidates"].items()
    )
    status = "全部候选完成" if receipt["status"] == "complete" else "候选族未完成，效应结论不可用"
    evidence = {
        "synthetic": "合成输入，仅验证软件",
        "retrospective": "回顾性历史输入，不增加独立样本",
        "user_provided": "用户提供，真实性及历史可得性需另行核验",
        "unspecified": "来源性质未声明",
    }[receipt["evidence_kind"]]
    document = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>择时仓位与风格反事实</title><style>"
        "body{font:16px/1.55 system-ui;background:#f3f6fa;color:#192b41;margin:0;padding:24px;"
        "overflow-wrap:anywhere}"
        "main{max-width:1150px;margin:auto}details{background:white;padding:18px;margin:16px 0;"
        "border-radius:10px}summary{cursor:pointer;font-weight:bold}.table{overflow:auto}"
        "table{border-collapse:collapse;width:100%;min-width:660px}td,th{padding:12px;"
        "text-align:left;border-bottom:1px solid #ddd;vertical-align:top}"
        "td:first-child,th:first-child{min-width:10em}"
        "td:nth-child(2),td:nth-child(3){white-space:nowrap}"
        "td:last-child{min-width:18em}td details{padding:0;margin:0}"
        ".note{padding:16px;background:#fff5dc;line-height:1.7}a{color:#164bc4}"
        "@media(max-width:600px){body{padding:12px}details{padding:12px}}"
        "</style><main><h1>择时仓位与风格反事实</h1>"
        f"<p>数据性质：{evidence}</p>"
        f"<p><strong>{status}</strong></p><ul>{states}</ul>"
        f"<p>{html.escape(receipt.get('source_error') or '')}</p>"
        "<p class='note'>固定干预、同一历史输入与费用。保留原预热和所有后续约束，"
        "取消模型降档不等于最终满仓；取消风格倾斜仍保留组基准和偏离限制。"
        "各路径重新执行，分折不重建账户、不拼接重叠窗口。"
        "效应与交互仅解释既有模拟样本，不选择新策略、不新增前向证据、不发布下游仓位。</p>"
        "<p><a href='periods.csv'>完整分折效应CSV</a> · <a href='protocol.json'>冻结协议</a> · "
        "<a href='result.json'>核验结果</a></p>"
        f"{''.join(sections)}</main></html>\n"
    )
    return {"periods.csv": stream.getvalue(), "report.html": document}
