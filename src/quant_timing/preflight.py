"""Read-only structural checks; no signal generation or research acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_timing.config import load_yaml, resolve_config
from quant_timing.contract import file_sha256, json_sha256
from quant_timing.inputs import input_paths, load_market
from quant_timing.study import validate_study_inputs
from quant_timing.walkforward import iter_folds


def identify_sources(paths: dict[str, Path]) -> dict[str, dict]:
    identities = {}
    for name, path in paths.items():
        stat = path.stat()
        digest = file_sha256(path)
        final = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise ValueError(f"source changed while being read: {name}")
        identities[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
    return identities


@dataclass(frozen=True)
class ResearchInputs:
    raw: dict
    config: dict
    prices: pd.DataFrame
    extras: dict
    sources: dict[str, Path]
    identities: dict[str, dict]
    evidence_kind: str

    def verify(self) -> None:
        if identify_sources(self.sources) != self.identities:
            raise ValueError("research sources changed during preparation or execution")

    def describe(self) -> dict:
        folds = iter_folds(self.prices.index, **self.config["validation"])
        return {
            "schema_version": "quant-timing.preflight/v1",
            "software_preflight": "not_run",
            "structural_inputs_valid": True,
            "read_only": True,
            "investable": False,
            "evidence_kind": self.evidence_kind,
            "rows": len(self.prices),
            "symbols": len(self.prices.columns),
            "folds": len(folds),
            "observed_window": {
                "start": self.prices.index[0].isoformat(),
                "end": self.prices.index[-1].isoformat(),
            },
            "effective_config": self.raw,
            "effective_config_sha256": json_sha256(self.raw),
            "input_files": self.identities,
            "performance_basis": "descriptive_full_sample_not_walk_forward_acceptance",
            "acceptance": "walk_forward_folds_and_causal_audit",
            "orders": "research_target_not_routed",
            "limitations": [
                "No signals, weights, backtest, leakage audit or position_scale are produced",
                "Folds are structural windows, not evidence of scored or profitable decisions",
                "Dynamic coverage, macro policy and latest publication gates run during research",
                "Source kind is a declaration, not independent market-data certification",
                "Inputs are re-read at execution; preflight does not lock source files",
            ],
        }

    def preflight(self) -> dict:
        folds = iter_folds(self.prices.index, **self.config["validation"])
        if not any(fold["decision_dates"][0] < self.prices.index[-1] for fold in folds):
            raise ValueError("insufficient history for a walk-forward fold with a later return")
        report = self.describe()
        report["software_preflight"] = "pass"
        return report


def prepare_inputs(config_path: Path) -> ResearchInputs:
    config_path = config_path.resolve()
    sources = {"config": config_path}
    identities = identify_sources(sources)
    raw = load_yaml(config_path)
    config = resolve_config(raw)
    source = raw.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be a mapping")
    kind = source.get("evidence_kind", "unspecified")
    if not isinstance(kind, str) or kind not in {
        "synthetic",
        "retrospective",
        "user_provided",
        "unspecified",
    }:
        raise ValueError("source.evidence_kind is unsupported")
    market_paths = input_paths(raw, config, config_path)
    sources.update(market_paths)
    identities.update(identify_sources(market_paths))
    prices, extras = load_market(raw, config, config_path, paths=market_paths)
    validate_study_inputs(prices, config, **extras)
    prepared = ResearchInputs(raw, config, prices, extras, sources, identities, kind)
    prepared.verify()
    return prepared
