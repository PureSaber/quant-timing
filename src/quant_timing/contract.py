"""Versioned research-run artifacts aligned with the standard/v1 column contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "1.0"
ARTIFACT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "returns": (
        "date",
        "strategy",
        "gross_return",
        "net_return",
        "nav",
        "benchmark_return",
    ),
    "positions": (
        "date",
        "strategy",
        "symbol",
        "quantity",
        "market_value",
        "weight",
        "side",
    ),
    "orders": (
        "timestamp",
        "strategy",
        "symbol",
        "side",
        "quantity",
        "target_weight",
        "order_type",
        "status",
    ),
    "costs": (
        "date",
        "strategy",
        "symbol",
        "commission",
        "slippage",
        "market_impact",
        "borrow_cost",
        "total_cost",
    ),
    "exposures": ("date", "strategy", "exposure_type", "name", "value"),
}


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: str
    sha256: str
    rows: int
    columns: list[str]


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    project: str
    run_id: str
    strategy: str
    created_at: str
    status: str
    code_version: str
    config_sha256: str
    dataset_snapshots: dict[str, str]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_standard_run(
    run_dir: Path,
    *,
    project: str,
    run_id: str,
    strategy: str,
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    config: dict[str, Any],
    code_version: str,
    dataset_snapshots: dict[str, str],
    tags: dict[str, str],
) -> RunManifest:
    standard_dir = Path(run_dir) / "standard"
    if standard_dir.exists():
        raise FileExistsError(f"standard run already exists: {standard_dir}")
    standard_dir.mkdir(parents=True, exist_ok=False)
    records: list[ArtifactRecord] = []
    for name, columns in ARTIFACT_SCHEMAS.items():
        frame = _normalize(name, frames.get(name))
        path = standard_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        records.append(
            ArtifactRecord(
                name=name,
                path=path.name,
                sha256=file_sha256(path),
                rows=len(frame),
                columns=list(frame.columns),
            )
        )
        if list(frame.columns)[: len(columns)] != list(columns):
            raise ValueError(f"artifact columns drifted: {name}")
    metrics_path = standard_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    records.append(
        ArtifactRecord(
            name="metrics",
            path=metrics_path.name,
            sha256=file_sha256(metrics_path),
            rows=1,
            columns=sorted(metrics),
        )
    )
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        project=project,
        run_id=run_id,
        strategy=strategy,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="complete",
        code_version=code_version,
        config_sha256=json_sha256(config),
        dataset_snapshots=dataset_snapshots,
        artifacts=records,
        tags=tags,
    )
    (standard_dir / "run_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _normalize(name: str, frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ARTIFACT_SCHEMAS[name]
    result = pd.DataFrame() if frame is None else frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.Series(dtype="object")
    ordered = [*columns, *[column for column in result.columns if column not in columns]]
    return result[ordered]


def validate_standard_run(run_dir: Path) -> RunManifest:
    standard_dir = Path(run_dir) / "standard"
    manifest_path = standard_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"] = [ArtifactRecord(**record) for record in payload.get("artifacts", [])]
    manifest = RunManifest(**payload)
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema {manifest.schema_version}")
    expected = {*ARTIFACT_SCHEMAS, "metrics"}
    actual = {record.name for record in manifest.artifacts}
    if actual != expected or len(manifest.artifacts) != len(expected):
        raise ValueError(f"artifact set mismatch: {actual ^ expected}")
    for record in manifest.artifacts:
        expected_path = "metrics.json" if record.name == "metrics" else f"{record.name}.csv"
        if record.path != expected_path:
            raise ValueError(f"artifact path must be canonical: {record.name}")
        path = standard_dir / record.path
        if not path.resolve().is_relative_to(Path(run_dir).resolve()):
            raise ValueError(f"artifact escaped run: {record.name}")
        if not path.is_file() or file_sha256(path) != record.sha256:
            raise ValueError(f"artifact missing or mutated: {path}")
        if record.name in ARTIFACT_SCHEMAS:
            columns = list(pd.read_csv(path, nrows=0).columns)
            required = set(ARTIFACT_SCHEMAS[record.name])
            if not required.issubset(columns):
                raise ValueError(f"artifact schema invalid: {record.name}")
    metrics = json.loads((standard_dir / "metrics.json").read_text(encoding="utf-8"))
    if "return_attribution" in metrics or "return_attribution" in manifest.tags:
        from quant_timing.attribution import SCHEMA, validate_attribution

        if manifest.tags.get("return_attribution") != SCHEMA or "return_attribution" not in metrics:
            raise ValueError("attribution manifest binding missing")
        validate_attribution(run_dir, metrics)
    return manifest
