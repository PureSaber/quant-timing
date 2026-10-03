"""Resolve and load the same local sources for research and read-only preflight."""

from __future__ import annotations

from pathlib import Path

from quant_timing.io import (
    load_macro_history,
    load_prices,
    load_scale_history,
    load_signal_panel,
    read_macro_context,
    read_position_scale,
    resolve_path,
)


def input_paths(raw: dict, resolved: dict, config_path: Path) -> dict[str, Path]:
    sources = {}
    for section, key, required in (
        ("input", "path", True),
        ("signals", "path", False),
        ("activity", "path", False),
        ("regime", "history", False),
        ("regime", "snapshot", False),
        ("macro", "context", False),
        ("macro", "history", False),
        ("futures", "path", False),
    ):
        if section == "futures" and resolved["overlay"]["mode"] != "futures":
            continue
        mapping = raw.get(section, {})
        if mapping is None and not required:
            continue
        if not isinstance(mapping, dict):
            raise ValueError(f"{section} must be a mapping")
        value = mapping.get(key)
        if value is None and not required:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{key} must be a non-empty path")
        sources[f"{section}.{key}"] = resolve_path(value, config_path)
    return sources


def load_market(
    raw: dict, resolved: dict, config_path: Path, *, paths: dict[str, Path] | None = None
) -> tuple:
    paths = input_paths(raw, resolved, config_path) if paths is None else paths
    input_cfg = raw["input"]
    prices = load_prices(
        paths["input.path"],
        date_col=str(input_cfg.get("date_col", "date")),
        fmt=str(input_cfg.get("format", "wide")),
    )
    if input_cfg.get("start"):
        prices = prices.loc[str(input_cfg["start"]) :]
    if input_cfg.get("end"):
        prices = prices.loc[: str(input_cfg["end"])]
    if prices.empty:
        raise ValueError("input date range contains no prices")
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
    for source, target, reader in (
        ("regime.history", "regime_history", load_scale_history),
        ("regime.snapshot", "regime_snapshot", read_position_scale),
        ("macro.context", "macro", read_macro_context),
        ("macro.history", "macro_history", load_macro_history),
        ("signals.path", "signal_panel", load_signal_panel),
        ("activity.path", "amounts", load_signal_panel),
    ):
        if source in paths:
            extras[target] = reader(paths[source])
    if resolved["valuation"] is not None:
        column = str(resolved["valuation"]["column"])
        panel = extras["signal_panel"]
        if panel is None or column not in panel.columns:
            raise ValueError(f"valuation column {column} is missing from the signal panel")
        extras["peg"] = panel[column]
    if resolved["cash"]["mode"] in {"yield", "price"}:
        column = str(resolved["cash"]["yield_column"])
        if column not in prices.columns:
            raise ValueError(
                f"cash {resolved['cash']['mode']} column {column} is missing from prices"
            )
        extras["cash_yield"] = prices[column]
        if resolved["cash"]["mode"] == "price":
            extras["cash_yield"] = prices[column].astype(float).ffill().pct_change()
        prices = prices.drop(columns=[column])
    if resolved["overlay"]["mode"] == "futures":
        if "futures.path" in paths:
            extras["futures_prices"] = load_signal_panel(paths["futures.path"])
        else:
            columns = list(resolved["overlay"]["contracts"])
            missing = [column for column in columns if column not in prices.columns]
            if missing:
                raise ValueError(f"futures price columns {missing} are missing from prices")
            extras["futures_prices"] = prices[columns]
            prices = prices.drop(columns=columns)
    return prices, extras
