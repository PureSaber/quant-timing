from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_path(raw: str, config_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    config_dir = config_path.parent
    for base in (config_dir, config_dir.parent):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return (config_dir / path).resolve()


def load_prices(path: Path, *, date_col: str = "date", fmt: str = "wide") -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns.duplicated().any():
        raise ValueError(f"duplicate columns in {path}")
    if date_col not in frame.columns:
        raise ValueError(f"missing date column {date_col!r} in {path}")
    if fmt == "wide":
        prices = frame.copy()
        prices[date_col] = pd.to_datetime(prices[date_col])
        if prices[date_col].duplicated().any():
            raise ValueError(f"duplicate price dates in {path}")
        prices = prices.set_index(date_col).sort_index()
    elif fmt == "long":
        required = {date_col, "sleeve", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"long prices missing {sorted(missing)}")
        frame = frame.copy()
        frame[date_col] = pd.to_datetime(frame[date_col])
        if frame.duplicated([date_col, "sleeve"]).any():
            raise ValueError(f"duplicate sleeve dates in {path}")
        prices = frame.pivot(index=date_col, columns="sleeve", values="close").sort_index()
    else:
        raise ValueError("price format must be wide or long")
    return _validate_prices(prices)


def load_scale_history(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "position_scale" not in frame.columns:
        raise ValueError(f"scale history needs date and position_scale: {path}")
    dates = pd.to_datetime(frame["date"])
    if dates.duplicated().any():
        raise ValueError(f"duplicate scale dates in {path}")
    values = pd.to_numeric(frame["position_scale"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).sort_index()
    series.index.name = "date"
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"position_scale history must be finite: {path}")
    if ((series < 0) | (series > 1)).any():
        raise ValueError(f"position_scale history must be between 0 and 1: {path}")
    return series


def load_signal_panel(path: Path, date_col: str = "date") -> pd.DataFrame:
    frame = pd.read_csv(path)
    if date_col not in frame.columns:
        raise ValueError(f"signal panel missing {date_col}")
    frame[date_col] = pd.to_datetime(frame[date_col])
    if frame[date_col].duplicated().any():
        raise ValueError(f"duplicate signal dates in {path}")
    panel = frame.set_index(date_col).sort_index()
    panel.index.name = "date"
    return panel.apply(pd.to_numeric, errors="raise")


def load_macro_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "series", "value", "available_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"macro history missing {sorted(missing)}")
    frame = frame.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    return frame


def load_yield(path: Path, column: str, date_col: str = "date") -> pd.Series:
    frame = pd.read_csv(path)
    if date_col not in frame.columns or column not in frame.columns:
        raise ValueError(f"yield file needs {date_col} and {column}")
    dates = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[column], errors="raise")
    series = pd.Series(values.to_numpy(), index=dates).sort_index()
    series.index.name = "date"
    if series.isna().any() or (series < 0).any():
        raise ValueError(f"yield {column} must be nonnegative")
    return series


def read_position_scale(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "position_scale" not in payload:
        raise ValueError(f"regime snapshot has no position_scale: {path}")
    value = payload["position_scale"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"position_scale must be a number: {path}")
    scale = float(value)
    if not math.isfinite(scale) or scale < 0 or scale > 1:
        raise ValueError(f"position_scale must be between 0 and 1: {path}")
    return scale


def read_macro_context(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"macro context must be a mapping: {path}")
    return payload


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("price panel is empty")
    if "CASH" in prices.columns:
        raise ValueError("CASH is reserved for the residual sleeve")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("price dates must be unique and sorted")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and positive")
    numeric.index.name = "date"
    numeric.columns.name = None
    return numeric
