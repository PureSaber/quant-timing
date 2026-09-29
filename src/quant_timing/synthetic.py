from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_prices(n: int = 360, seed: int = 7) -> pd.DataFrame:
    """Deterministic sleeve paths with a drawdown and a later volatility spike."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    market_return = rng.normal(0.0003, 0.008, n)
    market_return[180:200] -= 0.012
    market_return[250:280] = rng.normal(0.0, 0.035, 30)
    market = 100.0 * np.cumprod(1.0 + market_return)
    frame = pd.DataFrame(
        {
            "market": market,
            "large": market * np.cumprod(1.0 + rng.normal(0.0, 0.002, n)),
            "small": market * np.cumprod(1.0 + rng.normal(0.00015, 0.004, n)),
            "value": market * np.cumprod(1.0 + rng.normal(0.0, 0.003, n)),
            "growth": market * np.cumprod(1.0 + rng.normal(0.0001, 0.0035, n)),
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


def write_fixture(path: Path, n: int = 360, seed: int = 7) -> None:
    prices = synthetic_prices(n=n, seed=seed)
    output = prices.copy()
    output.insert(0, "date", prices.index.strftime("%Y-%m-%d"))
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, float_format="%.16g")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    write_fixture(root / "tests" / "fixtures" / "sleeves.csv")


if __name__ == "__main__":
    main()
