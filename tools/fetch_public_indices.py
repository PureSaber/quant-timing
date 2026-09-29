"""Download the public market snapshot used by the comparison study.

Most closes and volumes are the latest 2500 daily bars from Sina. CSI 300 value and
growth closes, and the published peg field used as the valuation spread, come from
the CSI index performance feed. Futures use cached individual quarterly contracts. The treasury
ETF close is the cash/bond return. Twenty-day average volume is the industry crowding
proxy. Earnings revisions are not in these feeds and are not invented.
"""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quant_timing.futures import (  # noqa: E402
    dominant_schedule,
    price_index_from_returns,
    quarter_contracts,
    stitch_returns,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public"
BARS = 2500
OFFLINE = False
INDICES = {
    "hs300": "sh000300",
    "csi500": "sh000905",
    "csi1000": "sh000852",
    "cni_value": "sz399371",
    "cni_growth": "sz399370",
    "dividend": "sh000015",
    "cyclical": "sh000928",
    "bank": "sz399986",
    "broker": "sz399975",
    "pharma": "sh000991",
    "electronics": "sz399811",
    "bond_etf": "sh511010",
}


def _get(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"},
            )
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry a flaky public endpoint
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(url) from last_error


def sina_daily(symbol: str) -> pd.DataFrame:
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={BARS}"
    )
    rows = json.loads(_get(url))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no rows for {symbol}")
    frame["date"] = pd.to_datetime(frame["day"])
    frame["close"] = pd.to_numeric(frame["close"])
    frame["volume"] = pd.to_numeric(frame["volume"])
    print(symbol, len(frame), frame["date"].min().date(), frame["date"].max().date())
    return frame.set_index("date")[["close", "volume"]].sort_index()


def futures_if() -> pd.Series:
    url = (
        "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
        "var%20_IF0=/InnerFuturesNewService.getDailyKLine?symbol=IF0"
    )
    text = _get(url).decode("utf-8", errors="replace")
    rows = json.loads(text[text.find("[") : text.rfind("]") + 1])
    series = pd.Series({row["d"]: float(row["c"]) for row in rows}, dtype=float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def csi_perf(code: str) -> pd.DataFrame:
    url = (
        "https://www.csindex.com.cn/csindex-home/perf/index-perf"
        f"?indexCode={code}&startDate=20160101&endDate=20260929"
    )
    payload = json.loads(_get(url))
    rows = payload.get("data") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no CSI rows for {code}")
    frame["date"] = pd.to_datetime(frame["tradeDate"])
    frame["close"] = pd.to_numeric(frame["close"])
    frame["peg"] = pd.to_numeric(frame["peg"])
    frame = frame.set_index("date")[["close", "peg"]].sort_index()
    print("csi", code, len(frame), frame.index.min().date(), frame.index.max().date())
    return frame


def sina_contract(symbol: str) -> pd.Series:
    cached = OUT / "contracts" / f"{symbol}.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["date"]).set_index("date")["close"]
    if OFFLINE:
        return pd.Series(dtype=float)
    url = (
        "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
        f"var%20_{symbol}=/InnerFuturesNewService.getDailyKLine?symbol={symbol}"
    )
    text = _get(url).decode("utf-8", errors="replace")
    body = text[text.find("[") : text.rfind("]") + 1]
    rows = json.loads(body) if body.startswith("[") else []
    if not rows:
        return pd.Series(dtype=float)
    series = pd.Series({row["d"]: float(row["c"]) for row in rows}, dtype=float)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    cached.parent.mkdir(parents=True, exist_ok=True)
    series.rename_axis("date").rename("close").to_csv(cached)
    return series


def _stitched(prefix: str, calendar: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.Series:
    contracts = quarter_contracts(prefix, calendar.min(), calendar.max())
    prices: dict[str, pd.Series] = {}
    for symbol, _expiry in contracts:
        series = sina_contract(symbol)
        if not series.empty:
            prices[symbol] = series
        if not OFFLINE:
            time.sleep(0.12)
    print(prefix, "contracts", len(prices), flush=True)
    schedule = dominant_schedule(calendar, contracts, trading_sessions=sessions)
    return price_index_from_returns(stitch_returns(prices, schedule))


def refresh_futures(calendar: pd.DatetimeIndex) -> None:
    """Rebuild from cached individual contracts and the independently frozen calendar."""
    sessions = pd.DatetimeIndex(
        pd.read_csv(OUT / "trading_sessions.csv", parse_dates=["date"])["date"]
    )
    futures = pd.DataFrame(
        {name: _stitched(name, calendar, sessions) for name in ("IF", "IC", "IM")}
    )
    _write(futures, OUT / "futures.csv")


def _write(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    output.insert(0, "date", output.index.strftime("%Y-%m-%d"))
    output.to_csv(path, index=False, float_format="%.6g")


def main() -> None:
    global OFFLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--futures-only", action="store_true")
    parser.add_argument(
        "--offline", action="store_true", help="Rebuild only from the committed contract snapshot"
    )
    args = parser.parse_args()
    OFFLINE = args.offline
    if OFFLINE and not args.futures_only:
        parser.error("--offline requires --futures-only")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.futures_only:
        dates = pd.DatetimeIndex(pd.read_csv(OUT / "indices.csv", parse_dates=["date"])["date"])
        refresh_futures(dates)
        return
    closes = {}
    volumes = {}
    for name, symbol in INDICES.items():
        frame = sina_daily(symbol)
        closes[name] = frame["close"]
        volumes[name] = frame["volume"]
        time.sleep(0.3)
    prices = pd.DataFrame(closes).sort_index()
    value = csi_perf("000919")
    growth = csi_perf("000918")
    hs300_peg = csi_perf("000300")["peg"]
    prices["csi300_value"] = value["close"]
    prices["csi300_growth"] = growth["close"]
    prices = prices.dropna()
    volume = pd.DataFrame(volumes).reindex(prices.index)
    valuation = (growth["peg"] - value["peg"]).reindex(prices.index)
    value_earnings = (value["close"] / value["peg"]).pct_change(60)
    growth_earnings = (growth["close"] / growth["peg"]).pct_change(60)
    signals = pd.DataFrame(
        {
            "value_growth": valuation,
            "revision": (value_earnings - growth_earnings).reindex(prices.index),
            "hs300_peg": hs300_peg.reindex(prices.index),
        },
        index=prices.index,
    )
    for name in ("bank", "broker", "pharma", "electronics"):
        signals[f"industry.{name}"] = volume[name].rolling(20).mean()
    signals = signals.dropna(subset=["value_growth", "hs300_peg"])
    common = prices.index.intersection(signals.index)
    prices = prices.loc[common]
    signals = signals.loc[common]
    volume = volume.reindex(common)
    _write(prices, OUT / "indices.csv")
    _write(signals, OUT / "signals.csv")
    _write(volume, OUT / "amounts.csv")
    refresh_futures(common)
    note = OUT / "SOURCE.md"
    note.write_text(
        "\n".join(
            [
                "# Public snapshot",
                "",
                "Downloaded 2026-09-29.",
                "",
                "Index and bond ETF bars are the latest 2500 daily closes and volumes from",
                "Sina `CN_MarketData.getKLineData`. `futures.csv` is a back-adjusted IF, IC, and IM",
                "series: each day's return uses the dominant quarterly contract's own previous",
                "close, rolled five trading days before the third Friday. IM before listing falls",
                "back to IF inside the study. `bond_etf` is SSE 511010 and is only a cash/bond",
                "total-return proxy. `amounts.csv` is Sina volume, used only for the industry",
                "volume-share benchmark. No monetary capacity is inferred. That benchmark is not the",
                "official historical industry weight of the CSI 300.",
                "",
                "`signals.csv` column `value_growth` is CSI 300 Growth published `peg` minus",
                "CSI 300 Value `peg` (000918 minus 000919). A positive number means value is",
                "cheaper on that published field. Those closes also come from the CSI feed",
                "because Sina's copies stop updating. Industry columns are 20-day average",
                "volumes, a crowding proxy. `revision` is the 60-day change in close divided by",
                "the published peg for CSI 300 value minus the same change for CSI 300 growth.",
                "It is an implied-earnings proxy, not an analyst revision feed.",
                "Dividend is SSE 000015 and the cyclical sleeve is CSI Energy 000928.",
                "The Eastmoney mutual-deal report publishes gross turnover and a text quota",
                "flag, not a signed northbound flow, so it is not used as a macro rule.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("prices", len(prices), prices.index.min().date(), prices.index.max().date())


if __name__ == "__main__":
    main()
