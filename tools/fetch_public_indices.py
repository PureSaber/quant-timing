"""Download the public market snapshot used by the comparison study.

Most closes and volumes are the latest 2500 daily bars from Sina. CSI 300 value and
growth closes, and the published peg field used as the valuation spread, come from
the CSI index performance feed. IF0 is Sina's continuous main contract. The treasury
ETF close is the cash/bond return. Twenty-day average volume is the industry crowding
proxy. Earnings revisions are not in these feeds and are not invented.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public"
BARS = 2500
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    closes = {}
    volumes = {}
    for name, symbol in INDICES.items():
        frame = sina_daily(symbol)
        closes[name] = frame["close"]
        volumes[name] = frame["volume"]
        time.sleep(0.3)
    prices = pd.DataFrame(closes).sort_index()
    prices["IF"] = futures_if()
    value = csi_perf("000919")
    growth = csi_perf("000918")
    prices["csi300_value"] = value["close"]
    prices["csi300_growth"] = growth["close"]
    prices = prices.dropna()
    volume = pd.DataFrame(volumes).reindex(prices.index)
    crowding = volume.rolling(20).mean()
    valuation = (growth["peg"] - value["peg"]).reindex(prices.index)
    signals = pd.DataFrame({"value_growth": valuation}, index=prices.index)
    for name in ("bank", "broker", "pharma", "electronics"):
        signals[f"industry.{name}"] = crowding[name]
    signals = signals.dropna()
    common = prices.index.intersection(signals.index)
    prices = prices.loc[common]
    signals = signals.loc[common]
    prices_out = prices.copy()
    prices_out.insert(0, "date", prices_out.index.strftime("%Y-%m-%d"))
    signals_out = signals.copy()
    signals_out.insert(0, "date", signals_out.index.strftime("%Y-%m-%d"))
    prices_out.to_csv(OUT / "indices.csv", index=False, float_format="%.6g")
    signals_out.to_csv(OUT / "signals.csv", index=False, float_format="%.6g")
    note = OUT / "SOURCE.md"
    note.write_text(
        "\n".join(
            [
                "# Public snapshot",
                "",
                "Downloaded 2026-09-29.",
                "",
                "Index and bond ETF bars are the latest 2500 daily closes and volumes from",
                "Sina `CN_MarketData.getKLineData`. `IF` is Sina `IF0`, the continuous main",
                "contract, not a custom roll. `bond_etf` is SSE 511010 and is used only as a",
                "cash/bond total-return proxy.",
                "",
                "`signals.csv` column `value_growth` is CSI 300 Growth published `peg` minus",
                "CSI 300 Value `peg` (000918 minus 000919). A positive number means value is",
                "cheaper on that published field. Those closes also come from the CSI feed",
                "because Sina's copies stop updating. Industry columns are 20-day average",
                "volumes, a crowding proxy. Earnings-revision history is not in this source.",
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
