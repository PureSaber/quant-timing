# Public snapshot

Downloaded 2026-09-29.

Index and bond ETF bars are the latest 2500 daily closes and volumes from
Sina `CN_MarketData.getKLineData`. `futures.csv` is a back-adjusted IF, IC, and IM
series: each day's return uses the dominant quarterly contract's own previous
close, rolled five trading days before the third Friday. IM before listing falls
back to IF inside the study. `bond_etf` is SSE 511010 and is only a cash/bond
total-return proxy. `amounts.csv` is Sina volume, used only for the industry
volume-share benchmark. It is not CNY turnover and cannot support monetary
participation or capacity estimates. That benchmark is not the
official historical industry weight of the CSI 300.

`signals.csv` column `value_growth` is CSI 300 Growth published `peg` minus
CSI 300 Value `peg` (000918 minus 000919). A positive number means value is
cheaper on that published field. Those closes also come from the CSI feed
because Sina's copies stop updating. Industry columns are 20-day average
volumes, a crowding proxy. `revision` is the 60-day change in close divided by
the published peg for CSI 300 value minus the same change for CSI 300 growth.
It is an implied-earnings proxy, not an analyst revision feed.
Dividend is SSE 000015 and the cyclical sleeve is CSI Energy 000928.
The Eastmoney mutual-deal report publishes gross turnover and a text quota
flag, not a signed northbound flow, so it is not used as a macro rule.

## Corrected futures snapshot (2026-09-29)

`contracts/` contains the 82 non-empty individual-contract close series returned
by Sina `InnerFuturesNewService.getDailyKLine?symbol=<contract>` during the refresh.
Missing historical contract responses remain unavailable, not flat prices. The
IF and IC indexes first become available on 2019-03-08; IM on 2022-07-25. These are
source coverage dates, not claims about the exchange's original listing dates.
No internal missing observations occur after these respective start dates.

`trading_sessions.csv` is a separately frozen XSHG session calendar from
exchange-calendars, covering 2015-01-05 through 2026-12-31. It is the exchange-session
proxy for this research snapshot, independent of the price panel's end date.
Nominal third-Friday expiries advance to the next session on holidays. Extending
this snapshot requires extending and checking this calendar before fetching new bars.

Rebuild without network access:

```powershell
python tools/fetch_public_indices.py --futures-only --offline
```

The comparison explicitly starts on 2019-03-08 for all twelve book/model rows,
so cash and futures use the same source-covered window. Earlier index observations
remain in the raw snapshot. Costs use the configured turnover-based impact model;
`capacity` and `median_participation` are unavailable. The previous comparison table
used fabricated flat pre-coverage futures prices and incorrect capacity units and
has been replaced, not carried forward as a valid result.
