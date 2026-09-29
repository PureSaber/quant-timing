# Public snapshot

Downloaded 2026-09-29.

Index and bond ETF bars are the latest 2500 daily closes and volumes from
Sina `CN_MarketData.getKLineData`. `futures.csv` is a back-adjusted IF, IC, and IM
series: each day's return uses the dominant quarterly contract's own previous
close, rolled five trading days before the third Friday. IM before listing falls
back to IF inside the study. `bond_etf` is SSE 511010 and is only a cash/bond
total-return proxy. `amounts.csv` is Sina volume, used as the activity unit for
impact and for the industry amount-share benchmark. That benchmark is not the
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
