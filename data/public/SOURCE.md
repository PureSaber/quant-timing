# Public snapshot

Downloaded 2026-09-29.

Index and bond ETF bars are the latest 2500 daily closes and volumes from
Sina `CN_MarketData.getKLineData`. `IF` is Sina `IF0`, the continuous main
contract, not a custom roll. `bond_etf` is SSE 511010 and is used only as a
cash/bond total-return proxy.

`signals.csv` column `value_growth` is CSI 300 Growth published `peg` minus
CSI 300 Value `peg` (000918 minus 000919). A positive number means value is
cheaper on that published field. Those closes also come from the CSI feed
because Sina's copies stop updating. Industry columns are 20-day average
volumes, a crowding proxy. Earnings-revision history is not in this source.
Dividend is SSE 000015 and the cyclical sleeve is CSI Energy 000928.
The Eastmoney mutual-deal report publishes gross turnover and a text quota
flag, not a signed northbound flow, so it is not used as a macro rule.
