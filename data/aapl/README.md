# AAPL frozen market data, as-of 2026-04-20

These files are **artifacts, not outputs**: the option surface cannot be regenerated from code,
because `yfinance`'s `Ticker.option_chain` returns whatever expiries are live on the day it is
called. Re-running the fetcher today returns a different surface. The file *is* the data.

| file | what | as-of |
|---|---|---|
| `aapl_option_chain_clean.csv` | 230 quotes, 5 expiries, bid/ask/mid/IV/volume/OI | 2026-04-20 |
| `aapl_stock_history.csv` | 2,840 daily bars, OHLC + Adj Close | 2015-01-02 - 2026-04-20 |

**Provenance.** Both were produced by the `yfinance` fetcher in `aapl.ipynb` (not in this repo;
kept with the author's records). The chain carries a single spot, 272.565002, on every row, and
the latest `lastTradeDate` is 2026-04-20 15:07:57, so it is one snapshot rather than a stitched
panel. 228 of 230 rows have both bid > 0 and ask > 0.

**The return series is reproducible** from `src_real/data/prices_yf.py` on branch `v2_aapl` with
pinned dates; `aapl_stock_history.csv` is committed as the frozen cross-check, so the experiment
runs identically offline and a vendor revision is detected rather than silently absorbed.

Split handling: `Close` is split-adjusted (largest one-day move over the window is 14.26 %, with
no split artifacts). `Close` is the price process; `Adj Close` additionally reinvests dividends,
and the gap between their drifts over this window is 1.075 %/yr.
