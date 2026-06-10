# Gold Price Predictor

Daily gold-price forecaster — XGBoost on engineered time-series features,
data from Yahoo Finance. Emits a next-day point forecast **plus a
walk-forward backtest of its own accuracy** against a random-walk baseline.

> Research pipeline, **not** trading advice. A point forecast with honest,
> dated error bars — never a buy/sell signal.

For full design notes, conventions, and modeling red-lines, see
[`CLAUDE.md`](./CLAUDE.md).

## Quickstart

```bash
uv sync                            # create .venv, install deps (Python 3.12)

uv run gold fetch                  # download GC=F history → data/raw/
# …or, if your network blocks Yahoo/Stooq (see note below):
uv run gold import-csv gold.csv    # import a local Date+Close CSV instead

uv run gold features               # engineer features      → data/processed/
uv run gold train                  # fit XGBoost            → models/
uv run gold backtest               # walk-forward vs random walk → reports/backtest.json
uv run gold predict                # next-day point forecast

uv run pytest                      # no-leakage invariants (offline)

uv run streamlit run streamlit_app.py   # local analysis dashboard → localhost:8501
```

> **Network note.** `fetch` pulls from Yahoo (via `requests` + `truststore`,
> not `yfinance`) with a Stooq fallback. Behind a restrictive/MITM proxy that
> intercepts finance hosts, both can be blocked — use `import-csv`, or allow
> `finance.yahoo.com` / `stooq.com` through your proxy. See `CLAUDE.md` §3.

## Pipeline

```
fetch → features → train → predict
                 └→ backtest        (honest out-of-sample eval)
```

- **Data** — `GC=F` (COMEX gold futures) from Yahoo via `requests` (Stooq
  fallback, CSV import offline); `GLD` as cross-check.
- **Target** — next-day log return (configurable in `src/gold/config.py`).
- **Model** — XGBoost on lags, moving-average distances, volatility, RSI, a
  multi-timeframe trend feature, and calendar effects. Macro factors
  (real rate, DXY) are a documented, off-by-default hook.

Every knob lives in [`src/gold/config.py`](./src/gold/config.py).
