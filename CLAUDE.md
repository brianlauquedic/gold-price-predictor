# CLAUDE.md — Gold Price Predictor

Onboarding for Claude Code sessions on this repo. Read once at session
start. Pointers, not encyclopedia.

---

## 0 · Status (read this first)

**Scaffold landed 2026-06-10.** The full `src/gold/` package, `pyproject.toml`
(uv, Python 3.12), CLI, and the offline no-leakage test all exist. The
pipeline `fetch → features → train → backtest → predict` is wired and runs.
Paths still marked `(planned)` in §2 have **not** been created — do not
assume a file exists because it is documented here.

**Product as of latest (2026-06-10):** the proxy is fixed (§3), so `gold fetch`
now pulls **real, current GC=F daily to today** (~$4,100/oz). The primary
deliverable is the **tri-lingual multi-timeframe candlestick dashboard**
(`streamlit_app.py`): weekly/daily/30-min candles + Elder Triple Screen
(周线方向 · 日线强弱 · 30分买卖点) + macro signal lights + live spot. The XGBoost
next-day forecast is now a secondary expander (it doesn't beat the random
walk — see below — so the chart-based method leads).

First real milestone (in progress): the price-only model beating a
random-walk baseline on out-of-sample RMSE in `reports/backtest.json`.
Until a run records `beats_baseline: true`, treat every accuracy number as
TBD — and note that *not* beating the baseline is a legitimate, reportable
result (CLAUDE.md §5.3), not a bug to paper over.

**First real-data run (2026-06-10)** — real GLD daily 2008–2018, price-only,
walk-forward OOS 2016–2018: RMSE skill **−2.0%**, dir-acc **49%** →
**does NOT beat the random walk**, as expected (daily gold ≈ martingale).
This is the bar the next levers (§6: macro factors, longer horizon) must
clear. Live `fetch` is blocked on this box (§3 network gotcha); the run used
a cached GLD set at `data/raw/GLD.parquet` pulled from GitHub via `gh api`.

**Decisions locked so far** (details in §6): target = next-day **log return**;
macro factors **off** (`config.USE_MACRO=False`) until the price-only baseline
clears the bar; package manager = **uv**, Python **3.12**.

---

## 1 · What this is

A **daily gold-price forecaster**. It pulls historical gold prices,
engineers time-series features, trains a gradient-boosted model
(XGBoost), and emits a next-day forecast **plus a walk-forward backtest
of its own accuracy** — so the error bars ship with the prediction.

- **Target** — next-day gold price (or its log-return; decide in EDA,
  §6). One step ahead, daily frequency.
- **Data** — Yahoo Finance (primary) + Stooq (fallback), pulled via
  `requests` — **not** `yfinance` (its `curl_cffi` TLS stack breaks behind
  MITM proxies; see §5). Primary series `GC=F` (COMEX gold futures); `GLD`
  ETF as a cross-check. Free, no API key. `gold import-csv <file>` is the
  offline escape hatch when network egress is blocked.
- **Model** — XGBoost regression on engineered features: lags, moving
  averages, realized volatility, calendar effects, and optional macro
  factors (DXY, real yields, oil).
- **Scope** — a research-grade forecasting *pipeline*. **Not** trading
  advice, **not** a live execution system. The deliverable is a point
  forecast with an honest, dated backtest — never a buy/sell signal.

**Honesty discipline (load-bearing):** every accuracy claim is reported
against a **random-walk / last-value baseline** on **out-of-sample**
dates. A model that does not beat "tomorrow ≈ today" is a non-result,
and we say so. No in-sample bragging, no cherry-picked windows.

---

## 2 · Where things live (target layout — mostly `planned`)

```
gold-price-predictor/
  CLAUDE.md                  this file
  README.md                  human-facing overview + quickstart
  streamlit_app.py           local analysis dashboard (Streamlit) — price/split, forecast, skill
  pyproject.toml             deps + tooling, managed by uv (Python 3.12)
  .python-version            pins 3.12
  .gitignore                 ignores .venv/ data/ models/ reports/

  src/gold/                  the package
    __init__.py
    config.py                paths, tickers, horizon, hyperparams — ONE source of truth
    i18n.py                  dashboard UI strings — EN / 繁體中文 / 日本語 (hand-authored)
    indicators.py            indicators + Triple Screen (tunable) + macro lights + order_plan
                             (multi-TF confluence_zones → S1/S2 buy, sell target, stop+R:R)
    live.py                  live spot price via gold-api.com (reachable even when Yahoo isn't)
    cli.py                   `gold` entry point: fetch / spot / features / train / predict / backtest
    data/
      fetch.py               Yahoo (requests+truststore, any interval via yahoo_ohlc)→Stooq; import_csv
      load.py                read + validate raw frames (sort, dedupe, drop NaN close)
    features/
      build.py               lags, MA-distance, vol, RSI, multi-TF trend — NO lookahead (§5)
                             `add_macro_features` reads data/raw/macro.parquet (off by default)
    models/
      train.py               XGBoost fit on pre-test window + persist artifact
      predict.py             load model → next-day point forecast
      backtest.py            walk-forward retrain + random-walk baseline → reports/

  data/                      git-ignored (.gitkeep only)
    raw/                     downloaded series (parquet)
    processed/               features.parquet — engineered matrix (generated)
  models/                    git-ignored — xgb_gold.json (generated by `train`)
  reports/                   backtest.json (generated by `backtest`); plots TBD
  notebooks/                 (planned, empty) 01-eda, 02-feature-exploration
  tests/                     pytest — test_features.py pins the no-leakage invariants (offline)
```

Rule of thumb: **`config.py` holds every knob** (tickers, train/test
split dates, horizon, feature windows, hyperparams). Scripts read from
it; they do not hardcode dates or paths.

---

## 3 · Common commands (tooling is `uv`)

```bash
uv sync                                 # create .venv, install pinned deps (Python 3.12)

# Pipeline (each writes to data/ , models/ , or reports/)
uv run gold fetch                       # download raw price history  → data/raw/
uv run gold fetch --cross               # also fetch GLD cross-check
uv run gold spot                        # live current spot via gold-api.com (no key, ~$4,100)
uv run gold import-csv path/to.csv      # OFFLINE: import a local Date+Close CSV → data/raw/
uv run gold features                    # build feature matrix         → data/processed/
uv run gold train                       # fit XGBoost, persist artifact → models/
uv run gold backtest                    # walk-forward eval + baseline → reports/backtest.json
uv run gold predict                     # next-day point forecast      → stdout

# Flags: --ticker (features/predict) and --test-start (train/backtest) override config.
# Reproduce the real GLD 2008–2018 demo from the cached data/raw/GLD.parquet:
uv run gold features --ticker GLD
uv run gold train     --test-start 2016-01-01
uv run gold backtest  --test-start 2016-01-01
uv run gold predict   --ticker GLD

# Dashboard (local, tri-lingual EN/繁中/日本語 — deep-link ?lang=en|zh-Hant|ja & ?unit=)
uv run streamlit run streamlit_app.py   # → http://localhost:8501
#   Multi-timeframe CANDLESTICKS (Elder Triple Screen): weekly (+MACD sub),
#   daily (+volume+RSI subs), 30-min (+volume) on LIVE Yahoo data + macro lights.
#   UNIT switch: USD/oz · 元/克 (CNY/g) · 円/g (JPY/g), defaults by language (zh→元/克);
#     prices = international × live FX (CNY=X/JPY=X) ÷ 31.1035g/oz.
#   Sidebar "indicators/sensitivity" expander: sensitivity preset (standard/fast/smooth
#     = shorter periods, less lag/more noise), RSI overbought, swing lookback, volume toggle.
#   "Order levels": S1/S2 buy + sell target from multi-TF confluence, each with
#     entry/stop/target/R:R + resonance (n_tf/3), shaded on the candles. DIRECTION-AWARE
#     (Elder): in a downtrend buys are flagged 接飞刀/counter-trend; method bias leads.
#   Freshness: 30s/180s TTLs, auto-refresh hero (st.fragment run_every), refresh button,
#     "data as of" stamp. NOTE: MACD/RSI/MA lag by construction — can't be removed, only
#     traded against noise; only candlesticks are lag-free. Live spot ~minute-level (free).

# Quality
uv run pytest                           # no-leakage invariants (offline, 4 tests)
uv run ruff check . && uv run ruff format .
```

**Network (proxy gotcha — usually RESOLVED by routing).** If a local proxy
blocks Yahoo/Stooq/FRED finance hosts (a finance host with no routing rule can
fall to a DIRECT/MATCH fallback and get reset or intercepted → OS-untrusted
cert), route those hosts (`finance.yahoo.com`, `query1/2.finance.yahoo.com`,
`stooq.com`) through a working proxy node — not DIRECT, no TLS interception.
`truststore` already routes verification through the OS trust store, so a
trusted proxy CA just works. After that, `gold fetch` pulls real GC=F daily to
today. Reachable even when finance hosts are blocked: `gold-api.com` (live spot,
`gold spot`) and GitHub. Offline fallback: `gold import-csv`.

---

## 4 · Conventions (set the grain early, do not drift)

- **Time-ordered everything.** Data is a time series. Never shuffle it,
  never random-split it. Train on the past, test on the future.
- **Config over constants.** New ticker, window, or split date → it goes
  in `config.py`, not inline in a script.
- **Reproducibility.** Pin deps (`uv.lock`), set seeds, and cache raw
  downloads to `data/raw/` so a rerun does not depend on a live network
  or a moved market.
- **Artifacts are git-ignored, reports are committed.** `data/`,
  `models/`, `.venv/` stay out of git. A dated metrics summary in
  `reports/` can be committed as a record of what a run produced.
- **Every metric carries a baseline.** Report MAE/RMSE/directional-
  accuracy *next to* the random-walk baseline on the same test dates.
- **Tri-lingual UI, hand-authored.** The dashboard reads `gold/i18n.py`
  (EN / 繁體中文 / 日本語). 繁中 and 日本語 are independently authored, **not**
  machine-translated from EN. Edit one locale's string → update all three;
  keys must stay in parity (a quick check: the three key-sets must be equal).
  Language is deep-linkable via `?lang=en|zh-Hant|ja`.

---

## 5 · Modeling red-lines (the easy ways to fool yourself)

1. **Lookahead / target leakage.** A feature must use only information
   available *before* the timestamp it predicts. Rolling stats,
   normalization, and the train/test split must all be causal. This is
   the #1 way a gold forecaster posts a fake 99%.

2. **Random K-fold on a time series — banned.** Use walk-forward /
   expanding-window CV. Shuffled folds leak the future into the past.

3. **No baseline, no claim.** Before celebrating any model, beat the
   naive "next day = today" (random walk). Daily gold is close to a
   martingale; clearing the baseline is the whole game.

4. **Out-of-sample only.** Headline error is computed on dates the model
   never saw in training (and never used to pick hyperparams). Hold a
   final, untouched test window.

5. **Survivorship / revision traps.** yfinance series can revise or gap.
   Snapshot raw data to `data/raw/` with a fetch date; do not silently
   re-pull and overwrite history mid-experiment.

---

## 6 · Decisions (settled ✓ / open ◻) — record the reason, not just the verdict

- ✓ **Target → next-day log return** (`config.TARGET="log_return"`,
  `HORIZON=1`). Returns are better-behaved for ML; a raw-price target
  invites trend-leakage and a flattering-but-fake fit. Switch is one
  config line; `price` is wired as the alternative.
- ✓ **Macro features → implemented, off by default** (`config.USE_MACRO`,
  `features --macro`). `add_macro_features` now reads `config.MACRO_PATH` (a
  date-indexed parquet of exogenous series) and adds causal per-series features
  (contemporaneous return + lag1 + 5-day mean). **Tested on real GLD 2008–2018
  (2026-06-10)** with SLV/USO/EURUSD/SPX: skill improved **−2.0% → −0.9%**,
  dir-acc 49.0% → 49.7% — measurably helpful, but **still below the random-walk
  baseline at the 1-day horizon**. Conclusion: the bottleneck is horizon, not
  features (slow macro variables bite at weeks/months, not next-day; CLAUDE.md
  §3). Real-rate (10y TIPS, FRED `DFII10`) + DXY (`DX-Y.NYB`) belong in
  MACRO_PATH too once a reachable source lands.
- ✓ **Retrain cadence → walk-forward, expanding window, retrain every
  21 trading days** (`config.RETRAIN_EVERY`). `train` fits the deployable
  model on the full pre-test window; `backtest` is the honest evaluator.
- ✓ **Tooling → uv + Python 3.12.** ML wheels (xgboost/numpy) are most
  reliable there; system Python on this machine is 3.14 (too new for some
  wheels) — do not switch the project to it.
- ◻ **Horizon (the leading next experiment):** the macro run above showed
  features aren't the bottleneck — the 1-day horizon is. Test 5-day / 20-day
  targets (`config.HORIZON`), where the slow macro factors (real rate, dollar,
  silver co-movement) actually move price. This is where skill, if anywhere,
  goes positive.
- ◻ **Primary series:** `GC=F` futures vs `GLD` ETF as the modeled target.
  Both are fetched; pick one as canonical after EDA compares their gaps.

When you settle an open item, flip ◻→✓ here with the reason, and update
the matching `config.py` constant — that pairing is how the next session
inherits the decision.

---

## Tail · who to ask

Solo project · Brian Lau (Hong Kong). Greenfield as of 2026-06-10.
Sibling repo `../quedic-app` (Sakura) is unrelated — do not import its
structure or conventions; this is a Python ML project, not a Solana one.
