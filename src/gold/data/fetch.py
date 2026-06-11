"""Download raw price history → data/raw/*.parquet.

Source = Yahoo Finance (primary) with a Stooq fallback, both free + key-less.
A user-supplied CSV can be imported instead (``import_csv``) for offline /
locked-down-network environments.

Two environment gotchas this module is built around (see CLAUDE.md §5):
  - The old ``yfinance`` lib uses ``curl_cffi`` (bundled BoringSSL), which
    fails TLS behind some local MITM proxies. We use plain ``requests`` instead.
  - Behind a MITM proxy, ``certifi`` does not trust the proxy's root CA, so
    verification fails. ``truststore`` routes verification through the OS trust
    store (macOS keychain / Windows store), which *does* trust it.

Snapshots are explicit: a rerun overwrites only when you ask (§5.5), so a
market revision never silently shifts an experiment mid-flight.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from .. import config

try:  # make Python's ssl use the OS trust store (no-op if unavailable)
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - environment dependent
    pass

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Yahoo ticker → Stooq symbol for the fallback source.
_STOOQ = {"GC=F": "gc.f", "GLD": "gld.us", "XAUUSD": "xauusd"}


def _safe_name(ticker: str) -> str:
    return ticker.replace("=", "_").replace("^", "_").replace("/", "_").replace(".", "_")


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[_OHLCV].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def _from_yahoo(ticker: str, start: str | None = None,
                interval: str = "1d", rng: str | None = None) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    if rng:  # intraday / relative window (e.g. range=60d, interval=30m)
        params = {"range": rng, "interval": interval}
    else:    # absolute window from `start` to now
        params = {"period1": int(pd.Timestamp(start).timestamp()),
                  "period2": int(time.time()), "interval": interval}
    r = requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=12)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q["open"], "High": q["high"], "Low": q["low"],
         "Close": q["close"], "Volume": q["volume"]},
        index=pd.to_datetime(res["timestamp"], unit="s"),
    )
    return _finalize(df.dropna(subset=["Close"]))


def yahoo_ohlc(ticker: str, interval: str = "1d", rng: str | None = None,
               start: str | None = None) -> pd.DataFrame:
    """Live OHLC from Yahoo for any interval. Used by the candlestick dashboard.

    Daily/weekly: pass start (default config.START_DATE). Intraday (30m/15m/…):
    pass rng (e.g. "60d") — Yahoo caps intraday history. Weekly is normally
    resampled from daily rather than fetched. Retries transient TLS/network drops.
    """
    last = None
    for attempt in range(3):
        try:
            return _from_yahoo(ticker, start=start or config.START_DATE, interval=interval, rng=rng)
        except Exception as e:  # transient SSL EOF / connection reset through the proxy
            last = e
            time.sleep(1 + attempt)
    raise last


def _from_stooq(ticker: str, start: str) -> pd.DataFrame:
    sym = _STOOQ.get(ticker, ticker.lower())
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
    r.raise_for_status()
    if not r.text.lstrip().lower().startswith("date"):
        raise RuntimeError(f"stooq returned no CSV for {sym!r} (got: {r.text[:60]!r})")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    df = df[df.index >= pd.Timestamp(start)]
    return _finalize(df)


def fetch_frame(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Return a clean OHLCV frame for one ticker (no disk write). Tries each source."""
    start = start or config.START_DATE
    errors = []
    for source in (_from_yahoo, _from_stooq):
        try:
            df = source(ticker, start)
            if not df.empty:
                return df
            errors.append(f"{source.__name__}: empty frame")
        except Exception as e:
            errors.append(f"{source.__name__}: {e}")
    raise RuntimeError(
        f"all sources failed for {ticker!r}:\n  " + "\n  ".join(errors) +
        "\n(behind a restrictive proxy? use `gold import-csv <file>` instead)"
    )


def _write(df: pd.DataFrame, ticker: str) -> Path:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = config.DATA_RAW / f"{_safe_name(ticker)}.parquet"
    df.to_parquet(out)
    print(
        f"saved {len(df):>5} rows  {df.index.min().date()} → {df.index.max().date()}  "
        f"{ticker}  → {out.relative_to(config.ROOT)}"
    )
    return out


def fetch(ticker: str | None = None, start: str | None = None) -> Path:
    ticker = ticker or config.PRIMARY_TICKER
    return _write(fetch_frame(ticker, start), ticker)


def import_csv(path: str, ticker: str | None = None) -> Path:
    """Import a local OHLC(V) CSV → data/raw/. Needs at least Date + Close columns.

    Missing OHLC columns are backfilled from Close; missing Volume → 0. This is
    the escape hatch when network egress to Yahoo/Stooq is blocked.
    """
    ticker = ticker or config.PRIMARY_TICKER
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    date_col = next((cols[k] for k in ("date", "timestamp", "time") if k in cols), None)
    close_col = next((cols[k] for k in ("close", "adj close", "adjclose", "price") if k in cols), None)
    if date_col is None or close_col is None:
        raise ValueError(f"CSV needs a date and a close column; got {list(raw.columns)}")

    df = raw.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)  # tolerate mixed formats
    df = df.set_index("date").sort_index()
    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    for c in ("Open", "High", "Low"):
        src = cols.get(c.lower())
        df[c] = pd.to_numeric(df[src], errors="coerce") if src else df["Close"]
    vol = cols.get("volume")
    df["Volume"] = pd.to_numeric(df[vol], errors="coerce") if vol else 0.0
    df = _finalize(df.dropna(subset=["Close"]))
    return _write(df, ticker)
