"""Engineer causal time-series features + the next-day target.

LEAKAGE RED-LINE (CLAUDE.md §5): every feature at row ``t`` uses only data up
to and including ``t``. The target is the FUTURE return (``t → t+HORIZON``),
created by a single forward shift — the only future-looking operation in this
module. Rolling windows are backward-looking (pandas default). No row ever
sees its own future. The test in ``tests/test_features.py`` pins this invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..data.load import load_raw


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_macro_features(out: pd.DataFrame) -> pd.DataFrame:
    """Add CAUSAL exogenous (macro / cross-asset) features. Off unless ``config.USE_MACRO``.

    Reads ``config.MACRO_PATH`` — a date-indexed parquet of exogenous series (e.g.
    SLV silver, USO oil, EURUSD ≈ inverse DXY, SPX). For each series we add its
    contemporaneous log return (``_ret0``, known at the close of ``t`` — NOT
    leakage, both markets close at the same time), one lag, and a 5-day mean.
    Series are aligned to gold's trading calendar and forward-filled.

    This is the concrete form of "borrow the *variables* macro investors watch,
    not their narratives" (CLAUDE.md §3). Each column widens the leakage surface,
    so it only earns its place if it beats the price-only baseline (§5.3, §6).
    The canonical real-rate (10y TIPS, FRED ``DFII10``) + DXY (``DX-Y.NYB``) belong
    in MACRO_PATH too, once a reachable data source for them lands.
    """
    if not config.USE_MACRO:
        return out
    if not config.MACRO_PATH.exists():
        raise FileNotFoundError(
            f"{config.MACRO_PATH} missing — USE_MACRO is on but no macro series file. "
            "Provide a date-indexed parquet of exogenous series (CLAUDE.md §6)."
        )
    m = pd.read_parquet(config.MACRO_PATH).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    m = m.reindex(out.index.union(m.index)).sort_index().ffill().reindex(out.index)
    for col in m.columns:
        r = np.log(m[col].astype(float)).diff()
        out[f"macro_{col}_ret0"] = r
        out[f"macro_{col}_ret_lag1"] = r.shift(1)
        out[f"macro_{col}_ret_mean5"] = r.rolling(5).mean()
    return out


def _engineer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pure feature engineering on an OHLCV frame (no I/O). Includes target column.

    Returns the full frame WITH warmup/tail NaNs intact — callers decide what to drop.
    """
    close = df["Close"].astype(float)
    lr = np.log(close).diff()  # today's log return — known at close of t

    out = pd.DataFrame(index=df.index)
    out["close"] = close                 # kept for price reconstruction, NOT a feature
    out["ret_0"] = lr                    # contemporaneous return (legitimately known at t)

    for lag in config.LAGS:
        out[f"ret_lag_{lag}"] = lr.shift(lag)

    for w in config.MA_WINDOWS:
        out[f"close_to_ma_{w}"] = close / close.rolling(w).mean() - 1.0
        out[f"ret_mean_{w}"] = lr.rolling(w).mean()

    for w in config.VOL_WINDOWS:
        out[f"vol_{w}"] = lr.rolling(w).std()

    out[f"rsi_{config.RSI_WINDOW}"] = _rsi(close, config.RSI_WINDOW)

    # Multi-timeframe trend, Triple-Screen-inspired (CLAUDE.md §3): a longer-horizon
    # direction feature — short MA relative to a longer MA.
    out["weekly_trend"] = close.rolling(5).mean() / close.rolling(25).mean() - 1.0

    # Calendar
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month

    out = add_macro_features(out)

    # --- Target: the ONLY forward-looking operation -------------------------
    if config.TARGET == "log_return":
        out["target"] = lr.shift(-config.HORIZON)
    elif config.TARGET == "price":
        out["target"] = close.shift(-config.HORIZON)
    else:
        raise ValueError(f"unknown TARGET {config.TARGET!r} (expected 'log_return' or 'price')")

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs = everything except the target and the raw price level."""
    return [c for c in df.columns if c not in ("target", "close")]


def build_features(ticker: str | None = None, save: bool = True, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full training matrix: engineer, drop warmup + unlabeled tail. Pass ``df`` to build
    from an in-memory OHLCV frame (e.g. live-fetched, hosted with no parquet); else load
    from data/raw/. Only persists when reading from disk."""
    out = _engineer_frame(df if df is not None else load_raw(ticker)).dropna()
    if save and df is None:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(config.FEATURES_PATH)
    return out


def latest_feature_row(ticker: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The most recent fully-featured row (its target is unknown) — for live predict."""
    out = _engineer_frame(df if df is not None else load_raw(ticker))
    out = out.dropna(subset=feature_columns(out))
    return out.iloc[[-1]]
