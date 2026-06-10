"""Technical indicators + multi-timeframe (Triple Screen) signals.

Implements "周线看方向 · 日线定强弱 · 30分钟找买卖点" — Alexander Elder's Triple
Screen: the weekly tide sets direction, the daily wave sets strength, the
intraday ripple times entry. Plus macro "signal lights" for the variables the
macro investors actually watch (dollar, rates, silver — CLAUDE.md §3). Pure
pandas/numpy; educational, NOT trading advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, low, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - low, (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig  # macd, signal, histogram


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV into weekly bars (week ending Friday)."""
    agg = {
        "Open": daily["Open"].resample("W-FRI").first(),
        "High": daily["High"].resample("W-FRI").max(),
        "Low": daily["Low"].resample("W-FRI").min(),
        "Close": daily["Close"].resample("W-FRI").last(),
        "Volume": daily["Volume"].resample("W-FRI").sum(),
    }
    return pd.DataFrame(agg).dropna()


# --- Triple Screen ----------------------------------------------------------
def weekly_trend(weekly: pd.DataFrame) -> str:
    """Screen 1 (the tide): direction from the weekly MACD-histogram slope."""
    _, _, hist = macd(weekly["Close"])
    if len(hist.dropna()) < 2:
        return "flat"
    last, prev = hist.iloc[-1], hist.iloc[-2]
    return "up" if last > prev else "down" if last < prev else "flat"


def daily_strength(daily: pd.DataFrame):
    """Screen 2 (the wave): strength from RSI + moving-average alignment."""
    r = float(rsi(daily["Close"]).iloc[-1])
    price = daily["Close"].iloc[-1]
    ma20 = sma(daily["Close"], 20).iloc[-1]
    ma50 = sma(daily["Close"], 50).iloc[-1]
    score = int(price > ma20) + int(ma20 > ma50) + int(r > 50)
    if score >= 2 and r < 70:
        label = "strong"
    elif score <= 1:
        label = "weak"
    else:
        label = "neutral"
    return label, r


def entry_levels(intraday: pd.DataFrame, lookback: int = 40) -> dict:
    """Screen 3 (the ripple): recent swing high/low as resistance / support."""
    recent = intraday.tail(lookback)
    return {"resistance": float(recent["High"].max()), "support": float(recent["Low"].min())}


def triple_screen(weekly: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame) -> dict:
    """Combine the three screens into one read. Elder: only enter with the tide."""
    trend = weekly_trend(weekly)
    strength, r = daily_strength(daily)
    levels = entry_levels(intraday)
    bias = {"up": "long", "down": "short", "flat": "none"}[trend]
    return {"trend": trend, "strength": strength, "rsi": r, "bias": bias, **levels}


# --- Macro signal lights (the variables macro investors watch) --------------
def _slope(df: pd.DataFrame, n: int) -> float:
    c = df["Close"].dropna()
    return float(c.iloc[-1] - c.iloc[-min(n, len(c) - 1)])


def macro_signals(gold: pd.DataFrame, dxy: pd.DataFrame, tnx: pd.DataFrame,
                  silver: pd.DataFrame, n: int = 20) -> dict:
    """Bullish/bearish-for-gold reads over the last n sessions.

    Dollar down → gold up; yields down → gold up (real-rate proxy); silver
    moving with gold confirms the metals bid (Dalio / Rogers variables).
    """
    out = {}
    out["dxy"] = "bull" if _slope(dxy, n) < 0 else "bear"
    out["rates"] = "bull" if _slope(tnx, n) < 0 else "bear"
    gold_up, silver_up = _slope(gold, n) > 0, _slope(silver, n) > 0
    out["silver"] = "bull" if gold_up == silver_up else "warn"
    return out
