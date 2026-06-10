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


def force_index(df: pd.DataFrame, n: int = 13) -> pd.Series:
    """Elder's Force Index: (price change × volume), EMA-smoothed. Positive =
    buyers in control. Needs real volume (GC=F has it; spot XAU does not)."""
    return (df["Close"].diff() * df["Volume"]).ewm(span=n, adjust=False).mean()


# --- Triple Screen ----------------------------------------------------------
def weekly_trend(weekly: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> str:
    """Screen 1 (the tide): direction from the weekly MACD-histogram slope."""
    _, _, hist = macd(weekly["Close"], fast, slow, signal)
    hist = hist.dropna()
    if len(hist) < 2:
        return "flat"
    last, prev = hist.iloc[-1], hist.iloc[-2]
    return "up" if last > prev else "down" if last < prev else "flat"


def daily_strength(daily: pd.DataFrame, rsi_n: int = 14, rsi_ob: int = 70,
                   ma_fast: int = 20, ma_slow: int = 50, use_force: bool = True):
    """Screen 2 (the wave): RSI + MA alignment + (optional) Force Index (volume)."""
    r = float(rsi(daily["Close"], rsi_n).iloc[-1])
    price = daily["Close"].iloc[-1]
    maf = sma(daily["Close"], ma_fast).iloc[-1]
    mas = sma(daily["Close"], ma_slow).iloc[-1]
    factors = [price > maf, maf > mas, r > 50]
    fi = 0.0
    if use_force and "Volume" in daily and float(daily["Volume"].fillna(0).sum()) > 0:
        fi = float(force_index(daily).iloc[-1])
        factors.append(fi > 0)
    ratio = sum(bool(x) for x in factors) / len(factors)
    label = "strong" if (ratio >= 0.66 and r < rsi_ob) else "weak" if ratio <= 0.34 else "neutral"
    return label, r, fi


def entry_levels(intraday: pd.DataFrame, lookback: int = 40) -> dict:
    """Screen 3 (the ripple): recent swing high/low as resistance / support."""
    recent = intraday.tail(lookback)
    return {"resistance": float(recent["High"].max()), "support": float(recent["Low"].min())}


def triple_screen(weekly: pd.DataFrame, daily: pd.DataFrame, intraday: pd.DataFrame, *,
                  macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                  rsi_n: int = 14, rsi_ob: int = 70, ma_fast: int = 20, ma_slow: int = 50,
                  swing: int = 40, use_force: bool = True) -> dict:
    """Combine the three screens into one read. Elder: only enter with the tide."""
    trend = weekly_trend(weekly, macd_fast, macd_slow, macd_signal)
    strength, r, fi = daily_strength(daily, rsi_n, rsi_ob, ma_fast, ma_slow, use_force)
    levels = entry_levels(intraday, swing)
    bias = {"up": "long", "down": "short", "flat": "none"}[trend]
    return {"trend": trend, "strength": strength, "rsi": r, "force": fi, "bias": bias, **levels}


# --- Order levels (multi-timeframe confluence) ------------------------------
def swing_pivots(df: pd.DataFrame, k: int = 3, recent: int = 12):
    """Recent swing highs & lows (a bar that is the local extreme over ±k bars)."""
    h, low = df["High"].to_numpy(), df["Low"].to_numpy()
    highs, lows = [], []
    for i in range(k, len(df) - k):
        if h[i] == h[i - k:i + k + 1].max():
            highs.append(float(h[i]))
        if low[i] == low[i - k:i + k + 1].min():
            lows.append(float(low[i]))
    return highs[-recent:], lows[-recent:]


def _level_candidates(weekly, daily, intraday):
    """(price, weight, timeframe) — swing pivots + key MAs per timeframe."""
    cands = []
    for df, wgt, tf, mas in ((weekly, 3, "W", (13, 30)),
                             (daily, 2, "D", (20, 50, 200)),
                             (intraday, 1, "M30", (20, 50))):
        hi, lo = swing_pivots(df)
        cands += [(p, wgt, tf) for p in hi + lo]
        for n in mas:
            v = sma(df["Close"], n).iloc[-1]
            if pd.notna(v):
                cands.append((float(v), wgt, tf))
    return cands


def _zone(members, price):
    w = sum(m[1] for m in members)
    center = sum(m[0] * m[1] for m in members) / w
    return {"center": center, "low": min(m[0] for m in members),
            "high": max(m[0] for m in members), "score": w,
            "n_tf": len({m[2] for m in members}),
            "kind": "support" if center <= price else "resistance"}


def confluence_zones(weekly, daily, intraday, price: float, tol: float):
    """Cluster candidates within `tol` into zones; return (supports, resistances)
    sorted by resonance (distinct timeframes), then score, then proximity to price."""
    cands = sorted(_level_candidates(weekly, daily, intraday), key=lambda c: c[0])
    zones, cur = [], []
    for c in cands:
        if cur and c[0] - cur[-1][0] > tol:
            zones.append(_zone(cur, price))
            cur = []
        cur.append(c)
    if cur:
        zones.append(_zone(cur, price))
    # nearest-first: S1/S2 are the closest actionable supports (resonance shown as quality)
    sups = sorted((z for z in zones if z["kind"] == "support"), key=lambda z: price - z["center"])
    ress = sorted((z for z in zones if z["kind"] == "resistance"), key=lambda z: z["center"] - price)
    return sups, ress


def order_plan(weekly, daily, intraday, *, bias: str, atr_d: float, **_):
    """Direction-aware buy/sell levels with stop + risk:reward.

    S1/S2 = the two highest-resonance support zones; sell target = nearest strong
    resistance; shorts = resistance zones. Buying is flagged counter-trend unless
    the weekly bias is long (Elder: only trade with the tide). Method-derived
    structure, NOT a prediction or advice.
    """
    price = float(daily["Close"].iloc[-1])
    sups, ress = confluence_zones(weekly, daily, intraday, price, 0.4 * atr_d)
    buf = 0.5 * atr_d
    res0, sup0 = (ress[0] if ress else None), (sups[0] if sups else None)

    def buy(z):
        entry, stop = z["center"], z["low"] - buf
        tgt = res0["center"] if res0 else None
        rr = (tgt - entry) / (entry - stop) if tgt and entry > stop else None
        return {**z, "side": "buy", "entry": entry, "stop": stop, "target": tgt, "rr": rr}

    def short(z):
        entry, stop = z["center"], z["high"] + buf
        tgt = sup0["center"] if sup0 else None
        rr = (entry - tgt) / (stop - entry) if tgt and stop > entry else None
        return {**z, "side": "short", "entry": entry, "stop": stop, "target": tgt, "rr": rr}

    # the "真最佳买入区" = the strongest multi-timeframe-confluence support (n_tf ≥ 2),
    # even if it is not the nearest. None when there is no real resonance support.
    res_sup = max((z for z in sups if z["n_tf"] >= 2), key=lambda z: (z["n_tf"], z["score"]), default=None)
    return {"price": price, "bias": bias, "counter_trend_buy": bias != "long",
            "buys": [buy(z) for z in sups[:2]], "shorts": [short(z) for z in ress[:2]],
            "sell_target": res0, "resonant_support": buy(res_sup) if res_sup else None}


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
