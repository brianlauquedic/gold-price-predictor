"""Event-driven backtest of the Triple Screen strategy — win-rate / expectancy.

Tests whether the method has an edge: weekly tide (confirmed, no repaint) + a daily
MACD-histogram zero-cross trigger in the tide's direction; ATR stop and target;
exit on stop / target / tide-flip. Reported against buy-and-hold AND random-timed
entries in the same direction — if it doesn't beat random timing, the daily trigger
adds nothing.

NO look-ahead: signals use data up to day t, entry at the OPEN of t+1, exits checked
on each subsequent daily bar (stop-before-target on a same-bar tie — conservative).
One position at a time. Educational; past results are not predictive, and this is
ONE concrete parameterization of the method, not the method itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def weekly_trend_series(weekly: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    """Per-week trend (up/down/flat) with the same deadband as indicators.weekly_trend."""
    _, _, hist = ind.macd(weekly["Close"], fast, slow, signal)
    chg = hist.diff()
    dead = 0.25 * chg.abs().rolling(slow).mean()
    s = pd.Series("flat", index=weekly.index, dtype=object)
    s[chg > dead] = "up"
    s[chg < -dead] = "down"
    return s


def _tide_daily(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.Series:
    """Confirmed weekly trend mapped to daily dates (shifted 1 week → no repaint)."""
    wt = weekly_trend_series(weekly).shift(1)
    return wt.reindex(daily.index, method="ffill")


def _simulate(daily, atr, i_entry, direction, rr, atr_stop, tide):
    """One trade opened at the OPEN of bar i_entry → (ret, reason, exit_i) or None."""
    opn, high, low, close = daily["Open"], daily["High"], daily["Low"], daily["Close"]
    a = atr.iloc[i_entry - 1]  # ATR known at the signal bar
    if not np.isfinite(a) or a <= 0:
        return None
    e = float(opn.iloc[i_entry])
    if direction == 1:
        stop, tgt = e - atr_stop * a, e + rr * atr_stop * a
    else:
        stop, tgt = e + atr_stop * a, e - rr * atr_stop * a
    for i in range(i_entry, len(daily)):
        hi, lo, cl = high.iloc[i], low.iloc[i], close.iloc[i]
        if direction == 1:
            if lo <= stop:
                return direction * (stop / e - 1), "stop", i
            if hi >= tgt:
                return direction * (tgt / e - 1), "target", i
        else:
            if hi >= stop:
                return direction * (stop / e - 1), "stop", i
            if lo <= tgt:
                return direction * (tgt / e - 1), "target", i
        td = tide.iloc[i]
        if (direction == 1 and td == "down") or (direction == -1 and td == "up"):
            return direction * (cl / e - 1), "tide_flip", i
    return direction * (float(close.iloc[-1]) / e - 1), "open_end", len(daily) - 1


def triple_screen_backtest(daily: pd.DataFrame, weekly: pd.DataFrame, *,
                           rr=1.5, atr_stop=2.0, fast=12, slow=26, signal=9, seed=0) -> dict:
    atr = ind.atr(daily)
    tide = _tide_daily(daily, weekly)
    _, _, dh = ind.macd(daily["Close"], fast, slow, signal)
    long_t = (dh.shift(1) < 0) & (dh > 0) & (tide == "up")
    short_t = (dh.shift(1) > 0) & (dh < 0) & (tide == "down")

    trades, i, n = [], 1, len(daily)
    while i < n - 1:
        direction = 1 if long_t.iloc[i] else (-1 if short_t.iloc[i] else 0)
        if direction == 0:
            i += 1
            continue
        res = _simulate(daily, atr, i + 1, direction, rr, atr_stop, tide)
        if res is None:
            i += 1
            continue
        ret, reason, exit_i = res
        trades.append({"entry_date": str(daily.index[i + 1].date()), "dir": direction,
                       "ret": float(ret), "reason": reason})
        i = exit_i + 1  # no overlapping positions

    return _metrics(trades, daily, tide, atr, rr, atr_stop, seed)


def _metrics(trades, daily, tide, atr, rr, atr_stop, seed) -> dict:
    bh = float(daily["Close"].iloc[-1] / daily["Close"].iloc[0] - 1)
    base = {"n_trades": len(trades), "buy_hold": bh, "rr": rr, "atr_stop": atr_stop,
            "period": f"{daily.index[0].date()} → {daily.index[-1].date()}"}
    if not trades:
        return base
    rets = np.array([t["ret"] for t in trades], float)
    wins, losses = rets[rets > 0], rets[rets <= 0]
    eq = np.cumprod(1 + rets)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())

    # random-entry control: same count basis, same with-the-tide direction, random days
    rng = np.random.default_rng(seed)
    valid = [j for j in range(1, len(daily) - 1)
             if tide.iloc[j] in ("up", "down") and np.isfinite(atr.iloc[j - 1])]
    rand = []
    for _ in range(300):
        j = int(rng.choice(valid))
        d = 1 if tide.iloc[j] == "up" else -1
        r = _simulate(daily, atr, j + 1, d, rr, atr_stop, tide)
        if r:
            rand.append(r[0])
    rand_exp = float(np.mean(rand)) if rand else float("nan")

    base.update({
        "win_rate": float((rets > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(rets.mean()),                 # avg return per trade
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "total_return": float(eq[-1] - 1),                # compounded, full-notional, sequential
        "max_drawdown": dd,
        "random_entry_expectancy": rand_exp,              # same direction, random timing
        "edge_vs_random": float(rets.mean() - rand_exp),  # >0 ⇒ the daily trigger adds value
        "reasons": {k: int(sum(t["reason"] == k for t in trades))
                    for k in ("target", "stop", "tide_flip", "open_end")},
    })
    return base
