"""Honest backtest lab for the Triple Screen strategy — net of costs, out-of-sample.

The method: weekly tide (confirmed, no repaint) + a daily MACD-histogram zero-cross
trigger in the tide's direction; ATR stop and target; exit on stop / target / tide-flip.

Rigour (so a backtest can't fool you):
  - **Transaction costs**: a round-trip cost is subtracted from every trade.
  - **Out-of-sample holdout**: ``backtest_with_holdout`` reports an in-sample head and an
    untouched tail separately; a big in-sample→holdout gap is the fingerprint of overfit.
  - **Baselines**: results shown against buy-and-hold and same-direction RANDOM entries.
  - **Sweep**: ``strategy_sweep`` ranks combos by HOLDOUT (not in-sample) — the best of N
    is still likely luck (multiple testing); the caller must say so.

NO look-ahead: signals use data up to day t, entry at the OPEN of t+1, exits checked on
each later daily bar (stop-before-target on a same-bar tie). One position at a time.
Educational; past results are not predictive.
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


def _simulate(daily, atr, i_entry, direction, rr, atr_stop, tide, cost=0.0):
    """One trade opened at the OPEN of bar i_entry → (ret_net_of_cost, reason, exit_i) or None."""
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
                return direction * (stop / e - 1) - cost, "stop", i
            if hi >= tgt:
                return direction * (tgt / e - 1) - cost, "target", i
        else:
            if hi >= stop:
                return direction * (stop / e - 1) - cost, "stop", i
            if lo <= tgt:
                return direction * (tgt / e - 1) - cost, "target", i
        td = tide.iloc[i]
        if (direction == 1 and td == "down") or (direction == -1 and td == "up"):
            return direction * (cl / e - 1) - cost, "tide_flip", i
    return direction * (float(close.iloc[-1]) / e - 1) - cost, "open_end", len(daily) - 1


def _run_trades(daily, weekly, *, rr, atr_stop, fast, slow, signal, cost, long_only):
    """Non-overlapping trade list for one parameter set (no look-ahead)."""
    atr = ind.atr(daily)
    tide = _tide_daily(daily, weekly)
    _, _, dh = ind.macd(daily["Close"], fast, slow, signal)
    long_t = (dh.shift(1) < 0) & (dh > 0) & (tide == "up")
    short_t = (dh.shift(1) > 0) & (dh < 0) & (tide == "down")
    trades, i, n = [], 1, len(daily)
    while i < n - 1:
        if long_t.iloc[i]:
            direction = 1
        elif short_t.iloc[i] and not long_only:
            direction = -1
        else:
            i += 1
            continue
        res = _simulate(daily, atr, i + 1, direction, rr, atr_stop, tide, cost)
        if res is None:
            i += 1
            continue
        ret, reason, exit_i = res
        trades.append({"entry_date": daily.index[i + 1], "dir": direction,
                       "ret": float(ret), "reason": reason})
        i = exit_i + 1  # no overlapping positions
    return trades, tide, atr


def _perf(rets) -> dict:
    """Core metrics from per-trade returns (already net of cost)."""
    rets = np.asarray(rets, float)
    if len(rets) == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "expectancy": float("nan"),
                "profit_factor": float("nan"), "total_return": 0.0, "max_drawdown": 0.0}
    wins, losses = rets[rets > 0], rets[rets <= 0]
    eq = np.cumprod(1 + rets)
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "expectancy": float(rets.mean()),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "total_return": float(eq[-1] - 1),
            "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min())}


def triple_screen_backtest(daily, weekly, *, rr=1.5, atr_stop=2.0, fast=12, slow=26, signal=9,
                           cost=0.0005, long_only=False, seed=0) -> dict:
    """Full backtest + random-entry control. ``cost`` = round-trip fraction per trade."""
    trades, tide, atr = _run_trades(daily, weekly, rr=rr, atr_stop=atr_stop, fast=fast,
                                    slow=slow, signal=signal, cost=cost, long_only=long_only)
    bh = float(daily["Close"].iloc[-1] / daily["Close"].iloc[0] - 1)
    out = {"period": f"{daily.index[0].date()} → {daily.index[-1].date()}",
           "rr": rr, "atr_stop": atr_stop, "cost": cost, "long_only": long_only,
           "buy_hold": bh, **_perf([t["ret"] for t in trades])}
    if not trades:
        return out
    rng = np.random.default_rng(seed)
    valid = [j for j in range(1, len(daily) - 1)
             if tide.iloc[j] in ("up", "down") and np.isfinite(atr.iloc[j - 1])]
    rand = []
    for _ in range(300):
        j = int(rng.choice(valid))
        d = 1 if tide.iloc[j] == "up" else -1
        r = _simulate(daily, atr, j + 1, d, rr, atr_stop, tide, cost)
        if r:
            rand.append(r[0])
    rand_exp = float(np.mean(rand)) if rand else float("nan")
    out["random_entry_expectancy"] = rand_exp
    out["edge_vs_random"] = float(out["expectancy"] - rand_exp)
    out["reasons"] = {k: int(sum(t["reason"] == k for t in trades))
                      for k in ("target", "stop", "tide_flip", "open_end")}
    return out


def backtest_with_holdout(daily, weekly, *, holdout_frac=0.3, rr=1.5, atr_stop=2.0,
                          fast=12, slow=26, signal=9, cost=0.0005, long_only=False) -> dict:
    """In-sample head vs untouched holdout tail, net of cost. Indicators use the full
    series (correct warmup, no look-ahead); the entry DATE decides the window. A large
    positive ``oos_gap`` (in-sample ≫ holdout) is the fingerprint of overfitting."""
    trades, _, _ = _run_trades(daily, weekly, rr=rr, atr_stop=atr_stop, fast=fast, slow=slow,
                               signal=signal, cost=cost, long_only=long_only)
    split = daily.index[int(len(daily) * (1 - holdout_frac))]
    ins = _perf([t["ret"] for t in trades if t["entry_date"] < split])
    oos = _perf([t["ret"] for t in trades if t["entry_date"] >= split])
    oos_close = daily["Close"][daily.index >= split]
    bh_oos = float(oos_close.iloc[-1] / oos_close.iloc[0] - 1) if len(oos_close) > 1 else float("nan")
    gap = (ins["expectancy"] - oos["expectancy"]) if (ins["n_trades"] and oos["n_trades"]) else float("nan")
    return {"split": str(split.date()), "holdout_frac": holdout_frac, "cost": cost,
            "rr": rr, "atr_stop": atr_stop, "long_only": long_only,
            "in_sample": ins, "holdout": oos, "oos_gap": gap, "holdout_buy_hold": bh_oos}


def strategy_sweep(daily, weekly, *, holdout_frac=0.3, cost=0.0005) -> dict:
    """Grid-search params; rank by HOLDOUT expectancy (not in-sample). The best of N is
    still likely luck — the caller must warn about multiple testing."""
    sens = ((6, 13, 5), (12, 26, 9), (19, 39, 9))
    rows = []
    for rr in (1.5, 2.0, 3.0):
        for atr_stop in (2.0, 3.0):
            for fast, slow, signal in sens:
                for long_only in (False, True):
                    r = backtest_with_holdout(daily, weekly, holdout_frac=holdout_frac, cost=cost,
                                              rr=rr, atr_stop=atr_stop, fast=fast, slow=slow,
                                              signal=signal, long_only=long_only)
                    rows.append({"rr": rr, "atr_stop": atr_stop, "macd": f"{fast}/{slow}/{signal}",
                                 "long_only": long_only, "in_exp": r["in_sample"]["expectancy"],
                                 "oos_exp": r["holdout"]["expectancy"], "oos_gap": r["oos_gap"],
                                 "oos_trades": r["holdout"]["n_trades"],
                                 "oos_total": r["holdout"]["total_return"]})
    usable = [x for x in rows if x["oos_trades"] >= 5 and x["oos_exp"] == x["oos_exp"]]
    usable.sort(key=lambda x: x["oos_exp"], reverse=True)
    return {"n_combos": len(rows), "rows": usable,
            "best_oos_positive": bool(usable and usable[0]["oos_exp"] > 0)}
