"""Walk-forward backtest with periodic retrain, vs a random-walk baseline.

Honesty (CLAUDE.md §5): the headline is model error reported NEXT TO the naive
"tomorrow = today" baseline, on out-of-sample dates only. Beating the baseline
is the whole game — daily gold is close to a martingale.

``walk_forward_series`` produces the per-day predicted/actual series (reused by
the dashboard); ``backtest`` aggregates it into metrics + a JSON report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .. import config
from ..features.build import feature_columns


def walk_forward_series(df: pd.DataFrame, test_start, retrain_every: int) -> pd.DataFrame:
    """Per-day out-of-sample predictions with an expanding-window retrain.

    Returns a frame indexed by test date with columns: pred, actual (both in the
    units of config.TARGET), and close (same-day price, for reconstruction).
    """
    cols = feature_columns(df)
    test_idx = df.index[df.index >= pd.Timestamp(test_start)]
    if len(test_idx) == 0:
        raise RuntimeError("no rows in test window — check test_start vs data range")

    preds, actuals, model = [], [], None
    for i, ts in enumerate(test_idx):
        if i % retrain_every == 0:                       # expanding-window retrain
            train_df = df[df.index < ts]
            model = xgb.XGBRegressor(**config.XGB_PARAMS)
            model.fit(train_df[cols], train_df["target"])
        row = df.loc[[ts]]
        preds.append(float(model.predict(row[cols])[0]))
        actuals.append(float(row["target"].iloc[0]))

    return pd.DataFrame(
        {"pred": preds, "actual": actuals, "close": df.loc[test_idx, "close"].to_numpy()},
        index=test_idx,
    )


def returns_from_series(out: pd.DataFrame):
    """Put pred/actual on a returns footing (random-walk baseline = 0 return)."""
    if config.TARGET == "log_return":
        return out["pred"].to_numpy(), out["actual"].to_numpy()
    close = out["close"].to_numpy()
    return out["pred"].to_numpy() / close - 1.0, out["actual"].to_numpy() / close - 1.0


def error_metrics(actual_ret: np.ndarray, pred_ret: np.ndarray) -> dict:
    err = pred_ret - actual_ret
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))}


def directional_acc(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float:
    return float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))


def evaluate(out: pd.DataFrame) -> dict:
    """Metrics dict comparing the model to the random-walk baseline."""
    model_ret, actual_ret = returns_from_series(out)
    baseline_ret = np.zeros_like(actual_ret)
    model_m = error_metrics(actual_ret, model_ret)
    model_m["directional_acc"] = directional_acc(actual_ret, model_ret)
    base_m = error_metrics(actual_ret, baseline_ret)
    skill = 1.0 - model_m["rmse"] / base_m["rmse"] if base_m["rmse"] else float("nan")
    return {
        "test_start": str(out.index.min().date()),
        "test_end": str(out.index.max().date()),
        "n_test": int(len(out)),
        "target": config.TARGET,
        "use_macro": config.USE_MACRO,
        "model": model_m,
        "baseline_random_walk": base_m,
        "rmse_skill_vs_baseline": skill,   # > 0 ⇒ model beats random walk
        "beats_baseline": bool(model_m["rmse"] < base_m["rmse"]),
    }


def backtest(features_path=None, test_start=None, retrain_every=None) -> dict:
    features_path = features_path or config.FEATURES_PATH
    test_start = pd.Timestamp(test_start or config.TEST_START)
    retrain_every = retrain_every or config.RETRAIN_EVERY

    df = pd.read_parquet(features_path)
    out = walk_forward_series(df, test_start, retrain_every)
    report = evaluate(out)
    report["retrain_every"] = retrain_every

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "backtest.json"
    path.write_text(json.dumps(report, indent=2))
    _print_report(report, path)
    return report


def _print_report(r: dict, out_path) -> None:
    print(f"\nwalk-forward backtest  {r['test_start']} → {r['test_end']}  "
          f"({r['n_test']} days, retrain/{r['retrain_every']}d)")
    print(f"  model     RMSE={r['model']['rmse']:.6f}  MAE={r['model']['mae']:.6f}  "
          f"dir-acc={r['model']['directional_acc']:.1%}")
    print(f"  baseline  RMSE={r['baseline_random_walk']['rmse']:.6f}  "
          f"MAE={r['baseline_random_walk']['mae']:.6f}  (random walk)")
    verdict = "BEATS baseline ✓" if r["beats_baseline"] else "does NOT beat baseline ✗"
    print(f"  skill={r['rmse_skill_vs_baseline']:+.2%} RMSE  →  {verdict}")
    print(f"  → {out_path}")
