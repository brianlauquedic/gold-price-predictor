"""No-leakage invariants for feature engineering (CLAUDE.md §5).

Runs offline on a synthetic random-walk price series — no network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold import config
from gold.features.build import _engineer_frame, feature_columns


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )


def test_target_is_exactly_the_future_return():
    df = _synthetic_ohlcv()
    close = df["Close"]
    feat = _engineer_frame(df)

    expected = np.log(close.shift(-config.HORIZON) / close)  # return t → t+HORIZON
    common = feat["target"].dropna().index
    assert np.allclose(feat.loc[common, "target"], expected.loc[common], atol=1e-12)


def test_contemporaneous_return_is_not_future():
    df = _synthetic_ohlcv()
    close = df["Close"]
    feat = _engineer_frame(df)

    # ret_0 at t must be the return INTO t (known at close), never t+1.
    expected = np.log(close / close.shift(1))
    common = feat["ret_0"].dropna().index
    assert np.allclose(feat.loc[common, "ret_0"], expected.loc[common], atol=1e-12)


def test_feature_columns_exclude_target_and_price_level():
    df = _synthetic_ohlcv()
    cols = feature_columns(_engineer_frame(df))
    assert "target" not in cols
    assert "close" not in cols
    assert len(cols) > 5


def test_no_feature_perfectly_predicts_target():
    # A causal feature cannot be perfectly correlated with the future return.
    df = _synthetic_ohlcv()
    feat = _engineer_frame(df).dropna()
    target = feat["target"]
    for col in feature_columns(feat):
        corr = np.corrcoef(feat[col], target)[0, 1]
        assert abs(corr) < 0.99, f"{col} suspiciously correlated with target ({corr:.3f})"
