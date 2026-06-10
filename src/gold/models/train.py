"""Fit XGBoost on the pre-TEST_START window and persist the artifact.

This trains the deployable model on all data before the test window. Honest
out-of-sample numbers come from ``backtest`` (walk-forward), not from here.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from .. import config
from ..features.build import feature_columns


def train(features_path=None, test_start: str | None = None, save: bool = True):
    features_path = features_path or config.FEATURES_PATH
    test_start = pd.Timestamp(test_start or config.TEST_START)

    df = pd.read_parquet(features_path)
    train_df = df[df.index < test_start]
    if train_df.empty:
        raise RuntimeError("empty training window — check TEST_START vs data range")

    cols = feature_columns(df)
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(train_df[cols], train_df["target"])

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(config.MODEL_PATH))
        print(
            f"trained on {len(train_df)} rows "
            f"({train_df.index.min().date()} → {train_df.index.max().date()})  "
            f"→ {config.MODEL_PATH.relative_to(config.ROOT)}"
        )
    return model
