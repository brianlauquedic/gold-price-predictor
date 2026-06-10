"""Load the trained model → next-day forecast from the latest fully-featured row."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .. import config
from ..features.build import feature_columns, latest_feature_row


def predict_next(ticker: str | None = None) -> dict:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.MODEL_PATH} missing — run `gold train` first"
        )

    row = latest_feature_row(ticker)
    cols = feature_columns(row)

    model = xgb.XGBRegressor()
    model.load_model(str(config.MODEL_PATH))
    pred = float(model.predict(row[cols])[0])

    asof = row.index[-1].date()
    last_close = float(row["close"].iloc[0])

    if config.TARGET == "log_return":
        next_price = last_close * float(np.exp(pred))
        print(
            f"as of {asof}  close={last_close:.2f}  "
            f"pred_return={pred:+.4%}  → next≈{next_price:.2f}"
        )
        result = {"asof": str(asof), "last_close": last_close,
                  "pred_return": pred, "next_price": next_price}
    else:
        print(f"as of {asof}  close={last_close:.2f}  → next≈{pred:.2f}")
        result = {"asof": str(asof), "last_close": last_close, "next_price": pred}

    print("note: a point forecast, not a trade signal (CLAUDE.md §1).")
    return result
