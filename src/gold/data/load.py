"""Load + validate raw frames from data/raw/."""

from __future__ import annotations

import pandas as pd

from .. import config
from .fetch import _safe_name


def load_raw(ticker: str | None = None) -> pd.DataFrame:
    ticker = ticker or config.PRIMARY_TICKER
    path = config.DATA_RAW / f"{_safe_name(ticker)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `gold fetch` first")
    df = pd.read_parquet(path).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Close"])
    return df
