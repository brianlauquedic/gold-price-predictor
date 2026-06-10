"""Single source of truth for paths, data, target, features, split, and model knobs.

Scripts read from here — they do not hardcode dates, tickers, or paths.
See CLAUDE.md §4 (Config over constants) and §6 (open decisions).
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# --- Data ------------------------------------------------------------------
PRIMARY_TICKER = "GC=F"   # COMEX gold futures (primary series)
CROSS_TICKER = "GLD"      # SPDR Gold Shares ETF (cross-check)
START_DATE = "2010-01-01"

# --- Target (CLAUDE.md §6: price vs log-return — currently log-return) ------
HORIZON = 1               # forecast 1 trading day ahead
TARGET = "log_return"     # "log_return" | "price"

# --- Features --------------------------------------------------------------
LAGS = [1, 2, 3, 5, 10, 21]
MA_WINDOWS = [5, 10, 21, 63]
VOL_WINDOWS = [10, 21]
RSI_WINDOW = 14

# Macro factors — OFF by default. The price-only baseline must clear the
# random-walk bar (CLAUDE.md §5) before macro earns its leakage surface.
# When enabled, real-rate (10y TIPS, FRED) is a documented hook; DXY + 10y
# nominal come from yfinance. See features/build.py:add_macro_features.
USE_MACRO = False

# --- Split (walk-forward test window) --------------------------------------
TEST_START = "2023-01-01"
RETRAIN_EVERY = 21        # trading days between walk-forward retrains

# --- Model -----------------------------------------------------------------
SEED = 42
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
}

MODEL_PATH = MODELS_DIR / "xgb_gold.json"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"
MACRO_PATH = DATA_RAW / "macro.parquet"   # date-indexed exogenous series (SLV/USO/EURUSD/SPX…)
