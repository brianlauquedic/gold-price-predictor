"""Gold — multi-timeframe candlestick dashboard (EN / 繁體中文 / 日本語).

Run:  uv run streamlit run streamlit_app.py

"周线看方向 · 日线定强弱 · 30分钟找买卖点" — Elder's Triple Screen on LIVE Yahoo
data (current to today): weekly / daily / 30-min candlesticks, a Triple Screen
read, macro signal lights (dollar / rates / silver), and a live spot price.
The optional ML next-day forecast + honest backtest lives in an expander.
Educational — not trading advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from gold import config
from gold import indicators as ind
from gold.data.fetch import yahoo_ohlc
from gold.features.build import build_features, feature_columns, latest_feature_row
from gold.i18n import LANGS, t
from gold.live import fx_rate, spot
from gold.models.backtest import evaluate, walk_forward_series

st.set_page_config(page_title="Gold candlesticks", layout="wide")

GOLD = "GC=F"
MA_C = ["#BA7517", "#378ADD", "#7F77DD"]  # amber / blue / purple
UP, DOWN = "#1D9E75", "#D85A30"
TTL = 600

# Display units: base prices are USD/oz; convert by a single factor per unit.
GRAMS_PER_TROY_OZ = 31.1034768
UNITS = {
    "usd_oz": {"sym": "$", "suffix": "/oz", "dec": 2},
    "cny_g": {"sym": "¥", "suffix": "/g", "dec": 2},   # 元/克
    "jpy_g": {"sym": "¥", "suffix": "/g", "dec": 0},   # 円/g
}
LANG_UNIT = {"en": "usd_oz", "zh-Hant": "cny_g", "ja": "jpy_g"}


# --- live data (cached with a short TTL so it stays current) ----------------
@st.cache_data(ttl=TTL, show_spinner="Loading daily…")
def daily(ticker=GOLD):
    return yahoo_ohlc(ticker, "1d", start="2015-01-01")


@st.cache_data(ttl=TTL, show_spinner="Loading 30-min…")
def intraday(ticker=GOLD):
    return yahoo_ohlc(ticker, "30m", rng="60d")


@st.cache_data(ttl=TTL, show_spinner=False)
def weekly(ticker=GOLD):
    return ind.resample_weekly(daily(ticker))


@st.cache_data(ttl=TTL, show_spinner=False)
def macro():
    out = {}
    for key, sym in {"dxy": "DX-Y.NYB", "tnx": "^TNX", "silver": "SI=F"}.items():
        try:
            out[key] = yahoo_ohlc(sym, "1d", start="2024-01-01")
        except Exception:
            out[key] = None
    return out


@st.cache_data(ttl=120, show_spinner=False)
def live_spot():
    try:
        return spot("XAU")
    except Exception:
        return None


@st.cache_data(ttl=TTL, show_spinner=False)
def fx():
    """USD/oz → unit multipliers. cny_g/jpy_g via live Yahoo FX; None on failure."""
    out = {"usd_oz": 1.0}
    for unit, sym in {"cny_g": "CNY=X", "jpy_g": "JPY=X"}.items():
        try:
            out[unit] = fx_rate(sym) / GRAMS_PER_TROY_OZ
        except Exception:
            out[unit] = None
    return out


@st.cache_data(ttl=3600, show_spinner="Backtesting…")
def ml_view(stem="GC_F"):
    df = build_features(stem, False)
    rep = evaluate(walk_forward_series(df, pd.Timestamp(config.TEST_START), config.RETRAIN_EVERY))
    cols = feature_columns(df)
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(df[cols], df["target"])
    row = latest_feature_row(stem)
    pred = float(model.predict(row[cols])[0])
    return rep, float(row["close"].iloc[0]) * float(np.exp(pred)), pred


# --- helpers ----------------------------------------------------------------
def candle(df, mas, height=300, hide_weekends=True, sup=None, res=None):
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=UP, decreasing_line_color=DOWN, showlegend=False))
    for n, color in mas:
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(df["Close"], n),
                                 line=dict(width=1, color=color), name=f"MA{n}"))
    if sup:
        fig.add_hline(y=sup, line=dict(dash="dot", color=UP), annotation_text="S")
    if res:
        fig.add_hline(y=res, line=dict(dash="dot", color=DOWN), annotation_text="R")
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6),
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)))
    if hide_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


_KIND = {"good": ("#0F6E56", "#E1F5EE"), "bad": ("#A32D2D", "#FCEBEB"),
         "warn": ("#854F0B", "#FAEEDA"), "neutral": ("#444441", "#F1EFE8")}


def badge(col, title, value, kind):
    fg, bg = _KIND[kind]
    col.markdown(
        f"<div style='background:{bg};color:{fg};padding:10px 12px;border-radius:10px'>"
        f"<div style='font-size:12px;opacity:.85'>{title}</div>"
        f"<div style='font-size:16px;font-weight:600'>{value}</div></div>",
        unsafe_allow_html=True)


# --- language ---------------------------------------------------------------
_dl = st.query_params.get("lang", "en")
if _dl not in LANGS:
    _dl = "en"
lang = st.sidebar.selectbox("Language · 語言 · 言語", list(LANGS),
                            index=list(LANGS).index(_dl), format_func=lambda c: LANGS[c])
if lang != st.query_params.get("lang"):
    st.query_params["lang"] = lang

# --- unit (default by language; deep-linkable via ?unit=) -------------------
rates = fx()
_du = st.query_params.get("unit") or LANG_UNIT[lang]
if _du not in UNITS or rates.get(_du) is None:
    _du = "usd_oz"
unit = st.sidebar.selectbox(t(lang, "unit"), list(UNITS),
                            index=list(UNITS).index(_du),
                            format_func=lambda u: t(lang, "unit_" + u))
if rates.get(unit) is None:        # FX unavailable → fall back to USD/oz
    unit = "usd_oz"
if unit != st.query_params.get("unit"):
    st.query_params["unit"] = unit
factor = rates.get(unit) or 1.0
_U = UNITS[unit]


def price(v):
    return f"{_U['sym']}{v * factor:,.{_U['dec']}f}{_U['suffix']}"


def conv(df):
    out = df.copy()
    out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]] * factor
    return out


# --- data -------------------------------------------------------------------
d, w, i30, mac, sp = daily(), weekly(), intraday(), macro(), live_spot()

st.title(t(lang, "app_title"))
st.caption(t(lang, "ts_method"))

last_close, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
chg = last_close / prev - 1
h1, h2 = st.columns(2)
h1.metric(t(lang, "live_spot"), price(sp["price"] if sp else last_close), delta=f"{chg:+.2%}")
h2.metric("GC=F", price(last_close), help=str(d.index[-1].date()))


# --- Triple Screen read -----------------------------------------------------
ts = ind.triple_screen(w, d, i30)
st.subheader(t(lang, "screen_read"))
b1, b2, b3, b4 = st.columns(4)
badge(b1, t(lang, "trend"), t(lang, "trend_" + ts["trend"]),
      {"up": "good", "down": "bad", "flat": "neutral"}[ts["trend"]])
badge(b2, t(lang, "strength"), t(lang, "str_" + ts["strength"]),
      {"strong": "good", "weak": "bad", "neutral": "neutral"}[ts["strength"]])
badge(b3, t(lang, "support"), price(ts["support"]), "good")
badge(b4, t(lang, "resistance"), price(ts["resistance"]), "bad")
st.caption("➤ " + t(lang, {"long": "bias_long", "short": "bias_short", "none": "bias_none"}[ts["bias"]]))


# --- macro signal lights ----------------------------------------------------
if all(mac.get(k) is not None for k in ("dxy", "tnx", "silver")):
    sig = ind.macro_signals(d, mac["dxy"], mac["tnx"], mac["silver"])
    km = {"bull": "good", "bear": "bad", "warn": "warn"}
    st.subheader(t(lang, "macro_lights"))
    m1, m2, m3 = st.columns(3)
    badge(m1, t(lang, "sig_dollar"), t(lang, "sig_" + sig["dxy"]), km[sig["dxy"]])
    badge(m2, t(lang, "sig_rates"), t(lang, "sig_" + sig["rates"]), km[sig["rates"]])
    badge(m3, t(lang, "sig_silver"), t(lang, "sig_" + sig["silver"]), km[sig["silver"]])


# --- the three screens (candlesticks) ---------------------------------------
st.subheader(t(lang, "tf_weekly"))
st.plotly_chart(candle(conv(w.tail(120)), [(13, MA_C[0]), (30, MA_C[2])], hide_weekends=False),
                use_container_width=True)
st.subheader(t(lang, "tf_daily"))
st.plotly_chart(candle(conv(d.tail(130)), [(20, MA_C[0]), (50, MA_C[1]), (200, MA_C[2])]),
                use_container_width=True)
st.subheader(t(lang, "tf_30m"))
st.plotly_chart(candle(conv(i30.tail(160)), [(20, MA_C[0]), (50, MA_C[1])],
                       sup=ts["support"] * factor, res=ts["resistance"] * factor),
                use_container_width=True)

st.info(t(lang, "method_note"))


# --- optional ML next-day forecast + honest backtest ------------------------
with st.expander(t(lang, "ml_section"), expanded=False):
    try:
        rep, next_price, pr = ml_view("GC_F")
        c1, c2, c3 = st.columns(3)
        c1.metric(t(lang, "hero_forecast_1"), price(next_price), delta=f"{pr:+.2%}")
        c2.metric(t(lang, "m_skill"), f"{rep['rmse_skill_vs_baseline']:+.2%}", help=t(lang, "skill_help"))
        c3.metric(t(lang, "m_diracc"), f"{rep['model']['directional_acc']:.1%}", help=t(lang, "diracc_help"))
        st.caption(t(lang, "beats_yes" if rep["beats_baseline"] else "beats_no", n=rep["n_test"]))
    except Exception as e:
        st.caption(f"ML view needs cached daily data (run `uv run gold fetch`). {e}")
