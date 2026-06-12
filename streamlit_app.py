"""Gold — multi-timeframe candlestick dashboard (EN / 繁體中文 / 日本語).

Run:  uv run streamlit run streamlit_app.py

"周线看方向 · 日线定强弱 · 30分钟找买卖点" — Elder's Triple Screen on LIVE Yahoo
data: weekly / daily / 30-min candlesticks with MACD / RSI / volume sub-panels,
a Triple Screen read, macro signal lights, a live (auto-refreshing) spot price,
unit switching (USD/oz · 元/克 · 円/g) and tunable indicator sensitivity.
Indicators confirm, they don't predict — only candlesticks are lag-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from plotly.subplots import make_subplots

from gold import config
from gold import indicators as ind
from gold.data.fetch import yahoo_ohlc
from gold.features.build import build_features, feature_columns, latest_feature_row
from gold.i18n import LANGS, t
from gold.live import fx_rate, spot
from gold.models.backtest import evaluate, walk_forward_series

st.set_page_config(page_title="Gold candlesticks", layout="wide")

# Mobile: stack column rows (badges / order cards / metrics) vertically on narrow screens.
st.markdown(
    "<style>@media (max-width:640px){"
    "[data-testid='stHorizontalBlock']{flex-direction:column;gap:8px}"
    "[data-testid='stColumn']{width:100%!important;min-width:100%!important;flex:1 1 100%!important}}"
    "</style>",
    unsafe_allow_html=True)

GOLD = "GC=F"
MA_C = ["#BA7517", "#378ADD", "#7F77DD"]  # amber / blue / purple
UP, DOWN = "#1D9E75", "#D85A30"

GRAMS_PER_TROY_OZ = 31.1034768
UNITS = {
    "usd_oz": {"sym": "$", "suffix": "/oz", "dec": 2},
    "cny_g": {"sym": "¥", "suffix": "/g", "dec": 2},   # 元/克
    "jpy_g": {"sym": "¥", "suffix": "/g", "dec": 0},   # 円/g
}
LANG_UNIT = {"en": "usd_oz", "zh-Hant": "cny_g", "ja": "jpy_g"}
# Sensitivity presets — shorter periods = less lag, more false signals.
SENS = {
    "standard": dict(macd_fast=12, macd_slow=26, macd_signal=9, rsi_n=14, ma_fast=20, ma_slow=50),
    "fast": dict(macd_fast=6, macd_slow=13, macd_signal=5, rsi_n=9, ma_fast=10, ma_slow=30),
    "smooth": dict(macd_fast=19, macd_slow=39, macd_signal=9, rsi_n=21, ma_fast=30, ma_slow=80),
}


# --- data (short TTL so it stays current; manual refresh clears caches) ------
@st.cache_data(ttl=180, show_spinner="Loading daily…")
def daily(ticker=GOLD):
    return yahoo_ohlc(ticker, "1d", start="2015-01-01")


@st.cache_data(ttl=180, show_spinner="Loading 30-min…")
def intraday(ticker=GOLD):
    return yahoo_ohlc(ticker, "30m", rng="60d")


@st.cache_data(ttl=180, show_spinner=False)
def weekly(ticker=GOLD):
    return ind.resample_weekly(daily(ticker))


@st.cache_data(ttl=180, show_spinner=False)
def macro():
    out = {}
    for key, sym in {"dxy": "DX-Y.NYB", "tnx": "^TNX", "silver": "SI=F"}.items():
        try:
            out[key] = yahoo_ohlc(sym, "1d", start="2024-01-01")
        except Exception:
            out[key] = None
    return out


@st.cache_data(ttl=30, show_spinner=False)
def live_spot():
    try:
        return spot("XAU")
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fx():
    out = {"usd_oz": 1.0}
    for unit, sym in {"cny_g": "CNY=X", "jpy_g": "JPY=X"}.items():
        try:
            out[unit] = fx_rate(sym) / GRAMS_PER_TROY_OZ
        except Exception:
            out[unit] = None
    return out


@st.cache_data(ttl=3600, show_spinner="Backtesting…")
def ml_view(last_date):
    d_ = daily()  # build features from LIVE data → works online (no parquet needed)
    df = build_features(df=d_, save=False)
    rep = evaluate(walk_forward_series(df, pd.Timestamp(config.TEST_START), config.RETRAIN_EVERY))
    cols = feature_columns(df)
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(df[cols], df["target"])
    row = latest_feature_row(df=d_)
    pred = float(model.predict(row[cols])[0])
    return rep, float(row["close"].iloc[0]) * float(np.exp(pred)), pred


@st.cache_data(ttl=1800, show_spinner="Backtesting…")
def lab_holdout(last_date, rr, atr_stop, cost_bp, holdout_pct, variant="macd"):
    from gold.strategy import backtest_with_holdout
    return backtest_with_holdout(daily(), weekly(), holdout_frac=holdout_pct / 100,
                                 rr=rr, atr_stop=atr_stop, cost=cost_bp / 10000, variant=variant)


@st.cache_data(ttl=1800, show_spinner="Sweeping…")
def lab_sweep_view(last_date, cost_bp, holdout_pct):
    from gold.strategy import strategy_sweep
    return strategy_sweep(daily(), weekly(), holdout_frac=holdout_pct / 100, cost=cost_bp / 10000)


@st.cache_data(ttl=3600, show_spinner="Backtesting strategy…")
def strategy_view(last_date, rr=1.5):
    from gold.strategy import triple_screen_backtest
    return triple_screen_backtest(daily(), weekly(), rr=rr)


# --- chart with optional MACD / RSI / volume sub-panels ---------------------
def panel(df, mas, subs=(), height=380, hide_weekends=True, sup=None, res=None, p=None, zones=None):
    p = p or {}
    rows = 1 + len(subs)
    rh = [0.58, *([round(0.42 / len(subs), 3)] * len(subs))] if subs else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=rh)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=UP, decreasing_line_color=DOWN, showlegend=False), row=1, col=1)
    for n, color in mas:
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(df["Close"], n),
                                 line=dict(width=1, color=color), name=f"MA{n}"), row=1, col=1)
    if sup is not None:
        fig.add_hline(y=sup, line=dict(dash="dot", color=UP, width=1), row=1, col=1)
    if res is not None:
        fig.add_hline(y=res, line=dict(dash="dot", color=DOWN, width=1), row=1, col=1)
    for lo, hi, kind, label in (zones or []):
        fig.add_hrect(y0=lo, y1=hi, line_width=0, opacity=0.13, row=1, col=1,
                      fillcolor=UP if kind == "support" else DOWN,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=10)

    r = 2
    for s in subs:
        if s == "volume":
            cols = [UP if c >= o else DOWN for o, c in zip(df["Open"], df["Close"])]
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=cols,
                                 name="Vol", showlegend=False), row=r, col=1)
        elif s == "macd":
            line, sig, hist = ind.macd(df["Close"], p.get("macd_fast", 12),
                                       p.get("macd_slow", 26), p.get("macd_signal", 9))
            fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD", showlegend=False,
                                 marker_color=[UP if h >= 0 else DOWN for h in hist]), row=r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=line, line=dict(width=1, color=MA_C[1]), name="MACD"), row=r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sig, line=dict(width=1, color=MA_C[0]), name="signal"), row=r, col=1)
        elif s == "rsi":
            ob = p.get("rsi_ob", 70)
            fig.add_trace(go.Scatter(x=df.index, y=ind.rsi(df["Close"], p.get("rsi_n", 14)),
                                     line=dict(width=1, color=MA_C[2]), name="RSI"), row=r, col=1)
            fig.add_hline(y=ob, line=dict(dash="dot", color=DOWN, width=1), row=r, col=1)
            fig.add_hline(y=100 - ob, line=dict(dash="dot", color=UP, width=1), row=r, col=1)
        r += 1

    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6),
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)))
    fig.update_xaxes(rangeslider_visible=False)
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


def order_card(col, label, o, risky):
    if o is None:
        col.caption(f"{label}: —")
        return
    fg, bg = _KIND["warn" if risky else ("good" if o["side"] == "buy" else "bad")]
    rr_s = f"{o['rr']:.1f}" if o.get("rr") else "—"
    tgt = price(o["target"]) if o.get("target") else "—"
    warn = f"<div style='font-size:11px;margin-top:3px'>⚠ {t(lang, 'dip_buy_risky')}</div>" if risky else ""
    col.markdown(
        f"<div style='background:{bg};color:{fg};padding:10px 12px;border-radius:10px'>"
        f"<div style='font-size:12px;opacity:.85'>{label} · {t(lang, 'o_reso')} {o['n_tf']}/3</div>"
        f"<div style='font-size:19px;font-weight:600;margin:2px 0'>{price(o['entry'])}</div>"
        f"<div style='font-size:12px'>{t(lang, 'o_stop')} {price(o['stop'])} → {t(lang, 'o_target')} {tgt}</div>"
        f"<div style='font-size:12px;font-weight:600'>{t(lang, 'o_rr')} {rr_s}</div>{warn}</div>",
        unsafe_allow_html=True)


def sell_card(col, label, z):
    if z is None:
        col.caption(f"{label}: —")
        return
    fg, bg = _KIND["bad"]
    col.markdown(
        f"<div style='background:{bg};color:{fg};padding:10px 12px;border-radius:10px'>"
        f"<div style='font-size:12px;opacity:.85'>{label} · {t(lang, 'o_reso')} {z['n_tf']}/3</div>"
        f"<div style='font-size:19px;font-weight:600;margin:2px 0'>{price(z['center'])}</div>"
        f"<div style='font-size:12px'>{price(z['low'])} – {price(z['high'])}</div></div>",
        unsafe_allow_html=True)


# --- sidebar: language / unit / sensitivity / refresh -----------------------
def _browser_lang():
    """Map the browser's Accept-Language to a supported locale (first visit only)."""
    try:
        first = (st.context.headers.get("Accept-Language") or "").lower().split(",")[0].strip()
    except Exception:
        first = ""
    if first.startswith("zh"):
        return "zh-Hant"
    if first.startswith("ja"):
        return "ja"
    return "en"


_dl = st.query_params.get("lang") or _browser_lang()
if _dl not in LANGS:
    _dl = "en"
lang = st.sidebar.selectbox("Language · 語言 · 言語", list(LANGS),
                            index=list(LANGS).index(_dl), format_func=lambda c: LANGS[c])
if lang != st.query_params.get("lang"):
    st.query_params["lang"] = lang

rates = fx()
_du = st.query_params.get("unit") or LANG_UNIT[lang]
if _du not in UNITS or rates.get(_du) is None:
    _du = "usd_oz"
unit = st.sidebar.selectbox(t(lang, "unit"), list(UNITS), index=list(UNITS).index(_du),
                            format_func=lambda u: t(lang, "unit_" + u))
if rates.get(unit) is None:
    unit = "usd_oz"
if unit != st.query_params.get("unit"):
    st.query_params["unit"] = unit
factor = rates.get(unit) or 1.0
_U = UNITS[unit]

# Indicators/sensitivity confuses beginners — hidden by default; power users opt in via ?expert=1.
if st.query_params.get("expert") in ("1", "true", "yes"):
    with st.sidebar.expander(t(lang, "adv"), expanded=False):
        sens = st.selectbox(t(lang, "sensitivity"), list(SENS),
                            format_func=lambda s: t(lang, "sens_" + s))
        rsi_ob = st.slider(t(lang, "rsi_ob_label"), 60, 85, 70)
        swing = st.slider(t(lang, "swing_label"), 10, 80, 40, step=5)
        use_vol = st.toggle(t(lang, "use_volume"), value=True)
    P = dict(SENS[sens], rsi_ob=rsi_ob, swing=swing, use_force=use_vol)
else:
    P = dict(SENS["standard"], rsi_ob=70, swing=40, use_force=True)  # locked standard defaults

auto = st.sidebar.toggle(t(lang, "autorefresh"), value=True)
if st.sidebar.button(t(lang, "refresh")):
    st.cache_data.clear()
    st.rerun()


def price(v):
    return f"{_U['sym']}{v * factor:,.{_U['dec']}f}{_U['suffix']}"


def conv(df):
    out = df.copy()
    out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]] * factor
    return out


# --- data + header ----------------------------------------------------------
d, w, i30, mac = daily(), weekly(), intraday(), macro()

st.title(t(lang, "app_title"))
st.caption(t(lang, "ts_method"))


@st.fragment(run_every="30s" if auto else None)
def hero():
    sp = live_spot()
    last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
    h1, h2 = st.columns(2)
    h1.metric(t(lang, "live_spot"), price(sp["price"] if sp else last), delta=f"{last / prev - 1:+.2%}")
    h2.metric("GC=F", price(last), help=str(d.index[-1].date()))
    asof = (sp["updated_at"] if sp else str(d.index[-1]))
    st.caption(f"{t(lang, 'data_time')}: {asof}")


hero()


# --- Triple Screen read -----------------------------------------------------
ts = ind.triple_screen(w, d, i30, **P)
st.subheader(t(lang, "screen_read"))
b1, b2, b3, b4 = st.columns(4)
badge(b1, t(lang, "trend"), t(lang, "trend_" + ts["trend"]),
      {"up": "good", "down": "bad", "flat": "neutral"}[ts["trend"]])
badge(b2, t(lang, "strength"), t(lang, "str_" + ts["strength"]),
      {"strong": "good", "weak": "bad", "neutral": "neutral"}[ts["strength"]])
badge(b3, t(lang, "support"), price(ts["support"]), "good")
badge(b4, t(lang, "resistance"), price(ts["resistance"]), "bad")
st.caption("➤ " + t(lang, {"long": "bias_long", "short": "bias_short", "none": "bias_none"}[ts["bias"]]))
st.caption(t(lang, "confirmed_note"))


# --- suggested order levels (multi-timeframe confluence) --------------------
op = ind.order_plan(w, d, i30, bias=ts["bias"], atr_d=float(ind.atr(d).iloc[-1]))
st.subheader(t(lang, "order_section"))
if op["bias"] == "long":
    st.success(t(lang, "with_trend"))
elif op["bias"] == "short":
    st.error(t(lang, "counter_trend_warn"))
else:
    st.warning(t(lang, "stand_aside_lv"))

ct = op["counter_trend_buy"]
o1, o2, o3 = st.columns(3)
order_card(o1, t(lang, "best_buy_s1"), op["buys"][0] if op["buys"] else None, ct)
order_card(o2, t(lang, "second_buy_s2"), op["buys"][1] if len(op["buys"]) > 1 else None, ct)
sell_card(o3, t(lang, "sell_target"), op["sell_target"])
if op["resonant_support"]:
    rs = op["resonant_support"]
    st.caption(f"{t(lang, 'resonant_buy')} · {price(rs['entry'])} · {t(lang, 'o_reso')} {rs['n_tf']}/3")
else:
    st.caption(t(lang, "no_resonant"))
st.caption(t(lang, "levels_note"))

# zones to shade on the candlesticks (price × unit factor)
cz = []
for lbl, b in (("S1", op["buys"][0] if op["buys"] else None),
               ("S2", op["buys"][1] if len(op["buys"]) > 1 else None)):
    if b:
        cz.append((b["low"] * factor, b["high"] * factor, "support", lbl))
if op["sell_target"]:
    s = op["sell_target"]
    cz.append((s["low"] * factor, s["high"] * factor, "resistance", "Sell"))

# does this method actually work? — honest backtest right next to the levels
with st.expander(t(lang, "strat_section"), expanded=False):
    sb = strategy_view(str(d.index[-1]))
    if sb.get("n_trades"):
        c1, c2, c3 = st.columns(3)
        c1.metric(t(lang, "strat_winrate"), f"{sb['win_rate']:.0%}")
        c2.metric(t(lang, "strat_exp"), f"{sb['expectancy']:+.2%}")
        c3.metric(t(lang, "strat_pf"), f"{sb['profit_factor']:.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric(t(lang, "strat_total"), f"{sb['total_return']:+.0%}")
        c5.metric(t(lang, "strat_bh"), f"{sb['buy_hold']:+.0%}")
        c6.metric(t(lang, "strat_edge"), f"{sb['edge_vs_random']:+.2%}")
        _bad = sb["expectancy"] <= 0 or sb["edge_vs_random"] <= 0
        (st.error if _bad else st.success)(
            t(lang, "strat_verdict_bad" if _bad else "strat_verdict_ok", period=sb["period"]))


# --- macro signal lights ----------------------------------------------------
if all(mac.get(k) is not None for k in ("dxy", "tnx", "silver")):
    sig = ind.macro_signals(d, mac["dxy"], mac["tnx"], mac["silver"])
    km = {"bull": "good", "bear": "bad", "warn": "warn"}
    st.subheader(t(lang, "macro_lights"))
    m1, m2, m3 = st.columns(3)
    badge(m1, t(lang, "sig_dollar"), t(lang, "sig_" + sig["dxy"]), km[sig["dxy"]])
    badge(m2, t(lang, "sig_rates"), t(lang, "sig_" + sig["rates"]), km[sig["rates"]])
    badge(m3, t(lang, "sig_silver"), t(lang, "sig_" + sig["silver"]), km[sig["silver"]])


# --- the three screens (candles + sub-panels) -------------------------------
st.subheader(t(lang, "tf_weekly"))
st.plotly_chart(panel(conv(w.tail(120)), [(13, MA_C[0]), (30, MA_C[2])],
                      subs=["macd"], height=380, hide_weekends=False, p=P), use_container_width=True)
st.subheader(t(lang, "tf_daily"))
st.plotly_chart(panel(conv(d.tail(160)), [(20, MA_C[0]), (50, MA_C[1]), (200, MA_C[2])],
                      subs=["volume", "rsi"], height=470, p=P, zones=cz), use_container_width=True)
st.subheader(t(lang, "tf_30m"))
st.plotly_chart(panel(conv(i30.tail(160)), [(20, MA_C[0]), (50, MA_C[1])], subs=["volume"],
                      height=380, p=P, zones=cz), use_container_width=True)

st.info(t(lang, "method_note"))
st.caption(t(lang, "lag_note"))


# --- optional ML next-day forecast + honest backtest (button-gated; runs online) ---
with st.expander(t(lang, "ml_section"), expanded=False):
    if st.button(t(lang, "run_btn"), key="run_ml"):
        st.session_state["ml_on"] = True
    if st.session_state.get("ml_on"):
        try:
            rep, next_price, pr = ml_view(str(d.index[-1]))
            c1, c2, c3 = st.columns(3)
            c1.metric(t(lang, "hero_forecast_1"), price(next_price), delta=f"{pr:+.2%}")
            c2.metric(t(lang, "m_skill"), f"{rep['rmse_skill_vs_baseline']:+.2%}", help=t(lang, "skill_help"))
            c3.metric(t(lang, "m_diracc"), f"{rep['model']['directional_acc']:.1%}", help=t(lang, "diracc_help"))
            st.caption(t(lang, "beats_yes" if rep["beats_baseline"] else "beats_no", n=rep["n_test"]))
        except Exception as e:
            st.caption(f"ML unavailable: {e}")


# --- Backtest Lab (?lab=1): out-of-sample, net of costs ---------------------
if st.query_params.get("lab") in ("1", "true", "yes"):
    st.divider()
    st.subheader(t(lang, "lab_section"))
    st.caption(t(lang, "lab_intro"))
    variant = st.selectbox(t(lang, "lab_strat"), ["macd", "elder"],
                           format_func=lambda v: t(lang, "var_" + v))
    lc1, lc2, lc3, lc4 = st.columns(4)
    rr_l = lc1.slider(t(lang, "lab_rr"), 1.0, 3.0, 1.5, step=0.5)
    atr_l = lc2.slider(t(lang, "lab_atr"), 1.0, 3.0, 2.0, step=0.5)
    cost_bp = lc3.slider(t(lang, "lab_cost"), 0, 20, 5)
    ho = lc4.slider(t(lang, "lab_holdout"), 20, 40, 30, step=5)

    hb = lab_holdout(str(d.index[-1]), rr_l, atr_l, cost_bp, ho, variant)
    ins, oos = hb["in_sample"], hb["holdout"]
    la, lb = st.columns(2)
    la.markdown(f"**{t(lang, 'lab_in')}**")
    la.metric(t(lang, "lab_exp"), f"{ins['expectancy']:+.2%}" if ins["n_trades"] else "—")
    la.caption(f"{t(lang, 'lab_trades')} {ins['n_trades']} · {t(lang, 'lab_total')} {ins['total_return']:+.0%}")
    lb.markdown(f"**{t(lang, 'lab_out')}**")
    lb.metric(t(lang, "lab_exp"), f"{oos['expectancy']:+.2%}" if oos["n_trades"] else "—")
    lb.caption(f"{t(lang, 'lab_trades')} {oos['n_trades']} · {t(lang, 'lab_total')} {oos['total_return']:+.0%} · B&H {hb['holdout_buy_hold']:+.0%}")

    # "survives" only if it actually made money out-of-sample (positive expectancy AND total)
    if oos["n_trades"] and oos["expectancy"] > 0 and oos["total_return"] > 0:
        st.success(t(lang, "lab_v_survives", n=oos["n_trades"]))
    elif oos["n_trades"]:
        st.error(t(lang, "lab_v_noedge"))
    if hb["oos_gap"] == hb["oos_gap"] and hb["oos_gap"] > 0.005:
        st.warning(t(lang, "lab_v_overfit", gap=f"{hb['oos_gap']:+.2%}"))

    if st.button(t(lang, "lab_sweep_btn"), key="run_sweep"):
        st.session_state["sweep_on"] = True
    if st.session_state.get("sweep_on"):
        sw = lab_sweep_view(str(d.index[-1]), cost_bp, ho)
        st.warning(t(lang, "lab_sweep_warn", n=sw["n_combos"]))
        tbl = pd.DataFrame(sw["rows"])
        if not tbl.empty:
            for c in ("in_exp", "oos_exp", "oos_gap", "oos_total"):
                tbl[c] = (tbl[c] * 100).round(2)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
