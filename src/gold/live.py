"""Live spot price via gold-api.com — no key, and reachable even behind the
finance-blocking proxy (Yahoo/Stooq/FRED are blocked; this host is not).

A current-price source and cross-check. It returns ONLY the latest spot — the
historical OHLC needed for candlesticks/backtests comes from Yahoo once the
proxy allows it (CLAUDE.md §3 network gotcha).
"""

from __future__ import annotations

import requests

try:  # use the OS trust store (handles the MITM-proxy root CA)
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

_URL = "https://api.gold-api.com/price/{symbol}"
_UA = "Mozilla/5.0"


def spot(symbol: str = "XAU") -> dict:
    """Latest spot price for a metal symbol (XAU gold, XAG silver, …)."""
    r = requests.get(_URL.format(symbol=symbol), headers={"User-Agent": _UA}, timeout=20)
    r.raise_for_status()
    d = r.json()
    return {
        "symbol": d["symbol"],
        "name": d.get("name", symbol),
        "price": float(d["price"]),
        "updated_at": d.get("updatedAt"),
    }
