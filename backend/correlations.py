"""
Multi-asset correlation engine.

Computes 20-period rolling Pearson correlation of daily returns between
XAU/USD and key correlated/anti-correlated assets. Results cached 30 min.
Uses yfinance (already a dependency).
"""

from __future__ import annotations

import time
from typing import Dict, Any

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
ASSETS: Dict[str, str] = {
    "Silver":    "SI=F",
    "S&P500":    "^GSPC",
    "DXY":       "DX-Y.NYB",
    "EUR/USD":   "EURUSD=X",
    "10Y Yield": "^TNX",
}

XAU_TICKER = "GC=F"   # Gold futures — daily close for XAU/USD proxy

# ---------------------------------------------------------------------------
# 30-minute in-process cache
# ---------------------------------------------------------------------------
_cache: Dict[str, Any] = {}
_cache_ts: float = 0.0
CACHE_TTL = 1800  # seconds


def _fetch_close(ticker: str, period: str = "60d", interval: str = "1d") -> pd.Series:
    """Download daily close prices and return a clean Series indexed by date."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).normalize()
    return close


def _trend(series: pd.Series, short: int = 5, long: int = 20) -> str:
    """
    Compare the mean of the last `short` values against the mean of the
    last `long` values to determine whether correlation is strengthening,
    weakening, or neutral.
    """
    if len(series) < long:
        return "neutral"
    recent_short = series.iloc[-short:].mean()
    recent_long = series.iloc[-long:].mean()
    diff = recent_short - recent_long
    if diff > 0.05:
        return "strengthening"
    if diff < -0.05:
        return "weakening"
    return "neutral"


def get_correlations() -> Dict[str, Any]:
    """
    Return a dict of per-asset correlation data.  Results are cached for
    CACHE_TTL seconds so that repeated fast requests don't hammer yfinance.

    Structure:
        {
          "Silver":    {"correlation": 0.72, "trend": "strengthening", "ticker": "SI=F"},
          "S&P500":    {"correlation": -0.31, "trend": "neutral",       "ticker": "^GSPC"},
          ...
        }
    """
    global _cache, _cache_ts
    now = time.time()
    if _cache and (now - _cache_ts) < CACHE_TTL:
        return _cache

    # Fetch XAU returns
    xau_close = _fetch_close(XAU_TICKER)
    xau_ret = xau_close.pct_change().dropna()

    result: Dict[str, Any] = {}
    for name, ticker in ASSETS.items():
        try:
            asset_close = _fetch_close(ticker)
            asset_ret = asset_close.pct_change().dropna()

            # Align on common dates
            combined = pd.concat([xau_ret, asset_ret], axis=1, join="inner")
            combined.columns = ["xau", "asset"]
            combined = combined.dropna()

            if len(combined) < 22:
                # Not enough data — report NaN
                result[name] = {"correlation": None, "trend": "neutral", "ticker": ticker}
                continue

            # 20-period rolling Pearson correlation
            rolling_corr = combined["xau"].rolling(window=20).corr(combined["asset"]).dropna()

            latest = float(rolling_corr.iloc[-1])
            # Guard against NaN slipping through
            if pd.isna(latest):
                latest_clean = None
                trend = "neutral"
            else:
                latest_clean = round(latest, 4)
                trend = _trend(rolling_corr)

            result[name] = {
                "correlation": latest_clean,
                "trend": trend,
                "ticker": ticker,
            }
        except Exception as exc:
            result[name] = {"correlation": None, "trend": "neutral", "ticker": ticker, "error": str(exc)}

    _cache = result
    _cache_ts = now
    return result
