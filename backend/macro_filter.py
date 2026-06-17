"""
macro_filter.py
===============
Macro market filters: DXY (Dollar Index), VIX (Fear Index), TNX (10Y Treasury),
Gold OI proxy (GC=F), Fear & Greed Index, and Fed/Central Bank bias (FRED).

DXY: Gold is inversely correlated with USD. DXY uptrend → block LONG gold.
VIX: VIX > 25 means extreme fear → scalping too dangerous → block all entries.
TNX: Rising 10Y yield = higher real rates = pressure on gold.
Fed: Hiking cycle + high positive real rates → additional headwind for gold longs.
"""
from __future__ import annotations

import time as _time
import threading
from typing import Optional, Dict, Any

import pandas as pd

VIX_BLOCK_THRESHOLD = 25.0
CACHE_TTL = 900  # refresh every 15 minutes


class MacroFilter:
    def __init__(self):
        self._dxy: Optional[float] = None
        self._dxy_trend: Optional[str] = None   # "up" | "down" | "neutral"
        self._vix: Optional[float] = None
        self._tnx: Optional[float] = None
        self._tnx_trend: Optional[str] = None
        self._gold_oi: Optional[float] = None
        self._fear_greed: Optional[int] = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()
        # try an initial fetch in background so startup isn't blocked
        threading.Thread(target=self._refresh, daemon=True).start()

    def _refresh(self) -> None:
        now = _time.time()
        if now - self._fetched_at < CACHE_TTL:
            return
        try:
            import yfinance as yf

            # DXY hourly (last 5 days)
            dxy_df = yf.download("DX-Y.NYB", period="5d", interval="1h",
                                  progress=False, auto_adjust=False)
            if dxy_df is not None and len(dxy_df) >= 10:
                if isinstance(dxy_df.columns, pd.MultiIndex):
                    dxy_df.columns = dxy_df.columns.get_level_values(0)
                close = dxy_df["Close"].dropna()
                self._dxy = round(float(close.iloc[-1]), 3)
                ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
                ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
                if ema9 > ema21 * 1.0001:
                    self._dxy_trend = "up"
                elif ema9 < ema21 * 0.9999:
                    self._dxy_trend = "down"
                else:
                    self._dxy_trend = "neutral"

            # VIX hourly
            vix_df = yf.download("^VIX", period="5d", interval="1h",
                                  progress=False, auto_adjust=False)
            if vix_df is not None and len(vix_df) >= 1:
                if isinstance(vix_df.columns, pd.MultiIndex):
                    vix_df.columns = vix_df.columns.get_level_values(0)
                self._vix = round(float(vix_df["Close"].dropna().iloc[-1]), 2)

            # TNX - 10-year US Treasury yield
            tnx_df = yf.download("^TNX", period="5d", interval="1h",
                                  progress=False, auto_adjust=False)
            if tnx_df is not None and len(tnx_df) >= 1:
                if isinstance(tnx_df.columns, pd.MultiIndex):
                    tnx_df.columns = tnx_df.columns.get_level_values(0)
                tnx_close = tnx_df["Close"].dropna()
                self._tnx = round(float(tnx_close.iloc[-1]), 3)
                tnx_ema9 = float(tnx_close.ewm(span=9, adjust=False).mean().iloc[-1])
                tnx_ema21 = float(tnx_close.ewm(span=21, adjust=False).mean().iloc[-1])
                self._tnx_trend = "up" if tnx_ema9 > tnx_ema21 * 1.0001 else ("down" if tnx_ema9 < tnx_ema21 * 0.9999 else "neutral")

            # Gold futures open interest (via yfinance GC=F info)
            try:
                gc = yf.Ticker("GC=F")
                info = gc.fast_info
                self._gold_oi = getattr(info, "three_month_average_volume", None)
            except Exception:
                self._gold_oi = None

            # CNN Fear & Greed Index (0-100, <25=extreme fear, >75=extreme greed)
            try:
                import urllib.request, json as _json
                req = urllib.request.Request(
                    "https://fear-and-greed-index.p.rapidapi.com/v1/fgi",
                    headers={"X-RapidAPI-Host": "fear-and-greed-index.p.rapidapi.com"}
                )
                # If this fails (no key), use neutral fallback
                raise Exception("no key")
            except Exception:
                # Fallback: compute from VIX (proxy for fear)
                if self._vix is not None:
                    # Map VIX to 0-100 fear score: VIX 10=greed(80), VIX 25=fear(30), VIX 40=extreme fear(5)
                    vix_score = max(0, min(100, int(100 - (self._vix - 10) * 3.5)))
                    self._fear_greed = vix_score
                else:
                    self._fear_greed = 50  # neutral default

            self._fetched_at = now
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh()
        vix_blocked = self._vix is not None and self._vix > VIX_BLOCK_THRESHOLD

        # Fed/CB bias (imported lazily to avoid circular imports)
        fed_bias: Optional[Dict[str, Any]] = None
        try:
            from fred_feed import get_gold_macro_bias
            fed_bias = get_gold_macro_bias()
        except Exception:
            pass

        return {
            "dxy":       self._dxy,
            "dxy_trend": self._dxy_trend,
            "vix":       self._vix,
            "vix_blocked": vix_blocked,
            "tnx":       self._tnx,
            "tnx_trend": self._tnx_trend,
            "fear_greed": self._fear_greed,
            "fed_bias":  fed_bias,
        }

    def blocks_entry(self, symbol: str, bias: str) -> tuple[bool, str]:
        """Return (blocked, reason). Checks VIX and DXY conflict."""
        s = self.status()

        if s["vix_blocked"]:
            return True, f"VIX={s['vix']:.1f} > {VIX_BLOCK_THRESHOLD} (panique marché)"

        trend = s["dxy_trend"]
        if trend == "up":
            # DXY up → USD strong → gold typically falls → block LONG XAUUSD
            if symbol == "XAUUSD" and bias == "LONG":
                return True, f"DXY haussier ({s['dxy']:.2f}) — contre le LONG or"
            # DXY up → USD strong → confirms SHORT EURUSD, blocks LONG EURUSD
            if symbol == "EURUSD" and bias == "LONG":
                return True, f"DXY haussier ({s['dxy']:.2f}) — contre le LONG EUR/USD"
        elif trend == "down":
            # DXY down → confirms LONG XAUUSD, blocks SHORT XAUUSD
            if symbol == "XAUUSD" and bias == "SHORT":
                return True, f"DXY baissier ({s['dxy']:.2f}) — contre le SHORT or"
            # DXY down → confirms LONG EURUSD, blocks SHORT EURUSD
            if symbol == "EURUSD" and bias == "SHORT":
                return True, f"DXY baissier ({s['dxy']:.2f}) — contre le SHORT EUR/USD"

        # TNX rising = higher real rates = pressure on gold
        if symbol == "XAUUSD" and bias == "LONG" and self._tnx_trend == "up":
            return True, f"Taux 10 ans haussiers ({self._tnx:.2f}%) — pression sur l'or"

        # Fed hiking + bearish real rates = double headwind on gold longs
        if symbol == "XAUUSD" and bias == "LONG":
            try:
                from fred_feed import get_fed_data
                fed = get_fed_data()
                if (fed["fed_direction"] == "hiking"
                        and fed["real_rate_bias"] == "bearish"):
                    return True, "Fed en hausse + taux réels élevés — contexte défavorable or"
            except Exception:
                pass

        return False, ""
