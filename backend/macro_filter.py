"""
macro_filter.py
===============
Macro market filters: DXY (Dollar Index) and VIX (Fear Index).

DXY: Gold is inversely correlated with USD. DXY uptrend → block LONG gold.
VIX: VIX > 25 means extreme fear → scalping too dangerous → block all entries.
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

            self._fetched_at = now
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh()
        vix_blocked = self._vix is not None and self._vix > VIX_BLOCK_THRESHOLD
        return {
            "dxy": self._dxy,
            "dxy_trend": self._dxy_trend,
            "vix": self._vix,
            "vix_blocked": vix_blocked,
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

        return False, ""
