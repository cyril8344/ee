"""
fred_feed.py
============
FRED (St. Louis Fed) + World Gold Council data feed.

Signals for XAU/USD trading:
- Fed rate direction: cutting = bullish gold, hiking = bearish gold
- Real interest rate (10Y - inflation breakeven): negative = bullish gold
- Central bank net gold purchases: buying = structural support

Environment variable:
  FRED_API_KEY  — free key from https://fred.stlouisfed.org/docs/api/api_key.html
                  Without it, only WGC/proxy data is available.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

# ── Constants ──────────────────────────────────────────────────────────────────
_FRED_BASE  = "https://api.stlouisfed.org/fred/series/observations"
_CACHE_TTL  = 3600      # 1 h for daily FRED data
_CB_TTL     = 21600     # 6 h for central bank data (monthly updates)

_cache:    Dict[str, Any]   = {}
_cache_ts: Dict[str, float] = {}


# ── Cache helper ───────────────────────────────────────────────────────────────
def _cached(key: str, fn, ttl: int = _CACHE_TTL) -> Any:
    now = time.time()
    if key in _cache and now - _cache_ts.get(key, 0) < ttl:
        return _cache[key]
    val = fn()
    if val is not None:
        _cache[key] = val
        _cache_ts[key] = now
    return val


# ── FRED fetch ─────────────────────────────────────────────────────────────────
def _fred(series_id: str, limit: int = 10) -> Optional[List[Dict]]:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return None
    params = urllib.parse.urlencode({
        "series_id":  series_id,
        "api_key":    api_key,
        "file_type":  "json",
        "sort_order": "desc",
        "limit":      limit,
    })
    url = f"{_FRED_BASE}?{params}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "xauusd-scalp-bot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return [o for o in data.get("observations", [])
                if o.get("value", ".") != "."]
    except Exception:
        return None


def _fval(obs: Optional[List[Dict]], idx: int = 0) -> Optional[float]:
    try:
        return float(obs[idx]["value"])  # type: ignore[index]
    except (TypeError, IndexError, ValueError):
        return None


# ── Fed data ───────────────────────────────────────────────────────────────────
def get_fed_data() -> Dict[str, Any]:
    """
    Returns:
      fed_rate        float | None   current target rate (upper bound)
      fed_direction   "cutting" | "hiking" | "hold" | "unknown"
      real_rate       float | None   10Y nominal - inflation breakeven
      real_rate_bias  "bullish" | "bearish" | "neutral"
      dgs10           float | None
      t10yie          float | None
      source          "FRED" | "unavailable"
    """
    def _fetch() -> Optional[Dict]:
        # Fed target upper bound (daily)
        fedfunds = _fred("DFEDTARU", 10)
        dgs10    = _fred("DGS10", 3)
        t10yie   = _fred("T10YIE", 3)

        if fedfunds is None:
            return None

        current = _fval(fedfunds, 0)
        # compare to 8 observations ago ≈ 8 business days ≈ ~2 weeks
        prev = _fval(fedfunds, min(8, len(fedfunds) - 1))

        direction = "hold"
        if current is not None and prev is not None:
            if current < prev - 0.01:
                direction = "cutting"
            elif current > prev + 0.01:
                direction = "hiking"

        dgs10_val  = _fval(dgs10)
        t10yie_val = _fval(t10yie)

        real_rate = None
        real_bias = "neutral"
        if dgs10_val is not None and t10yie_val is not None:
            real_rate = round(dgs10_val - t10yie_val, 3)
            if real_rate < 0:
                real_bias = "bullish"   # negative real rates = gold friendly
            elif real_rate > 1.5:
                real_bias = "bearish"   # high positive real rates = headwind

        return {
            "fed_rate":       current,
            "fed_direction":  direction,
            "real_rate":      real_rate,
            "real_rate_bias": real_bias,
            "dgs10":          dgs10_val,
            "t10yie":         t10yie_val,
            "source":         "FRED",
        }

    result = _cached("fed_data", _fetch)
    if result is None:
        has_key = bool(os.environ.get("FRED_API_KEY", "").strip())
        return {
            "fed_rate":       None,
            "fed_direction":  "unknown",
            "real_rate":      None,
            "real_rate_bias": "neutral",
            "dgs10":          None,
            "t10yie":         None,
            "source":         "no_key" if not has_key else "error",
        }
    return result


# ── Central bank gold ──────────────────────────────────────────────────────────
def get_cb_gold() -> Dict[str, Any]:
    """
    Returns central bank gold purchase trend.
    Primary:  World Gold Council page scrape
    Fallback: gold price momentum vs USD (FRED proxy)
    """
    def _fetch() -> Optional[Dict]:
        # ── Attempt 1: WGC page ─────────────────────────────────────────────
        try:
            from bs4 import BeautifulSoup
            req = urllib.request.Request(
                "https://www.gold.org/goldhub/data/central-bank-statistics",
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True).lower()

            buy_kw  = ["net buying", "net purchase", "added", "increased reserve"]
            sell_kw = ["net selling", "net sale", "reduced reserve", "sold"]
            bs = sum(text.count(k) for k in buy_kw)
            ss = sum(text.count(k) for k in sell_kw)

            trend = "neutral"
            if bs > ss + 1:
                trend = "buying"
            elif ss > bs + 1:
                trend = "selling"

            return {"trend": trend, "source": "WGC",
                    "note": "Banques centrales — achats nets (WGC)"}
        except Exception:
            pass

        # ── Attempt 2: FRED gold-price proxy ────────────────────────────────
        gold_obs = _fred("GOLDAMGBD228NLBM", 40)
        if gold_obs:
            try:
                prices = [float(o["value"]) for o in gold_obs
                          if o.get("value", ".") != "."]
                if len(prices) >= 20:
                    recent = sum(prices[:5]) / 5
                    older  = sum(prices[15:25]) / 10
                    trend  = ("buying"  if recent > older * 1.02 else
                               "selling" if recent < older * 0.98 else "neutral")
                    return {
                        "trend":  trend,
                        "source": "FRED proxy",
                        "note":   "Estimé via momentum prix or",
                    }
            except Exception:
                pass

        return {"trend": "unknown", "source": "unavailable", "note": ""}

    result = _cached("cb_gold", _fetch, ttl=_CB_TTL)
    return result or {"trend": "unknown", "source": "unavailable", "note": ""}


# ── Composite gold macro bias ──────────────────────────────────────────────────
def get_gold_macro_bias() -> Dict[str, Any]:
    """
    Composite bias for XAU/USD:
      score > 0 → bullish macro context
      score < 0 → bearish macro context
    """
    fed = get_fed_data()
    cb  = get_cb_gold()

    score   = 0
    signals: List[str] = []

    # 1. Fed direction
    if fed["fed_direction"] == "cutting":
        score += 2
        signals.append("Fed: baisses de taux → haussier or")
    elif fed["fed_direction"] == "hiking":
        score -= 2
        signals.append("Fed: hausses de taux → baissier or")
    elif fed["fed_direction"] == "hold":
        signals.append("Fed: taux stables")

    # 2. Real interest rate
    rr = fed.get("real_rate")
    if fed["real_rate_bias"] == "bullish":
        score += 2
        signals.append(
            f"Taux réels négatifs ({rr:.2f}%) → haussier or"
            if rr is not None else "Taux réels négatifs → haussier or"
        )
    elif fed["real_rate_bias"] == "bearish":
        score -= 1
        signals.append(
            f"Taux réels élevés ({rr:.2f}%) → pression sur l'or"
            if rr is not None else "Taux réels élevés → pression sur l'or"
        )

    # 3. Central bank positioning
    if cb["trend"] == "buying":
        score += 1
        signals.append("Banques centrales: acheteuses nettes → support structurel")
    elif cb["trend"] == "selling":
        score -= 1
        signals.append("Banques centrales: vendeuses nettes → pression")

    bias = "bullish" if score >= 2 else ("bearish" if score <= -2 else "neutral")

    return {
        "score":          score,
        "bias":           bias,
        "signals":        signals,
        "fed":            fed,
        "central_banks":  cb,
    }
