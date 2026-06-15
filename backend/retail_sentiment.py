"""
retail_sentiment.py
===================
Retail trader sentiment for XAU/USD and EUR/USD.

Primary source : Myfxbook community outlook (HTML scraping)
Fallback        : COT-derived proxy for XAUUSD, static 50/50 for EURUSD

Cache: 15 minutes
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

import requests

# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
CACHE_TTL = 15 * 60  # 15 minutes

_cache_data: Optional[Dict[str, Any]] = None
_cache_ts: float = 0.0

# --------------------------------------------------------------------------- #
# Myfxbook scraper
# --------------------------------------------------------------------------- #
MYFXBOOK_URL = "https://www.myfxbook.com/community/outlook"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.myfxbook.com/",
}

# Symbol aliases Myfxbook may use
_SYMBOL_VARIANTS = {
    "XAUUSD": ["XAUUSD", "XAU/USD", "Gold"],
    "EURUSD": ["EURUSD", "EUR/USD"],
}


def _scrape_myfxbook() -> Dict[str, Any]:
    """
    Scrape community outlook percentages from Myfxbook.
    Returns dict keyed by canonical symbol name.
    Raises on any failure so the caller can fall back.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 not installed — pip install beautifulsoup4")

    resp = requests.get(MYFXBOOK_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    result: Dict[str, Any] = {}

    # Myfxbook renders a table with rows containing symbol name + long%/short%
    # We look for rows where a cell text matches our symbol variants.
    rows = soup.find_all("tr")
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 3:
            continue
        cell_text = " ".join(cells[:3]).upper()

        matched_sym = None
        for sym, variants in _SYMBOL_VARIANTS.items():
            if any(v.upper() in cell_text for v in variants):
                matched_sym = sym
                break
        if matched_sym is None:
            continue

        # Extract first two percentage values found in the row
        pcts = re.findall(r"(\d{1,3}(?:\.\d{1,2})?)\s*%", " ".join(cells))
        if len(pcts) >= 2:
            long_pct = float(pcts[0])
            short_pct = float(pcts[1])
            result[matched_sym] = {
                "long_pct": round(long_pct, 1),
                "short_pct": round(short_pct, 1),
                "source": "myfxbook",
            }

    if not result:
        raise RuntimeError("Could not parse any sentiment from Myfxbook HTML")

    return result


# --------------------------------------------------------------------------- #
# COT-based fallback
# --------------------------------------------------------------------------- #
def _cot_fallback() -> Dict[str, Any]:
    """Derive rough retail sentiment from COT data (inverse of commercials)."""
    try:
        from cot_report import get_cot_data
        cot = get_cot_data()
        nc_net = cot.get("nc_net", 0)
        # Retail typically mirrors non-commercial positioning
        if nc_net > 0:
            xau_long = 60.0
        elif nc_net < 0:
            xau_long = 40.0
        else:
            xau_long = 50.0
    except Exception:
        xau_long = 50.0

    return {
        "XAUUSD": {
            "long_pct": round(xau_long, 1),
            "short_pct": round(100.0 - xau_long, 1),
            "source": "cot_proxy",
        },
        "EURUSD": {
            "long_pct": 50.0,
            "short_pct": 50.0,
            "source": "static_fallback",
        },
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_sentiment() -> Dict[str, Any]:
    """
    Return retail sentiment for XAUUSD and EURUSD.
    Tries Myfxbook first; falls back to COT proxy on any error.
    Result is cached for CACHE_TTL seconds.

    Returns
    -------
    {
        "XAUUSD": {"long_pct": 65.2, "short_pct": 34.8, "source": "myfxbook"},
        "EURUSD": {"long_pct": 48.1, "short_pct": 51.9, "source": "myfxbook"}
    }
    """
    global _cache_data, _cache_ts

    now = time.time()
    if _cache_data is not None and (now - _cache_ts) < CACHE_TTL:
        return _cache_data  # type: ignore[return-value]

    try:
        data = _scrape_myfxbook()
        # If scraping only partially succeeded, fill missing symbols from fallback
        if "XAUUSD" not in data or "EURUSD" not in data:
            fallback = _cot_fallback()
            for sym in ("XAUUSD", "EURUSD"):
                if sym not in data:
                    data[sym] = fallback[sym]
    except Exception:
        data = _cot_fallback()

    _cache_data = data
    _cache_ts = now
    return data
