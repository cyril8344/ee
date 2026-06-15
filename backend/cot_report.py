"""
cot_report.py
=============
Fetch and cache CFTC Disaggregated Commitments of Traders (COT) data
for Gold (XAU/USD) from the public CFTC Socrata API.

Endpoint: https://publicreporting.cftc.gov/resource/jun7-fc8e.json
Cache:    6 hours (data is published weekly on Fridays)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CFTC_URL = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
CACHE_TTL = 6 * 3600  # 6 hours

_cache_data: Optional[List[Dict[str, Any]]] = None
_cache_ts: float = 0.0


# --------------------------------------------------------------------------- #
# Internal fetch
# --------------------------------------------------------------------------- #
def _fetch_raw() -> List[Dict[str, Any]]:
    """Fetch the last 4 COT records for GOLD from CFTC."""
    params = {
        "contract_market_name": "GOLD",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": "4",
    }
    resp = requests.get(CFTC_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key fields from a raw CFTC row."""
    def _int(key: str) -> int:
        try:
            return int(float(row.get(key) or 0))
        except (ValueError, TypeError):
            return 0

    nc_long  = _int("noncomm_positions_long_all")
    nc_short = _int("noncomm_positions_short_all")
    nc_spread = _int("noncomm_positions_spread_all")
    c_long   = _int("comm_positions_long_all")
    c_short  = _int("comm_positions_short_all")

    return {
        "date":      (row.get("report_date_as_yyyy_mm_dd") or "")[:10],
        "nc_long":   nc_long,
        "nc_short":  nc_short,
        "nc_spread": nc_spread,
        "nc_net":    nc_long - nc_short,
        "c_long":    c_long,
        "c_short":   c_short,
        "c_net":     c_long - c_short,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_cot_data() -> Dict[str, Any]:
    """
    Return the latest COT week plus a 4-week history array.
    Result is cached for CACHE_TTL seconds.

    Returns
    -------
    {
        "date": "2024-01-16",
        "nc_long": 123456,
        "nc_short": 45678,
        "nc_spread": 11111,
        "nc_net": 77778,
        "c_long": 55555,
        "c_short": 111111,
        "c_net": -55556,
        "history": [ {...}, {...}, {...}, {...} ]   # newest first
    }
    """
    global _cache_data, _cache_ts

    now = time.time()
    if _cache_data is not None and (now - _cache_ts) < CACHE_TTL:
        return _cache_data  # type: ignore[return-value]

    try:
        raw = _fetch_raw()
    except Exception as exc:
        if _cache_data is not None:
            # Return stale cache rather than crashing
            return _cache_data  # type: ignore[return-value]
        raise RuntimeError(f"COT fetch failed and no cache available: {exc}") from exc

    history = [_parse_row(r) for r in raw]

    if not history:
        raise RuntimeError("COT API returned no data")

    latest = history[0].copy()
    latest["history"] = history

    _cache_data = latest
    _cache_ts = now
    return latest
