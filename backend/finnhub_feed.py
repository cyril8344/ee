"""
Finnhub live economic calendar + forex news.

Requires FINNHUB_API_KEY in .env. If not set, all functions return empty
results gracefully — the existing news_filter.py fallback takes over.

Free tier: 60 calls/min, calendar data available.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import requests

_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"
_NEWS_URL = "https://finnhub.io/api/v1/news"
_REQUEST_TIMEOUT = 8

# Impact label normalisation: Finnhub uses 1/2/3 numeric strings or text
_IMPACT_MAP: Dict[str, str] = {
    "1": "low",
    "2": "medium",
    "3": "high",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class FinnhubFeed:
    """Thin wrapper around the Finnhub REST API for economic calendar events
    and forex news. All public methods return empty lists when the API key is
    absent or any request fails, so callers never need to handle exceptions."""

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("FINNHUB_API_KEY", "")
        # Cache storage: (result, fetched_at_epoch)
        self._events_cache: Optional[tuple[List[dict], float]] = None
        self._news_cache: Optional[tuple[List[dict], float]] = None
        self._events_ttl: int = 300   # 5 minutes
        self._news_ttl: int = 120     # 2 minutes

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Return upcoming economic events for the next *hours_ahead* hours.

        Each item:
            {"time": "HH:MM", "event": "...", "currency": "USD", "impact": "high"}

        Returns [] if the API key is missing or the request fails.
        """
        if not self._api_key:
            return []

        # Serve from cache if fresh
        if self._events_cache is not None:
            cached_result, fetched_at = self._events_cache
            if time.monotonic() - fetched_at < self._events_ttl:
                return cached_result

        now_utc = datetime.now(timezone.utc)
        date_from = now_utc.strftime("%Y-%m-%d")
        date_to = (now_utc + timedelta(hours=hours_ahead)).strftime("%Y-%m-%d")

        try:
            resp = requests.get(
                _CALENDAR_URL,
                params={"from": date_from, "to": date_to, "token": self._api_key},
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": "xau-scalper/1.0"},
            )
            resp.raise_for_status()
            raw = resp.json()
        except (requests.RequestException, ValueError):
            return []

        events: List[Dict[str, Any]] = []
        economic_calendar = raw.get("economicCalendar", []) if isinstance(raw, dict) else (raw or [])
        for item in economic_calendar:
            try:
                event_name = str(item.get("event", "")).strip()
                currency = str(item.get("country", "")).upper().strip()
                impact_raw = str(item.get("impact", "")).lower().strip()
                impact = _IMPACT_MAP.get(impact_raw, "low")

                # time field can be epoch int or ISO string
                time_raw = item.get("time") or item.get("datetime") or ""
                event_time_str = self._parse_event_time(time_raw)
                if event_time_str is None:
                    continue

                if not event_name or not currency:
                    continue

                events.append({
                    "time": event_time_str,
                    "event": event_name,
                    "currency": currency,
                    "impact": impact,
                })
            except (KeyError, TypeError, ValueError):
                continue

        self._events_cache = (events, time.monotonic())
        return events

    def get_forex_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the latest forex news items.

        Each item:
            {"headline": "...", "datetime": <epoch int>, "url": "..."}

        Returns [] if the API key is missing or the request fails.
        """
        if not self._api_key:
            return []

        # Serve from cache if fresh
        if self._news_cache is not None:
            cached_result, fetched_at = self._news_cache
            if time.monotonic() - fetched_at < self._news_ttl:
                return cached_result[:limit]

        try:
            resp = requests.get(
                _NEWS_URL,
                params={"category": "forex", "token": self._api_key},
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": "xau-scalper/1.0"},
            )
            resp.raise_for_status()
            raw = resp.json()
        except (requests.RequestException, ValueError):
            return []

        news_items: List[Dict[str, Any]] = []
        for item in (raw or []):
            try:
                headline = str(item.get("headline", "")).strip()
                dt = item.get("datetime", 0)
                url = str(item.get("url", "")).strip()
                if not headline:
                    continue
                news_items.append({
                    "headline": headline,
                    "datetime": int(dt) if dt else 0,
                    "url": url,
                })
            except (KeyError, TypeError, ValueError):
                continue

        self._news_cache = (news_items, time.monotonic())
        return news_items[:limit]

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _parse_event_time(value: Any) -> Optional[str]:
        """Convert an epoch int or ISO string to 'HH:MM' UTC. Returns None on failure."""
        if not value:
            return None
        try:
            # Numeric epoch (seconds)
            epoch = int(value)
            dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
            return dt.strftime("%H:%M")
        except (ValueError, TypeError, OSError):
            pass
        # ISO string fallback
        try:
            s = str(value).strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%H:%M")
        except (ValueError, TypeError):
            return None


# Module-level singleton — instantiated once, reused across requests.
_feed: Optional[FinnhubFeed] = None


def get_feed() -> FinnhubFeed:
    global _feed
    if _feed is None:
        _feed = FinnhubFeed()
    return _feed
