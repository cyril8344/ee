"""
news_filter.py
==============
Economic-calendar filter for the XAU/USD scalping bot.

Goal: block trading in a window (default +/- 30 minutes) around major,
high-impact USD events that move gold violently:
    - NFP (Non-Farm Payrolls)
    - CPI (Consumer Price Index)
    - FOMC (rate decisions / statements / press conferences)
    - PCE, PPI, FOMC minutes, Fed Chair speeches (also high impact)

Data sources
------------
1. Primary: a public economic-calendar JSON feed (best effort, network).
2. Fallback: a deterministic generator for recurring events when the feed is
   unavailable (offline / rate-limited).  This guarantees the filter still
   works without network access — important for safety.

All event times are normalised to timezone-aware UTC datetimes.
"""

from __future__ import annotations

import calendar
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time, date
from typing import List, Optional, Dict, Any

import requests

# High-impact keywords used to classify events as "major".
MAJOR_KEYWORDS = (
    "non-farm", "nonfarm", "non farm", "nfp", "payroll",
    "cpi", "consumer price",
    "fomc", "federal funds", "interest rate decision", "rate decision",
    "fed chair", "powell", "press conference",
    "pce", "ppi", "producer price",
)

# Public feed (nfs.faireconomy.media) — ForexFactory-compatible weekly JSON.
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
REQUEST_TIMEOUT = 8


@dataclass
class NewsEvent:
    title: str
    time_utc: datetime
    currency: str
    impact: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "time_utc": self.time_utc.isoformat(),
            "currency": self.currency,
            "impact": self.impact,
        }


class NewsFilter:
    def __init__(self, window_minutes: int = 30, currencies=("USD",)):
        self.window = timedelta(minutes=window_minutes)
        self.currencies = tuple(c.upper() for c in currencies)
        self._events: List[NewsEvent] = []
        self._last_refresh: Optional[datetime] = None
        self._lock = threading.Lock()
        self.refresh()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def refresh(self, force: bool = False) -> None:
        """Refresh events at most once per hour unless forced."""
        now = datetime.now(timezone.utc)
        with self._lock:
            if (not force and self._last_refresh
                    and now - self._last_refresh < timedelta(hours=1)):
                return
            events = self._fetch_remote()
            if not events:
                events = self._fallback_events()
            # keep only future-ish (last 1 day .. next 14 days) major events
            horizon_start = now - timedelta(days=1)
            horizon_end = now + timedelta(days=14)
            self._events = [
                e for e in events
                if horizon_start <= e.time_utc <= horizon_end
            ]
            self._events.sort(key=lambda e: e.time_utc)
            self._last_refresh = now

    def _fetch_remote(self) -> List[NewsEvent]:
        try:
            resp = requests.get(
                CALENDAR_URL,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "xau-scalper/1.0"},
            )
            resp.raise_for_status()
            raw = resp.json()
        except (requests.RequestException, ValueError):
            return []

        events: List[NewsEvent] = []
        for item in raw:
            try:
                currency = str(item.get("country", "")).upper()
                impact = str(item.get("impact", "")).lower()
                title = str(item.get("title", ""))
                date_str = item.get("date")  # ISO 8601 w/ tz offset
                if not date_str:
                    continue
                dt = self._parse_iso(date_str)
                if dt is None:
                    continue
                if currency not in self.currencies:
                    continue
                if impact != "high" and not self._is_major(title):
                    continue
                events.append(NewsEvent(title, dt, currency, impact or "high"))
            except (KeyError, TypeError, ValueError):
                continue
        # Keep only the ones we consider major.
        return [e for e in events if self._is_major(e.title) or e.impact == "high"]

    @staticmethod
    def _parse_iso(date_str: str) -> Optional[datetime]:
        s = date_str.strip()
        # Handle trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _is_major(title: str) -> bool:
        t = title.lower()
        return any(k in t for k in MAJOR_KEYWORDS)

    # ------------------------------------------------------------------ #
    # Fallback recurring-event generator (offline safety net)
    # ------------------------------------------------------------------ #
    def _fallback_events(self) -> List[NewsEvent]:
        """
        Generate plausible recurring high-impact events for the current and
        next month.  Times are typical US release times in UTC:
            - NFP   : first Friday, 12:30 UTC (08:30 ET)
            - CPI   : ~10th business-ish day, 12:30 UTC
            - FOMC  : approximated mid-month Wednesday, 18:00 UTC
        These are conservative blocks; the real feed overrides them when
        available.
        """
        out: List[NewsEvent] = []
        base = datetime.now(timezone.utc)
        for month_offset in (0, 1):
            y = base.year + (base.month - 1 + month_offset) // 12
            m = (base.month - 1 + month_offset) % 12 + 1

            # NFP : first Friday 12:30 UTC
            first_friday = self._first_weekday(y, m, calendar.FRIDAY)
            out.append(NewsEvent(
                "Non-Farm Payrolls (NFP)",
                datetime(y, m, first_friday, 12, 30, tzinfo=timezone.utc),
                "USD", "high",
            ))

            # CPI : second Wednesday 12:30 UTC (approximation)
            second_wed = self._nth_weekday(y, m, calendar.WEDNESDAY, 2)
            out.append(NewsEvent(
                "Consumer Price Index (CPI)",
                datetime(y, m, second_wed, 12, 30, tzinfo=timezone.utc),
                "USD", "high",
            ))

            # FOMC : third Wednesday 18:00 UTC (approximation, ~8/year)
            third_wed = self._nth_weekday(y, m, calendar.WEDNESDAY, 3)
            out.append(NewsEvent(
                "FOMC Statement & Rate Decision",
                datetime(y, m, third_wed, 18, 0, tzinfo=timezone.utc),
                "USD", "high",
            ))
        return out

    @staticmethod
    def _first_weekday(year: int, month: int, weekday: int) -> int:
        for day in range(1, 8):
            if date(year, month, day).weekday() == weekday:
                return day
        return 1

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> int:
        count = 0
        days_in_month = calendar.monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):
            if date(year, month, day).weekday() == weekday:
                count += 1
                if count == n:
                    return day
        return min(28, days_in_month)

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def is_blocked(self, at: Optional[datetime] = None) -> bool:
        return self.active_event(at) is not None

    def active_event(self, at: Optional[datetime] = None) -> Optional[NewsEvent]:
        """Return the major event whose +/- window currently contains `at`."""
        now = at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        for e in self._events:
            if abs((e.time_utc - now).total_seconds()) <= self.window.total_seconds():
                return e
        return None

    def next_event(self, at: Optional[datetime] = None) -> Optional[NewsEvent]:
        now = at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        upcoming = [e for e in self._events if e.time_utc >= now]
        return upcoming[0] if upcoming else None

    def status(self, at: Optional[datetime] = None) -> Dict[str, Any]:
        now = at or datetime.now(timezone.utc)
        active = self.active_event(now)
        nxt = self.next_event(now)
        countdown = None
        if nxt:
            countdown = int((nxt.time_utc - now).total_seconds())
        return {
            "blocked": active is not None,
            "active_event": active.to_dict() if active else None,
            "next_event": nxt.to_dict() if nxt else None,
            "next_event_countdown_sec": countdown,
            "window_minutes": int(self.window.total_seconds() // 60),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
        }


if __name__ == "__main__":
    nf = NewsFilter()
    import json
    print(json.dumps(nf.status(), indent=2))
