"""Tests for the economic-calendar news filter (offline fallback)."""
from datetime import datetime, timedelta, timezone

from news_filter import NewsFilter, NewsEvent


def test_fallback_events_present_when_offline():
    nf = NewsFilter(window_minutes=30)
    # Force fallback by clearing remote and regenerating
    events = nf._fallback_events()
    titles = " ".join(e.title.lower() for e in events)
    assert "nfp" in titles or "payroll" in titles
    assert "cpi" in titles
    assert "fomc" in titles


def test_window_blocks_around_event():
    nf = NewsFilter(window_minutes=30)
    now = datetime.now(timezone.utc)
    nf._events = [NewsEvent("FOMC Statement", now + timedelta(minutes=10), "USD", "high")]
    assert nf.is_blocked(now) is True
    # outside the +/-30min window
    assert nf.is_blocked(now + timedelta(minutes=120)) is False


def test_next_event_and_countdown():
    nf = NewsFilter(window_minutes=30)
    now = datetime.now(timezone.utc)
    nf._events = [
        NewsEvent("CPI", now + timedelta(hours=2), "USD", "high"),
        NewsEvent("NFP", now + timedelta(days=1), "USD", "high"),
    ]
    nxt = nf.next_event(now)
    assert nxt is not None and nxt.title == "CPI"
    status = nf.status(now)
    assert status["next_event_countdown_sec"] > 0
    assert status["blocked"] is False


def test_is_major_classification():
    nf = NewsFilter()
    assert nf._is_major("US Non-Farm Payrolls") is True
    assert nf._is_major("Core CPI m/m") is True
    assert nf._is_major("FOMC Press Conference") is True
    assert nf._is_major("Random low-impact survey") is False
