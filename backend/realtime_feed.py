"""
realtime_feed.py
================
Twelve Data WebSocket real-time price feed.

Connects to wss://ws.twelvedata.com/v1/quotes/price and streams live ticks
for XAU/USD and EUR/USD into a module-level dict that the REST API can read.

If TWELVEDATA_API_KEY is not set, the module is a no-op (no error raised).

Public interface
----------------
    start_feed()          -> None   Launch daemon thread (call once at startup).
    get_latest(symbol)    -> Optional[dict]   Latest tick or None.
    is_connected()        -> bool   True while the WebSocket is open.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_latest: Dict[str, dict] = {}
_connected: bool = False
_lock = threading.Lock()
_started: bool = False

_WS_URL_TEMPLATE = "wss://ws.twelvedata.com/v1/quotes/price?apikey={apikey}"
_SYMBOLS = "XAU/USD,EUR/USD"
_RECONNECT_BASE = 5      # seconds
_RECONNECT_MAX = 60      # seconds cap for exponential backoff


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def start_feed() -> None:
    """Launch the WebSocket daemon thread (call once at startup)."""
    global _started
    apikey = os.environ.get("TWELVEDATA_API_KEY", "")
    if not apikey:
        logger.info("realtime_feed: TWELVEDATA_API_KEY not set — feed disabled.")
        return
    if _started:
        return
    _started = True
    t = threading.Thread(target=_run_feed, args=(apikey,), daemon=True, name="twelvedata-ws")
    t.start()
    logger.info("realtime_feed: WebSocket thread started.")


def get_latest(symbol: str) -> Optional[dict]:
    """Return latest tick dict or None.

    Tick format: {"price": float, "timestamp": int, "symbol": str}
    """
    with _lock:
        return _latest.get(symbol)


def is_connected() -> bool:
    """Return True if the WebSocket is currently open."""
    return _connected


# ---------------------------------------------------------------------------
# Internal feed logic
# ---------------------------------------------------------------------------

def _set_connected(value: bool) -> None:
    global _connected
    with _lock:
        _connected = value


def _handle_message(raw: str) -> None:
    """Parse a Twelve Data price event and store it."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return

    event = msg.get("event")

    if event == "price":
        symbol = msg.get("symbol")
        price_str = msg.get("price")
        ts = msg.get("timestamp")
        if symbol and price_str is not None:
            try:
                price = float(price_str)
            except (TypeError, ValueError):
                return
            tick = {"price": price, "timestamp": ts, "symbol": symbol}
            with _lock:
                _latest[symbol] = tick
            logger.debug("realtime_feed: %s @ %s", symbol, price)

    elif event == "subscribe-status":
        status = msg.get("status")
        logger.info("realtime_feed: subscribe-status = %s", status)

    elif event == "heartbeat":
        pass  # keep-alive, nothing to do

    else:
        logger.debug("realtime_feed: unhandled event type: %s", event)


def _run_feed(apikey: str) -> None:
    """Main reconnect loop — runs forever in a daemon thread."""
    # Import here so the module-level no-op path doesn't require websocket-client.
    try:
        import websocket  # type: ignore[import]
    except ImportError:
        logger.error(
            "realtime_feed: websocket-client is not installed. "
            "Run: pip install websocket-client"
        )
        return

    backoff = _RECONNECT_BASE

    while True:
        url = _WS_URL_TEMPLATE.format(apikey=apikey)
        ws_app: Optional[websocket.WebSocketApp] = None

        def on_open(ws: websocket.WebSocketApp) -> None:
            nonlocal backoff
            backoff = _RECONNECT_BASE  # reset on successful connect
            _set_connected(True)
            logger.info("realtime_feed: WebSocket connected.")
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "params": {"symbols": _SYMBOLS},
            })
            ws.send(subscribe_msg)

        def on_message(ws: websocket.WebSocketApp, message: str) -> None:
            _handle_message(message)

        def on_error(ws: websocket.WebSocketApp, error: Exception) -> None:
            logger.warning("realtime_feed: WebSocket error: %s", error)

        def on_close(
            ws: websocket.WebSocketApp,
            close_status_code: Optional[int],
            close_msg: Optional[str],
        ) -> None:
            _set_connected(False)
            logger.info(
                "realtime_feed: WebSocket closed (code=%s msg=%s).",
                close_status_code,
                close_msg,
            )

        try:
            ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws_app.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as exc:
            logger.warning("realtime_feed: run_forever raised: %s", exc)
        finally:
            _set_connected(False)

        logger.info("realtime_feed: reconnecting in %ds…", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, _RECONNECT_MAX)
