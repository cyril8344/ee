"""
data_provider.py
================
Unified 5-minute data layer for multiple markets.

Providers (selected via the XAU_DATA_PROVIDER env var, or auto):
    - "twelvedata"   : Twelve Data REST API  (key: TWELVEDATA_API_KEY)
    - "polygon"      : Polygon.io aggregates  (key: POLYGON_API_KEY)
    - "alphavantage" : Alpha Vantage FX intraday (key: ALPHAVANTAGE_API_KEY)
    - "yfinance"     : Yahoo Finance proxy (no key, ~60d M5 limit)
    - "synthetic"    : deterministic offline generator (always works)

Resolution order when XAU_DATA_PROVIDER is unset ("auto"):
    twelvedata -> polygon -> alphavantage -> yfinance -> synthetic
(only providers whose API key is present are attempted).

All providers return a tz-aware (UTC) DataFrame indexed by time with columns:
    open, high, low, close, volume

Set credentials in a `.env` file (see .env.example) or the real environment.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import numpy as np
import pandas as pd
import requests

REQUEST_TIMEOUT = 12

# --------------------------------------------------------------------------- #
# Throttle global des appels TwelveData
# Le plan gratuit autorise ~8 requêtes/min. On espace TOUS les appels (live +
# workers d'arrière-plan) d'un intervalle minimum pour ne jamais dépasser le
# quota et éviter les replis en données synthétiques.
# --------------------------------------------------------------------------- #
import threading as _threading
import time as _time_mod

_TD_MIN_INTERVAL = float(os.environ.get("TWELVEDATA_MIN_INTERVAL", "8.0"))  # secondes
_td_throttle_lock = _threading.Lock()
_td_last_call = [0.0]


def _td_throttle() -> None:
    """Bloque jusqu'à ce que l'intervalle minimum depuis le dernier appel soit écoulé."""
    with _td_throttle_lock:
        now = _time_mod.monotonic()
        wait = _TD_MIN_INTERVAL - (now - _td_last_call[0])
        if wait > 0:
            _time_mod.sleep(wait)
        _td_last_call[0] = _time_mod.monotonic()

# --------------------------------------------------------------------------- #
# Disk cache for historical M5 data
# Avoids re-downloading on every backtest/optimize run.
# Files stored in backend/data_cache/ as parquet.
# --------------------------------------------------------------------------- #
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
_CACHE_MAX_AGE_HOURS = 6  # refresh after 6 hours


def _cache_key(symbol: str, start: str, end: str) -> str:
    raw = f"{symbol}_{start}_{end}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{key}.parquet")


def _cache_load(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        path = _cache_path(_cache_key(symbol, start, end))
        if not os.path.exists(path):
            return None
        age_hours = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_hours > _CACHE_MAX_AGE_HOURS:
            return None
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception:
        return None


def _cache_save(symbol: str, start: str, end: str, df: pd.DataFrame) -> None:
    try:
        path = _cache_path(_cache_key(symbol, start, end))
        df.to_parquet(path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# .env loader (zero-dependency; avoids requiring python-dotenv)
# --------------------------------------------------------------------------- #
def _load_dotenv() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # don't override an already-set real env var
                os.environ.setdefault(key, value)
    except OSError:
        pass


_load_dotenv()

MARKET_SYMBOLS = {
    "XAUUSD": {
        "twelvedata": "XAU/USD",
        "polygon": "X:XAUUSD",
        "alphavantage": ("XAU", "USD"),
        "yfinance": "GC=F",
        "synthetic_price": 2000.0,
        "synthetic_vol": 0.0009,
        "synthetic_spread": 0.4,
    },
    "EURUSD": {
        "twelvedata": "EUR/USD",
        "polygon": "C:EURUSD",
        "alphavantage": ("EUR", "USD"),
        "yfinance": "EURUSD=X",
        "synthetic_price": 1.08,
        "synthetic_vol": 0.00015,
        "synthetic_spread": 0.00004,
    },
}

# Keep backward-compatible alias
SYMBOL_MAP = {
    "twelvedata": os.environ.get("XAU_TD_SYMBOL", "XAU/USD"),
    "polygon": os.environ.get("XAU_POLY_SYMBOL", "X:XAUUSD"),
    "alphavantage": ("XAU", "USD"),
    "yfinance": os.environ.get("XAU_YF_SYMBOL", "GC=F"),
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "time"
    return df.sort_index()


# --------------------------------------------------------------------------- #
# Provider implementations
# --------------------------------------------------------------------------- #
def _fetch_twelvedata(start: Optional[str], end: Optional[str], bars: int,
                      symbol: str = "XAUUSD") -> pd.DataFrame:
    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY not set")
    td_symbol = MARKET_SYMBOLS.get(symbol, MARKET_SYMBOLS["XAUUSD"])["twelvedata"]
    params = {
        "symbol": td_symbol, "interval": "5min",
        "apikey": key, "format": "JSON", "timezone": "UTC",
        "outputsize": min(max(bars, 1), 5000),
    }
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    _td_throttle()
    r = requests.get("https://api.twelvedata.com/time_series",
                     params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"TwelveData error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return _normalise(df)


def _fetch_twelvedata_range(start: str, end: str, symbol: str = "XAUUSD") -> pd.DataFrame:
    """Fetch a multi-year M5 dataset from Twelve Data using paginated requests.

    Each API call returns at most 5000 bars (≈17 days of M5). This function
    walks backwards in time from *end* to *start*, collecting chunks and
    concatenating them into a single sorted, deduplicated DataFrame.
    """
    import time as _time

    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY not set")

    td_symbol = MARKET_SYMBOLS.get(symbol, MARKET_SYMBOLS["XAUUSD"])["twelvedata"]

    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")

    chunks = []
    cursor = end_dt

    while cursor > start_dt:
        params = {
            "symbol": td_symbol, "interval": "5min",
            "apikey": key, "format": "JSON", "timezone": "UTC",
            "outputsize": 5000,
            "end_date": cursor.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _td_throttle()
        r = requests.get("https://api.twelvedata.com/time_series",
                         params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            break
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        chunk = _normalise(df)
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        earliest = chunk.index.min()
        if earliest <= start_dt:
            break
        cursor = earliest - pd.Timedelta(minutes=5)
        _time.sleep(0.5)

    if not chunks:
        raise RuntimeError("No data fetched from Twelve Data")

    result = pd.concat(chunks).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result = result[result.index >= start_dt]
    result = result[result.index <= end_dt]
    return result


def _fetch_polygon(start: Optional[str], end: Optional[str], bars: int,
                   symbol: str = "XAUUSD") -> pd.DataFrame:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    if not end:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not start:
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    sym = MARKET_SYMBOLS.get(symbol, MARKET_SYMBOLS["XAUUSD"])["polygon"]
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/5/minute/"
           f"{start}/{end}")
    r = requests.get(url, params={"adjusted": "true", "sort": "asc",
                                  "limit": 50000, "apiKey": key},
                     timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    results = data.get("results")
    if not results:
        raise RuntimeError(f"Polygon: no results ({data.get('status')})")
    df = pd.DataFrame(results)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("time").rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return _normalise(df)


def _fetch_alphavantage(start: Optional[str], end: Optional[str], bars: int,
                        symbol: str = "XAUUSD") -> pd.DataFrame:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    from_sym, to_sym = MARKET_SYMBOLS.get(symbol, MARKET_SYMBOLS["XAUUSD"])["alphavantage"]
    r = requests.get("https://www.alphavantage.co/query", params={
        "function": "FX_INTRADAY", "from_symbol": from_sym,
        "to_symbol": to_sym, "interval": "5min", "outputsize": "full",
        "apikey": key,
    }, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    ts_key = next((k for k in data if "Time Series" in k), None)
    if ts_key is None:
        raise RuntimeError(f"AlphaVantage error: {data.get('Note') or data.get('Error Message') or data}")
    rows = data[ts_key]
    df = pd.DataFrame(rows).T
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename(columns={
        "1. open": "open", "2. high": "high",
        "3. low": "low", "4. close": "close"})
    df["volume"] = 0.0
    out = _normalise(df)
    if start:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index <= pd.Timestamp(end, tz="UTC")]
    return out


def _fetch_yfinance(start: Optional[str], end: Optional[str], bars: int,
                    symbol: str = "XAUUSD") -> pd.DataFrame:
    import yfinance as yf
    sym = MARKET_SYMBOLS.get(symbol, MARKET_SYMBOLS["XAUUSD"])["yfinance"]
    if start and end:
        data = yf.download(sym, start=start, end=end, interval="5m",
                           progress=False, auto_adjust=False)
    else:
        data = yf.download(sym, period="5d", interval="5m",
                           progress=False, auto_adjust=False)
    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned no data")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return _normalise(data)


def _fetch_synthetic(start: Optional[str], end: Optional[str], bars: int,
                     symbol: str = "XAUUSD") -> pd.DataFrame:
    cfg = MARKET_SYMBOLS.get(symbol, MARKET_SYMBOLS["XAUUSD"])
    base_price = cfg["synthetic_price"]
    vol = cfg["synthetic_vol"]
    spread_scale = cfg["synthetic_spread"]

    if start:
        start_dt = pd.Timestamp(start, tz="UTC")
    else:
        start_dt = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(days=7)
    if end:
        end_dt = pd.Timestamp(end, tz="UTC")
    else:
        end_dt = pd.Timestamp.now(tz="UTC").floor("5min")
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=30)
    idx = pd.date_range(start_dt, end_dt, freq="5min", tz="UTC")
    idx = idx[idx.weekday < 5]
    n = len(idx)
    if n == 0:
        idx = pd.date_range(start_dt, start_dt + pd.Timedelta(days=5), freq="5min", tz="UTC")
        n = len(idx)
    # deterministic seed from start so backtests are reproducible
    seed = int(start_dt.timestamp()) // 300
    rng = np.random.default_rng(seed if seed else 42)
    rets = rng.normal(0, vol, n)
    hours = idx.hour.values
    boost = np.where(((hours >= 7) & (hours < 11)) | ((hours >= 13) & (hours < 17)), 1.6, 0.7)
    rets *= boost
    close = base_price * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, spread_scale * 1.5, n)) + spread_scale * 0.5
    open_ = np.concatenate([[base_price], close[:-1]])
    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum(close + spread, np.maximum(open_, close)),
        "low": np.minimum(close - spread, np.minimum(open_, close)),
        "close": close,
        "volume": (np.abs(rng.normal(1000, 300, n)) * boost).round(),
    }, index=idx)
    df.index.name = "time"
    return df


_PROVIDERS = {
    "twelvedata": _fetch_twelvedata,
    "polygon": _fetch_polygon,
    "alphavantage": _fetch_alphavantage,
    "yfinance": _fetch_yfinance,
    "synthetic": _fetch_synthetic,
}

_AUTO_ORDER = ["twelvedata", "polygon", "alphavantage", "yfinance", "synthetic"]

_KEY_ENV = {
    "twelvedata": "TWELVEDATA_API_KEY",
    "polygon": "POLYGON_API_KEY",
    "alphavantage": "ALPHAVANTAGE_API_KEY",
}


def available_providers() -> List[str]:
    """Providers usable right now (key present, or keyless)."""
    out = []
    for name in _AUTO_ORDER:
        env = _KEY_ENV.get(name)
        if env is None or os.environ.get(env):
            out.append(name)
    return out


def get_m5(start: Optional[str] = None, end: Optional[str] = None,
           bars: int = 500, symbol: str = "XAUUSD") -> tuple[pd.DataFrame, str]:
    """
    Return (dataframe, provider_name_used).

    Honors XAU_DATA_PROVIDER if set to a concrete provider; otherwise walks the
    auto order and falls back through providers until one succeeds. Synthetic is
    the guaranteed last resort so callers never get an empty result.
    """
    chosen = os.environ.get("XAU_DATA_PROVIDER", "auto").strip().lower()
    if chosen and chosen != "auto" and chosen in _PROVIDERS:
        order = [chosen]
        if chosen != "synthetic":
            order.append("synthetic")  # safety net
    else:
        order = [p for p in _AUTO_ORDER
                 if _KEY_ENV.get(p) is None or os.environ.get(_KEY_ENV[p])]
        if "synthetic" not in order:
            order.append("synthetic")

    # Disk cache for long-range requests (backtest / optimizer)
    _is_range = bool(start and end)
    if _is_range:
        cached = _cache_load(symbol, start, end)
        if cached is not None and len(cached) > 0:
            return cached, "cache"

    # For long-range requests via twelvedata, use the paginated range fetcher
    _use_td_range = (
        start and end
        and (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).days > 30
        and (chosen == "twelvedata" or (chosen == "auto" and os.environ.get("TWELVEDATA_API_KEY")))
    )
    if _use_td_range and "twelvedata" in order:
        try:
            df = _fetch_twelvedata_range(start, end, symbol)
            if df is not None and len(df) > 0:
                if _is_range:
                    _cache_save(symbol, start, end, df)
                return df, "twelvedata"
        except Exception:  # noqa: BLE001 - fall through to normal providers
            pass

    last_err = None
    for name in order:
        try:
            df = _PROVIDERS[name](start, end, bars, symbol)
            if df is not None and len(df) > 0:
                if _is_range:
                    _cache_save(symbol, start, end, df)
                return df, name
        except Exception as e:  # noqa: BLE001 - intentional fallthrough
            last_err = e
            continue
    # absolute last resort
    return _fetch_synthetic(start, end, bars, symbol), "synthetic"


if __name__ == "__main__":
    print("Available providers:", available_providers())
    df, used = get_m5(bars=50)
    print(f"Provider used: {used} | rows: {len(df)}")
    print(df.tail(3))
