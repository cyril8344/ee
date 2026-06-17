"""
backtest.py
===========
Event-driven backtester for the XAU/USD scalping strategy.

- Timeframe : M5 (entries), with M15 & H1 resampled from M5 for the
  multi-timeframe confirmation, exactly as the live engine.
- Costs     : simulated spread (default 0.3 pips) + slippage (0.1 pips).
  For gold, 1 pip = 0.1 price units, so 0.3 pip = 0.03 USD.
- Risk      : % of (compounding) equity per trade.

Outputs an exhaustive result dict consumed by the dashboard / BacktestPanel:
    equity curve, winrate (global + per session), profit factor,
    max drawdown ($ / %), hourly heatmap, avg duration win vs loss, ...

Data is pulled from yfinance ("GC=F" gold futures as XAU/USD proxy, since
yfinance has no spot symbol).  A synthetic generator is used as a fallback
when yfinance is unavailable so the endpoint never hard-fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from strategy import (
    add_indicators, evaluate, active_session,
    MAX_TRADE_MINUTES,
)
from risk_manager import CONTRACT_SIZE, MIN_LOT, LOT_STEP
import data_provider

PIP = 0.1  # 1 pip for gold = 0.1 price units


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    # yfinance may use 'adj close'; ensure required columns
    cols = {"open", "high", "low", "close"}
    if not cols.issubset(df.columns):
        raise ValueError("Missing OHLC columns")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    # ensure tz-aware UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "time"
    return df


def load_m5_data(start: str, end: str) -> pd.DataFrame:
    """Load 5-minute gold data via the unified data provider.

    Uses the configured/real provider (Twelve Data, Polygon, Alpha Vantage,
    yfinance) when a key is available, otherwise falls back to a deterministic
    synthetic series so the backtest never hard-fails.
    """
    try:
        df, _provider = data_provider.get_m5(start=start, end=end, bars=5000)
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass
    return _synthetic_m5(start, end)


def _synthetic_m5(start: str, end: str) -> pd.DataFrame:
    """Deterministic synthetic gold series (geometric random walk) so the
    backtester works fully offline."""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=30)
    idx = pd.date_range(start_dt, end_dt, freq="5min", tz="UTC")
    # only keep weekdays
    idx = idx[idx.weekday < 5]
    n = len(idx)
    rng = np.random.default_rng(42)
    price0 = 2000.0
    rets = rng.normal(0, 0.0009, n)
    # add mild intraday session volatility
    hours = idx.hour.values
    sess_boost = np.where(((hours >= 7) & (hours < 11)) | ((hours >= 13) & (hours < 17)), 1.6, 0.7)
    rets = rets * sess_boost
    close = price0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, 0.6, n)) + 0.2
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[price0], close[:-1]])
    volume = (np.abs(rng.normal(1000, 300, n)) * sess_boost).round()
    df = pd.DataFrame(
        {"open": open_, "high": np.maximum(high, np.maximum(open_, close)),
         "low": np.minimum(low, np.minimum(open_, close)),
         "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "time"
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out


# --------------------------------------------------------------------------- #
# Backtest config & engine
# --------------------------------------------------------------------------- #
@dataclass
class BacktestConfig:
    start: str
    end: str
    capital: float = 10000.0
    risk_pct: float = 1.0
    spread_pips: float = 0.3
    slippage_pips: float = 0.1
    max_trades_per_day: int = 4
    daily_stop_pct: float = 2.0


@dataclass
class BTTrade:
    direction: str
    session: str
    entry_time: datetime
    exit_time: datetime
    entry: float
    exit: float
    stop_loss: float
    tp1: float
    tp2: float
    volume: float
    pnl: float
    pnl_pct: float
    duration_min: float
    exit_reason: str
    entry_hour: int


def _round_lot(lot: float) -> float:
    if lot < MIN_LOT:
        return 0.0
    return round(round(lot / LOT_STEP) * LOT_STEP, 2)


def run_backtest(cfg: BacktestConfig) -> Dict[str, Any]:
    m5_raw = load_m5_data(cfg.start, cfg.end)
    if len(m5_raw) < 250:
        return {"error": "Not enough data for the selected period.",
                "config": cfg.__dict__}

    spread = cfg.spread_pips * PIP
    slippage = cfg.slippage_pips * PIP

    m5 = add_indicators(m5_raw)

    # Precompute the higher-timeframe indicator frames ONCE for the whole
    # period (instead of resampling a trailing window on every M5 bar). At each
    # step we slice these frames up to the current timestamp via searchsorted,
    # which is O(log n) and keeps a 6-12 month backtest fast. This also mirrors
    # the live engine, which feeds the strategy the full resampled history.
    m15_full = add_indicators(resample(m5_raw, "15min"))
    h1_full = add_indicators(resample(m5_raw, "60min"))
    m15_index = m15_full.index
    h1_index = h1_full.index

    equity = cfg.capital
    equity_curve: List[Dict[str, Any]] = [
        {"ts": m5.index[0].isoformat(), "equity": round(equity, 2)}
    ]
    trades: List[BTTrade] = []

    open_trade: Optional[Dict[str, Any]] = None

    # per-day risk state
    cur_day = None
    trades_today = 0
    day_start_equity = equity
    day_blocked = False

    warmup = 210  # need EMA200 on M5
    timestamps = m5.index

    for i in range(warmup, len(m5)):
        bar = m5.iloc[i]
        ts = timestamps[i]
        day = ts.date()

        # new day reset
        if day != cur_day:
            cur_day = day
            trades_today = 0
            day_start_equity = equity
            day_blocked = False

        # ---- Manage open position on this bar ----
        if open_trade is not None:
            exit_info = _try_exit(open_trade, bar, ts, slippage)
            if exit_info is not None:
                pnl, exit_price, reason = exit_info
                equity += pnl
                t = open_trade
                dur = (ts - t["entry_time"]).total_seconds() / 60.0
                trades.append(BTTrade(
                    direction=t["direction"], session=t["session"],
                    entry_time=t["entry_time"], exit_time=ts.to_pydatetime(),
                    entry=t["entry"], exit=exit_price,
                    stop_loss=t["stop_loss"], tp1=t["tp1"], tp2=t["tp2"],
                    volume=t["volume"], pnl=pnl,
                    pnl_pct=pnl / day_start_equity * 100.0,
                    duration_min=dur, exit_reason=reason,
                    entry_hour=t["entry_time"].astimezone(timezone.utc).hour,
                ))
                equity_curve.append({"ts": ts.isoformat(), "equity": round(equity, 2)})
                # daily stop check
                day_pnl = equity - day_start_equity
                if day_pnl <= -abs(day_start_equity * cfg.daily_stop_pct / 100.0):
                    day_blocked = True
                open_trade = None

        # ---- Look for a new entry (only if flat & allowed) ----
        if open_trade is None and not day_blocked and trades_today < cfg.max_trades_per_day:
            # Slice precomputed higher-timeframe frames up to the current time.
            j15 = m15_index.searchsorted(ts, side="right") - 1
            j1h = h1_index.searchsorted(ts, side="right") - 1
            if j15 < 2 or j1h < 0:
                sig = None
            else:
                sig = evaluate(
                    m5.iloc[: i + 1],
                    m15_full.iloc[: j15 + 1],
                    h1_full.iloc[: j1h + 1],
                    now=ts.to_pydatetime(),
                    check_session=True,
                )
            if sig is not None:
                vol = _round_lot(
                    (equity * cfg.risk_pct / 100.0)
                    / (sig.risk_distance * CONTRACT_SIZE)
                )
                if vol >= MIN_LOT:
                    # apply spread+slippage to entry (buy at ask / sell at bid)
                    if sig.direction == "long":
                        fill = sig.entry + spread + slippage
                    else:
                        fill = sig.entry - spread - slippage
                    open_trade = {
                        "direction": sig.direction,
                        "session": sig.session,
                        "entry_time": ts.to_pydatetime(),
                        "entry": fill,
                        "stop_loss": sig.stop_loss,
                        "tp1": sig.take_profit1,
                        "tp2": sig.take_profit2,
                        "volume": vol,
                        "tp1_done": False,
                        "remaining": vol,
                        "realised": 0.0,
                        "max_exit_time": ts.to_pydatetime() + timedelta(minutes=MAX_TRADE_MINUTES),
                    }
                    trades_today += 1

    return _build_report(cfg, trades, equity_curve)


def _try_exit(t: Dict[str, Any], bar, ts, slippage) -> Optional[tuple]:
    """
    Partial-TP exit model:
      - TP1 hit -> realise 60% of position, move on (remaining 40% runs to TP2/SL/timeout)
      - TP2 hit -> realise the rest
      - SL hit  -> realise the rest at stop
      - timeout -> realise the rest at close
    Returns (total_pnl_for_closed_portion, representative_exit_price, reason)
    only when the position is *fully* closed.  Otherwise records partial and
    returns None (position stays open).
    """
    direction = t["direction"]
    high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
    vol = t["volume"]
    sign = 1.0 if direction == "long" else -1.0

    def pnl_for(price, lots):
        return (price - t["entry"]) * sign * CONTRACT_SIZE * lots

    # 1) TP1 (partial 60%)
    if not t["tp1_done"]:
        hit_tp1 = (high >= t["tp1"]) if direction == "long" else (low <= t["tp1"])
        if hit_tp1:
            lots60 = _round_lot(vol * 0.6)
            lots60 = min(lots60, t["remaining"])
            fill = t["tp1"] - slippage * sign  # slight adverse slippage
            t["realised"] += pnl_for(fill, lots60)
            t["remaining"] = round(t["remaining"] - lots60, 2)
            t["tp1_done"] = True
            if t["remaining"] < MIN_LOT:
                return t["realised"], t["tp1"], "tp1"

    # 2) Stop loss (whole remaining)
    hit_sl = (low <= t["stop_loss"]) if direction == "long" else (high >= t["stop_loss"])
    if hit_sl:
        fill = t["stop_loss"] - slippage * sign
        t["realised"] += pnl_for(fill, t["remaining"])
        return t["realised"], t["stop_loss"], ("sl" if not t["tp1_done"] else "sl_after_tp1")

    # 3) TP2 (whole remaining)
    hit_tp2 = (high >= t["tp2"]) if direction == "long" else (low <= t["tp2"])
    if t["tp1_done"] and hit_tp2:
        fill = t["tp2"] - slippage * sign
        t["realised"] += pnl_for(fill, t["remaining"])
        return t["realised"], t["tp2"], "tp2"

    # 4) Timeout (force exit at close)
    if ts.to_pydatetime() >= t["max_exit_time"]:
        t["realised"] += pnl_for(close, t["remaining"])
        return t["realised"], close, "timeout"

    return None


# --------------------------------------------------------------------------- #
# Reporting / statistics
# --------------------------------------------------------------------------- #
def _build_report(cfg: BacktestConfig, trades: List[BTTrade],
                  equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    pnls = np.array([t.pnl for t in trades]) if n else np.array([])
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    gross_profit = float(sum(t.pnl for t in wins))
    gross_loss = float(abs(sum(t.pnl for t in losses)))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    final_equity = equity_curve[-1]["equity"] if equity_curve else cfg.capital
    total_return = final_equity - cfg.capital

    # Max drawdown
    eq = np.array([p["equity"] for p in equity_curve]) if equity_curve else np.array([cfg.capital])
    running_max = np.maximum.accumulate(eq)
    drawdowns = eq - running_max
    max_dd_usd = float(drawdowns.min()) if len(drawdowns) else 0.0
    dd_pct_series = drawdowns / running_max * 100.0
    max_dd_pct = float(dd_pct_series.min()) if len(dd_pct_series) else 0.0

    # Per-session winrate
    def _winrate(subset):
        if not subset:
            return 0.0
        w = sum(1 for t in subset if t.pnl > 0)
        return round(w / len(subset) * 100.0, 2)

    london = [t for t in trades if t.session == "London"]
    newyork = [t for t in trades if t.session == "NewYork"]

    # Hourly heatmap (UTC entry hour -> stats)
    heatmap = []
    for h in range(24):
        ht = [t for t in trades if t.entry_hour == h]
        if ht:
            heatmap.append({
                "hour": h,
                "trades": len(ht),
                "pnl": round(float(sum(t.pnl for t in ht)), 2),
                "winrate": _winrate(ht),
            })

    best_hour = max(heatmap, key=lambda x: x["pnl"], default=None)
    worst_hour = min(heatmap, key=lambda x: x["pnl"], default=None)

    avg_dur_win = round(float(np.mean([t.duration_min for t in wins])), 1) if wins else 0.0
    avg_dur_loss = round(float(np.mean([t.duration_min for t in losses])), 1) if losses else 0.0

    return {
        "config": cfg.__dict__,
        "summary": {
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "winrate": _winrate(trades),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else None,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(total_return, 2),
            "net_profit_pct": round(total_return / cfg.capital * 100.0, 2),
            "final_equity": round(final_equity, 2),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "avg_win": round(float(np.mean([t.pnl for t in wins])), 2) if wins else 0.0,
            "avg_loss": round(float(np.mean([t.pnl for t in losses])), 2) if losses else 0.0,
            "expectancy": round(float(pnls.mean()), 2) if n else 0.0,
        },
        "by_session": {
            "London": {"trades": len(london), "winrate": _winrate(london),
                       "pnl": round(float(sum(t.pnl for t in london)), 2)},
            "NewYork": {"trades": len(newyork), "winrate": _winrate(newyork),
                        "pnl": round(float(sum(t.pnl for t in newyork)), 2)},
        },
        "duration": {
            "avg_win_min": avg_dur_win,
            "avg_loss_min": avg_dur_loss,
        },
        "heatmap": heatmap,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "equity_curve": equity_curve,
        "trades": [
            {
                "direction": t.direction,
                "session": t.session,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry": round(t.entry, 3),
                "exit": round(t.exit, 3),
                "volume": t.volume,
                "pnl": round(t.pnl, 2),
                "duration_min": round(t.duration_min, 1),
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ],
    }


if __name__ == "__main__":
    import json
    cfg = BacktestConfig(
        start=(datetime.utcnow() - timedelta(days=45)).strftime("%Y-%m-%d"),
        end=datetime.utcnow().strftime("%Y-%m-%d"),
    )
    report = run_backtest(cfg)
    print(json.dumps(report.get("summary", report), indent=2))
