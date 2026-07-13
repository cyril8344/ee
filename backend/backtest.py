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
    MAX_TRADE_MINUTES, batch_signals,
    SL_ATR_MULT, last_swing_low, last_swing_high,
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


def load_m5_data(start: str, end: str, symbol: str = "XAUUSD") -> pd.DataFrame:
    """Load 5-minute gold data via the unified data provider.

    Uses the configured/real provider (Twelve Data, Polygon, Alpha Vantage,
    yfinance) when a key is available, otherwise falls back to a deterministic
    synthetic series so the backtest never hard-fails.
    """
    try:
        df, _provider = data_provider.get_m5(start=start, end=end, bars=5000, symbol=symbol)
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
    rets = rng.normal(0, 0.0004, n)
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
    spread_pips: float = 0.5   # 0.5 pip = spread conservateur XAU/USD (réaliste broker standard)
    slippage_pips: float = 0.1  # $0.01 de slippage par côté
    max_trades_per_day: int = 4
    daily_stop_pct: float = 2.0
    symbol: str = "XAUUSD"
    strategy_mode: str = "standard"  # "standard" | "ict"


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
    mae: float = 0.0   # Maximum Adverse Excursion  (en unités de prix)
    mfe: float = 0.0   # Maximum Favorable Excursion (en unités de prix)
    risk: float = 0.0  # distance SL initiale (pour normaliser en R)


def _round_lot(lot: float) -> float:
    if lot < MIN_LOT:
        return 0.0
    return round(round(lot / LOT_STEP) * LOT_STEP, 2)


def run_backtest(cfg: BacktestConfig, preloaded_data: "pd.DataFrame | None" = None) -> Dict[str, Any]:
    m5_raw = preloaded_data if preloaded_data is not None else load_m5_data(cfg.start, cfg.end, symbol=cfg.symbol)
    if len(m5_raw) < 250:
        return {"error": "Not enough data for the selected period.",
                "config": cfg.__dict__}

    contract_size = 100.0 if cfg.symbol == "XAUUSD" else 100000.0
    pip_size = 0.1 if cfg.symbol == "XAUUSD" else 0.0001

    spread = cfg.spread_pips * pip_size
    slippage = cfg.slippage_pips * pip_size

    m5 = add_indicators(m5_raw)

    # Pre-compute higher-TF frames ONCE
    m15_full = add_indicators(resample(m5_raw, "15min"))
    h1_full  = add_indicators(resample(m5_raw, "60min"))
    h4_full  = add_indicators(resample(m5_raw, "240min"))
    m15_index = m15_full.index
    h1_index  = h1_full.index

    # ── Signal pre-computation ────────────────────────────────────────────────
    # Standard mode: vectorised O(n) lookup via batch_signals().
    # ICT mode: per-bar evaluate_ict() — slower but correct for context-heavy logic.
    if cfg.strategy_mode in ("ict", "B"):
        from strategy_ict import evaluate_ict as _evaluate_ict
        precomputed_signals = None
    else:
        precomputed_signals = batch_signals(m5, m15_full, h1_full, h4=h4_full, check_session=True)
        _evaluate_ict = None

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
            exit_info = _try_exit(open_trade, bar, ts, slippage, contract_size)
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
                    mae=t.get("mae", 0.0),
                    mfe=t.get("mfe", 0.0),
                    risk=t.get("risk", 0.0),
                ))
                equity_curve.append({"ts": ts.isoformat(), "equity": round(equity, 2)})
                # daily stop check
                day_pnl = equity - day_start_equity
                if day_pnl <= -abs(day_start_equity * cfg.daily_stop_pct / 100.0):
                    day_blocked = True
                open_trade = None

        # ---- Look for a new entry (only if flat & allowed) ----
        if open_trade is None and not day_blocked and trades_today < cfg.max_trades_per_day:
            if cfg.strategy_mode in ("ict", "B"):
                # Pre-filter by session (London 8-12 CET / NY 14-18 CET) before
                # the expensive per-bar evaluate_ict() call — reduces ~26k bars to ~4k.
                from strategy import CET as _CET
                _hour = ts.astimezone(_CET).hour
                if (_hour < 8 or 12 <= _hour < 14 or _hour >= 18):
                    sig = None
                else:
                    m15_s = m15_full.iloc[:m15_full.index.searchsorted(ts, side="right")]
                    h1_s  = h1_full.iloc[:h1_full.index.searchsorted(ts, side="right")]
                    sig = _evaluate_ict(
                        m5.iloc[:i + 1], m15_s, h1_s,
                        now=ts.to_pydatetime(), check_session=True,
                    )
            else:
                direction = precomputed_signals.get(ts)
                if direction is None:
                    sig = None
                else:
                    bar_atr = float(bar.get("atr", pip_size * 10))
                    entry_p = float(bar["close"])
                    if direction == "long":
                        swing  = last_swing_low(m5.iloc[:i+1], lookback=10)
                        raw_sl = min(swing, entry_p - 1e-6)
                        sl     = max(raw_sl, entry_p - SL_ATR_MULT * bar_atr)
                    else:
                        swing  = last_swing_high(m5.iloc[:i+1], lookback=10)
                        raw_sl = max(swing, entry_p + 1e-6)
                        sl     = min(raw_sl, entry_p + SL_ATR_MULT * bar_atr)
                    risk = abs(entry_p - sl)
                    if risk <= 0:
                        sig = None
                    else:
                        from types import SimpleNamespace
                        sig = SimpleNamespace(
                            direction=direction,
                            entry=entry_p,
                            stop_loss=sl,
                            take_profit1=entry_p + 0.7*risk if direction == "long" else entry_p - 0.7*risk,
                            take_profit2=entry_p + 1.8*risk if direction == "long" else entry_p - 1.8*risk,
                            risk_distance=risk,
                            session=active_session(ts.to_pydatetime()) or "London",
                            max_duration_min=MAX_TRADE_MINUTES,
                        )

            if sig is not None:
                vol = _round_lot(
                    (equity * cfg.risk_pct / 100.0)
                    / (sig.risk_distance * contract_size)
                )
                if vol >= MIN_LOT:
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
                        "max_exit_time": ts.to_pydatetime() + timedelta(minutes=sig.max_duration_min),
                        "risk": sig.risk_distance,
                        "mae": 0.0,
                        "mfe": 0.0,
                    }
                    trades_today += 1

    return _build_report(cfg, trades, equity_curve)


def _try_exit(t: Dict[str, Any], bar, ts, slippage, contract_size: float) -> Optional[tuple]:
    """
    Split-TP exit model: 60% à TP1 (0.7R), SL → ATR trailing (TP1 − TRAIL_ATR_MULT×ATR), puis 40% à TP2 (1.4R).
    Returns (pnl, exit_price, reason) quand la position est fermée, sinon None.
    """
    direction = t["direction"]
    high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
    sign = 1.0 if direction == "long" else -1.0

    def pnl_for(price, lots):
        return (price - t["entry"]) * sign * contract_size * lots

    # MAE / MFE : mise à jour barre par barre
    if direction == "long":
        t["mfe"] = max(t.get("mfe", 0.0), high - t["entry"])
        t["mae"] = max(t.get("mae", 0.0), t["entry"] - low)
    else:
        t["mfe"] = max(t.get("mfe", 0.0), t["entry"] - low)
        t["mae"] = max(t.get("mae", 0.0), high - t["entry"])

    # Early exit: 15 min (3 bougies M5) sans conviction — MFE < 0.2R
    if not t["tp1_done"] and t.get("risk", 0) > 0:
        bars_elapsed = int((ts.to_pydatetime() - t["entry_time"]).total_seconds() / 300)
        if bars_elapsed >= 3 and t["mfe"] / t["risk"] < 0.2:
            t["realised"] += pnl_for(close, t["remaining"])
            return t["realised"], close, "early_exit"

    # 1) TP1 — sortie 50% (ou 100% si tp1_close_all) à 0.7R
    if not t["tp1_done"]:
        hit_tp1 = (high >= t["tp1"]) if direction == "long" else (low <= t["tp1"])
        if hit_tp1:
            close_ratio = 1.0 if t.get("tp1_close_all") else 0.5
            lots50 = _round_lot(t["volume"] * close_ratio)
            lots50 = min(lots50, t["remaining"])
            if lots50 < MIN_LOT:
                lots50 = t["remaining"]  # trop petit pour spliter → close total à TP1
            fill = t["tp1"] - slippage * sign
            t["realised"] += pnl_for(fill, lots50)
            t["remaining"] = round(t["remaining"] - lots50, 2)
            t["tp1_done"] = True
            if t["remaining"] < MIN_LOT:
                return t["realised"], t["tp1"], "tp1"
            # Déplacer SL à l'entrée (breakeven) si demandé par la stratégie
            # Return None pour ne pas checker le SL sur la même bougie que TP1
            if t.get("be_after_tp1"):
                t["stop_loss"] = t["entry"]
                return None

    # 2) Stop loss
    hit_sl = (low <= t["stop_loss"]) if direction == "long" else (high >= t["stop_loss"])
    if hit_sl:
        fill = t["stop_loss"] - slippage * sign
        t["realised"] += pnl_for(fill, t["remaining"])
        return t["realised"], t["stop_loss"], "sl" if not t["tp1_done"] else "sl_after_tp1"

    # 3) TP2 — sortie 50% restants à 1.0R
    if t["tp1_done"]:
        hit_tp2 = (high >= t["tp2"]) if direction == "long" else (low <= t["tp2"])
        if hit_tp2:
            fill = t["tp2"] - slippage * sign
            t["realised"] += pnl_for(fill, t["remaining"])
            return t["realised"], t["tp2"], "tp2"

    # 4) Timeout
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

    # MAE / MFE en multiples de R (normalisation par distance SL initiale)
    def _r(val, risk):
        return round(val / risk, 3) if risk > 0 else 0.0

    trades_with_risk = [t for t in trades if t.risk > 0]
    mae_r_all   = [_r(t.mae, t.risk) for t in trades_with_risk]
    mfe_r_all   = [_r(t.mfe, t.risk) for t in trades_with_risk]
    mae_r_wins  = [_r(t.mae, t.risk) for t in wins  if t.risk > 0]
    mfe_r_wins  = [_r(t.mfe, t.risk) for t in wins  if t.risk > 0]
    mae_r_loss  = [_r(t.mae, t.risk) for t in losses if t.risk > 0]
    mfe_r_loss  = [_r(t.mfe, t.risk) for t in losses if t.risk > 0]

    excursion = {
        "avg_mae_r":       round(float(np.mean(mae_r_all)),  3) if mae_r_all  else 0.0,
        "avg_mfe_r":       round(float(np.mean(mfe_r_all)),  3) if mfe_r_all  else 0.0,
        "avg_mae_r_wins":  round(float(np.mean(mae_r_wins)), 3) if mae_r_wins else 0.0,
        "avg_mfe_r_wins":  round(float(np.mean(mfe_r_wins)), 3) if mfe_r_wins else 0.0,
        "avg_mae_r_loss":  round(float(np.mean(mae_r_loss)), 3) if mae_r_loss else 0.0,
        "avg_mfe_r_loss":  round(float(np.mean(mfe_r_loss)), 3) if mfe_r_loss else 0.0,
        # Losers qui ont déjà atteint 0.5R avant de perdre = signal ok mais sortie prématurée
        "pct_loss_mfe_gt_half_r": round(
            sum(1 for v in mfe_r_loss if v >= 0.5) / len(mfe_r_loss) * 100, 1
        ) if mfe_r_loss else 0.0,
        # Gagnants avec MAE > 0.5R = entrée trop tôt, trade risqué avant d'être gagnant
        "pct_win_mae_gt_half_r": round(
            sum(1 for v in mae_r_wins if v >= 0.5) / len(mae_r_wins) * 100, 1
        ) if mae_r_wins else 0.0,
    }

    # Sharpe ratio — raw per-trade (no annualisation).
    # Annualising with sqrt(252) gives absurd values with <30 trades, so we
    # show the raw ratio: positive = profitable pattern, negative = losing.
    # Typical good strategies land between 0.3 and 1.5.
    if n > 1:
        returns = pnls / cfg.capital
        std = float(np.std(returns))
        sharpe = float(np.mean(returns) / std) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "config": cfg.__dict__,
        "summary": {
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "winrate": _winrate(trades),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else None,
            "sharpe_ratio": round(sharpe, 3),
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
        "excursion": excursion,
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
                "mae_r": _r(t.mae, t.risk),
                "mfe_r": _r(t.mfe, t.risk),
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
