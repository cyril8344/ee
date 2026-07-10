"""
pretrain_es.py
==============
Pré-entraînement de la stratégie Order Flow ES (S&P 500 E-mini).

Rejoue les trades bar-par-bar sur données historiques ES=F (yfinance)
en utilisant strategy_es.py comme moteur de signal.

Pas de ML Gate, pas de pattern recognition : uniquement le signal
volume-absorption + filtres EMA/RSI/session, exact reflet de la logique live
(où le signal vient du DOMScanner NinjaTrader au lieu du proxy volume).

Usage :
    from pretrain_es import run_pretrain_es, launch_pretrain_es, get_progress_es
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import yfinance as yf

import strategy_es as strat

# --------------------------------------------------------------------------- #
# État de progression (partagé avec l'API)
# --------------------------------------------------------------------------- #
_progress: Dict[str, Any] = {
    "running":     False,
    "pct":         0,
    "bars_done":   0,
    "bars_total":  0,
    "trades":      0,
    "wins":        0,
    "status":      "idle",
    "last_result": None,
    "error":       None,
}
_lock = threading.Lock()


def get_progress_es() -> Dict[str, Any]:
    with _lock:
        return dict(_progress)


def _set(**kwargs):
    with _lock:
        _progress.update(kwargs)


# --------------------------------------------------------------------------- #
# Chargement des données ES=F
# --------------------------------------------------------------------------- #
def _load_es_data(start: str, end: str) -> pd.DataFrame:
    """Télécharge les données ES=F depuis yfinance, fallback synthétique."""
    try:
        tk = yf.Ticker("ES=F")
        df = tk.history(start=start, end=end, interval="5m", auto_adjust=True)
        if df is not None and len(df) > 100:
            df = df.rename(columns=str.lower)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df.dropna()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index.name = "time"
            return df
    except Exception:
        pass
    return _synthetic_es(start, end)


def _synthetic_es(start: str, end: str) -> pd.DataFrame:
    """Données synthétiques ES pour les tests offline."""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC")
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=30)
    idx = pd.date_range(start_dt, end_dt, freq="5min", tz="UTC")
    idx = idx[idx.weekday < 5]
    n   = len(idx)
    rng = np.random.default_rng(99)
    price0 = 5800.0
    rets   = rng.normal(0, 0.0003, n)
    # Volatilité session RTH (14h30–21h UTC = 9h30–16h ET)
    hours  = idx.hour.values
    boost  = np.where((hours >= 14) & (hours < 21), 1.8, 0.6)
    rets   = rets * boost
    close  = price0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, 1.5, n)) + 0.5
    high   = close + spread
    low    = close - spread
    open_  = np.concatenate([[price0], close[:-1]])
    volume = (np.abs(rng.normal(2000, 500, n)) * boost).round()
    df = pd.DataFrame(
        {"open":   open_,
         "high":   np.maximum(high, np.maximum(open_, close)),
         "low":    np.minimum(low,  np.minimum(open_, close)),
         "close":  close,
         "volume": volume},
        index=idx,
    )
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------- #
# Simulation de sortie de trade ES
# --------------------------------------------------------------------------- #
MAX_TRADE_BARS = 9   # 45 min en M5 = 9 bougies

def _try_exit_es(trade: dict, bar: pd.Series, ts) -> Optional[tuple]:
    """
    Retourne (pnl_dollars, exit_price, exit_reason) ou None si encore ouvert.
    La simulation se fait sur les valeurs OHLC de la bougie (approximation event-driven).
    """
    direction   = trade["direction"]
    sl          = trade["sl"]
    tp1         = trade["tp1"]
    tp2         = trade["tp2"]
    entry       = trade["entry"]
    entry_time  = trade["entry_time"]
    contracts   = trade["contracts"]
    sl_hit_tp1  = trade.get("sl_hit_tp1", False)   # SL monté à BE après TP1

    high = float(bar.get("high", 0) or 0)
    low  = float(bar.get("low",  0) or 0)

    duration_bars = trade.get("duration_bars", 0) + 1
    trade["duration_bars"] = duration_bars

    tick = strat.TICK_SIZE
    tv   = strat.TICK_VALUE

    # --- Partial exit à TP1 (50% de la position) ---
    if not trade.get("tp1_hit", False):
        if direction == "long"  and high >= tp1:
            trade["tp1_hit"] = True
            # PnL partiel = 50% contrats × distance en ticks × tick value
            pnl_part = 0.5 * contracts * round((tp1 - entry) / tick) * tv
            trade["partial_pnl"] = pnl_part
            # Monter SL à entry (breakeven) après TP1
            trade["sl"] = entry
            trade["sl_hit_tp1"] = True
            sl = entry
        elif direction == "short" and low <= tp1:
            trade["tp1_hit"] = True
            pnl_part = 0.5 * contracts * round((entry - tp1) / tick) * tv
            trade["partial_pnl"] = pnl_part
            trade["sl"] = entry
            trade["sl_hit_tp1"] = True
            sl = entry

    # Màj MAE/MFE
    if direction == "long":
        mfe = high - entry
        mae = entry - low
    else:
        mfe = entry - low
        mae = high - entry
    trade["mfe"] = max(trade.get("mfe", 0.0), mfe)
    trade["mae"] = max(trade.get("mae", 0.0), mae)

    partial_pnl = trade.get("partial_pnl", 0.0)
    remaining   = 0.5 if trade.get("tp1_hit", False) else 1.0
    r_contracts = contracts * remaining

    def _pnl(exit_p: float) -> float:
        if direction == "long":
            ticks = round((exit_p - entry) / tick)
        else:
            ticks = round((entry - exit_p) / tick)
        return partial_pnl + r_contracts * ticks * tv

    # --- TP2 ---
    if direction == "long"  and high >= tp2:
        reason = "tp2" if trade.get("tp1_hit") else "tp_direct"
        return _pnl(tp2), tp2, reason
    if direction == "short" and low  <= tp2:
        reason = "tp2" if trade.get("tp1_hit") else "tp_direct"
        return _pnl(tp2), tp2, reason

    # --- SL ---
    if direction == "long"  and low  <= sl:
        reason = "sl_after_tp1" if trade.get("tp1_hit") else "sl"
        return _pnl(sl), sl, reason
    if direction == "short" and high >= sl:
        reason = "sl_after_tp1" if trade.get("tp1_hit") else "sl"
        return _pnl(sl), sl, reason

    # --- Timeout ---
    if duration_bars >= MAX_TRADE_BARS:
        mid = float(bar.get("close", entry) or entry)
        reason = "timeout_tp1" if trade.get("tp1_hit") else "timeout"
        return _pnl(mid), mid, reason

    return None


# --------------------------------------------------------------------------- #
# Moteur principal
# --------------------------------------------------------------------------- #
def run_pretrain_es(
    start:     str,
    end:       str,
    params:    Optional[dict] = None,
    capital:   float = 50_000.0,
    risk_pct:  float = 1.0,
) -> Dict[str, Any]:
    """Lance le pré-entraînement ES en mode bloquant (appelé depuis un thread)."""
    _set(running=True, pct=0, bars_done=0, trades=0, wins=0,
         status="running", error=None, last_result=None)
    try:
        _set(status="Chargement données ES=F…")
        raw = _load_es_data(start, end)
        if len(raw) < 250:
            raise ValueError("Pas assez de données pour la période sélectionnée.")

        data_start = raw.index[0].isoformat()[:10]
        data_end   = raw.index[-1].isoformat()[:10]
        bars_total = len(raw)

        df = strat.add_indicators(raw, params)

        warmup     = max(int((params or {}).get("ema_trend", 200)) + 10, 210)
        total      = len(df) - warmup

        open_trade    = None
        n_trades      = 0
        n_wins        = 0
        equity        = float(capital)
        equity_curve  = [{"ts": df.index[warmup].isoformat(), "equity": equity}]
        trades_log    = []
        pnl_wins      = []
        pnl_losses    = []
        _day_trades: Dict[str, int] = {}
        MAX_DAY = 4

        _set(bars_total=total, status="Analyse des trades…")

        for i in range(warmup, len(df)):
            ts  = df.index[i]
            bar = df.iloc[i]

            # Progression
            done = i - warmup
            if done % 100 == 0:
                _set(pct=round(done / total * 100), bars_done=done,
                     trades=n_trades, wins=n_wins)

            # ── Gérer le trade ouvert ──────────────────────────────────────────
            if open_trade is not None:
                exit_info = _try_exit_es(open_trade, bar, ts)
                if exit_info is not None:
                    pnl, exit_price, exit_reason = exit_info
                    won = pnl > 0
                    n_trades += 1
                    equity   += pnl
                    equity_curve.append({"ts": ts.isoformat(), "equity": round(equity, 2)})

                    if won:
                        n_wins += 1
                        pnl_wins.append(pnl)
                    else:
                        pnl_losses.append(abs(pnl))

                    risk  = open_trade.get("risk", 1.0)
                    mae_r = round(open_trade.get("mae", 0) / risk, 3) if risk > 0 else 0
                    mfe_r = round(open_trade.get("mfe", 0) / risk, 3) if risk > 0 else 0

                    trades_log.append({
                        "entry_ts":   open_trade["entry_time"].isoformat() if hasattr(open_trade["entry_time"], "isoformat") else str(open_trade["entry_time"]),
                        "exit_ts":    ts.isoformat(),
                        "direction":  open_trade["direction"],
                        "entry":      round(open_trade["entry"], 2),
                        "exit_price": round(float(exit_price), 2),
                        "exit_reason": exit_reason,
                        "pnl":        round(pnl, 2),
                        "won":        won,
                        "mae_r":      mae_r,
                        "mfe_r":      mfe_r,
                        "contracts":  open_trade["contracts"],
                        "rsi":        open_trade.get("rsi", 0),
                        "atr":        open_trade.get("atr", 0),
                        "vol_ratio":  open_trade.get("vol_ratio", 0),
                        "bias":       open_trade.get("bias", ""),
                    })
                    open_trade = None

            # ── Chercher une entrée ────────────────────────────────────────────
            if open_trade is not None:
                continue

            day_key = ts.strftime("%Y-%m-%d")
            if _day_trades.get(day_key, 0) >= MAX_DAY:
                continue

            sig = strat.evaluate(
                df.iloc[:i + 1],
                params=params,
                ts=ts.to_pydatetime(),
            )
            if sig is None:
                continue

            _day_trades[day_key] = _day_trades.get(day_key, 0) + 1

            bias      = sig["bias"]
            entry_p   = sig["entry"]
            sl_p      = sig["stop_loss"]
            tp1_p     = sig["take_profit1"]
            tp2_p     = sig["take_profit2"]
            sl_dist   = abs(entry_p - sl_p)

            contracts = strat.size_contracts(equity, risk_pct, int((params or {}).get("sl_ticks", 8)))
            risk_usd  = int((params or {}).get("sl_ticks", 8)) * strat.TICK_VALUE * contracts

            open_trade = {
                "direction":  "long" if bias == "LONG" else "short",
                "entry_time": ts.to_pydatetime(),
                "entry":      entry_p,
                "sl":         sl_p,
                "tp1":        tp1_p,
                "tp2":        tp2_p,
                "contracts":  contracts,
                "risk":       risk_usd,
                "rsi":        sig.get("rsi", 0),
                "atr":        sig.get("atr", 0),
                "vol_ratio":  sig.get("vol_ratio", 0),
                "bias":       bias,
                "mae":        0.0,
                "mfe":        0.0,
                "tp1_hit":    False,
                "partial_pnl": 0.0,
                "duration_bars": 0,
            }

        # ── Calcul métriques finales ──────────────────────────────────────────
        n_losses       = n_trades - n_wins
        win_rate       = round(n_wins / n_trades * 100, 1) if n_trades else 0
        gross_win      = sum(pnl_wins)
        gross_loss     = sum(pnl_losses)
        profit_factor  = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0
        total_pnl      = round(gross_win - gross_loss, 2)
        avg_win        = round(gross_win / len(pnl_wins),    2) if pnl_wins   else 0
        avg_loss       = round(gross_loss / len(pnl_losses), 2) if pnl_losses else 0

        # Drawdown max
        eq_vals  = [e["equity"] for e in equity_curve]
        peak     = eq_vals[0]
        max_dd   = 0.0
        for e in eq_vals:
            peak   = max(peak, e)
            max_dd = max(max_dd, peak - e)
        max_dd_pct = round(max_dd / capital * 100, 1) if capital > 0 else 0

        # SL direct %
        sl_direct = sum(1 for t in trades_log if t["exit_reason"] == "sl")
        sl_pct    = round(sl_direct / n_trades * 100, 1) if n_trades else 0

        # TP2 %
        tp2_count = sum(1 for t in trades_log if t["exit_reason"] == "tp2")
        tp2_pct   = round(tp2_count / n_trades * 100, 1) if n_trades else 0

        # Distribution par bias
        long_trades  = [t for t in trades_log if t["direction"] == "long"]
        short_trades = [t for t in trades_log if t["direction"] == "short"]
        long_wins    = sum(1 for t in long_trades  if t["won"])
        short_wins   = sum(1 for t in short_trades if t["won"])

        # Distribution horaire
        hourly: Dict[int, Dict[str, int]] = {}
        for t in trades_log:
            try:
                h = datetime.fromisoformat(t["entry_ts"].replace("Z", "+00:00")).hour
            except Exception:
                h = 0
            import pytz
            try:
                et_h = (datetime.fromisoformat(t["entry_ts"].replace("Z", "+00:00"))
                        .astimezone(pytz.timezone("America/New_York")).hour)
            except Exception:
                et_h = h
            bucket = hourly.setdefault(et_h, {"trades": 0, "wins": 0})
            bucket["trades"] += 1
            bucket["wins"]   += 1 if t["won"] else 0

        result = {
            "ok":            True,
            "n_trades":      n_trades,
            "n_wins":        n_wins,
            "n_losses":      n_losses,
            "win_rate":      win_rate,
            "profit_factor": profit_factor,
            "total_pnl":     total_pnl,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "max_dd":        round(max_dd, 2),
            "max_dd_pct":    max_dd_pct,
            "sl_direct_pct": sl_pct,
            "tp2_pct":       tp2_pct,
            "equity_curve":  equity_curve,
            "trades":        trades_log[-200:],     # derniers 200 pour le UI
            "long_trades":   len(long_trades),
            "long_wins":     long_wins,
            "short_trades":  len(short_trades),
            "short_wins":    short_wins,
            "hourly":        {str(h): v for h, v in sorted(hourly.items())},
            "data_start":    data_start,
            "data_end":      data_end,
            "bars_total":    bars_total,
            "params":        {**strat.DEFAULTS, **(params or {})},
        }

        _set(running=False, pct=100, status="done", last_result=result)
        return result

    except Exception as exc:
        import traceback
        err = f"{exc}\n{traceback.format_exc()}"
        _set(running=False, status="error", error=err)
        return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Lancement asynchrone (depuis l'API)
# --------------------------------------------------------------------------- #
def launch_pretrain_es(
    start:    str,
    end:      str,
    params:   Optional[dict] = None,
    capital:  float = 50_000.0,
    risk_pct: float = 1.0,
    on_complete: Optional[Callable] = None,
) -> None:
    def _run():
        result = run_pretrain_es(start, end, params=params,
                                  capital=capital, risk_pct=risk_pct)
        if on_complete:
            try:
                on_complete(result)
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True, name="pretrain_es")
    t.start()
