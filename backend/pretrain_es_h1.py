"""
pretrain_es_h1.py
==================
Pré-entraînement de la stratégie Order Flow ES en bougies H1
(voir strategy_es_h1.py).

Contrairement à pretrain_es.py (M5, qui rééchantillonne en H1 pour un simple
biais secondaire), ce module charge directement des données H1 — ce qui
permet un historique bien plus long chez yfinance (interval="60m" couvre
~2 ans, contre ~60 jours en interval="5m").

Walk-forward : run_walkforward_es_h1() pour validation OOS (mêmes critères
de robustesse que XAU/ES M5 : PF > 1.0 dans ≥ 75% des fenêtres, std_pf < 0.30).

Usage :
    from pretrain_es_h1 import run_pretrain_es_h1, launch_pretrain_es_h1, get_progress_es_h1
    from pretrain_es_h1 import run_walkforward_es_h1, launch_walkforward_es_h1, get_wf_progress_es_h1
"""

from __future__ import annotations

import hashlib
import threading
import time as _time_mod
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import yfinance as yf

import strategy_es_h1 as strat

# --------------------------------------------------------------------------- #
# État de progression — prétrain
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

# État de progression — walk-forward
_wf_progress: Dict[str, Any] = {
    "running":  False,
    "window":   0,
    "n_splits": 0,
    "result":   None,
    "error":    None,
}
_wf_lock = threading.Lock()


def get_progress_es_h1() -> Dict[str, Any]:
    with _lock:
        return dict(_progress)


def get_wf_progress_es_h1() -> Dict[str, Any]:
    with _wf_lock:
        return dict(_wf_progress)


def _set(**kwargs):
    with _lock:
        _progress.update(kwargs)


def _set_wf(**kwargs):
    with _wf_lock:
        _wf_progress.update(kwargs)


# --------------------------------------------------------------------------- #
# Chargement des données ES=F en H1
# --------------------------------------------------------------------------- #
def _load_es_h1_data(start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Télécharge ES=F en bougies 60 minutes depuis yfinance (avec 1 retry
    après backoff — Yahoo rate-limite volontiers des appels rapprochés, ex.
    plusieurs fenêtres de walk-forward enchaînées), fallback synthétique sinon.

    Retourne (df, source) avec source "yfinance" ou "synthetic" — le walk-
    forward doit pouvoir distinguer une vraie fenêtre OOS d'un fallback, sans
    quoi un rate-limit silencieux se fait passer pour un résultat réel.
    """
    for attempt in range(2):
        try:
            tk = yf.Ticker("ES=F")
            df = tk.history(start=start, end=end, interval="60m", auto_adjust=True)
            if df is not None and len(df) > 100:
                df = df.rename(columns=str.lower)
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df = df.dropna()
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                df.index.name = "time"
                return df, "yfinance"
        except Exception:
            pass
        if attempt == 0:
            _time_mod.sleep(2.0)
    return _synthetic_es_h1(start, end), "synthetic"


def _synthetic_es_h1(start: str, end: str) -> pd.DataFrame:
    """Données synthétiques ES H1 pour les tests offline / fallback réseau.

    Graine dérivée de (start, end) plutôt que fixe — deux fenêtres de walk-
    forward de même durée retombant toutes deux en fallback ne doivent pas
    produire un chemin de prix identique (ça a caché un rate-limit Yahoo
    qui se faisait passer pour deux fenêtres OOS réelles et cohérentes)."""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC")
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=180)
    idx = pd.date_range(start_dt, end_dt, freq="60min", tz="UTC")
    idx = idx[idx.weekday < 5]
    n   = len(idx)
    seed = int(hashlib.md5(f"{start}_{end}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    price0 = 5800.0
    rets   = rng.normal(0, 0.0011, n)
    # Volatilité renforcée en session RTH (14h30–21h UTC = 9h30–16h ET)
    hours = idx.hour.values
    boost = np.where((hours >= 14) & (hours < 21), 1.8, 0.6)
    rets  = rets * boost
    close = price0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, 3.5, n)) + 1.0
    high   = close + spread
    low    = close - spread
    open_  = np.concatenate([[price0], close[:-1]])
    volume = (np.abs(rng.normal(12000, 3000, n)) * boost).round()
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
# Simulation de sortie de trade ES H1
# --------------------------------------------------------------------------- #
def _try_exit_es_h1(trade: dict, bar: pd.Series, ts, params: dict) -> Optional[tuple]:
    """
    Retourne (pnl_dollars, exit_price, exit_reason) ou None si encore ouvert.

    Force la sortie au marché à la clôture de session RTH (jamais de position
    overnight/weekend) même si SL/TP/timeout ne sont pas encore atteints —
    même principe que XAU/EUR, appliqué ici à un actif qui gapperait
    fortement sur des news pendant la fermeture des marchés.
    """
    direction  = trade["direction"]
    sl         = trade["sl"]
    tp1        = trade["tp1"]
    tp2        = trade["tp2"]
    entry      = trade["entry"]
    contracts  = trade["contracts"]

    high = float(bar.get("high", 0) or 0)
    low  = float(bar.get("low",  0) or 0)

    duration_bars = trade.get("duration_bars", 0) + 1
    trade["duration_bars"] = duration_bars

    tick = strat.TICK_SIZE
    tv   = strat.TICK_VALUE

    # Màj MAE/MFE
    if direction == "long":
        trade["mfe"] = max(trade.get("mfe", 0.0), high - entry)
        trade["mae"] = max(trade.get("mae", 0.0), entry - low)
    else:
        trade["mfe"] = max(trade.get("mfe", 0.0), entry - low)
        trade["mae"] = max(trade.get("mae", 0.0), high - entry)

    # --- TP1 partiel (50%) ---
    if not trade.get("tp1_hit", False):
        hit_tp1 = (direction == "long" and high >= tp1) or \
                  (direction == "short" and low  <= tp1)
        if hit_tp1:
            trade["tp1_hit"] = True
            if direction == "long":
                pnl_part = 0.5 * contracts * round((tp1 - entry) / tick) * tv
            else:
                pnl_part = 0.5 * contracts * round((entry - tp1) / tick) * tv
            trade["partial_pnl"] = pnl_part
            trade["sl"] = entry    # SL -> BE après TP1
            sl = entry

    partial_pnl = trade.get("partial_pnl", 0.0)
    remaining   = 0.5 if trade.get("tp1_hit", False) else 1.0
    r_contracts = contracts * remaining

    def _pnl(exit_p: float) -> float:
        ticks = round((exit_p - entry) / tick) if direction == "long" \
                else round((entry - exit_p) / tick)
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

    # --- Clôture de session (flatten forcé, priorité sur le timeout) ---
    try:
        import pytz
        et_tz   = pytz.timezone("America/New_York")
        ts_et   = ts.astimezone(et_tz)
        dec_hr  = ts_et.hour + ts_et.minute / 60.0
        cls_dec = float(params.get("session_close_h", 16)) + float(params.get("session_close_m", 0)) / 60.0
        if dec_hr >= cls_dec:
            mid    = float(bar.get("close", entry) or entry)
            reason = "session_close_tp1" if trade.get("tp1_hit") else "session_close"
            return _pnl(mid), mid, reason
    except Exception:
        pass

    # --- Timeout ---
    max_bars = int(params.get("max_trade_hours", 8))
    if duration_bars >= max_bars:
        mid    = float(bar.get("close", entry) or entry)
        reason = "timeout_tp1" if trade.get("tp1_hit") else "timeout"
        return _pnl(mid), mid, reason

    return None


# --------------------------------------------------------------------------- #
# Moteur principal
# --------------------------------------------------------------------------- #
MAX_DAY = 2   # trades H1 max par jour — bien moins de signaux/jour qu'en M5


def run_pretrain_es_h1(
    start:     str,
    end:       str,
    params:    Optional[dict] = None,
    capital:   float = 50_000.0,
    risk_pct:  float = 1.0,
    _set_fn:   Optional[Callable] = None,   # permet d'utiliser _set_wf depuis walk-forward
    _raw_data: Optional[tuple] = None,      # (df, source) déjà chargé — évite un refetch (voir walk-forward)
) -> Dict[str, Any]:
    """Lance le prétrain ES H1 en mode bloquant."""
    set_fn = _set_fn or _set
    p = {**strat.DEFAULTS, **(params or {})}

    set_fn(running=True, pct=0, bars_done=0, trades=0, wins=0,
           status="running", error=None, last_result=None)
    try:
        if _raw_data is not None:
            raw, data_source = _raw_data
        else:
            set_fn(status="Chargement données ES=F (H1)…")
            raw, data_source = _load_es_h1_data(start, end)
        warmup = max(int(p.get("ema_trend", 200)) + 10, 210)
        if len(raw) < warmup + 20:
            raise ValueError("Pas assez de données H1 pour la période sélectionnée.")

        data_start = raw.index[0].isoformat()[:10]
        data_end   = raw.index[-1].isoformat()[:10]
        bars_total = len(raw)

        set_fn(status="Calcul des indicateurs H1…")
        h1 = strat.add_indicators(raw, params)

        total = len(h1) - warmup

        open_trade    = None
        n_trades      = 0
        n_wins        = 0
        equity        = float(capital)
        equity_curve  = [{"ts": h1.index[warmup].isoformat(), "equity": equity}]
        trades_log    = []
        pnl_wins      = []
        pnl_losses    = []
        _day_trades: Dict[str, int] = {}

        set_fn(bars_total=total, status="Analyse bar-par-bar…")

        for i in range(warmup, len(h1)):
            ts  = h1.index[i]
            bar = h1.iloc[i]

            done = i - warmup
            if done % 100 == 0:
                set_fn(pct=round(done / total * 100), bars_done=done,
                       trades=n_trades, wins=n_wins)

            # ── Gérer le trade ouvert ─────────────────────────────────────────
            if open_trade is not None:
                exit_info = _try_exit_es_h1(open_trade, bar, ts, p)
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

                    try:
                        import pytz
                        hour_et = ts.astimezone(pytz.timezone("America/New_York")).hour
                    except Exception:
                        hour_et = ts.hour

                    trades_log.append({
                        "entry_ts":    open_trade["entry_time"].isoformat(),
                        "exit_ts":     ts.isoformat(),
                        "direction":   open_trade["direction"],
                        "entry":       round(open_trade["entry"], 2),
                        "exit_price":  round(float(exit_price), 2),
                        "exit_reason": exit_reason,
                        "pnl":         round(pnl, 2),
                        "won":         won,
                        "mae_r":       mae_r,
                        "mfe_r":       mfe_r,
                        "contracts":   open_trade["contracts"],
                        "rsi":         open_trade.get("rsi", 0),
                        "atr":         open_trade.get("atr", 0),
                        "adx":         open_trade.get("adx", 0),
                        "bias":        open_trade.get("bias", ""),
                        "hour_et":     hour_et,
                    })
                    open_trade = None

            # ── Chercher une entrée ───────────────────────────────────────────
            if open_trade is not None:
                continue

            day_key = ts.strftime("%Y-%m-%d")
            if _day_trades.get(day_key, 0) >= MAX_DAY:
                continue

            sig = strat.evaluate(h1.iloc[:i + 1], params=params, ts=ts.to_pydatetime())
            if sig is None:
                continue

            _day_trades[day_key] = _day_trades.get(day_key, 0) + 1

            entry_p  = sig["entry"]
            sl_pts   = abs(entry_p - sig["stop_loss"])
            sl_ticks = max(1, round(sl_pts / strat.TICK_SIZE))
            contracts = strat.size_contracts(equity, risk_pct, sl_ticks)
            risk_usd  = sl_ticks * strat.TICK_VALUE * contracts

            open_trade = {
                "direction":     "long" if sig["bias"] == "LONG" else "short",
                "entry_time":    ts.to_pydatetime(),
                "entry":         entry_p,
                "sl":            sig["stop_loss"],
                "tp1":           sig["take_profit1"],
                "tp2":           sig["take_profit2"],
                "contracts":     contracts,
                "risk":          risk_usd,
                "rsi":           sig.get("rsi", 0),
                "atr":           sig.get("atr", 0),
                "adx":           sig.get("adx", 0),
                "bias":          sig["bias"],
                "mae":           0.0,
                "mfe":           0.0,
                "tp1_hit":       False,
                "partial_pnl":   0.0,
                "duration_bars": 0,
            }

        # ── Métriques finales ─────────────────────────────────────────────────
        n_losses      = n_trades - n_wins
        win_rate      = round(n_wins / n_trades * 100, 1) if n_trades else 0
        gross_win     = sum(pnl_wins)
        gross_loss    = sum(pnl_losses)
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0
        total_pnl     = round(gross_win - gross_loss, 2)
        avg_win       = round(gross_win  / len(pnl_wins),    2) if pnl_wins   else 0
        avg_loss      = round(gross_loss / len(pnl_losses),  2) if pnl_losses else 0

        eq_vals = [e["equity"] for e in equity_curve]
        peak    = eq_vals[0]
        max_dd  = 0.0
        for e in eq_vals:
            peak   = max(peak, e)
            max_dd = max(max_dd, peak - e)
        max_dd_pct = round(max_dd / capital * 100, 1) if capital > 0 else 0

        sl_direct  = sum(1 for t in trades_log if t["exit_reason"] == "sl")
        sl_pct     = round(sl_direct / n_trades * 100, 1) if n_trades else 0
        tp2_count  = sum(1 for t in trades_log if t["exit_reason"] in ("tp2", "tp_direct"))
        tp2_pct    = round(tp2_count / n_trades * 100, 1) if n_trades else 0
        flatten_count = sum(1 for t in trades_log if t["exit_reason"].startswith("session_close"))
        flatten_pct   = round(flatten_count / n_trades * 100, 1) if n_trades else 0

        long_trades  = [t for t in trades_log if t["direction"] == "long"]
        short_trades = [t for t in trades_log if t["direction"] == "short"]
        long_wins    = sum(1 for t in long_trades  if t["won"])
        short_wins   = sum(1 for t in short_trades if t["won"])

        hourly: Dict[int, Dict[str, int]] = {}
        for t in trades_log:
            h = t.get("hour_et", 0)
            bucket = hourly.setdefault(h, {"trades": 0, "wins": 0})
            bucket["trades"] += 1
            bucket["wins"]   += 1 if t["won"] else 0

        result = {
            "ok":             True,
            "n_trades":       n_trades,
            "n_wins":         n_wins,
            "n_losses":       n_losses,
            "win_rate":       win_rate,
            "profit_factor":  profit_factor,
            "total_pnl":      total_pnl,
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "max_dd":         round(max_dd, 2),
            "max_dd_pct":     max_dd_pct,
            "sl_direct_pct":  sl_pct,
            "tp2_pct":        tp2_pct,
            "flatten_pct":    flatten_pct,
            "equity_curve":   equity_curve,
            "trades":         trades_log[-200:],
            "long_trades":    len(long_trades),
            "long_wins":      long_wins,
            "short_trades":   len(short_trades),
            "short_wins":     short_wins,
            "hourly":         {str(h): v for h, v in sorted(hourly.items())},
            "data_start":     data_start,
            "data_end":       data_end,
            "bars_total":     bars_total,
            "data_source":    data_source,
            "params":         p,
        }

        set_fn(running=False, pct=100, status="done", last_result=result)
        return result

    except Exception as exc:
        import traceback as _tb
        err = f"{exc}\n{_tb.format_exc()}"
        set_fn(running=False, status="error", error=err)
        return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def run_walkforward_es_h1(
    start:    str,
    end:      str,
    n_splits: int = 4,
    params:   Optional[dict] = None,
    capital:  float = 50_000.0,
    risk_pct: float = 1.0,
) -> Dict[str, Any]:
    """
    Walk-forward sur n_splits fenêtres indépendantes.
    Critère robustesse (même que XAU/ES M5) : PF > 1.0 dans ≥ 75% des fenêtres
    et std_pf < 0.30.
    """
    _set_wf(running=True, window=0, n_splits=n_splits, result=None, error=None)

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    total_days  = (end_dt - start_dt).days
    window_days = max(total_days // n_splits, 30)

    # Un seul appel réseau pour toute la période, puis découpage local par
    # fenêtre — 4 fenêtres = 4 appels yfinance indépendants auparavant, ce
    # qui déclenchait le rate-limit Yahoo dès la 3e requête enchaînée. Une
    # fenêtre synthétique invalide déjà la robustesse (voir plus bas) ; ce
    # fetch unique réduit fortement le risque d'en avoir ne serait-ce qu'une.
    full_raw, full_source = _load_es_h1_data(start, end)

    windows = []
    for i in range(n_splits):
        w_start = (start_dt + pd.Timedelta(days=i * window_days)).strftime("%Y-%m-%d")
        w_end   = (start_dt + pd.Timedelta(days=min((i + 1) * window_days, total_days))).strftime("%Y-%m-%d")
        _set_wf(window=i + 1)

        window_raw = full_raw.loc[w_start:w_end]

        _local: Dict[str, Any] = {}
        def _noop(**kw): _local.update(kw)

        r = run_pretrain_es_h1(w_start, w_end, params=params,
                                capital=capital, risk_pct=risk_pct, _set_fn=_noop,
                                _raw_data=(window_raw, full_source))
        windows.append({
            "window":        i + 1,
            "start":         w_start,
            "end":           w_end,
            "n_trades":      r.get("n_trades", 0),
            "win_rate":      r.get("win_rate", 0),
            "profit_factor": r.get("profit_factor", 0.0),
            "total_pnl":     r.get("total_pnl", 0.0),
            "sl_direct_pct": r.get("sl_direct_pct", 0.0),
            "data_source":   r.get("data_source", "unknown"),
        })

    pfs = [w["profit_factor"] for w in windows if w.get("profit_factor") is not None]
    mean_pf  = round(sum(pfs) / len(pfs), 2) if pfs else 0
    std_pf   = round(float(pd.Series(pfs).std()), 2) if len(pfs) > 1 else 0
    pct_prof = round(sum(1 for p in pfs if p >= 1.0) / len(pfs) * 100, 0) if pfs else 0
    synthetic_windows = sum(1 for w in windows if w.get("data_source") == "synthetic")

    result = {
        "windows":           windows,
        "mean_pf":           mean_pf,
        "std_pf":            std_pf,
        "pct_profitable":    pct_prof,
        "synthetic_windows": synthetic_windows,
        # Une fenêtre retombée en synthétique invalide la validation OOS,
        # même si les critères PF/std_pf sont par ailleurs remplis.
        "robust":            pct_prof >= 75 and std_pf < 0.30 and synthetic_windows == 0,
    }
    _set_wf(running=False, window=n_splits, result=result)
    return result


# --------------------------------------------------------------------------- #
# Lancement asynchrone
# --------------------------------------------------------------------------- #
def launch_pretrain_es_h1(
    start:       str,
    end:         str,
    params:      Optional[dict] = None,
    capital:     float = 50_000.0,
    risk_pct:    float = 1.0,
    on_complete: Optional[Callable] = None,
) -> None:
    def _run():
        result = run_pretrain_es_h1(start, end, params=params,
                                     capital=capital, risk_pct=risk_pct)
        if on_complete:
            try:
                on_complete(result)
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True, name="pretrain_es_h1").start()


def launch_walkforward_es_h1(
    start:    str,
    end:      str,
    n_splits: int = 4,
    params:   Optional[dict] = None,
    capital:  float = 50_000.0,
    risk_pct: float = 1.0,
) -> None:
    def _run():
        run_walkforward_es_h1(start, end, n_splits=n_splits, params=params,
                               capital=capital, risk_pct=risk_pct)

    threading.Thread(target=_run, daemon=True, name="walkforward_es_h1").start()
