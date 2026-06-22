"""
feature_logger.py
=================
Logging des features ML pour la stratégie B (ICT/SMC).

Principe :
  - À l'ENTRÉE  : on stocke les features en mémoire (pending).
  - À la CLÔTURE : on fusionne features + label et on écrit la ligne complète en CSV.

Garantie no-look-ahead :
  Les features sont figées au moment de l'entrée — aucune donnée future ne peut
  les contaminer. Le label (result, r_multiple…) est ajouté séparément à la clôture,
  dans des colonnes distinctes, jamais au moment de l'entrée.

Format CSV (une ligne par trade B) :
  - Colonnes features  : tout ce qui est connu à l'entrée
  - Colonnes label     : result, r_multiple, exit_reason, duration_min
  - Séparateur visuel dans les noms : les colonnes label n'ont pas de préfixe.

Usage :
  from feature_logger import log_entry, log_exit
  log_entry(trade_id, sig.meta, ts_utc, session, session_hour)
  log_exit(trade_id, pnl, risk_amount, exit_reason, duration_min)
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# Chemin du fichier CSV — lu à chaque appel pour respecter les overrides de test
def _features_path() -> Path:
    return Path(os.getenv("FEATURES_B_PATH", "data/features_B.csv"))

# Colonnes dans l'ordre d'écriture.
# Séparation claire features (entrée) / label (clôture).
FEATURE_COLUMNS = [
    # Contexte temporel
    "trade_id", "ts_utc", "session", "session_hour",
    # Structure M15
    "signal_type", "direction", "m15_swing_dist_atr",
    # Order Block
    "ob_size_atr", "ob_distance_atr", "ob_age_bars",
    # FVG
    "fvg_present", "fvg_size_atr", "fvg_entry_pct",
    # Sweep
    "sweep_amplitude_atr",
    # Tendance / volatilité H1
    "h1_ema50_dist_atr", "h1_ema200_dist_atr",
    "vwap_distance_atr", "h1_ema50_slope", "atr_entry",
    # Golden Pocket
    "gp_pct",
    # Paramètres trade
    "sl_distance_atr", "rr_target",
]
LABEL_COLUMNS = ["result", "r_multiple", "exit_reason", "duration_min"]
ALL_COLUMNS   = FEATURE_COLUMNS + LABEL_COLUMNS

_pending: Dict[int, Dict[str, Any]] = {}
_lock    = threading.Lock()


def _ensure_file() -> Path:
    """Crée le répertoire et écrit l'entête CSV si le fichier n'existe pas. Retourne le path."""
    p = _features_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=ALL_COLUMNS).writeheader()
    return p


def log_entry(
    trade_id: int,
    meta: Dict[str, Any],
    ts_utc: str,
    session: str,
    session_hour: float,
) -> None:
    """
    Enregistre les features au moment de l'ENTRÉE en trade.
    Aucune donnée future n'est disponible ici — les colonnes label restent vides.

    trade_id     : identifiant du trade (clé de jointure avec la clôture)
    meta         : sig.meta retourné par evaluate_ict()
    ts_utc       : timestamp ISO UTC de l'entrée
    session      : "London" | "NY"
    session_hour : heure écoulée depuis le début de la session (ex: 1.5 = 1h30)
    """
    row: Dict[str, Any] = {
        "trade_id":           trade_id,
        "ts_utc":             ts_utc,
        "session":            session,
        "session_hour":       round(session_hour, 2),
        "signal_type":        meta.get("signal_type", ""),
        "direction":          meta.get("direction", ""),
        "m15_swing_dist_atr": meta.get("m15_swing_dist_atr", 0.0),
        "ob_size_atr":        meta.get("ob_size_atr", 0.0),
        "ob_distance_atr":    meta.get("ob_distance_atr", 0.0),
        "ob_age_bars":        meta.get("ob_age_bars", 0),
        "fvg_present":        meta.get("fvg_present", 0),
        "fvg_size_atr":       meta.get("fvg_size_atr", 0.0),
        "fvg_entry_pct":      meta.get("fvg_entry_pct", 0.0),
        "sweep_amplitude_atr":meta.get("sweep_amplitude_atr", 0.0),
        "h1_ema50_dist_atr":  meta.get("h1_ema50_dist_atr", 0.0),
        "h1_ema200_dist_atr": meta.get("h1_ema200_dist_atr", 0.0),
        "vwap_distance_atr":  meta.get("vwap_distance_atr", 0.0),
        "h1_ema50_slope":     meta.get("h1_ema50_slope", 0.0),
        "atr_entry":          meta.get("atr_entry", 0.0),
        "gp_pct":             meta.get("gp_pct", 0.0),
        "sl_distance_atr":    meta.get("sl_distance_atr", 0.0),
        "rr_target":          meta.get("rr_target", 1.5),
        # Label — vide à l'entrée, rempli à la clôture
        "result":             "",
        "r_multiple":         "",
        "exit_reason":        "",
        "duration_min":       "",
    }
    with _lock:
        _pending[trade_id] = row


def log_exit(
    trade_id: int,
    pnl: float,
    risk_amount: float,
    exit_reason: str,
    duration_min: float,
) -> None:
    """
    Complète la ligne avec le label au moment de la CLÔTURE et écrit en CSV.

    pnl          : P&L en devise du trade
    risk_amount  : risque initial en devise (pour calculer R multiple)
    exit_reason  : "tp1" | "tp2" | "sl" | "timeout" | …
    duration_min : durée du trade en minutes
    """
    with _lock:
        row = _pending.pop(trade_id, None)
    if row is None:
        return

    result     = "win" if pnl > 0 else "loss"
    r_multiple = round(pnl / risk_amount, 3) if risk_amount and risk_amount > 0 else 0.0

    row["result"]       = result
    row["r_multiple"]   = r_multiple
    row["exit_reason"]  = exit_reason
    row["duration_min"] = round(duration_min, 1)

    p = _ensure_file()
    with _lock:
        with p.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=ALL_COLUMNS).writerow(row)


def get_pending_count() -> int:
    """Nombre de trades B ouverts dont les features sont en attente de label."""
    with _lock:
        return len(_pending)
