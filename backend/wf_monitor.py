"""
wf_monitor.py
=============
Walk-forward automatique périodique — surveille la robustesse OOS de la
stratégie sans jamais modifier ses paramètres (contrairement à
researcher_agent.py et adaptive_agent.py, désactivés — voir CLAUDE.md).

Purement diagnostique : relance le walk-forward à 12 fenêtres à intervalle
régulier, hors session et sans position active, et pousse une alerte
dashboard uniquement si les critères de robustesse du CLAUDE.md ne sont
plus respectés (PF > 1.0 dans ≥75% des fenêtres ET std_pf < 0.30). Silence
si tout va bien — pas d'alerte à chaque run réussi.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MIN_INTERVAL_S = float(os.environ.get("WF_MONITOR_INTERVAL_HOURS", "168")) * 3600.0  # 7j par défaut
_N_SPLITS = 12
_WINDOW_MONTHS_BACK = 24


class WalkForwardMonitor:
    """Lance périodiquement pretrain.run_walk_forward et alerte si hors critères."""

    def __init__(self, symbol: str = "XAUUSD") -> None:
        self.symbol = symbol
        self._last_ts = 0.0
        self._running = False
        self._last_result: Optional[Dict[str, Any]] = None
        logger.info("[WFMonitor:%s] initialisé — intervalle=%.0fh, %d fenêtres",
                    symbol, _MIN_INTERVAL_S / 3600.0, _N_SPLITS)

    # ---- Public API ---- #

    def maybe_run(self, has_active_position: bool) -> None:
        """Appelé depuis la boucle principale toutes les N secondes."""
        if has_active_position or self._running:
            return
        if time.time() - self._last_ts < _MIN_INTERVAL_S:
            return
        if not self._is_off_session():
            return
        t = threading.Thread(target=self._thread_run, daemon=True)
        t.start()

    def run_now(self) -> Dict[str, Any]:
        """Force un run immédiat (appelé depuis l'endpoint API)."""
        if self._running:
            return {"error": "Un run est déjà en cours"}
        self._running = True
        try:
            return self._analyze()
        finally:
            self._running = False
            self._last_ts = time.time()

    def status(self) -> Dict[str, Any]:
        history = []
        try:
            import database as _db
            history = _db.wf_monitor_history(self.symbol, limit=20)
        except Exception as exc:
            logger.debug("[WFMonitor] read history: %s", exc)
        return {
            "running":   self._running,
            "last_run":  self._last_result,
            "interval_hours": _MIN_INTERVAL_S / 3600.0,
            "history":   history,
        }

    # ---- Internals ---- #

    def _is_off_session(self) -> bool:
        try:
            import pytz
            _CET = pytz.timezone("Europe/Paris")
            h = datetime.now(timezone.utc).astimezone(_CET).hour
            return not (8 <= h < 12 or 14 <= h < 18)
        except Exception:
            return True

    def _thread_run(self) -> None:
        self._running = True
        try:
            self._analyze()
        finally:
            self._running = False
            self._last_ts = time.time()

    def _analyze(self) -> Dict[str, Any]:
        try:
            import pretrain as _pt
            end = datetime.now(timezone.utc).date().isoformat()
            start = (datetime.now(timezone.utc) - timedelta(days=_WINDOW_MONTHS_BACK * 30)).date().isoformat()
            result = _pt.run_walk_forward(
                start=start, end=end, n_splits=_N_SPLITS, symbol=self.symbol,
            )
        except Exception as exc:
            logger.warning("[WFMonitor:%s] erreur: %s", self.symbol, exc)
            return {"error": str(exc)}

        summary = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "period":          f"{start} → {end}",
            "avg_pf":          result.get("avg_pf"),
            "std_pf":          result.get("std_pf"),
            "pct_profitable":  result.get("pct_profitable"),
            "is_robust":       result.get("is_robust"),
        }
        self._last_result = summary

        try:
            import database as _db
            _db.wf_monitor_log_run(
                self.symbol, summary["period"], summary["avg_pf"], summary["std_pf"],
                summary["pct_profitable"], bool(summary["is_robust"]),
            )
        except Exception as exc:
            logger.debug("[WFMonitor] persist history: %s", exc)

        if not result.get("is_robust"):
            try:
                import main as _main
                _main.state.push_alert(
                    "warn",
                    f"[WFMonitor:{self.symbol}] walk-forward hors critères — "
                    f"PF moy {summary['avg_pf']} ± {summary['std_pf']}, "
                    f"{summary['pct_profitable']}% fenêtres rentables "
                    f"(seuils CLAUDE.md : ≥75% et std_pf<0.30)",
                )
            except Exception as exc:
                logger.debug("[WFMonitor] push_alert: %s", exc)

        logger.info("[WFMonitor:%s] run terminé — robust=%s (PF %s±%s, %s%%)",
                    self.symbol, summary["is_robust"], summary["avg_pf"],
                    summary["std_pf"], summary["pct_profitable"])
        return summary
