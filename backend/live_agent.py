"""
live_agent.py
=============
Agent adaptatif live pour XAUUSD — apprend uniquement des trades paper réels.
Aucun backtest. Aucune interaction avec pretrain.

Fonctionnement :
- Chaque trade fermé appelle on_trade_closed()
- Tous les BATCH_SIZE trades : évalue la performance et ajuste 3 paramètres
- Les paramètres sont modifiés directement sur le module strategy (sans redémarrage)
- L'état est persisté en DB (table live_agent)

Paramètres ajustés :
  strategy.RSI_M5_LONG_MIN   : seuil RSI M5 LONG  (bornes 40–52)
  strategy.RSI_M5_SHORT_MAX  : seuil RSI M5 SHORT (bornes 48–60)
  strategy.ATR_REGIME_MIN_RATIO : filtre régime   (bornes 0.60–0.90)
  strategy.ADX_MIN           : force tendance minimale (bornes 15–30)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BATCH_SIZE   = 10    # ajustement tous les N trades
WINDOW       = 20    # fenêtre glissante de trades pour évaluer le WR
WR_TARGET    = 0.52  # WR cible
WR_LOW       = 0.42  # en dessous → resserrer les filtres
WR_HIGH      = 0.62  # au-dessus  → assouplir légèrement

# Boucle d'apprentissage autonome
BOOTSTRAP_EXIT_TRADES     = 50   # trades live avant désactivation de BOOTSTRAP_MODE
AUTO_PRETRAIN_MONTHS      = 3    # fenêtre du pretrain automatique (mois glissants)

BOUNDS = {
    "RSI_M5_LONG_MIN":      (38.0, 52.0),
    "RSI_M5_SHORT_MAX":     (48.0, 62.0),
    "RSI_LOW":              (30.0, 45.0),
    "RSI_HIGH":             (55.0, 70.0),
    "ATR_REGIME_MIN_RATIO": (0.60, 0.90),
    "ADX_MIN":              (15.0, 30.0),
}

STEP = {
    "RSI_M5_LONG_MIN":      1.0,
    "RSI_M5_SHORT_MAX":     1.0,
    "RSI_LOW":              1.0,
    "RSI_HIGH":             1.0,
    "ATR_REGIME_MIN_RATIO": 0.05,
    "ADX_MIN":              2.0,
}


class LiveAdaptiveAgent:
    """
    Agent live qui ajuste les paramètres de strategy.py
    en fonction des résultats des trades paper XAUUSD.
    """

    def __init__(self, symbol: str = "XAUUSD") -> None:
        self.symbol = symbol
        self._lock  = threading.Lock()
        self._trade_log: List[Dict[str, Any]] = []
        self._adjustments: List[Dict[str, Any]] = []
        self._total_trades = 0
        self._params: Dict[str, float] = self._default_params()
        self._bootstrap_exit_pending = False   # True une seule fois, quand BOOTSTRAP→filtres
        self._load()

    # ------------------------------------------------------------------ #

    def _default_params(self) -> Dict[str, float]:
        import strategy as st
        return {
            "RSI_M5_LONG_MIN":      getattr(st, "RSI_M5_LONG_MIN",      42.0),
            "RSI_M5_SHORT_MAX":     getattr(st, "RSI_M5_SHORT_MAX",      58.0),
            "RSI_LOW":              getattr(st, "RSI_LOW",               35.0),
            "RSI_HIGH":             getattr(st, "RSI_HIGH",              65.0),
            "ATR_REGIME_MIN_RATIO": getattr(st, "ATR_REGIME_MIN_RATIO",  0.65),
            "ADX_MIN":              getattr(st, "ADX_MIN",               20.0),
        }

    def _load(self) -> None:
        try:
            import database as db
            data = db.live_agent_load(self.symbol)
            if data:
                saved = data.get("params", {})
                # Params dont une valeur BASSE est plus permissive (seuils plancher)
                _lower_is_looser = {"RSI_M5_LONG_MIN", "RSI_LOW", "ATR_REGIME_MIN_RATIO", "ADX_MIN"}
                for k in self._params:
                    if k in saved:
                        saved_val = float(saved[k])
                        default_val = self._params[k]
                        # Toujours prendre la valeur la plus permissive entre sauvegardée et défaut module
                        if k in _lower_is_looser:
                            self._params[k] = min(saved_val, default_val)
                        else:  # RSI_M5_SHORT_MAX, RSI_HIGH — valeur haute = plus permissive
                            self._params[k] = max(saved_val, default_val)
                self._trade_log = data.get("trade_log", [])
                self._total_trades = len(self._trade_log)
                self._apply_to_strategy()
                logger.info("[LiveAgent:%s] état chargé — %d trades, params=%s",
                            self.symbol, self._total_trades, self._params)
        except Exception as e:
            logger.warning("[LiveAgent:%s] erreur chargement: %s", self.symbol, e)

    def _save(self) -> None:
        try:
            import database as db
            db.live_agent_save(self.symbol, self._params, self._trade_log[-WINDOW:])
        except Exception as e:
            logger.warning("[LiveAgent:%s] erreur sauvegarde: %s", self.symbol, e)

    def _apply_to_strategy(self) -> None:
        import strategy as st
        for k, v in self._params.items():
            if hasattr(st, k):
                setattr(st, k, v)

    # ------------------------------------------------------------------ #

    def on_trade_closed(self, won: bool, pnl: float, features: Optional[Dict] = None) -> None:
        with self._lock:
            self._trade_log.append({"won": won, "pnl": pnl})
            self._total_trades += 1

            # Transition automatique BOOTSTRAP_MODE → filtres calibrés
            import strategy as st
            if (self._total_trades >= BOOTSTRAP_EXIT_TRADES
                    and st.BOOTSTRAP_MODE
                    and not self._bootstrap_exit_pending):
                st.BOOTSTRAP_MODE = False
                self._bootstrap_exit_pending = True
                logger.info("[LiveAgent:%s] %d trades live — BOOTSTRAP_MODE désactivé, pretrain demandé",
                            self.symbol, self._total_trades)

            if self._total_trades % BATCH_SIZE == 0:
                self._evaluate_and_adjust()

            self._save()

    def consume_bootstrap_exit(self) -> bool:
        """Retourne True une seule fois au moment de la transition BOOTSTRAP → filtres."""
        with self._lock:
            if self._bootstrap_exit_pending:
                self._bootstrap_exit_pending = False
                return True
            return False

    def _evaluate_and_adjust(self) -> None:
        window = self._trade_log[-WINDOW:]
        if len(window) < BATCH_SIZE:
            return

        wr = sum(1 for t in window if t["won"]) / len(window)
        logger.info("[LiveAgent:%s] évaluation — WR=%.0f%% sur %d trades",
                    self.symbol, wr * 100, len(window))

        if WR_LOW <= wr <= WR_HIGH:
            return  # zone acceptable, pas d'ajustement

        changes = {}

        if wr < WR_LOW:
            # Resserrer tous les filtres
            changes["RSI_M5_LONG_MIN"]      = +STEP["RSI_M5_LONG_MIN"]
            changes["RSI_M5_SHORT_MAX"]     = -STEP["RSI_M5_SHORT_MAX"]
            changes["ATR_REGIME_MIN_RATIO"] = +STEP["ATR_REGIME_MIN_RATIO"]
            changes["ADX_MIN"]              = +STEP["ADX_MIN"]
        else:
            # WR > WR_HIGH : assouplir légèrement pour augmenter le nombre de trades
            changes["RSI_M5_LONG_MIN"]      = -STEP["RSI_M5_LONG_MIN"]
            changes["RSI_M5_SHORT_MAX"]     = +STEP["RSI_M5_SHORT_MAX"]
            changes["ATR_REGIME_MIN_RATIO"] = -STEP["ATR_REGIME_MIN_RATIO"]
            changes["ADX_MIN"]              = -STEP["ADX_MIN"]

        applied = {}
        for k, delta in changes.items():
            lo, hi = BOUNDS[k]
            old = self._params[k]
            new = max(lo, min(hi, old + delta))
            if new != old:
                self._params[k] = new
                applied[k] = {"from": round(old, 3), "to": round(new, 3)}

        if applied:
            self._apply_to_strategy()
            self._adjustments.append({
                "trades": self._total_trades,
                "wr": round(wr, 3),
                "changes": applied,
            })
            logger.info("[LiveAgent:%s] ajustements: %s", self.symbol, applied)

    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        import strategy as st
        with self._lock:
            window = self._trade_log[-WINDOW:]
            wr = sum(1 for t in window if t["won"]) / len(window) if window else None
            return {
                "symbol":           self.symbol,
                "total_trades":     self._total_trades,
                "rolling_wr":       round(wr, 3) if wr is not None else None,
                "params":           {k: round(v, 3) for k, v in self._params.items()},
                "last_adj":         self._adjustments[-1] if self._adjustments else None,
                "n_adjustments":    len(self._adjustments),
                "bootstrap_mode":   st.BOOTSTRAP_MODE,
                "trades_to_exit":   max(0, BOOTSTRAP_EXIT_TRADES - self._total_trades),
            }
