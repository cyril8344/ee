"""
live_agent.py
=============
Agent adaptatif live pour XAUUSD — apprend uniquement des trades paper réels.
Aucun backtest. Aucune interaction avec pretrain.

AJUSTEMENT AUTOMATIQUE DÉSACTIVÉ (août 2026) — voir AUTO_ADJUST_ENABLED ci-dessous.
L'agent ne fait plus que compter les trades et gérer la sortie de BOOTSTRAP_MODE.

Motifs de la désactivation :
- Il évaluait le WR sur une fenêtre glissante de 20 trades (WINDOW). L'erreur-type
  d'un WR sur 20 trades est de ±11 points : une stratégie qui gagne réellement 52 %
  descend sous le seuil WR_LOW (42 %) environ une évaluation sur cinq, par pur hasard.
  L'agent ne réagissait donc pas à une dégradation, mais au bruit.
- _evaluate_and_adjust() modifiait QUATRE paramètres simultanément, alors que les
  règles anti-overfitting du projet fixent un maximum de trois (voir CLAUDE.md).
- Aucune validation walk-forward, en production, sur de l'argent réel — exactement
  le motif pour lequel researcher_agent.py et adaptive_agent.py ont été désactivés
  en juillet 2026.
- Effet constaté : _load() prenait la valeur la PLUS PERMISSIVE entre la valeur
  sauvegardée et le défaut module (min() pour les seuils plancher). Un desserrage
  devenait donc irréversible — aucun redémarrage ne le rattrapait. strategy.ADX_MIN
  est resté bloqué à 20 alors que le code déclarait 28, pendant des semaines, sans
  la moindre alerte (le garde-fou _near_bound ne se déclenche qu'à 18).
- Bilan réel : un seul ajustement en plus d'un mois, pour cette divergence.

Le forçage manuel depuis le dashboard (force_params) reste actif : c'est une action
humaine explicite et tracée, pas une boucle autonome.

Fonctionnement restant :
- Chaque trade fermé appelle on_trade_closed() : comptage + sortie de BOOTSTRAP_MODE
- L'état est persisté en DB (table live_agent)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Boucle d'ajustement autonome. False = l'agent n'écrit plus jamais dans le module
# strategy de lui-même : les valeurs déclarées dans strategy.py font foi et sont
# auditables. Ne repasser à True qu'avec une validation walk-forward de la boucle
# elle-même, pas seulement des paramètres qu'elle produit.
AUTO_ADJUST_ENABLED = False

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

# Distance à la borne (en fraction de la plage BOUNDS) en dessous de laquelle on
# considère qu'un paramètre est "coincé" en bout de course — signal qu'il a dérivé
# au maximum permis sans qu'on s'en rende compte (cas vécu : ADX_MIN à 17, presque
# au plancher 15, resté invisible plusieurs jours faute d'alerte).
_NEAR_BOUND_FRACTION = 0.20

# Paramètres dont une valeur BASSE est la direction permissive/risquée (seuils
# plancher) — cohérent avec la logique déjà utilisée dans _load(). Les autres
# (RSI_M5_SHORT_MAX, RSI_HIGH) ont leur direction risquée du côté haut. On
# n'alerte que dans la direction permissive : une valeur proche de la borne
# STRICTE (ex. ADX_MIN proche de 30) n'est pas un problème, l'inverse si.
_LOWER_IS_LOOSER = {"RSI_M5_LONG_MIN", "RSI_LOW", "ATR_REGIME_MIN_RATIO", "ADX_MIN"}


def _near_bound(key: str, value: float) -> Optional[str]:
    """Retourne 'min'/'max' si value est à moins de _NEAR_BOUND_FRACTION de la borne
    du côté permissif, None sinon (y compris si elle est proche de la borne stricte —
    ce n'est pas un problème). Ignore les clés sans bornes connues."""
    bounds = BOUNDS.get(key)
    if not bounds:
        return None
    lo, hi = bounds
    span = hi - lo
    if span <= 0:
        return None
    looser_side = "min" if key in _LOWER_IS_LOOSER else "max"
    if looser_side == "min" and value - lo <= _NEAR_BOUND_FRACTION * span:
        return "min"
    if looser_side == "max" and hi - value <= _NEAR_BOUND_FRACTION * span:
        return "max"
    return None


def _push_near_bound_alert(symbol: str, applied: Dict[str, Any]) -> None:
    """Alerte dashboard si un des changements appliqués atterrit près d'une borne."""
    warnings = []
    for k, v in applied.items():
        side = _near_bound(k, v["to"])
        if side:
            lo, hi = BOUNDS[k]
            bound_val = lo if side == "min" else hi
            warnings.append(f"{k}={v['to']} (borne {side}={bound_val})")
    if not warnings:
        return
    try:
        import main as _main
        _main.state.push_alert(
            "warn",
            f"[LiveAgent:{symbol}] paramètre(s) proche de leur borne — " + ", ".join(warnings),
        )
    except Exception as exc:
        logger.debug("[LiveAgent] push_alert near-bound: %s", exc)


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
                # Boucle autonome désactivée : on NE restaure PAS les paramètres
                # sauvegardés. Ils sont l'héritage de l'ancienne boucle, et la règle
                # min()/max() ci-dessous retenait toujours la valeur la plus permissive
                # — un desserrage ne pouvait donc jamais être annulé par un redémarrage.
                # Les valeurs déclarées dans strategy.py font foi. Conséquence assumée :
                # un forçage manuel ne survit plus à un redéploiement ; pour le rendre
                # permanent, la valeur doit être écrite dans strategy.py, qui est de
                # toute façon sa place.
                for k in (self._params if AUTO_ADJUST_ENABLED else {}):
                    if k in saved:
                        saved_val = float(saved[k])
                        default_val = self._params[k]
                        # Toujours prendre la valeur la plus permissive entre sauvegardée et défaut module
                        if k in _LOWER_IS_LOOSER:
                            new_val = min(saved_val, default_val)
                        else:  # RSI_M5_SHORT_MAX, RSI_HIGH — valeur haute = plus permissive
                            new_val = max(saved_val, default_val)
                        # Clamp aux BOUNDS — une valeur DB corrompue/hors bornes ne doit
                        # jamais s'appliquer telle quelle au module strategy live.
                        if k in BOUNDS:
                            lo, hi = BOUNDS[k]
                            new_val = max(lo, min(hi, new_val))
                        self._params[k] = new_val
                self._trade_log = data.get("trade_log", [])
                self._total_trades = len(self._trade_log)
                self._apply_to_strategy()
                logger.info("[LiveAgent:%s] état chargé — %d trades, params=%s",
                            self.symbol, self._total_trades, self._params)
            # Historique des ajustements — persistant, survit aux redémarrages/redéploiements
            # (contrairement à trade_log/params ci-dessus qui ne gardent que l'état courant).
            history = db.live_agent_adjustments_history(self.symbol)
            self._adjustments = list(reversed(history))  # plus ancien → plus récent
        except Exception as e:
            logger.warning("[LiveAgent:%s] erreur chargement: %s", self.symbol, e)

    def _save(self) -> None:
        try:
            import database as db
            db.live_agent_save(self.symbol, self._params, self._trade_log[-WINDOW:])
        except Exception as e:
            logger.warning("[LiveAgent:%s] erreur sauvegarde: %s", self.symbol, e)

    def _apply_to_strategy(self) -> None:
        # Boucle autonome désactivée : seul un forçage manuel explicite peut encore
        # écrire dans le module strategy (force_params passe par _force_to_strategy).
        # Sans ce garde-fou, _load() réappliquerait au démarrage les valeurs héritées
        # de l'ancienne boucle et ferait diverger le live de ce que déclare strategy.py.
        if not AUTO_ADJUST_ENABLED:
            return
        self._force_to_strategy()

    def _force_to_strategy(self) -> None:
        import strategy as st
        for k, v in self._params.items():
            if hasattr(st, k):
                setattr(st, k, v)

    def force_params(self, overrides: Dict[str, float]) -> Dict[str, float]:
        """Force-apply specific params, persist to DB. Returns updated params."""
        with self._lock:
            applied = {}
            for k, v in overrides.items():
                if k in self._params and k in BOUNDS:
                    lo, hi = BOUNDS[k]
                    old = self._params[k]
                    new = max(lo, min(hi, float(v)))
                    if new != old:
                        self._params[k] = new
                        applied[k] = {"from": round(old, 3), "to": round(new, 3)}
            # Forçage manuel : action humaine explicite et tracée, elle écrit dans le
            # module même quand la boucle autonome est désactivée.
            self._force_to_strategy()
            self._save()
            if applied:
                window = self._trade_log[-WINDOW:]
                wr = sum(1 for t in window if t["won"]) / len(window) if window else 0.0
                record = {"trades": self._total_trades, "wr": round(wr, 3),
                          "changes": applied, "manual": True}
                self._adjustments.append(record)
                try:
                    import database as db
                    db.live_agent_log_adjustment(self.symbol, self._total_trades, wr, applied)
                except Exception as e:
                    logger.warning("[LiveAgent:%s] erreur journalisation forçage manuel: %s", self.symbol, e)
                logger.info("[LiveAgent:%s] forçage manuel: %s", self.symbol, applied)
                _push_near_bound_alert(self.symbol, applied)
        return dict(self._params)

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
        if not AUTO_ADJUST_ENABLED:
            return
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
            try:
                import database as db
                db.live_agent_log_adjustment(self.symbol, self._total_trades, wr, applied)
            except Exception as e:
                logger.warning("[LiveAgent:%s] erreur journalisation ajustement: %s", self.symbol, e)
            logger.info("[LiveAgent:%s] ajustements: %s", self.symbol, applied)
            _push_near_bound_alert(self.symbol, applied)
            # Notifier ResearcherAgent pour valider ces params via pretrain
            self._notify_researcher()

    def _notify_researcher(self) -> None:
        """Demande au ResearcherAgent de valider les nouveaux params via un pretrain court."""
        try:
            import main as _main
            researcher = getattr(_main.state, "researcher", None)
            if researcher is not None:
                researcher.request_validation()
                logger.info("[LiveAgent:%s] ResearcherAgent notifié pour validation", self.symbol)
        except Exception as exc:
            logger.debug("[LiveAgent] notify researcher: %s", exc)

    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        import strategy as st
        with self._lock:
            window = self._trade_log[-WINDOW:]
            wr = sum(1 for t in window if t["won"]) / len(window) if window else None
            near_bounds = {
                k: _near_bound(k, v) for k, v in self._params.items()
                if _near_bound(k, v) is not None
            }
            return {
                "symbol":           self.symbol,
                "total_trades":     self._total_trades,
                "rolling_wr":       round(wr, 3) if wr is not None else None,
                "params":           {k: round(v, 3) for k, v in self._params.items()},
                "near_bounds":      near_bounds,
                "last_adj":         self._adjustments[-1] if self._adjustments else None,
                "n_adjustments":    len(self._adjustments),
                "adjustment_history": list(reversed(self._adjustments[-20:])),  # plus récent d'abord
                # False = plus aucun ajustement automatique ; les valeurs affichées
                # sont celles déclarées dans strategy.py, pas un état appris.
                "auto_adjust_enabled": AUTO_ADJUST_ENABLED,
                "bootstrap_mode":   st.BOOTSTRAP_MODE,
                "trades_to_exit":   max(0, BOOTSTRAP_EXIT_TRADES - self._total_trades),
            }
