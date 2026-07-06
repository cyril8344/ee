"""
adaptive_agent.py
=================
Agent IA autonome qui analyse les performances du bot et ajuste
les paramètres de stratégie directement, sans intervention humaine.

Toutes les 6h (hors session, sans position active), lit le rapport
de trades, appelle Claude Haiku pour décider des ajustements, et
applique les changements sur le module strategy à chaud.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import strategy

logger = logging.getLogger(__name__)

_LLM_MODEL       = "claude-haiku-4-5-20251001"
_MIN_INTERVAL_S  = 6 * 3600   # 6h entre deux runs automatiques
_MIN_TRADES      = 30          # pas assez de données sous ce seuil

# Bornes pour les paramètres modifiables
_PARAM_BOUNDS = {
    "RSI_M5_LONG_MIN":  (38.0, 52.0),
    "RSI_M5_SHORT_MAX": (48.0, 62.0),
    "ADX_MIN":          (12.0, 30.0),
}

# Heures de session qu'on ne peut jamais bloquer (cœur des meilleures heures)
_PROTECTED_HOURS = {9, 15, 16}

# Max 2 nouvelles heures bloquées par run (pour ne pas trop dégrader le volume)
_MAX_NEW_BLOCKS_PER_RUN = 2


class AdaptiveAgent:
    """
    Agent IA autonome de contrôle de la stratégie.
    Lit les stats de trades, demande à Claude Haiku quoi changer,
    et applique directement sur strategy.* à chaud.
    """

    def __init__(self) -> None:
        self._last_ts   = 0.0
        self._running   = False
        self._lock      = threading.Lock()
        self._history:  List[Dict[str, Any]] = []
        self._run_count = 0
        logger.info("[Adaptive] initialisé — intervalle=6h, min_trades=%d", _MIN_TRADES)

    # ---- Public API ---- #

    def maybe_run(self, has_active_position: bool) -> None:
        """Appelé depuis la boucle principale toutes les N secondes."""
        if has_active_position:
            return
        if self._running:
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
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
            return {"error": "Un run est déjà en cours, patiente quelques secondes"}
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {"error": "ANTHROPIC_API_KEY non configuré dans les variables Railway"}
        self._running = True
        try:
            return self._analyze_and_act()
        finally:
            self._running = False
            self._last_ts = time.time()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            last = self._history[-1] if self._history else None
            count = self._run_count
        return {
            "running":    self._running,
            "run_count":  count,
            "last_run":   last,
            "history":    list(self._history[-5:]),  # 5 derniers
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
            self._analyze_and_act()
        finally:
            self._running = False
            self._last_ts = time.time()

    def _analyze_and_act(self) -> Dict[str, Any]:
        # Lire les données de performance
        try:
            import database as _db
            report = _db.get_trade_report(limit=500)
        except Exception as exc:
            logger.warning("[Adaptive] erreur lecture rapport: %s", exc)
            return {"error": str(exc)}

        total = report.get("stats", {}).get("total", 0)
        if total < _MIN_TRADES:
            msg = f"seulement {total} trades (min {_MIN_TRADES}) — run ignoré"
            logger.info("[Adaptive] %s", msg)
            return {"skipped": msg}

        # Construire le contexte pour Claude
        stats      = report.get("stats", {})
        by_hour    = report.get("by_hour", {})
        by_session = report.get("by_session", {})
        by_dir     = report.get("by_direction", {})

        current = {
            "RSI_M5_LONG_MIN":  getattr(strategy, "RSI_M5_LONG_MIN",  45.0),
            "RSI_M5_SHORT_MAX": getattr(strategy, "RSI_M5_SHORT_MAX", 55.0),
            "ADX_MIN":          getattr(strategy, "ADX_MIN",           20.0),
            "BAD_HOURS_CET":    sorted(getattr(strategy, "BAD_HOURS_CET", set())),
        }

        hour_lines = "\n".join(
            f"  {h}h: WR={v['wr']}% n={v['n']} pnl={v['pnl']:.0f}$"
            for h, v in sorted(by_hour.items(), key=lambda x: int(x[0]))
        )

        session_lines = "\n".join(
            f"  {s}: WR={v['wr']}% n={v['n']} pnl={v['pnl']:.0f}$"
            for s, v in by_session.items()
        )

        dir_lines = "\n".join(
            f"  {d}: WR={v['wr']}% n={v['n']} pnl={v['pnl']:.0f}$"
            for d, v in by_dir.items()
        )

        prompt = f"""Tu es l'agent de contrôle autonome d'un bot de scalping XAU/USD (or).
Tu peux modifier directement ses paramètres pour améliorer les performances.
Le bot trade London (8-12h CET) et NY (14-18h CET). Hors de ces plages = interdit.

PARAMÈTRES ACTUELS:
- RSI_M5_LONG_MIN  : {current['RSI_M5_LONG_MIN']} (plage autorisée : 38-52)
- RSI_M5_SHORT_MAX : {current['RSI_M5_SHORT_MAX']} (plage autorisée : 48-62)
- ADX_MIN          : {current['ADX_MIN']} (plage autorisée : 12-30)
- Heures bloquées  : {current['BAD_HOURS_CET']}

PERFORMANCES ({total} trades):
- WR global : {stats.get('win_rate', 0)}% | PF : {stats.get('profit_factor', 0)} | PnL : {stats.get('total_pnl', 0):.0f}$
- Gain moyen : +{stats.get('avg_win', 0):.1f}$ | Perte moyenne : {stats.get('avg_loss', 0):.1f}$

WR PAR HEURE CET:
{hour_lines}

PAR SESSION:
{session_lines}

PAR DIRECTION:
{dir_lines}

RÈGLES DE DÉCISION:
1. Heures protégées (ne jamais bloquer) : 9h, 15h, 16h
2. Bloquer une heure si WR < 35% ET n >= 20 trades
3. Maximum 2 nouvelles heures bloquées par run
4. Si WR global < 40% → resserrer RSI (augmenter LONG_MIN et/ou baisser SHORT_MAX de 1-2 pts)
5. Si PF < 0.8 → envisager augmenter ADX_MIN (filtrer les tendances faibles)
6. Débloquer une heure si elle était bloquée mais n'a plus de données récentes

Réponds UNIQUEMENT avec du JSON valide (rien d'autre) :
{{
  "analysis": "2-3 phrases expliquant ce qui ne va pas et pourquoi",
  "actions": [
    {{"type": "BLOCK_HOUR", "hour": 17, "reason": "WR 18% sur 38 trades"}},
    {{"type": "SET_RSI_LONG_MIN", "value": 47.0, "reason": "WR global 36%, resserrer momentum"}}
  ]
}}

Types d'actions : BLOCK_HOUR, UNBLOCK_HOUR, SET_RSI_LONG_MIN, SET_RSI_SHORT_MAX, SET_ADX_MIN
Si rien à changer → "actions": []
"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp = client.messages.create(
                model=_LLM_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start < 0 or end <= 0:
                raise ValueError("pas de JSON dans la réponse LLM")
            decision = json.loads(raw[start:end])
        except Exception as exc:
            logger.warning("[Adaptive] erreur LLM: %s", exc)
            return {"error": str(exc)}

        # Appliquer les actions avec garde-fous
        actions_taken = []
        new_blocks = 0
        for action in decision.get("actions", []):
            atype = action.get("type", "")
            if atype == "BLOCK_HOUR" and new_blocks >= _MAX_NEW_BLOCKS_PER_RUN:
                continue
            result = self._apply_action(action)
            if result:
                actions_taken.append(result)
                if atype == "BLOCK_HOUR":
                    new_blocks += 1

        run_record = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "trades_analyzed": total,
            "analysis":        decision.get("analysis", ""),
            "actions_taken":   actions_taken,
        }

        self._run_count += 1
        with self._lock:
            self._history.append(run_record)
            if len(self._history) > 20:
                self._history = self._history[-20:]

        logger.info("[Adaptive] run #%d — %d action(s) appliquée(s)",
                    self._run_count, len(actions_taken))
        for a in actions_taken:
            logger.info("[Adaptive]   → %s", a)

        return run_record

    def _apply_action(self, action: Dict[str, Any]) -> Optional[str]:
        atype  = action.get("type", "")
        reason = action.get("reason", "")

        if atype == "BLOCK_HOUR":
            hour = int(action.get("hour", -1))
            if not (0 <= hour <= 23):
                return None
            if hour in _PROTECTED_HOURS:
                logger.info("[Adaptive] heure %dh protégée, blocage refusé", hour)
                return None
            bad = set(getattr(strategy, "BAD_HOURS_CET", set()))
            if hour in bad:
                return None
            bad.add(hour)
            setattr(strategy, "BAD_HOURS_CET", bad)
            return f"BLOCK {hour}h — {reason}"

        if atype == "UNBLOCK_HOUR":
            hour = int(action.get("hour", -1))
            bad = set(getattr(strategy, "BAD_HOURS_CET", set()))
            if hour not in bad:
                return None
            bad.discard(hour)
            setattr(strategy, "BAD_HOURS_CET", bad)
            return f"UNBLOCK {hour}h — {reason}"

        param_map = {
            "SET_RSI_LONG_MIN":  "RSI_M5_LONG_MIN",
            "SET_RSI_SHORT_MAX": "RSI_M5_SHORT_MAX",
            "SET_ADX_MIN":       "ADX_MIN",
        }
        if atype in param_map:
            param = param_map[atype]
            val   = float(action.get("value", 0))
            lo, hi = _PARAM_BOUNDS[param]
            val = max(lo, min(hi, val))
            old = getattr(strategy, param, None)
            if old is not None and abs(old - val) < 0.001:
                return None
            setattr(strategy, param, val)
            return f"{param}: {old} → {val} — {reason}"

        return None
