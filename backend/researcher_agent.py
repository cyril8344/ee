"""
researcher_agent.py
===================
Agent chercheur qui optimise les paramètres de stratégie en tâche de fond.

Toutes les 4h (hors session London/NY, sans position active), teste des
combinaisons de RSI_M5_LONG_MIN / RSI_M5_SHORT_MAX / ADX_MIN sur 1 mois
de données historiques. Applique la meilleure combinaison à la stratégie live.

Optionnel : utilise Claude API (claude-haiku-4-5-20251001) pour suggérer
le prochain ensemble d'expériences basé sur les résultats précédents.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytz
import strategy

logger = logging.getLogger(__name__)

_LLM_MODEL   = "claude-haiku-4-5-20251001"
_CET         = pytz.timezone("Europe/Paris")
_MIN_INTERVAL_S = 4 * 3600   # 4h entre deux batches
_WINDOW_MONTHS  = 1           # pretrain court pour chaque test
_MIN_TRADES     = 20          # ignorer résultats avec < 20 trades (pas fiable)

# Grille initiale
_DEFAULT_GRID: List[Dict[str, float]] = [
    {"RSI_M5_LONG_MIN": 42.0, "RSI_M5_SHORT_MAX": 58.0, "ADX_MIN": 15.0},
    {"RSI_M5_LONG_MIN": 45.0, "RSI_M5_SHORT_MAX": 55.0, "ADX_MIN": 20.0},  # référence
    {"RSI_M5_LONG_MIN": 48.0, "RSI_M5_SHORT_MAX": 52.0, "ADX_MIN": 20.0},
    {"RSI_M5_LONG_MIN": 45.0, "RSI_M5_SHORT_MAX": 55.0, "ADX_MIN": 25.0},
    {"RSI_M5_LONG_MIN": 48.0, "RSI_M5_SHORT_MAX": 52.0, "ADX_MIN": 25.0},
]

# Bornes pour la génération de grilles (cohérentes avec live_agent.BOUNDS)
_BOUNDS = {
    "RSI_M5_LONG_MIN":  (38.0, 52.0),
    "RSI_M5_SHORT_MAX": (48.0, 62.0),
    "ADX_MIN":          (12.0, 30.0),
}


def _score(result: Dict[str, Any]) -> float:
    """Score composite : PF × WR × couverture statistique."""
    pf  = float(result.get("profit_factor", 1.0))
    wr  = float(result.get("win_rate", 0.5))
    n   = int(result.get("total_trades", 0))
    coverage = min(1.0, n / 50.0)
    return pf * wr * coverage


def _clamp(val: float, key: str) -> float:
    lo, hi = _BOUNDS[key]
    return max(lo, min(hi, val))


class ResearcherAgent:
    """
    Chercheur de paramètres autonome.
    Teste des combinaisons de RSI/ADX et applique les meilleures
    à la stratégie live via setattr sur le module strategy.
    """

    def __init__(self, capital: float = 1000.0) -> None:
        self._capital   = capital
        self._results:  List[Dict[str, Any]] = []
        self._queue:    List[Dict[str, float]] = []
        self._lock      = threading.Lock()
        self._running   = False
        self._last_ts   = 0.0
        self._best_params: Optional[Dict[str, float]] = None
        self._batch_count = 0
        logger.info("[Researcher] initialisé — fenêtre=%dM, grid_init=%d",
                    _WINDOW_MONTHS, len(_DEFAULT_GRID))

    # ---- Public API -------------------------------------------------------- #

    def maybe_run(self, has_active_position: bool) -> None:
        """
        Appelé depuis la boucle principale (toutes les N secondes).
        Lance un experiment si les conditions sont réunies.
        """
        if has_active_position:
            return
        if self._running:
            return
        if strategy.BOOTSTRAP_MODE:
            return
        if time.time() - self._last_ts < _MIN_INTERVAL_S:
            return
        if not self._is_off_session():
            return

        with self._lock:
            if not self._queue:
                self._queue = self._generate_grid()
                self._batch_count += 1
                logger.info("[Researcher] batch #%d — %d expériences",
                            self._batch_count, len(self._queue))

        self._start_next()

    def request_validation(self) -> None:
        """
        Déclenché par LiveAdaptiveAgent après un ajustement de params.
        Insère les params live actuels en tête de queue pour validation rapide.
        """
        if self._running or strategy.BOOTSTRAP_MODE:
            return
        current = {
            "RSI_M5_LONG_MIN":  float(getattr(strategy, "RSI_M5_LONG_MIN",  45.0)),
            "RSI_M5_SHORT_MAX": float(getattr(strategy, "RSI_M5_SHORT_MAX", 55.0)),
            "ADX_MIN":          float(getattr(strategy, "ADX_MIN",           20.0)),
        }
        with self._lock:
            # Éviter les doublons : ne pas re-tester les mêmes params
            if current not in self._queue:
                self._queue.insert(0, current)
                logger.info("[Researcher] validation demandée par LiveAgent: %s", current)
        if self._is_off_session():
            self._start_next()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            results_copy = list(self._results)
            queue_size   = len(self._queue)
        top3 = sorted(results_copy, key=lambda r: r.get("score", 0.0), reverse=True)[:3]
        return {
            "running":       self._running,
            "queue_size":    queue_size,
            "results_total": len(results_copy),
            "batch_count":   self._batch_count,
            "best_params":   self._best_params,
            "top3":          top3,
        }

    # ---- Grid generation --------------------------------------------------- #

    def _generate_grid(self) -> List[Dict[str, float]]:
        if self._batch_count == 0 or not self._results:
            return list(_DEFAULT_GRID)
        llm_grid = self._suggest_via_llm()
        if llm_grid:
            return llm_grid
        return self._heuristic_grid()

    def _heuristic_grid(self) -> List[Dict[str, float]]:
        """Raffine autour des params live actuels (posés par LiveAdaptiveAgent ou AdaptiveAgent)."""
        # Point de départ = ce que le live utilise MAINTENANT (pas juste notre meilleur résultat)
        rsi_lo = float(getattr(strategy, "RSI_M5_LONG_MIN",  45.0))
        rsi_hi = float(getattr(strategy, "RSI_M5_SHORT_MAX", 55.0))
        adx    = float(getattr(strategy, "ADX_MIN",           20.0))
        # Si on a des résultats et que le meilleur est bien meilleur que l'actuel, on centre sur lui
        if self._results:
            best_r = max(self._results, key=lambda r: r.get("score", 0.0))
            bp = best_r.get("params", {})
            rsi_lo = float(bp.get("RSI_M5_LONG_MIN",  rsi_lo))
            rsi_hi = float(bp.get("RSI_M5_SHORT_MAX", rsi_hi))
            adx    = float(bp.get("ADX_MIN",           adx))
        grid = []
        for drsi in (-1.0, 0.0, 1.0):
            for dadx in (-2.5, 0.0, 2.5):
                combo = {
                    "RSI_M5_LONG_MIN":  _clamp(rsi_lo + drsi, "RSI_M5_LONG_MIN"),
                    "RSI_M5_SHORT_MAX": _clamp(rsi_hi - drsi, "RSI_M5_SHORT_MAX"),
                    "ADX_MIN":          _clamp(adx    + dadx, "ADX_MIN"),
                }
                if combo not in grid:
                    grid.append(combo)
        return grid[:6]

    def _suggest_via_llm(self) -> Optional[List[Dict[str, float]]]:
        """Demande à Claude de suggérer les prochains paramètres."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            top5 = sorted(self._results, key=lambda r: r.get("score", 0.0), reverse=True)[:5]
            results_text = json.dumps(top5, indent=2, default=str)

            # Injecter l'historique de trades réels si disponible
            trade_summary = ""
            try:
                import database as _db
                report = _db.get_trade_report(limit=200)
                trade_summary = (
                    "\n\nHISTORIQUE TRADES RÉELS (live/paper) :\n"
                    + report.get("llm_summary", "")
                    + "\n"
                )
            except Exception:
                pass

            prompt = (
                "Tu es un optimiseur de stratégie scalping XAU/USD.\n"
                f"Résultats des {len(self._results)} expériences de pretrain (top 5) :\n"
                f"{results_text}\n"
                f"{trade_summary}\n"
                "En tenant compte des résultats pretrain ET de l'historique réel, "
                "propose 5 nouvelles combinaisons de paramètres à tester. "
                "Réponds UNIQUEMENT avec du JSON valide (tableau) :\n"
                '[{"RSI_M5_LONG_MIN":45.0,"RSI_M5_SHORT_MAX":55.0,"ADX_MIN":20.0}, ...]\n'
                "Règles : RSI_M5_LONG_MIN ∈ [38,52], RSI_M5_SHORT_MAX ∈ [48,62], "
                "ADX_MIN ∈ [12,30]. Score = PF × WR × min(trades/50,1). Plus élevé = meilleur."
            )
            resp = client.messages.create(
                model=_LLM_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start < 0 or end <= 0:
                return None
            parsed = json.loads(raw[start:end])
            if not isinstance(parsed, list):
                return None
            grid = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                grid.append({
                    "RSI_M5_LONG_MIN":  _clamp(float(item.get("RSI_M5_LONG_MIN",  45.0)), "RSI_M5_LONG_MIN"),
                    "RSI_M5_SHORT_MAX": _clamp(float(item.get("RSI_M5_SHORT_MAX", 55.0)), "RSI_M5_SHORT_MAX"),
                    "ADX_MIN":          _clamp(float(item.get("ADX_MIN",           20.0)), "ADX_MIN"),
                })
            if grid:
                logger.info("[Researcher] LLM a suggéré %d expériences", len(grid))
                return grid
        except Exception as exc:
            logger.warning("[Researcher] erreur LLM suggestion: %s", exc)
        return None

    # ---- Experiment runner ------------------------------------------------- #

    def _is_off_session(self) -> bool:
        h = datetime.now(timezone.utc).astimezone(_CET).hour
        return not (8 <= h < 12 or 14 <= h < 18)

    def _is_pretrain_free(self) -> bool:
        try:
            import pretrain as _pt
            return not _pt.get_progress().get("running", False)
        except Exception:
            return False

    def _start_next(self) -> None:
        with self._lock:
            if not self._queue:
                return
            params = self._queue.pop(0)

        if not self._is_pretrain_free():
            with self._lock:
                self._queue.insert(0, params)
            return

        self._running = True
        end_d   = date.today().isoformat()
        start_d = (date.today() - timedelta(days=_WINDOW_MONTHS * 30)).isoformat()
        logger.info("[Researcher] test: %s (fenêtre %s→%s)", params, start_d, end_d)

        import pretrain as _pt

        def _done() -> None:
            try:
                prog   = _pt.get_progress()
                result = prog.get("last_result") or {}
                s      = _score(result)
                entry  = {
                    "params":       dict(params),
                    "score":        round(s, 4),
                    "profit_factor": round(float(result.get("profit_factor", 1.0)), 3),
                    "win_rate":     round(float(result.get("win_rate",       0.0)),  3),
                    "total_trades": int(result.get("total_trades", 0)),
                    "sl_direct_pct": round(float(result.get("sl_direct_pct", 0.0)), 3),
                }
                with self._lock:
                    self._results.append(entry)
                logger.info(
                    "[Researcher] score=%.3f PF=%.2f WR=%.0f%% n=%d → %s",
                    s, entry["profit_factor"], entry["win_rate"] * 100,
                    entry["total_trades"], params,
                )
                self._maybe_apply_best()
            finally:
                self._running = False
                self._last_ts = time.time()
                # Enchaîner l'expérience suivante si la queue n'est pas vide
                if self._queue and self._is_off_session():
                    self._start_next()

        _pt.launch_pretrain(
            start_d, end_d,
            symbol="XAUUSD",
            reset=False,           # conserver le ML Gate, tester seulement les filtres
            capital=self._capital,
            on_complete=_done,
            extra_overrides=dict(params),
        )

    # ---- Apply best -------------------------------------------------------- #

    def _maybe_apply_best(self) -> None:
        with self._lock:
            valid = [r for r in self._results if r["total_trades"] >= _MIN_TRADES]
        if not valid:
            return
        best = max(valid, key=lambda r: r["score"])

        # Ne pas appliquer si ce sont les mêmes params qu'actuellement
        if best["params"] == self._best_params:
            return

        # Comparer avec la baseline (référence = params actuel de strategy)
        baseline_score = 0.0
        baseline_combo = {
            "RSI_M5_LONG_MIN":  getattr(strategy, "RSI_M5_LONG_MIN",  45.0),
            "RSI_M5_SHORT_MAX": getattr(strategy, "RSI_M5_SHORT_MAX", 55.0),
            "ADX_MIN":          getattr(strategy, "ADX_MIN",           20.0),
        }
        for r in self._results:
            if r["params"] == baseline_combo and r["total_trades"] >= _MIN_TRADES:
                baseline_score = r["score"]
                break

        if baseline_score and best["score"] <= baseline_score * 1.02:
            logger.info("[Researcher] gain marginal (%.3f vs %.3f) — pas de changement",
                        best["score"], baseline_score)
            return

        for k, v in best["params"].items():
            if hasattr(strategy, k):
                old = getattr(strategy, k)
                setattr(strategy, k, v)
                if abs(old - v) > 0.001:
                    logger.info("[Researcher] %s: %.1f → %.1f", k, old, v)

        self._best_params = dict(best["params"])
        logger.info(
            "[Researcher] meilleurs params appliqués: score=%.3f PF=%.2f WR=%.0f%%",
            best["score"], best["profit_factor"], best["win_rate"] * 100,
        )

        # Synchroniser la DB de LiveAdaptiveAgent pour éviter qu'il écrase nos params
        try:
            import database as _db
            live_data = _db.live_agent_load("XAUUSD") or {}
            live_params = dict(live_data.get("params", {}))
            live_params.update({k: v for k, v in best["params"].items()})
            _db.live_agent_save("XAUUSD", live_params, live_data.get("trade_log", []))
            logger.info("[Researcher] DB LiveAgent synchronisée avec les meilleurs params")
        except Exception as exc:
            logger.warning("[Researcher] erreur sync DB LiveAgent: %s", exc)
