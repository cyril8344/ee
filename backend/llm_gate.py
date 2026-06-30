"""
llm_gate.py
===========
Porte LLM optionnelle (claude-haiku-4-5-20251001) pour valider les signaux M5.

Activée uniquement quand ANTHROPIC_API_KEY est défini et BOOTSTRAP_MODE=False.
Timeout 4s pour ne pas bloquer la boucle M5 de 5s.
En cas d'erreur/timeout → passe le signal sans blocage (fail-open).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_CONFIDENCE_THRESHOLD = 0.55   # en dessous → HOLD implicite

_SYSTEM = (
    "Tu es un analyste scalping XAU/USD. Reçois les indicateurs d'un signal M5 "
    "et réponds UNIQUEMENT avec du JSON valide (sans markdown) :\n"
    '{"action":"LONG"|"SHORT"|"HOLD","confidence":0.0-1.0,"reason":"<1-2 phrases>"}\n'
    "- LONG / SHORT si les indicateurs confirment clairement le signal\n"
    "- HOLD si les indicateurs sont contradictoires, marginaux ou le risque est élevé\n"
    f"- confidence ≥ {_CONFIDENCE_THRESHOLD} pour valider, sinon HOLD implicite"
)

_TMPL = (
    "Signal {direction} XAUUSD — Session {session}\n\n"
    "RSI M5      = {rsi_m5:.1f}  (LONG >45, SHORT <55)\n"
    "RSI M15     = {rsi_m15:.1f}\n"
    "ATR M5      = {atr:.2f}\n"
    "ADX H1      = {adx_h1:.1f}  (seuil 20)\n"
    "Biais H1    = {bias}\n"
    "VWAP side   = {vwap}\n"
    "Patterns    = {patterns}\n"
    "Poids total = {weight:.2f}\n"
    "Score ML    = {ml_score:.2f}\n"
    "EMA200 dist = {ema200_dist:.2f}×ATR\n\n"
    "Confirmes-tu ce signal {direction} ?"
)

_client_cache: Optional[Any] = None
_client_lock = threading.Lock()


def _get_client() -> Optional[Any]:
    global _client_cache
    with _client_lock:
        if _client_cache is not None:
            return _client_cache
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        try:
            import anthropic
            _client_cache = anthropic.Anthropic(api_key=api_key)
            return _client_cache
        except ImportError:
            logger.warning("[LLMGate] anthropic SDK non installé — pip install anthropic")
            return None


class LLMGate:
    """
    Gate LLM (claude-haiku) — valide chaque signal M5 avec un LLM rapide.
    Désactivée automatiquement si ANTHROPIC_API_KEY absent ou BOOTSTRAP_MODE.
    Thread-safe, timeout 4s, fail-open en cas d'erreur.
    """

    def __init__(self) -> None:
        self._enabled = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        self._n_pass = 0
        self._n_hold = 0
        self._n_error = 0
        if self._enabled:
            logger.info("[LLMGate] activé — modèle=%s", _MODEL)
        else:
            logger.info("[LLMGate] désactivé (ANTHROPIC_API_KEY absent)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def analyze(
        self,
        snap: Dict[str, Any],
        direction: str,
        timeout: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Analyse le snapshot de marché et valide le signal.

        Retourne {"action": str, "confidence": float, "reason": str}.
        En cas d'erreur/timeout → passe le signal (fail-open).
        """
        if not self._enabled:
            return {"action": direction, "confidence": 0.5, "reason": "désactivé"}

        client = _get_client()
        if client is None:
            return {"action": direction, "confidence": 0.5, "reason": "client absent"}

        prompt = _TMPL.format(
            direction=direction.upper(),
            session=snap.get("session", "?"),
            rsi_m5=snap.get("rsi_m5", 50.0),
            rsi_m15=snap.get("rsi_m15", 50.0),
            atr=snap.get("atr", 0.0),
            adx_h1=snap.get("adx_h1", 0.0),
            bias=snap.get("bias", "?"),
            vwap="au-dessus" if snap.get("vwap_side", 1) else "en-dessous",
            patterns=", ".join(snap.get("patterns", [])) or "aucun",
            weight=snap.get("pattern_weight", 0.0),
            ml_score=snap.get("ml_score", 0.5),
            ema200_dist=snap.get("ema200_dist", 0.0),
        )

        # Default: fail-open (don't block signal on error)
        result: Dict[str, Any] = {
            "action": direction.upper(),
            "confidence": 0.5,
            "reason": "timeout",
        }
        box = [result]

        def _call() -> None:
            try:
                resp = client.messages.create(
                    model=_MODEL,
                    max_tokens=150,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                parsed = json.loads(raw)
                box[0] = {
                    "action": str(parsed.get("action", direction)).upper(),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "reason": str(parsed.get("reason", "")),
                }
            except Exception as exc:
                logger.warning("[LLMGate] erreur API: %s", exc)
                self._n_error += 1
                box[0] = {
                    "action": direction.upper(),
                    "confidence": 0.5,
                    "reason": f"erreur: {exc}",
                }

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        r = box[0]
        if r["action"] == "HOLD" or r["confidence"] < _CONFIDENCE_THRESHOLD:
            self._n_hold += 1
        else:
            self._n_pass += 1
        logger.info("[LLMGate] %s → %s conf=%.2f (%s)",
                    direction.upper(), r["action"], r["confidence"], r["reason"][:60])
        return r

    def status(self) -> Dict[str, Any]:
        total = self._n_pass + self._n_hold + self._n_error
        return {
            "enabled": self._enabled,
            "total_calls": total,
            "passed": self._n_pass,
            "held": self._n_hold,
            "errors": self._n_error,
            "pass_rate": round(self._n_pass / max(total, 1), 3),
        }
