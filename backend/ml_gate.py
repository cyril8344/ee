"""
ml_gate.py
==========
Deux mécanismes d'apprentissage complémentaires :

1. OnlineLogisticRegression — filtre probabiliste sur 9 features.
   Apprend des DEUX côtés : wins et losses ajustent les poids via SGD.
   Série noire : après 3 pertes consécutives le seuil monte automatiquement
   (+5 % par perte, plafonné à +30 %), reset dès la première victoire.

2. AdaptiveThresholds — adapte ATR_MIN, tolérance EMA9 M5 et EMA M15.
   Wins  : tire les seuils vers ce qui a marché (α = 0.08).
   Losses : pousse les seuils dans la direction opposée (α = 0.04,
             plus prudent pour ne pas sur-réagir à une seule mauvaise entrée).
   Actif après 10 trades au total (pas seulement des wins).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

N_MIN_TRADES  = 20    # trades min avant activation du ML gate
THRESHOLD     = 0.45  # probabilité de base pour autoriser une entrée
STREAK_BOOST  = 0.05  # boost du seuil par perte au-delà de 2 consécutives
STREAK_CAP    = 0.30  # boost maximum (+30 % → seuil max 0.75)
LEARNING_RATE = 0.05
L2_LAMBDA     = 0.01


# --------------------------------------------------------------------------- #
# Seuils adaptatifs
# --------------------------------------------------------------------------- #

class AdaptiveThresholds:
    """
    Adapte ATR_MIN, la tolérance EMA9 M5 et la tolérance EMA M15
    en apprenant de chaque trade — victoire ET défaite.

    Victoire → seuils se déplacent vers ce contexte (le bot l'accepte mieux).
    Défaite  → seuils s'éloignent de ce contexte (le bot le filtre davantage).

    Taux d'apprentissage win = 2 × taux loss pour ne pas sur-réagir aux pertes.
    Planchers et plafonds garantissent qu'aucun paramètre ne dérive.
    """

    ATR_RATIO_FLOOR  = 0.40
    ATR_RATIO_CEIL   = 2.50
    EMA9_MULT_FLOOR  = 0.15
    EMA9_MULT_CEIL   = 1.20
    M15_MULT_FLOOR   = 0.05
    M15_MULT_CEIL    = 0.70

    N_MIN       = 10    # trades totaux avant d'activer l'adaptation
    ALPHA_WIN   = 0.08  # taux apprentissage sur victoires
    ALPHA_LOSS  = 0.04  # taux apprentissage sur défaites (2× plus lent)

    def __init__(self, atr_min_default: float = 0.8, symbol: str = "XAUUSD"):
        self.symbol          = symbol
        self.atr_min_default = atr_min_default
        self.atr_min   = atr_min_default
        self.ema9_mult = 0.5
        self.m15_mult  = 0.3
        self.n_wins    = 0
        self.n_losses  = 0
        self.n_total   = 0
        self._load()

    def _load(self) -> None:
        try:
            import database as db
            data = db.load_adaptive_thresholds(self.symbol)
            if data:
                self.atr_min   = data.get("atr_min",   self.atr_min_default)
                self.ema9_mult = data.get("ema9_mult",  0.5)
                self.m15_mult  = data.get("m15_mult",   0.3)
                self.n_wins    = data.get("n_wins",     0)
                self.n_losses  = data.get("n_losses",   0)
                self.n_total   = data.get("n_total",    0)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            import database as db
            db.save_adaptive_thresholds(self.symbol, {
                "atr_min":   self.atr_min,
                "ema9_mult": self.ema9_mult,
                "m15_mult":  self.m15_mult,
                "n_wins":    self.n_wins,
                "n_losses":  self.n_losses,
                "n_total":   self.n_total,
            })
        except Exception:
            pass

    def update(self, ml_features: list, entry_price: float, won: bool) -> None:
        """Mise à jour après un trade clôturé (win ou loss)."""
        self.n_total += 1
        if won:
            self.n_wins += 1
        else:
            self.n_losses += 1

        if self.n_total < self.N_MIN or len(ml_features) < 3:
            self._save()
            return

        atr_norm       = ml_features[0]
        ema9_gap_ratio = abs(ml_features[1])
        m15_gap_ratio  = abs(ml_features[2])
        atr_at_entry   = atr_norm * entry_price

        if won:
            # Victoire → élargit / baisse les seuils vers ce contexte
            self.atr_min = (1 - self.ALPHA_WIN) * self.atr_min + self.ALPHA_WIN * (atr_at_entry * 0.90)
            self.ema9_mult = (1 - self.ALPHA_WIN) * self.ema9_mult + self.ALPHA_WIN * ema9_gap_ratio
            self.m15_mult  = (1 - self.ALPHA_WIN) * self.m15_mult  + self.ALPHA_WIN * m15_gap_ratio
        else:
            # Défaite → remonte ATR_MIN + resserre les tolérances EMA
            self.atr_min   = (1 - self.ALPHA_LOSS) * self.atr_min + self.ALPHA_LOSS * (atr_at_entry * 1.15)
            self.ema9_mult = self.ema9_mult * (1 - self.ALPHA_LOSS * 0.15)   # resserre de ~0.6 % par perte
            self.m15_mult  = self.m15_mult  * (1 - self.ALPHA_LOSS * 0.15)

        self.atr_min = max(
            self.atr_min_default * self.ATR_RATIO_FLOOR,
            min(self.atr_min_default * self.ATR_RATIO_CEIL, self.atr_min),
        )
        self.ema9_mult = max(self.EMA9_MULT_FLOOR, min(self.EMA9_MULT_CEIL, self.ema9_mult))
        self.m15_mult  = max(self.M15_MULT_FLOOR,  min(self.M15_MULT_CEIL,  self.m15_mult))

        self._save()

    @property
    def is_ready(self) -> bool:
        return self.n_total >= self.N_MIN

    def win_rate(self) -> Optional[float]:
        if self.n_total == 0:
            return None
        return round(self.n_wins / self.n_total, 3)

    def status(self) -> Dict:
        return {
            "ready":           self.is_ready,
            "n_wins":          self.n_wins,
            "n_losses":        self.n_losses,
            "n_total":         self.n_total,
            "n_min":           self.N_MIN,
            "win_rate":        self.win_rate(),
            "atr_min":         round(self.atr_min,   4),
            "atr_min_default": self.atr_min_default,
            "ema9_mult":       round(self.ema9_mult, 3),
            "m15_mult":        round(self.m15_mult,  3),
        }


# --------------------------------------------------------------------------- #
# ML Gate — régression logistique online
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    "atr_norm",
    "ema9_gap_m5",
    "ema_gap_m15",
    "hour_sin",
    "hour_cos",
    "session_london",
    "session_newyork",
    "bias_long",
    "pattern_weight_sum",
]
N_FEATURES = len(FEATURE_NAMES)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


class OnlineLogisticRegression:
    """
    Régression logistique online avec détecteur de série noire.

    Série noire : après 3 pertes consécutives, le seuil d'entrée monte
    de +5 % par perte supplémentaire (plafonné à +30 %).
    Reset automatique dès la première victoire.
    """

    def __init__(self):
        self.weights:           List[float] = [0.0] * N_FEATURES
        self.bias_w:            float = 0.0
        self.n_samples:         int   = 0
        self.consecutive_losses: int  = 0
        self._load()

    def _load(self) -> None:
        try:
            import database as db
            data = db.load_ml_weights()
            if data:
                self.weights            = data.get("weights",            [0.0] * N_FEATURES)
                self.bias_w             = data.get("bias_w",             0.0)
                self.n_samples          = data.get("n_samples",          0)
                self.consecutive_losses = data.get("consecutive_losses", 0)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            import database as db
            db.save_ml_weights(
                self.weights, self.bias_w, self.n_samples,
                consecutive_losses=self.consecutive_losses,
            )
        except Exception:
            pass

    def predict(self, features: List[float]) -> float:
        z = self.bias_w + sum(w * x for w, x in zip(self.weights, features))
        return _sigmoid(z)

    def update(self, features: List[float], won: bool) -> None:
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        y     = 1.0 if won else 0.0
        y_hat = self.predict(features)
        error = y - y_hat
        self.bias_w += LEARNING_RATE * error
        for i, x in enumerate(features):
            self.weights[i] += LEARNING_RATE * (
                error * x - L2_LAMBDA * self.weights[i]
            )
        self.n_samples += 1
        self._save()

    @property
    def streak_boost(self) -> float:
        """Boost du seuil dû à la série noire courante."""
        extra = max(0, self.consecutive_losses - 2)
        return min(STREAK_CAP, extra * STREAK_BOOST)

    @property
    def effective_threshold(self) -> float:
        return THRESHOLD + self.streak_boost

    @property
    def is_ready(self) -> bool:
        return self.n_samples >= N_MIN_TRADES

    def gate(self, features: List[float]) -> Tuple[bool, float]:
        if not self.is_ready:
            return True, -1.0
        prob = self.predict(features)
        return prob >= self.effective_threshold, round(prob, 3)

    def feature_importance(self) -> Dict[str, float]:
        return {n: round(w, 4) for n, w in zip(FEATURE_NAMES, self.weights)}

    def status(self) -> Dict:
        return {
            "ready":               self.is_ready,
            "n_samples":           self.n_samples,
            "n_min":               N_MIN_TRADES,
            "threshold":           round(self.effective_threshold, 3),
            "threshold_base":      THRESHOLD,
            "consecutive_losses":  self.consecutive_losses,
            "streak_boost":        round(self.streak_boost, 3),
            "bias_w":              round(self.bias_w, 4),
            "importance":          self.feature_importance(),
        }


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(
    m5,
    m15,
    bias: str,
    session: str,
    pattern_weight_sum: float,
    ts,
) -> List[float]:
    import math as _math

    cur5  = m5.iloc[-1]
    cur15 = m15.iloc[-1] if len(m15) > 0 else cur5

    price    = float(cur5.get("close", 1) or 1)
    atr5     = float(cur5.get("atr",   0) or 0)
    ema9_5   = float(cur5.get("ema9",  price) or price)
    atr15    = float(cur15.get("atr",  atr5) or atr5)
    ema9_15  = float(cur15.get("ema9",  price) or price)
    ema21_15 = float(cur15.get("ema21", price) or price)

    atr_norm    = atr5 / price if price > 0 else 0.0
    ema9_gap_m5 = (price - ema9_5) / atr5 if atr5 > 0 else 0.0
    ema_gap_m15 = (ema9_15 - ema21_15) / atr15 if atr15 > 0 else 0.0

    hour       = ts.hour + ts.minute / 60.0
    hour_sin   = _math.sin(2 * _math.pi * hour / 24.0)
    hour_cos   = _math.cos(2 * _math.pi * hour / 24.0)

    sess_low        = (session or "").lower()
    session_london  = 1.0 if "london" in sess_low else 0.0
    session_newyork = 1.0 if "new" in sess_low or "york" in sess_low else 0.0
    bias_long       = 1.0 if bias == "LONG" else 0.0

    return [
        atr_norm,
        ema9_gap_m5,
        ema_gap_m15,
        hour_sin,
        hour_cos,
        session_london,
        session_newyork,
        bias_long,
        float(pattern_weight_sum),
    ]
