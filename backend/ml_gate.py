"""
ml_gate.py
==========
Régression logistique online (SGD) pour filtrer les entrées.

Après chaque trade clôturé, le modèle se met à jour avec les features
du contexte d'entrée et le résultat (gagné/perdu). Il prédit la
probabilité de gagner avant chaque entrée.

Avantages vs Laplace smoothing :
  - Apprend les COMBINAISONS perdantes (ex: ema9_gap faible + fin session NY)
  - Pas de sur-apprentissage : seulement 9 paramètres, L2 régularisation
  - Reste interprétable : les poids révèlent ce qui compte

Inactive pendant les 20 premiers trades (pas assez de données).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

N_MIN_TRADES  = 20    # n° de trades min avant activation du gate
THRESHOLD     = 0.45  # probabilité minimum pour autoriser l'entrée
LEARNING_RATE = 0.05  # taux d'apprentissage SGD
L2_LAMBDA     = 0.01  # régularisation L2 (évite les poids extrêmes)

FEATURE_NAMES = [
    "atr_norm",           # ATR / prix (volatilité normalisée)
    "ema9_gap_m5",        # (close - ema9) / ATR → proximité EMA9
    "ema_gap_m15",        # (ema9 - ema21) / ATR_m15 → force tendance M15
    "hour_sin",           # encodage cyclique de l'heure (sin)
    "hour_cos",           # encodage cyclique de l'heure (cos)
    "session_london",     # 1 si session London
    "session_newyork",    # 1 si session New York
    "bias_long",          # 1 si LONG, 0 si SHORT
    "pattern_weight_sum", # somme des poids patterns (système existant)
]
N_FEATURES = len(FEATURE_NAMES)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


class OnlineLogisticRegression:
    """
    Régression logistique entraînée online après chaque trade.
    Les poids sont persistés en base via database.py.
    """

    def __init__(self):
        self.weights:  List[float] = [0.0] * N_FEATURES
        self.bias_w:   float = 0.0
        self.n_samples: int  = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            import database as db
            data = db.load_ml_weights()
            if data:
                self.weights   = data.get("weights",   [0.0] * N_FEATURES)
                self.bias_w    = data.get("bias_w",    0.0)
                self.n_samples = data.get("n_samples", 0)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            import database as db
            db.save_ml_weights(self.weights, self.bias_w, self.n_samples)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    def predict(self, features: List[float]) -> float:
        """Probabilité de gagner ∈ [0, 1]."""
        z = self.bias_w + sum(w * x for w, x in zip(self.weights, features))
        return _sigmoid(z)

    def update(self, features: List[float], won: bool) -> None:
        """Mise à jour SGD après un trade clôturé."""
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

    # ------------------------------------------------------------------ #
    # Gate
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        return self.n_samples >= N_MIN_TRADES

    def gate(self, features: List[float]) -> Tuple[bool, float]:
        """
        Retourne (autorisé, probabilité).
        Si pas encore entraîné : autorisé = True, prob = -1.0 (non calculée).
        """
        if not self.is_ready:
            return True, -1.0
        prob = self.predict(features)
        return prob >= THRESHOLD, round(prob, 3)

    # ------------------------------------------------------------------ #
    # Interpretability
    # ------------------------------------------------------------------ #
    def feature_importance(self) -> Dict[str, float]:
        return {n: round(w, 4) for n, w in zip(FEATURE_NAMES, self.weights)}

    def status(self) -> Dict:
        return {
            "ready":      self.is_ready,
            "n_samples":  self.n_samples,
            "n_min":      N_MIN_TRADES,
            "threshold":  THRESHOLD,
            "bias_w":     round(self.bias_w, 4),
            "importance": self.feature_importance(),
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
    """
    Construit le vecteur de 9 features à partir des données de marché.
    Doit être appelé avec les mêmes données qu'à l'entrée du trade.
    """
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

    sess_low = (session or "").lower()
    session_london  = 1.0 if "london" in sess_low else 0.0
    session_newyork = 1.0 if "new" in sess_low or "york" in sess_low else 0.0

    bias_long = 1.0 if bias == "LONG" else 0.0

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
