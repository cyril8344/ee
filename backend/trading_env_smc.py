"""
trading_env_smc.py
==================
SMC/ICT-guided Gymnasium trading environment.

Étend TradingEnv de base avec :
  - Détection des Order Blocks (OB) et Fair Value Gaps (FVG)
  - Features SMC injectées dans l'espace d'état
  - Fonction de récompense alignée sur les concepts ICT/SMC

Espace d'observation : 117 features (109 base + 8 SMC)
  Base (109) : 20 bougies OHLCV + indicateurs + portfolio
  SMC (8)    : distances OB, statut OB, distances FVG, type FVG, wick rejet

Fonction de récompense SMC :
  +FVG_TP_BONUS       : fermeture dans la zone 50-100% du FVG opposé
  -OB_PIERCE_PENALTY  : bougie clôture >50% dans l'OB sans mèche de rejet
  -OB_ORIGIN_PENALTY  : prix dépasse l'origine (low/high) de l'OB = invalidation totale
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from trading_env import TradingEnv, LOOKBACK


# ─────────────────────────────────────────────────────────────────────────────
# SMC reward constants
# ─────────────────────────────────────────────────────────────────────────────
FVG_TP_BONUS      =  0.004   # bonus pour fermeture dans zone FVG 50-100%
OB_PIERCE_PENALTY = -0.008   # pénalité : OB transpercé >50% sans wick
OB_ORIGIN_PENALTY = -0.018   # pénalité maximale : origine OB franchie


# ─────────────────────────────────────────────────────────────────────────────
# SMC feature detectors (standalone, no pandas dependency in hot path)
# ─────────────────────────────────────────────────────────────────────────────

def _find_order_blocks(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    atr: float,
    lookback: int = 30,
) -> List[Dict[str, Any]]:
    """
    Détecte les Order Blocks sur la fenêtre.

    Bullish OB : dernière bougie baissière avant une impulsion haussière.
    Bearish OB : dernière bougie haussière avant une impulsion baissière.

    Retourne : liste de dicts {type, low, high, origin}
      - type   : 'bullish' | 'bearish'
      - low    : borne basse de la zone OB
      - high   : borne haute de la zone OB
      - origin : niveau mathématique inviolable (low de l'OB bullish, high du bearish)
    """
    n = len(closes)
    sub_start = max(0, n - lookback)
    obs: List[Dict[str, Any]] = []

    for i in range(sub_start + 1, n - 1):
        impulse = abs(closes[i + 1] - opens[i + 1])
        if impulse < 0.5 * atr:
            continue
        # Bullish OB : bougie baissière suivie d'une impulsion haussière
        if closes[i] < opens[i] and closes[i + 1] > opens[i + 1]:
            obs.append({
                "type":   "bullish",
                "low":    float(closes[i]),   # borne basse (close baissier)
                "high":   float(opens[i]),    # borne haute (open baissier)
                "origin": float(lows[i]),     # origine = low de la bougie OB
            })
        # Bearish OB : bougie haussière suivie d'une impulsion baissière
        elif closes[i] > opens[i] and closes[i + 1] < opens[i + 1]:
            obs.append({
                "type":   "bearish",
                "low":    float(opens[i]),    # borne basse (open haussier)
                "high":   float(closes[i]),   # borne haute (close haussier)
                "origin": float(highs[i]),    # origine = high de la bougie OB
            })

    return obs


def _find_fvgs(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 30,
) -> List[Dict[str, Any]]:
    """
    Détecte les Fair Value Gaps (imbalances à 3 bougies).

    Bullish FVG : lows[i] > highs[i-2]  → gap haussier entre wick[i-2] et wick[i]
    Bearish FVG : highs[i] < lows[i-2]  → gap baissier

    Retourne : liste de dicts {type, low, high, midpoint, pct50, pct100}
    """
    n = len(highs)
    sub_start = max(2, n - lookback)
    fvgs: List[Dict[str, Any]] = []

    for i in range(sub_start, n):
        # Bullish FVG : creux de la bougie i > sommet de la bougie i-2
        if lows[i] > highs[i - 2]:
            fvg_low  = float(highs[i - 2])
            fvg_high = float(lows[i])
            fvgs.append({
                "type":     "bullish",
                "low":      fvg_low,
                "high":     fvg_high,
                "midpoint": (fvg_low + fvg_high) / 2,
                "pct50":    fvg_low + (fvg_high - fvg_low) * 0.50,
                "pct100":   fvg_high,
            })
        # Bearish FVG : sommet de la bougie i < creux de la bougie i-2
        elif highs[i] < lows[i - 2]:
            fvg_low  = float(highs[i])
            fvg_high = float(lows[i - 2])
            fvgs.append({
                "type":     "bearish",
                "low":      fvg_low,
                "high":     fvg_high,
                "midpoint": (fvg_low + fvg_high) / 2,
                "pct50":    fvg_low + (fvg_high - fvg_low) * 0.50,
                "pct100":   fvg_high,
            })

    return fvgs


def _has_rejection_wick(
    open_: float, high: float, low: float, close: float,
    position: int,
    ob_50pct: float,
    atr: float,
) -> bool:
    """
    Détecte la présence d'une mèche de rejet significative (>= 0.3 ATR).

    Pour un LONG (position=+1) : mèche inférieure (rejet du bas).
    Pour un SHORT (position=-1): mèche supérieure (rejet du haut).
    """
    body_low  = min(open_, close)
    body_high = max(open_, close)
    min_wick  = 0.3 * atr

    if position == 1:    # long : on cherche une mèche basse de rejet
        lower_wick = body_low - low
        return lower_wick >= min_wick and close > ob_50pct
    elif position == -1: # short : mèche haute de rejet
        upper_wick = high - body_high
        return upper_wick >= min_wick and close < ob_50pct
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SMC Environment
# ─────────────────────────────────────────────────────────────────────────────

class SmcTradingEnv(TradingEnv):
    """
    Version SMC/ICT de TradingEnv.

    Ajoute 8 features SMC à l'observation et 3 composantes de récompense :
      1. Bonus TP-FVG
      2. Pénalité OB transpercé
      3. Pénalité maximale OB origine franchie
    """

    N_SMC_FEATURES = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Étendre l'espace d'observation
        n_base = LOOKBACK * 5 + 7 + 2           # 109
        n_total = n_base + self.N_SMC_FEATURES   # 117
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_total,),
            dtype=np.float32,
        )

        # OB de référence stocké à l'entrée de position
        self._entry_ob: Optional[Dict[str, Any]] = None

    # ── Gymnasium overrides ──────────────────────────────────────────────────

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = super().reset(**kwargs)
        self._entry_ob = None
        return self._observe_smc(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Capturer l'état AVANT le step pour les calculs de récompense
        price   = float(self.df.loc[self._cursor, "close"])
        prev_pos = self._position

        # Récupérer la fenêtre courante pour les détections SMC
        start  = self._cursor - LOOKBACK
        window = self.df.iloc[start: self._cursor]
        highs  = window["high"].values.astype(np.float64)
        lows   = window["low"].values.astype(np.float64)
        opens  = window["open"].values.astype(np.float64)
        closes = window["close"].values.astype(np.float64)
        atr    = self._atr(highs, lows, closes, 14)

        ob_list  = _find_order_blocks(highs, lows, opens, closes, atr)
        fvg_list = _find_fvgs(highs, lows)

        # 2. Exécuter le step de base (reward brut + transition d'état)
        _, base_reward, terminated, truncated, info = super().step(action)

        # 3. Calculer les composantes de récompense SMC
        smc_reward = self._smc_reward(
            action=action,
            prev_position=prev_pos,
            price=price,
            ob_list=ob_list,
            fvg_list=fvg_list,
            window_open=float(self.df.loc[self._cursor - 1, "open"])
                if self._cursor > 0 else price,
            window_high=float(self.df.loc[self._cursor - 1, "high"])
                if self._cursor > 0 else price,
            window_low=float(self.df.loc[self._cursor - 1, "low"])
                if self._cursor > 0 else price,
            atr=atr,
        )

        total_reward = base_reward + smc_reward

        return self._observe_smc(), float(total_reward), terminated, truncated, info

    # ── SMC Reward ───────────────────────────────────────────────────────────

    def _smc_reward(
        self,
        action: int,
        prev_position: int,
        price: float,
        ob_list: List[Dict],
        fvg_list: List[Dict],
        window_open: float,
        window_high: float,
        window_low: float,
        atr: float,
    ) -> float:
        reward = 0.0

        # ── Stocker l'OB de référence à l'ouverture d'une position ──────────
        if prev_position == 0 and self._position != 0:
            self._entry_ob = self._nearest_ob(price, self._position, ob_list)

        # ── Réinitialiser l'OB de référence à la fermeture ──────────────────
        if prev_position != 0 and self._position == 0:
            self._entry_ob = None

        # ── 1. Bonus TP-FVG ─────────────────────────────────────────────────
        # Si l'agent ferme sa position dans la zone 50-100% du FVG opposé
        if action == 2 and prev_position == 1:  # fermeture d'un LONG
            fvg = self._nearest_fvg(price, direction=1, fvg_list=fvg_list)
            if fvg is not None and fvg["pct50"] <= price <= fvg["pct100"]:
                # L'agent a correctement ciblé le FVG comme zone de TP
                reward += FVG_TP_BONUS
        elif action == 2 and prev_position == -1:  # fermeture d'un SHORT
            fvg = self._nearest_fvg(price, direction=-1, fvg_list=fvg_list)
            if fvg is not None and fvg["pct50"] >= price >= fvg["low"]:
                reward += FVG_TP_BONUS

        # ── 2. Pénalité OB transpercé (>50% sans mèche de rejet) ────────────
        # L'OB de référence est celui qui était actif à l'entrée
        if self._position != 0 and self._entry_ob is not None:
            ob   = self._entry_ob
            ob50 = ob["low"] + (ob["high"] - ob["low"]) * 0.50

            has_wick = _has_rejection_wick(
                window_open, window_high, window_low, price,
                self._position, ob50, atr,
            )

            if self._position == 1:   # LONG : l'OB de support est EN-DESSOUS
                # Pénalité si le prix clôture sous les 50% de l'OB sans wick
                if price < ob50 and not has_wick:
                    reward += OB_PIERCE_PENALTY

            elif self._position == -1:  # SHORT : l'OB de résistance est AU-DESSUS
                if price > ob50 and not has_wick:
                    reward += OB_PIERCE_PENALTY

        # ── 3. Pénalité maximale : origine OB franchie ───────────────────────
        # Le low/high de l'OB est la barrière mathématique inviolable en SMC
        if self._position != 0 and self._entry_ob is not None:
            ob = self._entry_ob
            if self._position == 1 and price < ob["origin"]:
                # Pour un long : prix sous le low de l'OB = invalidation
                reward += OB_ORIGIN_PENALTY
            elif self._position == -1 and price > ob["origin"]:
                # Pour un short : prix au-dessus du high de l'OB = invalidation
                reward += OB_ORIGIN_PENALTY

        return reward

    # ── SMC Observation ─────────────────────────────────────────────────────

    def _observe_smc(self) -> np.ndarray:
        """Observation de base + 8 features SMC normalisées."""
        base_obs = self._observe()   # 109 features de TradingEnv

        start  = self._cursor - LOOKBACK
        window = self.df.iloc[start: self._cursor]
        highs  = window["high"].values.astype(np.float64)
        lows   = window["low"].values.astype(np.float64)
        opens  = window["open"].values.astype(np.float64)
        closes = window["close"].values.astype(np.float64)
        price  = closes[-1]
        atr    = self._atr(highs, lows, closes, 14) + 1e-8

        ob_list  = _find_order_blocks(highs, lows, opens, closes, atr)
        fvg_list = _find_fvgs(highs, lows)

        # OB le plus proche dans le sens du bias courant
        ob_ref = (self._entry_ob if self._entry_ob is not None
                  else self._nearest_ob(price, self._position or 1, ob_list))

        if ob_ref is not None:
            ob_dist_high = np.clip((price - ob_ref["high"]) / atr, -3, 3) / 3  # [-1,1]
            ob_dist_low  = np.clip((price - ob_ref["low"])  / atr, -3, 3) / 3
            ob_inside    = 1.0 if ob_ref["low"] <= price <= ob_ref["high"] else -1.0
            ob_type      = 1.0 if ob_ref["type"] == "bullish" else -1.0
        else:
            ob_dist_high = ob_dist_low = 0.0
            ob_inside    = 0.0
            ob_type      = 0.0

        # FVG cible (opposé à la direction de position)
        fvg_ref = self._nearest_fvg(price, self._position or 1, fvg_list)

        if fvg_ref is not None:
            fvg_dist_50  = np.clip((fvg_ref["pct50"]  - price) / atr, -3, 3) / 3
            fvg_dist_100 = np.clip((fvg_ref["pct100"] - price) / atr, -3, 3) / 3
            fvg_type     = 1.0 if fvg_ref["type"] == "bearish" else -1.0
        else:
            fvg_dist_50 = fvg_dist_100 = 0.0
            fvg_type = 0.0

        # Mèche de rejet sur la dernière bougie
        cur_open  = float(window["open"].iloc[-1])
        cur_high  = float(window["high"].iloc[-1])
        cur_low   = float(window["low"].iloc[-1])
        ob50      = ((ob_ref["low"] + ob_ref["high"]) / 2) if ob_ref else price
        wick_flag = 1.0 if _has_rejection_wick(
            cur_open, cur_high, cur_low, price,
            self._position or 1, ob50, atr
        ) else -1.0

        smc_features = np.array([
            ob_dist_high,   # distance prix → borne haute OB  [-1,1]
            ob_dist_low,    # distance prix → borne basse OB  [-1,1]
            ob_inside,      # prix dans l'OB (+1) ou dehors (-1)
            ob_type,        # OB haussier (+1) ou baissier (-1)
            fvg_dist_50,    # distance au FVG 50%  [-1,1]
            fvg_dist_100,   # distance au FVG 100% [-1,1]
            fvg_type,       # type FVG  ±1
            wick_flag,      # mèche de rejet présente (+1) ou non (-1)
        ], dtype=np.float32)

        return np.concatenate([base_obs, smc_features])

    # ── Helpers SMC ──────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_ob(
        price: float,
        position: int,
        ob_list: List[Dict],
    ) -> Optional[Dict]:
        """
        OB le plus pertinent selon la direction :
        - LONG  (+1) : cherche un OB BULLISH en-dessous (support)
        - SHORT (-1) : cherche un OB BEARISH au-dessus (résistance)
        """
        if not ob_list:
            return None
        if position >= 0:  # long ou flat → OB bullish sous le prix
            candidates = [ob for ob in ob_list
                          if ob["type"] == "bullish" and ob["high"] <= price]
        else:              # short → OB bearish au-dessus
            candidates = [ob for ob in ob_list
                          if ob["type"] == "bearish" and ob["low"] >= price]
        if not candidates:
            # Fallback : n'importe quel OB
            candidates = ob_list
        return max(candidates, key=lambda ob: ob["high"] if position >= 0
                                              else -ob["low"])

    @staticmethod
    def _nearest_fvg(
        price: float,
        direction: int,
        fvg_list: List[Dict],
    ) -> Optional[Dict]:
        """
        FVG cible selon la direction :
        - LONG  (+1) : cherche un FVG BEARISH au-dessus (zone de TP)
        - SHORT (-1) : cherche un FVG BULLISH en-dessous (zone de TP)
        """
        if not fvg_list:
            return None
        if direction >= 0:
            targets = [f for f in fvg_list
                       if f["type"] == "bearish" and f["low"] > price]
        else:
            targets = [f for f in fvg_list
                       if f["type"] == "bullish" and f["high"] < price]
        if not targets:
            return None
        return min(targets, key=lambda f: abs(f["midpoint"] - price))
