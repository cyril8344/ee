"""Paramètres par défaut du scanner SMC (§ "Paramètres par défaut" de la spec).

Un seul endroit pour les seuils, pour qu'un réglage se change sans relire le
code — et pour qu'on voie d'un coup d'œil ce qui a bougé depuis la v1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# Fractales : n bougies de chaque côté pour valider un pivot.
FRACTAL_N: Dict[str, int] = {"D1": 3, "H4": 3, "H1": 3, "M15": 2, "M5": 2}

# Zones
OB_MIN_DISPLACEMENT_ATR = 1.5   # déplacement minimum de l'impulsion qui crée l'OB
FVG_MIN_SIZE_ATR        = 0.3   # taille minimale d'un Fair Value Gap
FVG_FILL_RATIO          = 0.5   # comblé à 50 % → retiré des zones actives
ZONE_MAX_AGE_DAYS       = 4     # fraîcheur : au-delà, la zone est périmée

# Sweep de liquidité — OPTIONNEL, jamais une condition de signal (§5).
SWEEP_MAX_DEPTH_ATR     = 0.5   # au-delà, c'est une vraie cassure, pas un sweep
SWEEP_TO_CHOCH_BARS     = 8     # fenêtre M15 entre le sweep et le CHoCH déclencheur

# Confluence
ZONE_TOUCH_LOOKBACK_BARS = 8    # le prix doit être dans la zone, ou l'avoir touchée
                                # dans les 8 dernières M15

# Affichage
RR_MIN_NORMAL           = 1.5   # en dessous : mention "⚠️ R:R faible"
SL_MARGIN_ATR           = 0.2   # marge sous le low du sweep / de la zone

# ATR utilisé partout dans la spec
ATR_PERIOD = 14


@dataclass(frozen=True)
class SessionWindow:
    """Fenêtre horaire en heure de Paris (la spec est écrite en heure de Paris ;
    l'interne reste en UTC, cf. §10)."""
    start_h: int
    start_m: int
    end_h: int
    end_m: int


# Londres 08h–17h, New York 14h30–22h (Paris).
FOREX_SESSIONS: Tuple[SessionWindow, ...] = (
    SessionWindow(8, 0, 17, 0),
    SessionWindow(14, 30, 22, 0),
)
CRYPTO_SESSIONS: Tuple[SessionWindow, ...] = ()   # vide = 24h


@dataclass(frozen=True)
class InstrumentConfig:
    """Un instrument surveillé. La liste vit ici pour que l'ajout de BTC/USDT
    (via CCXT) soit une ligne de configuration, pas une modification de code."""
    symbol: str
    feed: str = "auto"                  # "auto" | "mt5" | "ccxt"
    sessions: Tuple[SessionWindow, ...] = FOREX_SESSIONS
    max_spread_points: float = 50.0     # XAU/USD : 50 points
    point_size: float = 0.01            # 1 point XAU/USD


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "XAUUSD": InstrumentConfig(symbol="XAUUSD"),
    # "BTCUSDT": InstrumentConfig(symbol="BTCUSDT", feed="ccxt",
    #                             sessions=CRYPTO_SESSIONS, point_size=0.1),
}
