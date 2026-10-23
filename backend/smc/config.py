"""Paramètres par défaut du scanner SMC (§ "Paramètres par défaut" de la spec).

Un seul endroit pour les seuils, pour qu'un réglage se change sans relire le
code — et pour qu'on voie d'un coup d'œil ce qui a bougé depuis la v1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# Fractales : n bougies de chaque côté pour valider un pivot.
FRACTAL_N: Dict[str, int] = {"D1": 3, "H4": 3, "H1": 3, "M15": 2, "M5": 2}

# --------------------------------------------------------------------------- #
# Définitions de structure — alignement sur l'indicateur réellement utilisé
# --------------------------------------------------------------------------- #
# Le scanner ne sert à rien s'il ne désigne pas les mêmes zones que celles qu'on
# voit à l'écran. Trois définitions divergeaient.
#
# PIVOT_MODE
#   "symmetric" : fractale classique, n bougies plus basses de CHAQUE côté.
#   "rolling"   : une bougie est un swing haut dès qu'aucune des n bougies
#                 SUIVANTES ne la dépasse — rien n'est exigé à sa gauche. S'y
#                 ajoute une alternance stricte haut/bas/haut.
#
#   MESURÉ (7 marches aléatoires de 800 bougies à corps réel, n=3) : "rolling"
#   donne 0 à 19 % de pivots en MOINS, jamais plus, sur 7 séries sur 7. L'effet
#   vient de l'alternance, pas de l'absence de contrainte à gauche. Attention
#   au protocole : sur une série dégénérée où open == close, les deux modes
#   s'équivalent — c'est ce qu'une première mesure a conclu à tort. Il faut des
#   bougies avec corps pour que la question ait un sens.
#   **Reste à vérifier contre une capture d'écran réelle.**
PIVOT_MODE = "rolling"

# Deux échelles de structure lues ensemble : la cassure « interne » (celle qui
# déclenche une entrée) et la cassure « de swing » (celle qui donne la
# tendance). Une seule échelle oblige à choisir entre ces deux rôles.
STRUCTURE_LENGTH_INTERNAL = 5
STRUCTURE_LENGTH_SWING    = 50

# OB_STYLE
#   "last_opposite" : dernière bougie de couleur contraire avant la cassure.
#   "leg_extreme"   : la bougie la plus EXTRÊME de la jambe comprise entre le
#                     swing cassé et la cassure, zone = son high→low complet.
#
#   MESURÉ (195 Order Blocks sur 7 marches aléatoires) : les deux définitions
#   désignent une bougie DIFFÉRENTE dans 67 % des cas, et la zone est en médiane
#   74 % plus haute (2.54 contre 1.46). Ce n'est donc pas un détail de
#   convention : c'est l'écart qui faisait que le scanner et le graphique
#   montraient deux zones à deux endroits, sans qu'aucun des deux ait l'air en
#   panne. C'est l'écart principal corrigé ici.
OB_STYLE = "leg_extreme"

# Un OB plus haut que ce multiple de l'ATR n'est pas une zone d'entrée : le stop
# qu'il impose rend tout R:R mécaniquement faux. Ce filtre manquait. Il sert
# aussi d'exclusion de volatilité pendant le choix de la bougie : une bougie de
# nouvelle imprime un extrême qui n'est l'empreinte d'aucun ordre, seulement
# d'un trou de liquidité, et c'est la bougie suivante la plus extrême qui est
# alors retenue.
OB_MAX_HEIGHT_ATR = 3.0

# OB_MITIGATION : "wick" (une mèche dans la zone suffit) ou "close" (il faut une
# clôture dedans). Le choix change le nombre de zones encore vivantes, jamais
# leur emplacement.
OB_MITIGATION = "wick"

# Zones
OB_MIN_DISPLACEMENT_ATR = 1.5   # déplacement minimum de l'impulsion qui crée l'OB
FVG_MIN_SIZE_ATR        = 0.3   # taille minimale d'un Fair Value Gap
FVG_FILL_RATIO          = 0.5   # comblé à 50 % → retiré des zones actives
ZONE_MAX_AGE_DAYS       = 4     # fraîcheur : au-delà, la zone est périmée

# Sweep de liquidité — OPTIONNEL, jamais une condition de signal (§5).
SWEEP_MAX_DEPTH_ATR     = 0.5   # au-delà, c'est une vraie cassure, pas un sweep
SWEEP_TO_CHOCH_BARS     = 8     # fenêtre M15 entre le sweep et le CHoCH déclencheur

# Poches de liquidité (Equal Highs / Equal Lows). Mêmes valeurs que
# strategy_ict._find_equal_levels, dont la définition est reprise : deux swings
# ou plus à moins de tol×ATR l'un de l'autre forment un cluster.
EQUAL_LEVEL_LOOKBACK      = 50
EQUAL_LEVEL_TOLERANCE_ATR = 0.15

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
