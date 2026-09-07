"""Zones : Order Blocks, Fair Value Gaps, premium/discount, fraîcheur (§4).

Une zone est un endroit où l'on accepterait d'entrer. Quatre conditions dans la
spec, et chacune retire des candidats :

1. **Order Block** — dernière bougie baissière avant une impulsion haussière
   **qui casse une structure** (BOS/CHoCH), et miroir pour la vente. Le lien
   avec la cassure est ce qui distingue un OB d'une simple bougie rouge : sans
   lui on retombe sur le détecteur du graphique principal, qui empile 5 à 7
   zones dont la plupart sont déjà mortes.

2. **Déplacement ≥ 1,5 × ATR(14)** du timeframe. Une cassure molle ne laisse pas
   d'ordre institutionnel derrière elle.

3. **Fraîcheur** — jamais retestée, et 4 jours calendaires maximum. Une zone
   touchée est `mitigated` et ne génère plus de signal ; une zone vieille est
   `expired`. Les deux sont conservées dans la liste avec leur état plutôt que
   supprimées : « aucune zone active » et « trois zones toutes mitigées » sont
   deux situations différentes, et l'alerte doit pouvoir le dire.

4. **Premium/discount** — on n'achète que sous les 50 % du range H1 courant, on
   ne vend qu'au-dessus. C'est le filtre qui empêche d'acheter un sommet.

Anti look-ahead, comme dans structure.py : la mitigation d'une zone n'est jugée
que sur les bougies **postérieures** à sa formation, et une zone n'existe qu'à
partir de la bougie qui a confirmé la cassure — pas de celle qui l'a créée.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from smc import config
from smc.structure import StructureEvent, find_events, last_swings

ZoneKind = Literal["OB", "FVG"]
ZoneSide = Literal["bullish", "bearish"]
ZoneState = Literal["active", "mitigated", "expired"]


def atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """True Range moyen. Réimplémenté ici plutôt qu'importé de `strategy` : ce
    paquet ne doit dépendre d'aucun module de la boucle de trading, pour qu'un
    réglage du bot ne puisse jamais déplacer une zone du scanner."""
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


@dataclass
class Zone:
    """Une zone de prix, avec l'état qui dit si elle vaut encore quelque chose."""
    kind: ZoneKind
    side: ZoneSide
    low: float
    high: float
    index: int                 # bougie qui a formé la zone
    time: Any
    created_index: int         # bougie à partir de laquelle la zone est connue
    timeframe: str
    displacement_atr: float = 0.0
    state: ZoneState = "active"
    mitigated_index: Optional[int] = None
    filled_ratio: float = 0.0
    discount_pct: Optional[float] = None    # position dans le range H1, 0=bas 100=haut

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    @property
    def height(self) -> float:
        return self.high - self.low

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "side": self.side, "low": self.low,
                "high": self.high, "mid": self.mid, "height": self.height,
                "index": self.index, "time": self.time,
                "created_index": self.created_index, "timeframe": self.timeframe,
                "displacement_atr": self.displacement_atr, "state": self.state,
                "mitigated_index": self.mitigated_index,
                "filled_ratio": self.filled_ratio,
                "discount_pct": self.discount_pct}


# --------------------------------------------------------------------------- #
# Order Blocks
# --------------------------------------------------------------------------- #
def find_order_blocks(df: pd.DataFrame, timeframe: str,
                      events: Optional[List[StructureEvent]] = None,
                      min_displacement_atr: float = config.OB_MIN_DISPLACEMENT_ATR,
                      ) -> List[Zone]:
    """OB attachés à une cassure de structure, un par événement au plus.

    Pour un BOS/CHoCH haussier en `e.index` : on remonte depuis la bougie de
    cassure jusqu'à la dernière bougie **baissière**, et c'est elle l'OB. Zone =
    open→low (§4). Miroir exact pour un événement baissier (open→high).

    Le déplacement est mesuré de l'extrême de la bougie OB jusqu'à la clôture qui
    a cassé — c'est l'impulsion réellement produite, pas la taille d'une bougie.
    """
    n = config.FRACTAL_N.get(timeframe.upper(), 3)
    events = events if events is not None else find_events(df, n)
    if not events:
        return []

    atr_series = atr(df).values
    opens, highs = df["open"].values, df["high"].values
    lows, closes = df["low"].values, df["close"].values
    idx = df.index

    zones: List[Zone] = []
    for e in events:
        atr_val = float(atr_series[e.index]) if e.index < len(atr_series) else float("nan")
        if not (atr_val > 0):
            continue

        # Remonter jusqu'à la dernière bougie contrariante avant la cassure.
        ob_i = None
        for j in range(e.index, max(-1, e.index - 20), -1):
            if j < 0:
                break
            if e.direction == "bullish" and closes[j] < opens[j]:
                ob_i = j
                break
            if e.direction == "bearish" and closes[j] > opens[j]:
                ob_i = j
                break
        if ob_i is None:
            continue

        if e.direction == "bullish":
            displacement = float(closes[e.index]) - float(lows[ob_i])
            low, high = float(lows[ob_i]), float(opens[ob_i])
        else:
            displacement = float(highs[ob_i]) - float(closes[e.index])
            low, high = float(opens[ob_i]), float(highs[ob_i])

        if high <= low:
            continue
        ratio = displacement / atr_val
        if ratio < min_displacement_atr:
            continue

        zones.append(Zone(
            kind="OB", side=e.direction, low=low, high=high,
            index=ob_i, time=idx[ob_i],
            # La zone n'est connue qu'à la clôture qui a cassé la structure.
            created_index=e.index,
            timeframe=timeframe.upper(), displacement_atr=ratio,
        ))
    return zones


# --------------------------------------------------------------------------- #
# Fair Value Gaps
# --------------------------------------------------------------------------- #
def find_fvgs(df: pd.DataFrame, timeframe: str,
              min_size_atr: float = config.FVG_MIN_SIZE_ATR) -> List[Zone]:
    """Imbalances à trois bougies : gap entre le high de la 1re et le low de la
    3e (haussier), miroir pour la baisse. Taille minimale 0,3 × ATR(14)."""
    if len(df) < 3:
        return []
    atr_series = atr(df).values
    highs, lows = df["high"].values, df["low"].values
    idx = df.index

    zones: List[Zone] = []
    for i in range(2, len(df)):
        atr_val = float(atr_series[i])
        if not (atr_val > 0):
            continue
        h1, l1 = float(highs[i - 2]), float(lows[i - 2])
        h3, l3 = float(highs[i]), float(lows[i])

        if l3 > h1 and (l3 - h1) >= min_size_atr * atr_val:
            zones.append(Zone(kind="FVG", side="bullish", low=h1, high=l3,
                              index=i - 1, time=idx[i - 1], created_index=i,
                              timeframe=timeframe.upper(),
                              displacement_atr=(l3 - h1) / atr_val))
        elif l1 > h3 and (l1 - h3) >= min_size_atr * atr_val:
            zones.append(Zone(kind="FVG", side="bearish", low=h3, high=l1,
                              index=i - 1, time=idx[i - 1], created_index=i,
                              timeframe=timeframe.upper(),
                              displacement_atr=(l1 - h3) / atr_val))
    return zones


# --------------------------------------------------------------------------- #
# Mitigation, comblement, fraîcheur
# --------------------------------------------------------------------------- #
def update_states(df: pd.DataFrame, zones: List[Zone],
                  upto: Optional[int] = None,
                  max_age_days: int = config.ZONE_MAX_AGE_DAYS,
                  fill_ratio: float = config.FVG_FILL_RATIO) -> List[Zone]:
    """Marque chaque zone `active` / `mitigated` / `expired` à la bougie `upto`.

    - OB : mitigée dès que le prix **retouche** la zone (§4).
    - FVG : comblé à 50 % → `mitigated` (la spec dit `filled`, même effet — la
      zone sort des actives ; on garde un seul vocabulaire d'état).
    - Fraîcheur : plus de `max_age_days` jours calendaires → `expired`.

    Seules les bougies **postérieures** à `created_index` sont examinées : une
    zone ne peut pas être mitigée par la bougie qui l'a créée, ni par le passé.
    """
    limit = len(df) - 1 if upto is None else upto
    if limit < 0 or df.empty:
        return zones
    highs, lows = df["high"].values, df["low"].values
    idx = df.index
    now_ts = idx[limit]

    for z in zones:
        z.state = "active"
        z.mitigated_index = None
        z.filled_ratio = 0.0

        if z.created_index > limit:
            # Pas encore connue à cette bougie : on ne la présente pas comme active.
            z.state = "expired"
            continue

        for j in range(z.created_index + 1, limit + 1):
            bar_low, bar_high = float(lows[j]), float(highs[j])
            if bar_high < z.low or bar_low > z.high:
                continue
            if z.kind == "OB":
                z.state = "mitigated"
                z.mitigated_index = j
                break
            # FVG : profondeur de pénétration depuis le bord d'entrée.
            if z.side == "bullish":
                penetration = (z.high - max(bar_low, z.low)) / z.height
            else:
                penetration = (min(bar_high, z.high) - z.low) / z.height
            z.filled_ratio = max(z.filled_ratio, min(1.0, penetration))
            if z.filled_ratio >= fill_ratio:
                z.state = "mitigated"
                z.mitigated_index = j
                break

        if z.state == "active":
            age = pd.Timestamp(now_ts) - pd.Timestamp(z.time)
            if age > timedelta(days=max_age_days):
                z.state = "expired"

    return zones


# --------------------------------------------------------------------------- #
# Premium / discount
# --------------------------------------------------------------------------- #
def h1_range(h1: pd.DataFrame, upto: Optional[int] = None
             ) -> Optional[Dict[str, float]]:
    """Range H1 courant = dernier swing high ↔ dernier swing bas H1 (§4).

    `None` si l'un des deux swings manque : sans range, il n'y a pas de notion de
    premium ni de discount, et un signal pris « par défaut » serait exactement le
    genre de repli muet que ce dépôt paie cher.
    """
    swings = last_swings(h1, config.FRACTAL_N["H1"], upto=upto)
    hi, lo = swings["high"], swings["low"]
    if hi is None or lo is None or hi.price <= lo.price:
        return None
    return {"high": hi.price, "low": lo.price,
            "mid": (hi.price + lo.price) / 2,
            "height": hi.price - lo.price}


def discount_pct(price: float, rng: Dict[str, float]) -> float:
    """Position du prix dans le range, 0 % = bas (discount), 100 % = haut."""
    return (price - rng["low"]) / rng["height"] * 100


def in_valid_half(side: ZoneSide, price: float, rng: Dict[str, float]) -> bool:
    """Achat valide seulement sous les 50 %, vente seulement au-dessus (§4)."""
    pct = discount_pct(price, rng)
    return pct < 50.0 if side == "bullish" else pct > 50.0


def active_zones(df: pd.DataFrame, timeframe: str,
                 h1: Optional[pd.DataFrame] = None,
                 upto: Optional[int] = None) -> List[Zone]:
    """Les zones exploitables à la bougie `upto` : OB + FVG, fraîches, non
    mitigées, et du bon côté du range H1 quand ce range existe.

    Le filtre premium/discount est appliqué au **milieu de la zone**, pas au prix
    courant : c'est la zone qui doit être en discount pour un achat, et elle ne
    bouge pas, alors que le prix, lui, la traverse.
    """
    n = config.FRACTAL_N.get(timeframe.upper(), 3)
    sub = df if upto is None else df.iloc[:upto + 1]
    events = find_events(sub, n)
    zones = find_order_blocks(sub, timeframe, events=events) + find_fvgs(sub, timeframe)
    update_states(sub, zones, upto=None)

    out = [z for z in zones if z.state == "active"]
    rng = h1_range(h1 if h1 is not None else sub)
    if rng is None:
        return out

    kept: List[Zone] = []
    for z in out:
        z.discount_pct = discount_pct(z.mid, rng)
        if in_valid_half(z.side, z.mid, rng):
            kept.append(z)
    return kept
