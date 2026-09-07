"""Logique de confluence : biais H4 → zone H1 → déclencheur M15 (§6).

Un signal d'achat est émis **si et seulement si** les trois conditions sont
réunies :

1. **Biais H4** — dernier événement structurel H4 = BOS haussier. Un CHoCH H4
   récent non confirmé par un BOS donne un biais **neutre**, donc aucun signal,
   dans aucun sens. C'est `structure.current_bias()` qui porte cette règle.
2. **Zone H1** — le prix entre dans un OB ou un FVG haussier H1 actif : non
   mitigé, frais, en discount.
3. **Déclencheur M15** — CHoCH haussier M15 pendant que le prix est dans la
   zone, ou l'a touchée dans les 8 dernières bougies M15.

Et deux règles de gestion :
- **1 signal maximum par zone.** Une zone traversée sans déclencheur est `dead`
  et ne repassera jamais.
- **Miroir exact** pour les ventes.

Le D1 est calculé et joint au signal à titre d'information (aligné ou contraire
au biais H4). Il ne bloque rien en v1 — c'est explicite dans la spec, et le noter
ici évite qu'on le transforme en filtre par inadvertance dans six mois.

Le sweep (§5) est **taggé, jamais exigé** : `signal.sweep` vaut True ou False, et
les deux valeurs produisent un signal. Voir sweep.py pour pourquoi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from smc import config
from smc.structure import (Direction, StructureEvent, current_bias, find_events,
                           last_swings)
from smc.sweep import Sweep, find_sweeps, sweep_before
from smc.zones import Zone, active_zones, atr, h1_range


@dataclass
class Signal:
    """Un signal d'analyse. Aucun ordre n'en découle automatiquement : il part en
    alerte, et c'est l'utilisateur qui décide."""
    time: Any
    index: int                       # position M15
    symbol: str
    direction: Direction
    bias_h4: Direction
    d1_aligned: Optional[bool]
    zone: Zone
    choch: StructureEvent
    price: float
    sweep: bool
    sweep_obj: Optional[Sweep] = None
    atr_m15: float = 0.0
    atr_h1: float = 0.0

    @property
    def zone_key(self) -> Tuple[str, str, int, float, float]:
        """Identité d'une zone, pour la règle « 1 signal max par zone »."""
        return (self.zone.timeframe, self.zone.kind, self.zone.index,
                self.zone.low, self.zone.high)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time, "index": self.index, "symbol": self.symbol,
            "direction": self.direction, "bias_h4": self.bias_h4,
            "d1_aligned": self.d1_aligned,
            "zone_type": self.zone.kind, "zone_tf": self.zone.timeframe,
            "zone_low": self.zone.low, "zone_high": self.zone.high,
            "discount_pct": self.zone.discount_pct,
            "zone_age_hours": None, "price": self.price,
            "sweep": self.sweep,
            "sweep_depth_atr": self.sweep_obj.depth_atr if self.sweep_obj else None,
            "choch_index": self.choch.index, "choch_level": self.choch.level,
            "atr_m15": self.atr_m15, "atr_h1": self.atr_h1,
        }


def _zone_touched_recently(m15: pd.DataFrame, zone: Zone, upto: int,
                           lookback: int = config.ZONE_TOUCH_LOOKBACK_BARS) -> bool:
    """Le prix est dans la zone, ou l'a touchée dans les `lookback` dernières M15.

    §6 : le déclencheur vaut « pendant que le prix est dans la zone (ou l'a
    touchée dans les 8 dernières bougies M15) ». Sans cette tolérance, il faudrait
    que le CHoCH tombe exactement sur la bougie de contact — ce qui n'arrive
    presque jamais, le retournement prenant quelques bougies à se dessiner.
    """
    lo = max(0, upto - lookback + 1)
    fenetre = m15.iloc[lo: upto + 1]
    if fenetre.empty:
        return False
    return bool(((fenetre["low"] <= zone.high) & (fenetre["high"] >= zone.low)).any())


def d1_alignment(d1: Optional[pd.DataFrame], bias: Direction) -> Optional[bool]:
    """Structure D1 alignée ou contraire au biais H4 — information seule (§6).

    `None` quand le D1 est neutre ou indisponible : « on ne sait pas » et
    « contraire » ne doivent pas se confondre dans le message d'alerte.
    """
    if d1 is None or len(d1) < 2 * config.FRACTAL_N["D1"] + 2:
        return None
    b = current_bias(find_events(d1, config.FRACTAL_N["D1"]))
    return None if b is None else (b == bias)


def evaluate(m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame,
             d1: Optional[pd.DataFrame] = None,
             symbol: str = "XAUUSD",
             upto: Optional[int] = None,
             dead_zones: Optional[Set[Any]] = None,
             signalled_zones: Optional[Set[Any]] = None) -> Optional[Signal]:
    """Le signal à la bougie M15 `upto`, ou None.

    `dead_zones` et `signalled_zones` sont fournis et mis à jour par l'appelant
    (le replay ou la boucle live) : ce sont des états qui traversent les cycles,
    pas des faits recalculables depuis les seules bougies.
    """
    i = len(m15) - 1 if upto is None else upto
    if i < 0 or i >= len(m15):
        return None
    signalled_zones = signalled_zones if signalled_zones is not None else set()
    dead_zones = dead_zones if dead_zones is not None else set()

    # 1) Biais H4 — un CHoCH terminal rend None, donc aucun signal.
    bias = current_bias(find_events(h4, config.FRACTAL_N["H4"]))
    if bias is None:
        return None

    # 3) Déclencheur M15 : le CHoCH doit tomber SUR la bougie courante, sinon on
    #    ré-émettrait le même signal à chaque bougie suivante.
    m15_events = find_events(m15.iloc[:i + 1], config.FRACTAL_N["M15"])
    if not m15_events:
        return None
    trigger = m15_events[-1]
    if trigger.index != i or trigger.kind != "CHoCH" or trigger.direction != bias:
        return None

    # 2) Zone H1 du bon côté, vierge jusqu'à ce retest, touchée récemment.
    #    `retest_since` = début de la fenêtre de contact M15 : la zone doit être
    #    restée intacte jusque-là, et c'est ce premier retest qu'on trade (cf.
    #    zones.active_zones pour la contradiction §4/§6 que ça résout).
    rng = h1_range(h1)
    if rng is None:
        return None
    window_start = m15.index[max(0, i - config.ZONE_TOUCH_LOOKBACK_BARS + 1)]
    zones_h1 = [z for z in active_zones(h1, "H1", h1=h1, retest_since=window_start)
                if z.side == bias]
    if not zones_h1:
        return None

    price = float(m15["close"].iloc[i])
    candidates: List[Zone] = []
    for z in zones_h1:
        key = (z.timeframe, z.kind, z.index, z.low, z.high)
        if key in signalled_zones or key in dead_zones:
            continue
        if _zone_touched_recently(m15, z, i):
            candidates.append(z)
    if not candidates:
        return None

    # La zone la plus proche du prix : celle que le marché vient de travailler.
    zone = min(candidates, key=lambda z: abs(z.mid - price))

    sweeps = find_sweeps(m15.iloc[:i + 1])
    sw = sweep_before(sweeps, i, bias)

    atr_m15 = float(atr(m15.iloc[:i + 1]).iloc[-1])
    atr_h1 = float(atr(h1).iloc[-1])

    return Signal(
        time=m15.index[i], index=i, symbol=symbol, direction=bias,
        bias_h4=bias, d1_aligned=d1_alignment(d1, bias),
        zone=zone, choch=trigger, price=price,
        sweep=sw is not None, sweep_obj=sw,
        atr_m15=atr_m15 if atr_m15 == atr_m15 else 0.0,
        atr_h1=atr_h1 if atr_h1 == atr_h1 else 0.0,
    )


def mark_dead_zones(m15: pd.DataFrame, zones_h1: List[Zone], upto: int,
                    dead: Set[Any]) -> Set[Any]:
    """Une zone traversée de part en part sans déclencheur est morte (§6).

    « Traversée » = le prix est ressorti de l'autre côté, pas seulement touché :
    une zone touchée qui tient est justement le scénario recherché.
    """
    if upto < 0 or upto >= len(m15):
        return dead
    bar_low = float(m15["low"].iloc[upto])
    bar_high = float(m15["high"].iloc[upto])
    for z in zones_h1:
        key = (z.timeframe, z.kind, z.index, z.low, z.high)
        if key in dead:
            continue
        if z.side == "bullish" and bar_low < z.low:
            dead.add(key)
        elif z.side == "bearish" and bar_high > z.high:
            dead.add(key)
    return dead
