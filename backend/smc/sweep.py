"""Sweep de liquidité — détecté et TAGGÉ, jamais exigé (§5).

Point le plus important de ce module, et il est dans la spec : **le sweep n'est
pas une condition de signal.** Il est enregistré comme attribut (`sweep:
true/false`) sur chaque signal, pour l'analyse statistique et les features ML.

Ce n'est pas une précaution de style. Ce dépôt a déjà testé « sweep obligatoire »
en conditions réelles sur EUR/USD (strategy_ict.py, `OB_REQUIRE_LIQUIDITY`,
laissé à False) : trop de confluence exigée = trop peu de signaux, sans gain de
robustesse démontré en out-of-sample. Exiger le sweep ici referait la même
erreur, en pire — on n'aurait même pas les données pour la mesurer, puisque les
signaux sans sweep n'existeraient plus. En le taggant, on garde de quoi trancher
la question plus tard, sur des chiffres.

Détection, côté achat (miroir exact pour la vente) :
  1. dernier swing bas M15 **confirmé** ;
  2. une bougie dont la mèche passe sous ce bas mais qui **clôture au-dessus**,
     ou dont le prix reclôture au-dessus dans les 2 bougies suivantes ;
  3. profondeur maximale 0,5 × ATR(14) M15 sous le niveau — au-delà, c'est une
     vraie cassure, donc pas de tag ;
  4. le CHoCH déclencheur doit survenir dans les 8 bougies M15 après le sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from smc import config
from smc.structure import Direction, find_pivots
from smc.zones import atr

# Nombre de bougies laissées pour reclôturer au-dessus du niveau (§5).
RECLOSE_BARS = 2


@dataclass(frozen=True)
class Sweep:
    """Une prise de liquidité datée. `depth_atr` est conservé tel quel : c'est
    une feature du dataset ML, pas un seuil de décision."""
    index: int
    time: Any
    direction: Direction      # "bullish" = sweep d'un bas (achat)
    level: float              # le swing balayé
    extreme: float            # la mèche atteinte
    depth_atr: float
    reclosed_index: int       # bougie où le prix a reclôturé du bon côté

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "time": self.time,
                "direction": self.direction, "level": self.level,
                "extreme": self.extreme, "depth_atr": self.depth_atr,
                "reclosed_index": self.reclosed_index}


def find_sweeps(m15: pd.DataFrame,
                max_depth_atr: float = config.SWEEP_MAX_DEPTH_ATR,
                n: Optional[int] = None) -> List[Sweep]:
    """Tous les sweeps de la série, dans l'ordre chronologique.

    Anti look-ahead : un swing n'est balayable qu'à partir de sa bougie de
    confirmation, et la reclôture est cherchée uniquement dans les bougies
    suivantes. Un sweep n'est donc jamais « vu » avant d'exister.
    """
    n = n if n is not None else config.FRACTAL_N["M15"]
    if len(m15) < 2 * n + 2:
        return []

    atr_series = atr(m15).values
    highs, lows, closes = m15["high"].values, m15["low"].values, m15["close"].values
    idx = m15.index

    pivots = find_pivots(m15, n)
    lows_p = [p for p in pivots if p.kind == "low"]
    highs_p = [p for p in pivots if p.kind == "high"]

    out: List[Sweep] = []

    def _scan(pivs, direction: Direction) -> None:
        for p in pivs:
            atr_ref = None
            # Le niveau devient balayable dès sa confirmation.
            for i in range(p.confirmed_index, len(m15)):
                if direction == "bullish":
                    pierced = float(lows[i]) < p.price
                    extreme = float(lows[i])
                    depth = p.price - extreme
                else:
                    pierced = float(highs[i]) > p.price
                    extreme = float(highs[i])
                    depth = extreme - p.price
                if not pierced:
                    continue

                atr_ref = float(atr_series[i])
                if not (atr_ref > 0):
                    break
                if depth > max_depth_atr * atr_ref:
                    break            # vraie cassure : ce niveau ne sera plus balayé

                # Reclôture du bon côté : cette bougie, ou l'une des 2 suivantes.
                reclosed = None
                for j in range(i, min(i + RECLOSE_BARS + 1, len(m15))):
                    ok = (float(closes[j]) > p.price if direction == "bullish"
                          else float(closes[j]) < p.price)
                    if ok:
                        reclosed = j
                        break
                if reclosed is None:
                    break            # le prix est resté de l'autre côté

                out.append(Sweep(index=i, time=idx[i], direction=direction,
                                 level=p.price, extreme=extreme,
                                 depth_atr=depth / atr_ref,
                                 reclosed_index=reclosed))
                break                # un niveau n'est balayé qu'une fois

    _scan(lows_p, "bullish")
    _scan(highs_p, "bearish")
    out.sort(key=lambda s: s.index)
    return out


def sweep_before(sweeps: List[Sweep], choch_index: int, direction: Direction,
                 window: int = config.SWEEP_TO_CHOCH_BARS) -> Optional[Sweep]:
    """Le sweep qui précède immédiatement un CHoCH, s'il est dans la fenêtre.

    §5 : « Le CHoCH déclencheur doit survenir dans les 8 bougies M15 après le
    sweep pour que le tag s'applique. » Au-delà, les deux événements ne sont plus
    liés — les rapprocher quand même produirait un tag qui ne veut rien dire, et
    donc une feature ML bruitée.

    Renvoie None sans que ce soit une anomalie : la majorité des signaux n'auront
    pas de sweep, et c'est précisément ce qu'on veut pouvoir comparer.
    """
    candidats = [s for s in sweeps
                 if s.direction == direction
                 and s.index <= choch_index
                 and choch_index - s.index <= window]
    return max(candidats, key=lambda s: s.index) if candidats else None
