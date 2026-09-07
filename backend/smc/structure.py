"""Détection de structure : pivots fractals, BOS et CHoCH (§3 de la spec).

Trois règles, et elles décident de tout le reste :

1. **Pivot** = fractale. Pivot haut si le high est *strictement* supérieur aux n
   bougies de chaque côté (n=3 en D1/H4/H1, n=2 en M15). Idem, en miroir, pour
   les pivots bas. Le « strictement » compte : avec `>=`, un plateau de bougies
   identiques produit n pivots au même prix.

2. **Cassure en clôture, jamais en mèche.** Une mèche qui dépasse un swing puis
   revient n'est pas une cassure — c'est même souvent l'inverse (un sweep, §5).

3. **Anti look-ahead.** Un pivot en i n'existe qu'à partir de la bougie i+n : il
   faut avoir vu les n bougies suivantes pour savoir que c'en est un. Toute la
   détection est donc écrite en *rejouant* la série bougie par bougie, en
   n'utilisant à l'instant i que ce qui était connu en i.

   Ce point n'est pas cosmétique. Un détecteur qui repère les pivots sur toute
   la série puis « constate » les cassures voit les retournements avant qu'ils
   soient confirmés, et produit un backtest flatteur qui ne se reproduit jamais
   en live. `test_structure.py` le vérifie par la seule preuve qui vaille :
   calculer sur df[:k] doit rendre exactement le préfixe du calcul sur df.

BOS vs CHoCH : c'est la même cassure, lue par rapport à la tendance en cours.
Dans le sens de la tendance → BOS (continuation). Contre elle → CHoCH (premier
signal de retournement). La toute première cassure de la série établit la
tendance, et est comptée comme un BOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

Direction = Literal["bullish", "bearish"]
EventKind = Literal["BOS", "CHoCH"]


@dataclass(frozen=True)
class Pivot:
    """Un swing confirmé. `confirmed_index` est la bougie à partir de laquelle il
    est utilisable — c'est elle qui porte l'anti look-ahead."""
    index: int
    time: Any
    price: float
    kind: Literal["high", "low"]
    confirmed_index: int

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "time": self.time, "price": self.price,
                "kind": self.kind, "confirmed_index": self.confirmed_index}


@dataclass(frozen=True)
class StructureEvent:
    """Une cassure de structure validée en clôture."""
    index: int
    time: Any
    kind: EventKind
    direction: Direction
    level: float          # le swing cassé
    close: float          # la clôture qui l'a cassé
    pivot_index: int      # position du swing cassé

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "time": self.time, "kind": self.kind,
                "direction": self.direction, "level": self.level,
                "close": self.close, "pivot_index": self.pivot_index}


def find_pivots(df: pd.DataFrame, n: int) -> List[Pivot]:
    """Pivots fractals de la série, chacun daté de sa bougie de confirmation.

    Les n dernières bougies ne peuvent, par construction, porter aucun pivot
    confirmé : il manque les bougies de droite. C'est voulu.
    """
    if n < 1:
        raise ValueError("n doit valoir au moins 1")
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index
    out: List[Pivot] = []
    for i in range(n, len(df) - n):
        window_h = highs[i - n: i + n + 1]
        window_l = lows[i - n: i + n + 1]
        # Strictement supérieur/inférieur à TOUS les voisins (le centre exclu).
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            out.append(Pivot(i, idx[i], float(highs[i]), "high", i + n))
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            out.append(Pivot(i, idx[i], float(lows[i]), "low", i + n))
    out.sort(key=lambda p: (p.confirmed_index, p.index))
    return out


def find_events(df: pd.DataFrame, n: int) -> List[StructureEvent]:
    """Suite chronologique des BOS/CHoCH, en ne connaissant à chaque bougie que
    le passé.

    Le déroulé, bougie par bougie :
      - on active les pivots dont `confirmed_index` est atteint ;
      - on garde le dernier swing haut et le dernier swing bas non cassés ;
      - une clôture au-delà d'un de ces niveaux produit un événement, et le
        niveau est consommé (il ne peut pas casser deux fois).
    """
    pivots = find_pivots(df, n)
    by_confirm: Dict[int, List[Pivot]] = {}
    for p in pivots:
        by_confirm.setdefault(p.confirmed_index, []).append(p)

    closes = df["close"].values
    idx = df.index

    pending_high: Optional[Pivot] = None
    pending_low: Optional[Pivot] = None
    trend: Optional[Direction] = None
    events: List[StructureEvent] = []

    for i in range(len(df)):
        for p in by_confirm.get(i, []):
            # Un pivot ne remplace le précédent que s'il est plus récent : les
            # pivots sont ajoutés dans l'ordre de confirmation, donc c'est le cas.
            if p.kind == "high":
                pending_high = p
            else:
                pending_low = p

        close = float(closes[i])

        if pending_high is not None and close > pending_high.price:
            kind: EventKind = "CHoCH" if trend == "bearish" else "BOS"
            events.append(StructureEvent(i, idx[i], kind, "bullish",
                                         pending_high.price, close,
                                         pending_high.index))
            trend = "bullish"
            pending_high = None
        elif pending_low is not None and close < pending_low.price:
            kind = "CHoCH" if trend == "bullish" else "BOS"
            events.append(StructureEvent(i, idx[i], kind, "bearish",
                                         pending_low.price, close,
                                         pending_low.index))
            trend = "bearish"
            pending_low = None

    return events


def current_bias(events: List[StructureEvent]) -> Optional[Direction]:
    """Biais du timeframe = direction du dernier événement structurel.

    §6 : « Si CHoCH H4 récent non confirmé par un BOS → biais neutre → aucun
    signal, dans aucun sens. » Un CHoCH est le *premier* signe d'un retournement,
    pas sa confirmation ; l'exiger confirmé par un BOS évite d'acheter le premier
    sursaut d'une tendance baissière. D'où `None` sur un CHoCH terminal.
    """
    if not events:
        return None
    last = events[-1]
    if last.kind == "CHoCH":
        return None
    return last.direction


def last_swings(df: pd.DataFrame, n: int,
                upto: Optional[int] = None) -> Dict[str, Optional[Pivot]]:
    """Dernier pivot haut et dernier pivot bas connus à la bougie `upto`.

    Sert au range premium/discount (§4) et au sweep (§5). `upto=None` = fin de
    série. Le filtre porte sur `confirmed_index`, pas sur `index` : c'est la
    différence entre « ce pivot existe » et « on pouvait le savoir ».
    """
    limit = len(df) - 1 if upto is None else upto
    out: Dict[str, Optional[Pivot]] = {"high": None, "low": None}
    for p in find_pivots(df, n):
        if p.confirmed_index > limit:
            continue
        out[p.kind] = p
    return out
