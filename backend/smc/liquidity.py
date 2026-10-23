"""Poches de liquidité : Equal Highs / Equal Lows (EQH / EQL).

Un niveau où plusieurs swings se sont arrêtés au même prix est un endroit où des
stops se sont accumulés. C'est ce que les méthodes SMC appellent une poche de
liquidité, et c'est une cible naturelle : le prix va la chercher.

Définition reprise telle quelle de `strategy_ict._find_equal_levels` — deux
swings ou plus à moins de `tol_atr × ATR` l'un de l'autre forment un cluster, et
on renvoie le niveau moyen du cluster. Elle est réimplémentée ici plutôt
qu'importée, pour la même raison que `zones.atr()` : ce paquet ne doit dépendre
d'aucun module de la boucle de trading, sinon un réglage du bot déplacerait un
niveau du scanner. La duplication est assumée et le seuil est le même (0,15×ATR).

Le nombre de swings du cluster est renvoyé avec le niveau : un EQH formé de
quatre touches ne vaut pas un EQH formé de deux, et c'est à l'appelant de le
voir plutôt qu'à ce module d'en décider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from smc import config
from smc.structure import find_pivots
from smc.zones import atr

LevelKind = Literal["EQH", "EQL"]


@dataclass(frozen=True)
class EqualLevel:
    kind: LevelKind
    price: float
    n_swings: int
    first_index: int
    last_index: int
    spread: float          # écart entre le plus haut et le plus bas du cluster

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "price": self.price, "n_swings": self.n_swings,
                "first_index": self.first_index, "last_index": self.last_index,
                "spread": self.spread}


def equal_levels(df: pd.DataFrame, kind: LevelKind,
                 lookback: int = config.EQUAL_LEVEL_LOOKBACK,
                 tol_atr: float = config.EQUAL_LEVEL_TOLERANCE_ATR,
                 n: Optional[int] = None) -> List[EqualLevel]:
    """Clusters de swings égaux dans les `lookback` dernières bougies.

    Anti look-ahead : on ne retient que les pivots confirmés dans la fenêtre,
    donc un cluster n'existe qu'à partir du moment où son dernier swing a été
    confirmé — pas depuis la bougie qui l'a formé.
    """
    if len(df) < 10:
        return []
    sub = df.tail(lookback)
    atr_val = float(atr(sub).iloc[-1]) if len(sub) else 0.0
    if not (atr_val > 0):
        return []
    tol = tol_atr * atr_val

    want = "high" if kind == "EQH" else "low"
    n = n if n is not None else 2
    pivots = [p for p in find_pivots(sub, n) if p.kind == want]
    if len(pivots) < 2:
        return []

    pivots = sorted(pivots, key=lambda p: p.price)
    out: List[EqualLevel] = []
    cluster = [pivots[0]]
    for p in pivots[1:]:
        if p.price - cluster[-1].price <= tol:
            cluster.append(p)
            continue
        if len(cluster) >= 2:
            out.append(_build(kind, cluster))
        cluster = [p]
    if len(cluster) >= 2:
        out.append(_build(kind, cluster))
    return out


def _build(kind: LevelKind, cluster: List[Any]) -> EqualLevel:
    prix = [p.price for p in cluster]
    idx = [p.index for p in cluster]
    return EqualLevel(kind=kind, price=sum(prix) / len(prix), n_swings=len(cluster),
                      first_index=min(idx), last_index=max(idx),
                      spread=max(prix) - min(prix))


def nearest_above(levels: List[EqualLevel], price: float) -> Optional[EqualLevel]:
    au_dessus = [lv for lv in levels if lv.price > price]
    return min(au_dessus, key=lambda lv: lv.price) if au_dessus else None


def nearest_below(levels: List[EqualLevel], price: float) -> Optional[EqualLevel]:
    en_dessous = [lv for lv in levels if lv.price < price]
    return max(en_dessous, key=lambda lv: lv.price) if en_dessous else None
