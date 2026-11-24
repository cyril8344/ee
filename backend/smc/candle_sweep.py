"""Balayage puis récupération de la bougie précédente, en cascade multi-timeframes.

La méthode est une seule règle, appliquée trois fois à trois échelles — c'est ce
qui la rend intégralement mécanique, donc testable, là où tout ce qui a été
regardé jusqu'ici demandait un jugement :

    balayage        : low(bougie en cours) < low(bougie précédente)
    récupération    : close(bougie en cours) > low(bougie précédente)
    → lecture haussière

et le miroir exact pour la baisse (high dépassé, clôture repassée dessous).

Appliquée au Daily elle donne la **direction** (le plus bas de la veille est le
PDL) ; au H1/M30 le **setup** ; au M15→M1 l'**entrée**. Trois étages, une seule
fonction.

────────────────────────────────────────────────────────────────────────────
CE QUE CE MODULE N'AFFIRME PAS
────────────────────────────────────────────────────────────────────────────
Aucune espérance, aucun score, aucune probabilité. Il répond à « la condition
est-elle remplie ? », jamais à « est-ce que ça va marcher ? ». Rien dans ce
dépôt ne mesure la seconde question tant qu'un walk-forward n'a pas tourné.

Et un avertissement qui doit rester lisible dans le code, parce qu'il est
contre-intuitif : **les trois étages ne sont pas trois confirmations
indépendantes**. Le balayage H1 qui « confirme » le balayage Daily est, le plus
souvent, *le même mouvement de prix* vu de plus près — la mèche basse de la
bougie journalière EST la descente visible en H1. Empiler trois conditions
corrélées donne l'impression d'une preuve trois fois plus forte alors que
l'information ajoutée est faible. `independence_report()` mesure exactement ce
recouvrement, pour qu'on ne se raconte pas d'histoire.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class Sweep:
    """Un balayage récupéré, avec les trois nombres qui l'ont produit.

    `niveau` est le plus bas (ou plus haut) de la bougie précédente : c'est lui
    qu'on recopie sur le graphique. `depassement` et `recuperation` sont les
    distances, en prix, de part et d'autre — elles disent si le balayage est une
    mèche franche ou une égalité au centième.
    """
    index: int
    time: Any
    direction: Direction
    niveau: float          # low (haussier) ou high (baissier) de la précédente
    extreme: float         # jusqu'où le prix est allé au-delà
    close: float
    depassement: float     # niveau − extreme (haussier) : > 0 par construction
    recuperation: float    # close − niveau (haussier)   : > 0 par construction

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "time": self.time,
                "direction": self.direction, "niveau": self.niveau,
                "extreme": self.extreme, "close": self.close,
                "depassement": self.depassement,
                "recuperation": self.recuperation}


def sweep_at(df: pd.DataFrame, i: int,
             direction: Optional[Direction] = None) -> Optional[Sweep]:
    """La bougie `i` balaie-t-elle puis récupère-t-elle la bougie `i-1` ?

    `direction=None` teste les deux sens. Une bougie ne peut pas être les deux à
    la fois dans la pratique, mais si les deux conditions sont remplies (bougie
    englobante qui clôture à l'intérieur), le sens haussier est renvoyé et le
    cas est, par construction, ambigu — c'est pourquoi `direction` existe : en
    cascade on sait déjà quel sens on cherche.

    Anti look-ahead : tout est lu sur la bougie `i` **close**. Un appel sur la
    dernière bougie d'une série en cours de formation juge donc une bougie qui
    n'est pas terminée ; c'est à l'appelant de ne pas le faire (cf.
    `data_feed.drop_unclosed`).
    """
    if i <= 0 or i >= len(df):
        return None
    prev_low = float(df["low"].values[i - 1])
    prev_high = float(df["high"].values[i - 1])
    low = float(df["low"].values[i])
    high = float(df["high"].values[i])
    close = float(df["close"].values[i])
    idx = df.index

    if direction in (None, "bullish") and low < prev_low and close > prev_low:
        return Sweep(i, idx[i], "bullish", prev_low, low, close,
                     prev_low - low, close - prev_low)
    if direction in (None, "bearish") and high > prev_high and close < prev_high:
        return Sweep(i, idx[i], "bearish", prev_high, high, close,
                     high - prev_high, prev_high - close)
    return None


def find_sweeps(df: pd.DataFrame,
                direction: Optional[Direction] = None) -> List[Sweep]:
    """Tous les balayages récupérés de la série, dans l'ordre chronologique."""
    out: List[Sweep] = []
    for i in range(1, len(df)):
        s = sweep_at(df, i, direction)
        if s is not None:
            out.append(s)
    return out


def rate(df: pd.DataFrame, direction: Optional[Direction] = None) -> float:
    """Part des bougies qui remplissent la condition.

    Le chiffre qui remet les choses à leur place : une condition vérifiée une
    bougie sur trois ne « sélectionne » pas grand-chose à elle seule. À
    connaître **avant** de s'enthousiasmer pour un exemple bien choisi.
    """
    if len(df) < 2:
        return 0.0
    return len(find_sweeps(df, direction)) / (len(df) - 1)


# --------------------------------------------------------------------------- #
# Cascade : direction → setup → entrée
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cascade:
    """L'état des trois étages à un instant donné.

    `aligne` n'est vrai que si les trois étages existent et vont dans le même
    sens. Un étage manquant n'est pas un étage neutre : c'est une condition non
    remplie, et elle est nommée dans `manquants`.
    """
    direction: Optional[Direction]
    etages: Dict[str, Optional[Sweep]]
    manquants: List[str]

    @property
    def aligne(self) -> bool:
        return self.direction is not None and not self.manquants

    def as_dict(self) -> Dict[str, Any]:
        return {"direction": self.direction, "aligne": self.aligne,
                "manquants": self.manquants,
                "etages": {k: (v.as_dict() if v else None)
                           for k, v in self.etages.items()}}


def cascade(frames: Dict[str, pd.DataFrame],
            ordre: Optional[List[str]] = None) -> Cascade:
    """Applique la règle au dernier chandelier clos de chaque timeframe.

    `frames` associe un nom d'étage à sa série, par exemple
    `{"direction": daily, "setup": h1, "entree": m15}`. Le **premier** étage
    fixe le sens ; les suivants doivent le confirmer. C'est l'ordre de la
    méthode : la direction d'abord, jamais l'inverse — sinon une mèche de M15
    déciderait du biais journalier.
    """
    noms = ordre if ordre is not None else list(frames.keys())
    etages: Dict[str, Optional[Sweep]] = {}
    manquants: List[str] = []
    sens: Optional[Direction] = None

    for rang, nom in enumerate(noms):
        df = frames.get(nom)
        if df is None or len(df) < 2:
            etages[nom] = None
            manquants.append(nom)
            continue
        s = sweep_at(df, len(df) - 1, direction=sens if rang else None)
        etages[nom] = s
        if s is None:
            manquants.append(nom)
        elif rang == 0:
            sens = s.direction

    return Cascade(sens, etages, manquants)


def independence_report(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Ce que chaque étage retire vraiment, et ce qu'il ne retire pas.

    Pourquoi cette fonction existe : trois conditions empilées se présentent
    comme trois confirmations, alors que le balayage H1 et le balayage Daily
    décrivent souvent **le même mouvement**. Si les étages étaient
    indépendants, la part de bougies alignées vaudrait le produit des taux
    individuels. Elle est renvoyée à côté du taux réellement observé : l'écart
    entre les deux **est** la dépendance.

    Aucune interprétation n'est produite ici. Deux nombres, et le lecteur
    conclut.
    """
    taux = {nom: rate(df) for nom, df in frames.items()}
    produit = 1.0
    for t in taux.values():
        produit *= t
    return {"taux_par_etage": taux,
            "produit_si_independants": produit,
            "note": ("Comparer `produit_si_independants` au taux d'alignement "
                     "réellement mesuré sur la même période. Un taux observé "
                     "nettement supérieur signifie que les étages se répètent "
                     "au lieu de s'ajouter.")}
