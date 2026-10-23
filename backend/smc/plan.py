"""Plan de trade : entrée, stop, cible, R:R — tous dérivés de nombres mesurés.

Règle de conception, et c'est la seule qui compte ici : **aucun champ de ce
module n'existe sans la valeur qui l'a produit.** Chaque critère porte sa mesure
et son seuil, pour qu'on puisse contester la conclusion en regardant le chiffre
plutôt qu'en discutant d'une impression.

Conséquence assumée : **il n'y a pas de verdict « VALIDE ».** Le module compte
les critères remplis et donne le R:R. Dire « ce trade est bon » supposerait de
savoir qu'il va marcher, ce que rien ici ne mesure — et un avis tranché se
recopie dans la plateforme, contrairement à un décompte. La prédiction reste
hors du code tant que les résultats réels ne sont pas journalisés.

Conventions de niveaux, reprises du §8 de la spec :

- **Entrée** : le bord PROXIMAL de la zone, celui que le prix rencontre en
  premier (le bas d'une zone de vente, le haut d'une zone d'achat). C'est le
  niveau auquel il faut être prêt, pas le prix d'une bougie déjà passée.
- **Stop** : sous le creux du sweep s'il y en a un, sinon au-delà du bord
  DISTAL de la zone, plus une marge de SL_MARGIN_ATR × ATR.
- **Cible** : le premier obstacle réel devant le prix — poche de liquidité
  (EQH/EQL), zone opposée, ou dernier swing. La source retenue est renvoyée :
  un R:R n'a pas le même sens selon ce qu'il vise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from smc import config
from smc.liquidity import EqualLevel, equal_levels, nearest_above, nearest_below
from smc.structure import Direction, current_bias, find_events, last_swings
from smc.sweep import find_sweeps, sweep_before
from smc.zones import Zone, atr, find_order_blocks, update_states


@dataclass
class Critere:
    """Un critère, sa mesure, son seuil. Jamais un booléen seul."""
    nom: str
    rempli: Optional[bool]        # None = informatif, ne compte pas
    valeur: Any
    seuil: Any
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {"nom": self.nom, "rempli": self.rempli, "valeur": self.valeur,
                "seuil": self.seuil, "detail": self.detail}


@dataclass
class Plan:
    direction: Direction
    entry: float
    stop: float
    target: float
    rr: float
    target_source: str
    zone_low: float
    zone_high: float
    zone_kind: str
    atr: float
    criteres: List[Critere] = field(default_factory=list)

    @property
    def risque(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def gain(self) -> float:
        return abs(self.target - self.entry)

    @property
    def n_remplis(self) -> int:
        return sum(1 for c in self.criteres if c.rempli is True)

    @property
    def n_evalues(self) -> int:
        return sum(1 for c in self.criteres if c.rempli is not None)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": round(self.entry, 5), "stop": round(self.stop, 5),
            "target": round(self.target, 5),
            "risque": round(self.risque, 5), "gain": round(self.gain, 5),
            "rr": round(self.rr, 2), "target_source": self.target_source,
            "zone": {"kind": self.zone_kind, "low": round(self.zone_low, 5),
                     "high": round(self.zone_high, 5)},
            "atr": round(self.atr, 5),
            "criteres": [c.as_dict() for c in self.criteres],
            "n_remplis": self.n_remplis, "n_evalues": self.n_evalues,
        }


@dataclass
class Analyse:
    """Ce que le scanner a mesuré, qu'il y ait un plan ou non.

    `raison` est renseignée quand aucun plan n'est produit : « rien trouvé » et
    « biais neutre » sont deux situations différentes, et l'utilisateur doit
    pouvoir distinguer un marché sans setup d'un détecteur qui n'a pas marché.
    """
    bias: Optional[Direction]
    price: float
    atr: float
    zones: List[Zone] = field(default_factory=list)
    eqh: List[EqualLevel] = field(default_factory=list)
    eql: List[EqualLevel] = field(default_factory=list)
    plan: Optional[Plan] = None
    raison: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bias": self.bias, "price": round(self.price, 5),
            "atr": round(self.atr, 5),
            "eqh": [lv.as_dict() for lv in self.eqh],
            "eql": [lv.as_dict() for lv in self.eql],
            "plan": self.plan.as_dict() if self.plan else None,
            "raison": self.raison,
        }


def _target_for(direction: Direction, entry: float, df: pd.DataFrame,
                eqh: List[EqualLevel], eql: List[EqualLevel],
                zones: List[Zone], n: int) -> tuple[Optional[float], str]:
    """Le premier obstacle devant le prix, et d'où il vient.

    Ordre de préférence : poche de liquidité, puis zone opposée, puis dernier
    swing. Une poche d'abord parce que c'est là que vont les stops, donc là que
    le prix est attiré ; le swing en dernier parce que c'est le plus faible des
    trois — un extrême isolé n'a pas été validé par une répétition.
    """
    if direction == "bullish":
        poche = nearest_above(eqh, entry)
        if poche is not None:
            return poche.price, f"EQH ({poche.n_swings} swings)"
        opposees = [z for z in zones if z.side == "bearish" and z.low > entry]
        if opposees:
            z = min(opposees, key=lambda z: z.low)
            return z.low, f"zone opposée ({z.kind})"
        sw = last_swings(df, n)["high"]
        return (sw.price, "dernier swing haut") if sw and sw.price > entry else (None, "")

    poche = nearest_below(eql, entry)
    if poche is not None:
        return poche.price, f"EQL ({poche.n_swings} swings)"
    opposees = [z for z in zones if z.side == "bullish" and z.high < entry]
    if opposees:
        z = max(opposees, key=lambda z: z.high)
        return z.high, f"zone opposée ({z.kind})"
    sw = last_swings(df, n)["low"]
    return (sw.price, "dernier swing bas") if sw and sw.price < entry else (None, "")


def analyse(df: pd.DataFrame, timeframe: str = "M5",
            htf_bias: Optional[Direction] = None) -> Analyse:
    """Mesure la structure, les zones, la liquidité, et en déduit un plan.

    `htf_bias` surcharge le biais lu sur `df` : c'est là qu'on branche
    l'alignement H1/H4, qui ne se lit pas sur une capture M5.
    """
    n = config.FRACTAL_N.get(timeframe.upper(), 2)
    price = float(df["close"].iloc[-1])
    atr_val = float(atr(df).iloc[-1]) if len(df) >= config.ATR_PERIOD else 0.0
    if not (atr_val > 0):
        return Analyse(bias=None, price=price, atr=0.0,
                       raison=f"ATR indisponible : moins de {config.ATR_PERIOD} "
                              f"bougies exploitables.")

    events = find_events(df, n)
    bias_ltf = current_bias(events)
    eqh = equal_levels(df, "EQH")
    eql = equal_levels(df, "EQL")

    zones = find_order_blocks(df, timeframe, events=events)
    update_states(df, zones)

    res = Analyse(bias=htf_bias or bias_ltf, price=price, atr=atr_val,
                  zones=zones, eqh=eqh, eql=eql)

    if res.bias is None:
        res.raison = ("Biais neutre : le dernier événement de structure est un "
                      "CHoCH non confirmé par un BOS. Aucun sens n'est validé.")
        return res

    cote = [z for z in zones if z.side == res.bias and z.state == "active"]
    if not cote:
        total = len([z for z in zones if z.side == res.bias])
        res.raison = (f"Aucune zone {res.bias} vierge. {total} détectée(s), "
                      f"toutes mitigées ou périmées.")
        return res

    # La zone la plus proche du prix : c'est celle que le marché travaille.
    zone = min(cote, key=lambda z: min(abs(price - z.low), abs(price - z.high)))
    entry = zone.low if res.bias == "bearish" else zone.high

    sweeps = find_sweeps(df, n=n)
    sw = sweep_before(sweeps, len(df) - 1, res.bias, window=config.SWEEP_TO_CHOCH_BARS)

    marge = config.SL_MARGIN_ATR * atr_val
    if sw is not None:
        stop = sw.extreme - marge if res.bias == "bullish" else sw.extreme + marge
        origine_stop = "mèche du sweep"
    else:
        stop = zone.low - marge if res.bias == "bullish" else zone.high + marge
        origine_stop = "bord opposé de la zone"

    target, source = _target_for(res.bias, entry, df, eqh, eql, zones, n)
    if target is None:
        res.raison = ("Aucune cible devant le prix : ni poche de liquidité, ni "
                      "zone opposée, ni swing. Un R:R serait inventé.")
        return res

    risque = abs(entry - stop)
    if risque <= 0:
        res.raison = "Stop confondu avec l'entrée : risque nul, R:R indéfini."
        return res
    rr = abs(target - entry) / risque

    distance_atr = min(abs(price - zone.low), abs(price - zone.high)) / atr_val
    dans_la_zone = zone.low <= price <= zone.high

    res.plan = Plan(
        direction=res.bias, entry=entry, stop=stop, target=target, rr=rr,
        target_source=source, zone_low=zone.low, zone_high=zone.high,
        zone_kind=zone.kind, atr=atr_val,
        criteres=[
            Critere("Biais directionnel",
                    True, res.bias,
                    "BOS confirmé",
                    "fourni par le HTF" if htf_bias else
                    "dernier événement de structure sur ce timeframe"),
            Critere("Zone vierge (jamais retestée)",
                    zone.state == "active", zone.state, "active",
                    f"OB {zone.side} formé en bougie {zone.index}"),
            Critere("Déplacement de l'impulsion",
                    zone.displacement_atr >= config.OB_MIN_DISPLACEMENT_ATR,
                    round(zone.displacement_atr, 2),
                    config.OB_MIN_DISPLACEMENT_ATR,
                    "mesuré de l'extrême de la zone à la clôture qui a cassé"),
            Critere("Fraîcheur de la zone",
                    (zone.age_days or 0) <= config.ZONE_MAX_AGE_DAYS,
                    round(zone.age_days or 0, 2), config.ZONE_MAX_AGE_DAYS,
                    "en jours calendaires"),
            Critere("Prix à portée de la zone",
                    dans_la_zone or distance_atr <= 1.0,
                    round(distance_atr, 2), 1.0,
                    "dans la zone" if dans_la_zone else "distance en ATR"),
            Critere("R:R",
                    rr >= config.RR_MIN_NORMAL, round(rr, 2),
                    config.RR_MIN_NORMAL,
                    f"cible = {source}, stop sous {origine_stop}"),
            # Informatif : le sweep n'est JAMAIS une condition (§5). Il est
            # mesuré pour qu'on puisse trancher un jour sur des chiffres.
            Critere("Sweep de liquidité", None,
                    round(sw.depth_atr, 2) if sw else False,
                    "non exigé",
                    "profondeur en ATR" if sw else "aucun dans la fenêtre"),
        ],
    )
    return res
