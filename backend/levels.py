"""Niveaux à recopier à la main sur MetaTrader 5.

Aucune décision de trading ici : ce module ne produit ni signal, ni score, ni
espérance. Il met en chiffres ce que l'œil doit tracer — quelques Order Blocks,
un support, une résistance, un canal parallèle — pour qu'on puisse les reporter
dans MT5 sans les relire sur un graphique.

Deux choix structurants, contre le panneau OB actuel du graphique :

1. **Le détecteur ICT, pas celui de la stratégie A.** `strategy.find_order_blocks`
   accepte toute bougie rouge suivie d'une verte avec 0.5×ATR d'impulsion, sur 40
   bougies, **sans vérifier la mitigation** — d'où les 5-7 zones empilées, dont la
   plupart ont déjà été traversées par le prix. `strategy_ict._find_order_blocks`
   exige corps, hauteur bornée, 1.0×ATR d'impulsion et zone non mitiguée ; on lui
   demande en plus le BOS. Ici le coût d'un filtre trop strict est « moins de
   niveaux », pas du capital mal engagé.

2. **Un plafond dur.** Au plus MAX_OB_PER_SIDE zones par côté, les plus proches du
   prix. Une liste qu'on ne peut pas tracer en une minute ne sert à rien.

Le canal est marqué `fiable: False` quand un de ses bords a moins de
CHANNEL_MIN_TOUCHES touches distinctes : deux droites parallèles calées sur les
extrêmes contiennent le prix **par construction**, ça ne prouve rien. Même
convention que la distribution du MFE, dont les niveaux censurés sont grisés.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import strategy

# Une bougie "touche" un niveau si elle passe à moins de tol du niveau.
# 0.5×ATR : même tolérance que strategy.h1_sr_levels, pour que le nombre de
# touches affiché soit exactement celui qui a validé le niveau.
SR_TOL_ATR = 0.5
SR_MIN_TOUCHES = 2
SR_LOOKBACK = 120

MAX_OB_PER_SIDE = 2
OB_LOOKBACK_BARS = 60

CHANNEL_LOOKBACK = 120
CHANNEL_TOUCH_TOL_ATR = 0.25
CHANNEL_MIN_TOUCHES = 2
# Deux bougies voisines calées sur le même creux ne font qu'une touche.
CHANNEL_TOUCH_MIN_GAP = 3

DIGITS = {"XAUUSD": 2, "EURUSD": 5}


def digits_for(symbol: str) -> int:
    return DIGITS.get(symbol, 2)


def _tol_unit(df: pd.DataFrame) -> float:
    """ATR de la dernière bougie, ou un repli en % du prix si la colonne manque."""
    if "atr" in df.columns and len(df):
        val = float(df["atr"].iloc[-1])
        if val == val and val > 0:  # NaN-safe
            return val
    return float(df["close"].iloc[-1]) * 0.0008 if len(df) else 0.0


def _least_squares(y: List[float]) -> Tuple[float, float]:
    """Pente et ordonnée à l'origine d'une régression sur l'indice de bougie."""
    n = len(y)
    mean_x = (n - 1) / 2
    mean_y = sum(y) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    if denom == 0:
        return 0.0, mean_y
    num = sum((i - mean_x) * (y[i] - mean_y) for i in range(n))
    slope = num / denom
    return slope, mean_y - slope * mean_x


def _distinct_touches(indices: List[int], min_gap: int = CHANNEL_TOUCH_MIN_GAP) -> int:
    """Compte les touches distinctes : deux bougies à moins de min_gap l'une de
    l'autre appartiennent au même contact du bord, pas à deux."""
    count = 0
    last = None
    for i in sorted(indices):
        if last is None or i - last >= min_gap:
            count += 1
        last = i
    return count


def count_touches(df: pd.DataFrame, level: float, tol: float) -> int:
    """Nombre de contacts DISTINCTS entre le prix et le niveau.

    Deux points :

    - Test d'intervalle, pas de distance à l'extrême : une bougie qui traverse le
      niveau de part en part le touche, même si son high et son low en sont loin.
    - Contacts distincts, pas bougies. La tolérance vaut 0.5×ATR : dans un range,
      des dizaines de bougies consécutives stationnent dans la bande. Les compter
      une par une donnait « 28 touches » pour un niveau visité **une seule fois** —
      un chiffre qui se lit comme une force et n'en mesure aucune.
    """
    lows = df["low"].values
    highs = df["high"].values
    hits = [i for i in range(len(df)) if lows[i] - tol <= level <= highs[i] + tol]
    return _distinct_touches(hits)


# --------------------------------------------------------------------------- #
# Supports / résistances
# --------------------------------------------------------------------------- #
def sr_levels(df: pd.DataFrame, price: float,
              lookback: int = SR_LOOKBACK) -> List[Dict[str, Any]]:
    """Le support le plus proche sous le prix et la résistance la plus proche
    au-dessus, avec leur nombre de touches.

    La détection est déléguée à `strategy.h1_sr_levels` — le détecteur déjà
    calibré, qui exige une validation par répétition — et on ne fait qu'y ajouter
    le compte, que la fonction d'origine ne renvoie pas.
    """
    if len(df) < 8:
        return []
    sr = strategy.h1_sr_levels(df, lookback=lookback,
                               min_touches=SR_MIN_TOUCHES, tol_atr=SR_TOL_ATR)
    tol = SR_TOL_ATR * _tol_unit(df)
    sub = df.tail(lookback)

    # Un niveau à moins d'une tolérance du prix n'en est pas un : on est dessus.
    # Le tracer donnerait un "support" 0.23 sous le prix, inutilisable.
    out: List[Dict[str, Any]] = []
    supports = [lv for lv in sr["support"] if lv <= price - tol]
    resistances = [lv for lv in sr["resistance"] if lv >= price + tol]
    if supports:
        lv = max(supports)
        out.append({"kind": "support", "price": lv, "distance": price - lv,
                    "touches": count_touches(sub, lv, tol)})
    if resistances:
        lv = min(resistances)
        out.append({"kind": "resistance", "price": lv, "distance": lv - price,
                    "touches": count_touches(sub, lv, tol)})
    return out


# --------------------------------------------------------------------------- #
# Order Blocks
# --------------------------------------------------------------------------- #
def order_blocks(df: pd.DataFrame, price: float,
                 max_per_side: int = MAX_OB_PER_SIDE) -> List[Dict[str, Any]]:
    """Les zones non mitiguées les plus proches du prix, plafonnées par côté.

    Un OB haussier au-dessus du prix (ou baissier en dessous) est écarté : le prix
    l'a dépassé, il n'y a plus de retest à attendre dessus.
    """
    from strategy_ict import _find_order_blocks  # import tardif : évite un cycle

    atr_val = _tol_unit(df)
    if atr_val <= 0:
        return []
    sub = df.tail(OB_LOOKBACK_BARS)

    found: List[Dict[str, Any]] = []
    for direction, kind in (("LONG", "haussier"), ("SHORT", "baissier")):
        raw = _find_order_blocks(sub, direction, atr_val, require_bos=True)
        side: List[Dict[str, Any]] = []
        for ob in raw:
            low, high = float(ob["low"]), float(ob["high"])
            if kind == "haussier" and low > price:
                continue
            if kind == "baissier" and high < price:
                continue
            gap = 0.0 if low <= price <= high else min(abs(price - low), abs(price - high))
            ts = ob.get("ts")
            side.append({
                "type": kind,
                "low": low,
                "high": high,
                "mid": (low + high) / 2,
                "height": high - low,
                "distance": gap,
                "distance_atr": gap / atr_val,
                "price_inside": low <= price <= high,
                "time": ts.isoformat() if hasattr(ts, "isoformat") else None,
            })
        side.sort(key=lambda o: o["distance"])
        found.extend(side[:max_per_side])

    found.sort(key=lambda o: o["distance"])
    return found


# --------------------------------------------------------------------------- #
# Canal parallèle
# --------------------------------------------------------------------------- #
def parallel_channel(df: pd.DataFrame,
                     lookback: int = CHANNEL_LOOKBACK) -> Optional[Dict[str, Any]]:
    """Deux droites parallèles encadrant les `lookback` dernières bougies.

    Pente prise sur la régression des clôtures (elle utilise toutes les bougies,
    là qu'un tracé sur deux creux choisis dépend entièrement de ces deux points),
    puis translation jusqu'au plus bas et au plus haut résiduels. Les trois
    ancres renvoyées sont directement celles d'un OBJ_CHANNEL MT5.

    Le canal encadre le prix par construction : ce sont les touches, pas la
    largeur, qui disent s'il est réel — d'où `fiable`.
    """
    sub = df.tail(lookback)
    n = len(sub)
    if n < 20:
        return None

    closes = [float(v) for v in sub["close"].values]
    lows = [float(v) for v in sub["low"].values]
    highs = [float(v) for v in sub["high"].values]

    slope, intercept = _least_squares(closes)
    line = [slope * i + intercept for i in range(n)]

    b_low = intercept + min(lows[i] - line[i] for i in range(n))
    b_high = intercept + max(highs[i] - line[i] for i in range(n))
    width = b_high - b_low
    if width <= 0:
        return None

    tol = CHANNEL_TOUCH_TOL_ATR * _tol_unit(sub)
    low_hits = [i for i in range(n) if lows[i] <= slope * i + b_low + tol]
    high_hits = [i for i in range(n) if highs[i] >= slope * i + b_high - tol]
    touches_low = _distinct_touches(low_hits)
    touches_high = _distinct_touches(high_hits)

    mean_y = sum(closes) / n
    ss_tot = sum((c - mean_y) ** 2 for c in closes)
    ss_res = sum((closes[i] - line[i]) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    drift = slope * (n - 1)
    if drift > 0.25 * width:
        direction = "haussier"
    elif drift < -0.25 * width:
        direction = "baissier"
    else:
        direction = "plat"

    t0, t1 = sub.index[0], sub.index[-1]
    price = closes[-1]
    bottom_now = slope * (n - 1) + b_low
    top_now = slope * (n - 1) + b_high

    def _pt(ts, value):
        return {"time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "price": value}

    return {
        "direction": direction,
        "bars": n,
        "slope_per_bar": slope,
        "width": width,
        "r2": r2,
        "touches_low": touches_low,
        "touches_high": touches_high,
        "fiable": touches_low >= CHANNEL_MIN_TOUCHES and touches_high >= CHANNEL_MIN_TOUCHES,
        "bottom_now": bottom_now,
        "top_now": top_now,
        "position_pct": (price - bottom_now) / width * 100,
        # OBJ_CHANNEL : points 1 et 2 = droite principale (le bas), point 3 = la
        # parallèle. MT5 reconstruit le haut tout seul à partir du point 3.
        "mt5": {
            "point1": _pt(t0, slope * 0 + b_low),
            "point2": _pt(t1, bottom_now),
            "point3": _pt(t0, slope * 0 + b_high),
        },
    }


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def _round_deep(obj: Any, digits: int) -> Any:
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round_deep(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_deep(v, digits) for v in obj]
    return obj


def build_levels(df: pd.DataFrame, *, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Les niveaux du symbole/timeframe affiché, prêts à tracer.

    `df` doit porter les indicateurs (colonne `atr`) — passer add_indicators().
    """
    digits = digits_for(symbol)
    if len(df) < 8:
        return {"symbol": symbol, "timeframe": timeframe, "price": None,
                "digits": digits, "order_blocks": [], "sr": [], "channel": None,
                "note": "Pas assez de bougies pour calculer des niveaux."}

    price = float(df["close"].iloc[-1])
    obs = order_blocks(df, price)
    channel = parallel_channel(df)
    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": price,
        "digits": digits,
        "atr": _tol_unit(df),
        "last_bar": df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else None,
        "order_blocks": obs,
        "sr": sr_levels(df, price),
        "channel": channel,
    }
    # Les prix se lisent avec les décimales du symbole ; ni les compteurs, ni le
    # r2, ni les pourcentages ne sont des prix — d'où l'arrondi ciblé.
    for key in ("price", "atr"):
        out[key] = round(out[key], digits)
    out["order_blocks"] = [
        {**ob, **{k: round(ob[k], digits) for k in ("low", "high", "mid", "height", "distance")},
         "distance_atr": round(ob["distance_atr"], 2)}
        for ob in out["order_blocks"]
    ]
    out["sr"] = [{**lv, "price": round(lv["price"], digits),
                  "distance": round(lv["distance"], digits)} for lv in out["sr"]]
    if channel:
        for key in ("width", "bottom_now", "top_now"):
            channel[key] = round(channel[key], digits)
        channel["slope_per_bar"] = round(channel["slope_per_bar"], digits + 2)
        channel["r2"] = round(channel["r2"], 3)
        channel["position_pct"] = round(channel["position_pct"], 1)
        channel["mt5"] = {k: {"time": v["time"], "price": round(v["price"], digits)}
                          for k, v in channel["mt5"].items()}
    return out


def as_text(data: Dict[str, Any]) -> str:
    """Rendu texte copiable — ce qu'on colle à côté de MT5 pendant qu'on trace."""
    d = data.get("digits", 2)
    f = lambda v: f"{v:.{d}f}"  # noqa: E731
    lines = [f"{data['symbol']} {data['timeframe']} — prix {f(data['price'])}"
             if data.get("price") is not None else f"{data['symbol']} {data['timeframe']}"]
    if data.get("note"):
        lines.append(data["note"])
        return "\n".join(lines)

    if data["order_blocks"]:
        lines.append("")
        lines.append("Order Blocks (zones non mitiguées) :")
        for ob in data["order_blocks"]:
            tag = " ← prix dedans" if ob["price_inside"] else f" ({f(ob['distance'])} du prix)"
            lines.append(f"  {ob['type']:9s} {f(ob['low'])} → {f(ob['high'])}{tag}")
    else:
        lines.append("")
        lines.append("Order Blocks : aucun non mitigué à portée.")

    if data["sr"]:
        lines.append("")
        lines.append("Supports / résistances :")
        for lv in data["sr"]:
            lines.append(f"  {lv['kind']:11s} {f(lv['price'])}  ({lv['touches']} touches)")

    ch = data.get("channel")
    if ch:
        lines.append("")
        flag = "" if ch["fiable"] else "  [peu fiable : trop peu de touches]"
        lines.append(f"Canal {ch['direction']} sur {ch['bars']} bougies{flag}")
        lines.append(f"  bas  {f(ch['bottom_now'])}   haut {f(ch['top_now'])}"
                     f"   largeur {f(ch['width'])}")
        lines.append(f"  touches {ch['touches_low']} bas / {ch['touches_high']} haut"
                     f"   position du prix {ch['position_pct']}%")
        lines.append("  MT5 (OBJ_CHANNEL, 3 ancres) :")
        for name in ("point1", "point2", "point3"):
            p = ch["mt5"][name]
            lines.append(f"    {name} {p['time']}  {f(p['price'])}")
    return "\n".join(lines)
