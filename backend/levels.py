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
# Fenêtre S/R VOLONTAIREMENT découplée de celle du canal. Le canal décrit le
# mouvement en cours, donc une fenêtre courte ; un support, lui, reste un support
# à 300 bougies de distance. Les deux partageaient 120 bougies : quand le prix
# atteint l'extrême de cette fenêtre — exactement le moment où on veut savoir où
# est le niveau suivant — il n'y avait plus rien d'un côté PAR CONSTRUCTION, et
# le panneau se taisait sans dire pourquoi.
SR_LOOKBACK = 400

MAX_OB_PER_SIDE = 2
# 5 h de M5 ne laissaient presque jamais d'OB non mitiguée survivante.
OB_LOOKBACK_BARS = 150

CHANNEL_LOOKBACK = 120
CHANNEL_TOUCH_TOL_ATR = 0.25
CHANNEL_MIN_TOUCHES = 2
# Deux bougies voisines calées sur le même creux ne font qu'une touche.
CHANNEL_TOUCH_MIN_GAP = 3

FIB_LOOKBACK = 120
# Un seul jeu de ratios, une seule formule : prix(r) = fin − r × (fin − départ).
# r=0 tombe sur l'extrémité du mouvement, r=1 sur son origine, r>1 au-delà —
# exactement ce que trace l'outil Fibonacci de MT5, dont les niveaux au-dessus de
# 100 % dépassent la première ancre. Pas de seconde convention à côté.
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618)
# Sous ce seuil le "mouvement" n'est que du bruit de range : les sept niveaux
# seraient séparés par moins d'une mèche.
FIB_MIN_LEG_ATR = 1.5

DIGITS = {"XAUUSD": 2, "EURUSD": 5}


def digits_for(symbol: str) -> int:
    return DIGITS.get(symbol, 2)


def _iso(ts: Any) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


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
              lookback: int = SR_LOOKBACK) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Le support le plus proche sous le prix et la résistance la plus proche
    au-dessus, avec leur nombre de touches — et la raison quand il n'y en a pas.

    La détection est déléguée à `strategy.h1_sr_levels` — le détecteur déjà
    calibré, qui exige une validation par répétition — et on ne fait qu'y ajouter
    le compte, que la fonction d'origine ne renvoie pas.

    Renvoie (niveaux, notes). Un côté vide sans explication se lit comme une
    panne ; « le prix est au plus bas des N bougies » est une information.
    """
    if len(df) < 8:
        return [], []
    sr = strategy.h1_sr_levels(df, lookback=lookback,
                               min_touches=SR_MIN_TOUCHES, tol_atr=SR_TOL_ATR)
    tol = SR_TOL_ATR * _tol_unit(df)
    sub = df.tail(lookback)
    n = len(sub)

    # Un niveau à moins d'une tolérance du prix n'en est pas un : on est dessus.
    # Le tracer donnerait un "support" 0.23 sous le prix, inutilisable.
    out: List[Dict[str, Any]] = []
    notes: List[str] = []
    supports = [lv for lv in sr["support"] if lv <= price - tol]
    resistances = [lv for lv in sr["resistance"] if lv >= price + tol]

    if supports:
        lv = max(supports)
        out.append({"kind": "support", "price": lv, "distance": price - lv,
                    "touches": count_touches(sub, lv, tol)})
    elif price <= float(sub["low"].min()) + tol:
        notes.append(f"Aucun support : le prix est au plus bas des {n} bougies.")
    else:
        notes.append(f"Aucun support validé ({SR_MIN_TOUCHES} contacts minimum) "
                     f"sous le prix sur {n} bougies.")

    if resistances:
        lv = min(resistances)
        out.append({"kind": "resistance", "price": lv, "distance": lv - price,
                    "touches": count_touches(sub, lv, tol)})
    elif price >= float(sub["high"].max()) - tol:
        notes.append(f"Aucune résistance : le prix est au plus haut des {n} bougies.")
    else:
        notes.append(f"Aucune résistance validée ({SR_MIN_TOUCHES} contacts minimum) "
                     f"au-dessus du prix sur {n} bougies.")

    return out, notes


# --------------------------------------------------------------------------- #
# Order Blocks
# --------------------------------------------------------------------------- #
def order_blocks(df: pd.DataFrame, price: float,
                 max_per_side: int = MAX_OB_PER_SIDE
                 ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Les zones non mitiguées les plus proches du prix, plafonnées par côté.

    Un OB haussier au-dessus du prix (ou baissier en dessous) est écarté : le prix
    l'a dépassé, il n'y a plus de retest à attendre dessus.

    Renvoie (zones, notes) — même raison que sr_levels : « aucune » sans motif se
    lit comme une panne. On distingue « le détecteur n'a rien trouvé » de « il a
    trouvé, mais le prix les a toutes dépassées ».
    """
    from strategy_ict import _find_order_blocks  # import tardif : évite un cycle

    atr_val = _tol_unit(df)
    if atr_val <= 0:
        return [], []
    sub = df.tail(OB_LOOKBACK_BARS)

    found: List[Dict[str, Any]] = []
    notes: List[str] = []
    n_raw = 0
    n_passed = 0
    for direction, kind in (("LONG", "haussier"), ("SHORT", "baissier")):
        raw = _find_order_blocks(sub, direction, atr_val, require_bos=True)
        n_raw += len(raw)
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
        n_passed += len(side)
        found.extend(side[:max_per_side])

    found.sort(key=lambda o: o["distance"])
    if not found:
        if n_raw == 0:
            notes.append(f"Aucune zone non mitiguée sur {len(sub)} bougies "
                         f"(corps, hauteur, impulsion et cassure de structure exigés).")
        else:
            notes.append(f"{n_raw} zone(s) détectée(s), toutes déjà dépassées par le prix.")
    elif n_passed > len(found):
        notes.append(f"{n_passed - len(found)} zone(s) plus lointaine(s) masquée(s) "
                     f"par le plafond de {max_per_side} par côté.")
    return found, notes


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
        return {"time": _iso(ts), "price": value}

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
# Fibonacci
# --------------------------------------------------------------------------- #
def fibonacci(df: pd.DataFrame,
              lookback: int = FIB_LOOKBACK) -> Optional[Dict[str, Any]]:
    """Retracements de la dernière jambe marquante de la fenêtre.

    **Le seul vrai choix d'un Fibonacci est le segment.** Les ratios ne se
    discutent pas et les prix en découlent. Ici il est pris mécaniquement — plus
    haut et plus bas de la fenêtre, dans l'ordre où ils arrivent — pour être
    reproductible à l'identique dans MT5. Le segment retenu est renvoyé en clair
    (dates et prix des deux extrémités) : si ce n'est pas celui qu'on aurait tracé
    à l'œil, on voit immédiatement qu'il faut ignorer le bloc.

    Aucune valeur prédictive n'est affirmée. Rien dans ce dépôt n'a jamais validé
    que les niveaux de Fibonacci tiennent sur l'or ; c'est une convention de
    tracé. Le seul chiffre mesuré ici est `touches` — le nombre de contacts
    distincts qu'un niveau a réellement eus depuis la fin de la jambe. Il ne dit
    pas que la méthode marche, il dit si ce marché-ci s'y arrête en ce moment.
    """
    sub = df.tail(lookback)
    n = len(sub)
    if n < 20:
        return None

    highs = sub["high"].values
    lows = sub["low"].values
    i_hi = int(highs.argmax())
    i_lo = int(lows.argmin())
    if i_hi == i_lo:
        return None

    if i_hi > i_lo:
        direction = "haussier"
        i_start, start_price = i_lo, float(lows[i_lo])
        i_end, end_price = i_hi, float(highs[i_hi])
    else:
        direction = "baissier"
        i_start, start_price = i_hi, float(highs[i_hi])
        i_end, end_price = i_lo, float(lows[i_lo])

    delta = end_price - start_price
    atr_val = _tol_unit(sub)
    if atr_val <= 0 or abs(delta) < FIB_MIN_LEG_ATR * atr_val:
        return None

    tol = CHANNEL_TOUCH_TOL_ATR * atr_val
    after = sub.iloc[i_end:]          # bougies depuis la fin de la jambe
    price = float(sub["close"].iloc[-1])

    niveaux = []
    for r in FIB_RATIOS:
        lv = end_price - r * delta
        niveaux.append({
            "ratio": r,
            "price": lv,
            "distance": abs(lv - price),
            "touches": count_touches(after, lv, tol),
        })

    return {
        "direction": direction,
        "bars": n,
        "leg": abs(delta),
        "leg_atr": abs(delta) / atr_val,
        "depart": {"time": _iso(sub.index[i_start]), "price": start_price,
                   "kind": "creux" if direction == "haussier" else "sommet"},
        "arrivee": {"time": _iso(sub.index[i_end]), "price": end_price,
                    "kind": "sommet" if direction == "haussier" else "creux"},
        # Où en est le retracement : 0 % = le prix est encore sur l'extrémité de
        # la jambe, 100 % = il est revenu à son origine, > 100 % = il l'a dépassée.
        "retracement_pct": (end_price - price) / delta * 100,
        "bars_since": len(after) - 1,
        "niveaux": niveaux,
        # OBJ_FIBO : 2 ancres. Avec la formule prix(r) = fin − r×(fin − départ),
        # l'ancre 1 porte le niveau 100 % et l'ancre 2 le niveau 0 %.
        "mt5": {
            "point1": {"time": _iso(sub.index[i_start]), "price": start_price},
            "point2": {"time": _iso(sub.index[i_end]), "price": end_price},
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
                "fib": None, "ob_notes": [], "sr_notes": [],
                "note": "Pas assez de bougies pour calculer des niveaux."}

    price = float(df["close"].iloc[-1])
    obs, ob_notes = order_blocks(df, price)
    sr, sr_notes = sr_levels(df, price)
    channel = parallel_channel(df)
    fib = fibonacci(df)
    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": price,
        "digits": digits,
        "atr": _tol_unit(df),
        "last_bar": _iso(df.index[-1]),
        "order_blocks": obs,
        "ob_notes": ob_notes,
        "sr": sr,
        "sr_notes": sr_notes,
        "channel": channel,
        "fib": fib,
        "sr_lookback": SR_LOOKBACK,
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
    if fib:
        fib["leg"] = round(fib["leg"], digits)
        fib["leg_atr"] = round(fib["leg_atr"], 2)
        fib["retracement_pct"] = round(fib["retracement_pct"], 1)
        for key in ("depart", "arrivee"):
            fib[key]["price"] = round(fib[key]["price"], digits)
        fib["niveaux"] = [{**lv, "price": round(lv["price"], digits),
                           "distance": round(lv["distance"], digits)}
                          for lv in fib["niveaux"]]
        fib["mt5"] = {k: {"time": v["time"], "price": round(v["price"], digits)}
                      for k, v in fib["mt5"].items()}
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

    lines.append("")
    lines.append("Order Blocks (zones non mitiguées) :")
    for ob in data["order_blocks"]:
        tag = " ← prix dedans" if ob["price_inside"] else f" ({f(ob['distance'])} du prix)"
        lines.append(f"  {ob['type']:9s} {f(ob['low'])} → {f(ob['high'])}{tag}")
    for note in data.get("ob_notes", []):
        lines.append(f"  {note}")

    lines.append("")
    lines.append("Supports / résistances :")
    for lv in data["sr"]:
        lines.append(f"  {lv['kind']:11s} {f(lv['price'])}  ({lv['touches']} contacts,"
                     f" {f(lv['distance'])} du prix)")
    for note in data.get("sr_notes", []):
        lines.append(f"  {note}")

    fib = data.get("fib")
    if fib:
        lines.append("")
        lines.append(f"Fibonacci — jambe {fib['direction']} de {f(fib['leg'])} "
                     f"({fib['leg_atr']}×ATR), retracé à {fib['retracement_pct']}%")
        lines.append(f"  départ  {fib['depart']['kind']:6s} {f(fib['depart']['price'])}"
                     f"  {fib['depart']['time'][:16].replace('T', ' ')}")
        lines.append(f"  arrivée {fib['arrivee']['kind']:6s} {f(fib['arrivee']['price'])}"
                     f"  {fib['arrivee']['time'][:16].replace('T', ' ')}")
        for lv in fib["niveaux"]:
            lines.append(f"    {lv['ratio'] * 100:5.1f}%  {f(lv['price'])}"
                         f"   {lv['touches']} contact(s) depuis")
        lines.append(f"  ({fib['bars_since']} bougies depuis la fin de la jambe — un "
                     f"niveau récent n'a pas encore eu l'occasion d'être touché)")
        lines.append("  MT5 (OBJ_FIBO, 2 ancres) :")
        for name in ("point1", "point2"):
            p = fib["mt5"][name]
            lines.append(f"    {name} {p['time']}  {f(p['price'])}")

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
