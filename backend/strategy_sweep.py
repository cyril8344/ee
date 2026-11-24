"""Stratégie C — balayage/récupération en cascade D1 → H1 → M15.

Transcription de la méthode lue image par image (Hyvirtus France), **telle
qu'énoncée et rien de plus** :

    Étape 1 — Daily   : la bougie d'hier a balayé le plus bas de l'avant-veille
                        (PDL) puis a CLÔTURÉ au-dessus → direction = achat.
    Étape 2 — H1      : même figure une échelle plus bas → setup. Le niveau
                        balayé devient le niveau de référence.
    Étape 3 — M15     : le prix CASSE ce niveau dans le sens du biais, puis
                        revient sur l'Order Block (dernière bougie contraire
                        avant l'impulsion) → entrée.
                        SL sous l'Order Block. TP1 un peu sous le plus haut H1.
                        TP2 sur la liquidité journalière opposée (PDH).

────────────────────────────────────────────────────────────────────────────
CE QUI N'EST PAS ICI, ET POURQUOI
────────────────────────────────────────────────────────────────────────────
Aucun filtre du mode A : pas de session, pas d'ADX, pas de RSI, pas de VWAP,
pas de patterns, pas de TP1=0.7R/TP2=1.8R. C'est délibéré et c'est tout
l'intérêt du module.

Greffer la règle sur le pipeline existant répondrait à « survit-elle à nos neuf
filtres ? » et non à « vaut-elle quelque chose ? ». Pire : les survivants
seraient biaisés. Mesuré sur séries calibrées or, le seul filtre VWAP écarte
51 % des signaux LONG de cette méthode — et ceux qui passent sont justement les
balayages qui ressemblent à de la continuation de tendance, c'est-à-dire le
sous-ensemble qui n'est pas la méthode. On mesurerait l'ancienne stratégie une
seconde fois en croyant en tester une nouvelle.

Ce dépôt a déjà payé cette erreur : le replay §11.3 du scanner rendait 0 signal,
12 candidats sur 12 tués par une seule condition contradictoire, et on a failli
corriger les mauvais seuils.

────────────────────────────────────────────────────────────────────────────
CE QUE LE MODULE N'AFFIRME PAS
────────────────────────────────────────────────────────────────────────────
Rien sur la rentabilité. La condition de base tombe sur ~25 % des bougies
journalières d'une marche aléatoire (mesuré, cf. `smc.candle_sweep`) : elle
n'est pas rare, et un exemple bien choisi ne prouve rien. Seul un walk-forward
tranche, et tant qu'il n'a pas tourné cette stratégie ne doit pas devenir celle
de l'or en live.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from smc.candle_sweep import sweep_at
from strategy import Signal

# --------------------------------------------------------------------------- #
# Paramètres — tous déclarés ici, tous testables en walk-forward
# --------------------------------------------------------------------------- #
# La vidéo dit « un peu en dessous » du plus haut H1 pour le TP1 et ne chiffre
# pas. On refuse de laisser la formule implicite : c'est une marge en ATR,
# nommée, et elle se teste.
TP1_MARGIN_ATR = 0.2
SL_MARGIN_ATR = 0.2

# Fenêtres de validité. Un setup H1 vieux de deux jours ne décrit plus le
# marché courant ; sans plafond la stratégie entrerait sur n'importe quoi.
H1_SETUP_MAX_BARS = 12        # le balayage H1 doit dater de moins de 12 bougies
M15_BREAK_MAX_BARS = 16       # la cassure M15 doit dater de moins de 16 bougies
# Recherche de l'Order Block avant la cassure. Attention en lisant un
# résultat : à 12 bougies de recul, une bougie de couleur contraire existe
# presque toujours, donc cette condition ne filtre quasiment rien. Ce qui lie
# réellement, c'est le RETOUR du prix dans la zone.
OB_LOOKBACK_BARS = 12

# R:R minimum. La vidéo n'en impose aucun — elle dit seulement « SL proche,
# TP loin, moins de chances ». 0.0 = aucun filtre, valeur par défaut fidèle.
MIN_RR = 0.0

ATR_PERIOD = 14
MAX_DURATION_MIN = 240        # 4 h : l'horloge de la méthode est le H1, pas le M5


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """ATR du dernier chandelier, ou NaN si la série est trop courte."""
    if len(df) < period + 1:
        return float("nan")
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False,
                        min_periods=period).mean().iloc[-1])


def _closed(df: pd.DataFrame, now: datetime, minutes: int) -> pd.DataFrame:
    """Ne garde que les bougies **terminées** à l'instant `now`.

    C'est le garde-fou anti look-ahead du module, et il n'est pas optionnel :
    la bougie journalière en cours connaît son plus bas définitif seulement à
    minuit. Juger le balayage d'hier sur une bougie encore ouverte revient à
    lire le futur, et produit un backtest qui ne se reproduit jamais.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    borne = pd.Timestamp(now)
    tz = df.index.tz
    if tz is not None:
        borne = borne.tz_localize(tz) if borne.tzinfo is None \
            else borne.tz_convert(tz)
    elif borne.tzinfo is not None:
        borne = borne.tz_localize(None)
    return df[(df.index + pd.Timedelta(minutes=minutes)) <= borne]


def evaluate_sweep(m5: pd.DataFrame,
                   m15: pd.DataFrame,
                   h1: pd.DataFrame,
                   d1: pd.DataFrame,
                   now: Optional[datetime] = None,
                   tp1_margin_atr: float = TP1_MARGIN_ATR,
                   sl_margin_atr: float = SL_MARGIN_ATR,
                   h1_setup_max_bars: int = H1_SETUP_MAX_BARS,
                   m15_break_max_bars: int = M15_BREAK_MAX_BARS,
                   ob_lookback_bars: int = OB_LOOKBACK_BARS,
                   min_rr: float = MIN_RR,
                   _reject_log: Optional[Dict[str, int]] = None,
                   ) -> Optional[Signal]:
    """Les trois étapes, dans l'ordre. `None` dès qu'une n'est pas remplie.

    L'ordre compte : le journalier fixe le sens, jamais l'inverse. Sans ça une
    mèche de M15 déciderait du biais de la journée.
    """
    def rejet(motif: str) -> None:
        if _reject_log is not None:
            _reject_log[motif] = _reject_log.get(motif, 0) + 1

    if m5 is None or m5.empty:
        rejet("m5_vide"); return None
    now = now or m5.index[-1].to_pydatetime()
    prix = float(m5["close"].iloc[-1])

    d1c = _closed(d1, now, 24 * 60)
    h1c = _closed(h1, now, 60)
    m15c = _closed(m15, now, 15)

    # --- Étape 1 : direction, sur la bougie journalière close ---------------
    if len(d1c) < 2:
        rejet("d1_court"); return None
    jour = sweep_at(d1c, len(d1c) - 1)
    if jour is None:
        rejet("etape1_pas_de_balayage_d1"); return None
    sens = jour.direction
    direction = "long" if sens == "bullish" else "short"

    # --- Étape 2 : setup H1, même sens, récent ------------------------------
    if len(h1c) < 2:
        rejet("h1_court"); return None
    setup = None
    for k in range(len(h1c) - 1, max(0, len(h1c) - 1 - h1_setup_max_bars), -1):
        s = sweep_at(h1c, k, direction=sens)
        if s is not None:
            setup = s
            break
    if setup is None:
        rejet("etape2_pas_de_setup_h1"); return None

    # Les deux traits « PLUS HAUT 1H » / « PLUS BAS 1H » de la vidéo sont les
    # extrêmes de la bougie de RÉFÉRENCE — celle qui précède le balayage, donc
    # celle dont le niveau a été pris. `sweep_at` renvoie déjà son plus bas (ou
    # plus haut) comme `niveau` ; l'autre bord sert de cible au TP1.
    ref = setup.index - 1
    if ref < 0:
        rejet("etape2_pas_de_bougie_de_reference"); return None
    niveau = setup.niveau
    extreme_h1 = (float(h1c["high"].values[ref]) if sens == "bullish"
                  else float(h1c["low"].values[ref]))

    # --- Étape 3 : cassure M15 puis retour sur l'Order Block ----------------
    if len(m15c) < 3:
        rejet("m15_court"); return None
    # La cassure est cherchée APRÈS la clôture de la bougie H1 du setup, et pas
    # à l'intérieur. Deux raisons, et la seconde est dirimante :
    #
    #  - anti look-ahead : on ne sait que la bougie H1 a récupéré son niveau
    #    qu'une fois qu'elle est close ; agir avant, c'est lire le futur ;
    #  - sinon la condition serait VIDE. La clôture H1 au-dessus du niveau est,
    #    par construction, la clôture de sa dernière M15 : chercher une M15 qui
    #    clôture au-dessus à l'intérieur de l'heure en trouverait toujours une.
    #    L'étape 3 n'ajouterait rien à l'étape 2 — c'est exactement le piège des
    #    « trois confirmations » qui n'en sont qu'une.
    t_setup_fin = h1c.index[setup.index] + pd.Timedelta(minutes=60)
    apres = m15c[m15c.index >= t_setup_fin]
    if apres.empty:
        rejet("etape3_pas_de_m15_apres_setup"); return None

    opens = apres["open"].values
    closes = apres["close"].values
    highs = apres["high"].values
    lows = apres["low"].values

    cassure = None
    for j in range(len(apres)):
        if sens == "bullish" and float(closes[j]) > niveau:
            cassure = j
            break
        if sens == "bearish" and float(closes[j]) < niveau:
            cassure = j
            break
    if cassure is None:
        rejet("etape3_pas_de_cassure"); return None
    if (len(apres) - 1 - cassure) > m15_break_max_bars:
        rejet("etape3_cassure_trop_ancienne"); return None

    # Order Block : dernière bougie de couleur contraire avant l'impulsion.
    # C'est la bougie que la vidéo entoure ; définition volontairement
    # littérale, pas celle du scanner (qui prend l'extrême de la jambe).
    ob = None
    for j in range(cassure, max(-1, cassure - ob_lookback_bars), -1):
        if j < 0:
            break
        rouge = float(closes[j]) < float(opens[j])
        verte = float(closes[j]) > float(opens[j])
        if (sens == "bullish" and rouge) or (sens == "bearish" and verte):
            ob = j
            break
    if ob is None:
        rejet("etape3_pas_d_order_block"); return None

    ob_haut, ob_bas = float(highs[ob]), float(lows[ob])
    if ob_haut <= ob_bas:
        rejet("ob_degenere"); return None

    # Déclencheur : le prix est REVENU dans la zone. Avant ça il n'y a pas
    # d'entrée — c'est le « le prix revient, certains entrent » de la vidéo.
    if not (ob_bas <= prix <= ob_haut):
        rejet("etape3_prix_hors_order_block"); return None

    atr = _atr(m15c)
    if not (atr > 0):
        rejet("atr_indisponible"); return None

    pdh, pdl = float(d1c["high"].iloc[-1]), float(d1c["low"].iloc[-1])
    if sens == "bullish":
        entry = ob_haut
        stop = ob_bas - sl_margin_atr * atr
        tp1 = extreme_h1 - tp1_margin_atr * atr
        tp2 = max(pdh, tp1)                        # PDH : liquidité opposée
        valide = stop < entry < tp1
    else:
        entry = ob_bas
        stop = ob_haut + sl_margin_atr * atr
        tp1 = extreme_h1 + tp1_margin_atr * atr
        tp2 = min(pdl, tp1)
        valide = tp1 < entry < stop
    if not valide:
        rejet("geometrie_incoherente"); return None
    # La vidéo place TP2 au-delà de TP1, sur la liquidité journalière. Rien ne
    # garantit que le PDH soit effectivement au-delà : quand il ne l'est pas,
    # TP2 retombe sur TP1 (cible unique) et le `meta` le dit, plutôt que de
    # rendre une seconde cible fabriquée.
    tp2_sur_liquidite = (tp2 != tp1)

    risque = abs(entry - stop)
    if risque <= 0:
        rejet("risque_nul"); return None
    rr = abs(tp1 - entry) / risque
    if min_rr > 0 and rr < min_rr:
        rejet("rr_insuffisant"); return None

    return Signal(
        direction=direction,
        bias="LONG" if direction == "long" else "SHORT",
        session="sweep",
        entry=entry,
        stop_loss=stop,
        take_profit1=tp1,
        take_profit2=tp2,
        atr=atr,
        reason=f"balayage D1 {sens} → setup H1 {niveau:.2f} → OB M15",
        risk_distance=risque,
        timestamp=now,
        max_duration_min=MAX_DURATION_MIN,
        meta={
            "niveau_h1": niveau,
            "extreme_h1": extreme_h1,
            "pdl": pdl,
            "pdh": pdh,
            "tp2_sur_liquidite": tp2_sur_liquidite,
            "ob_ts": apres.index[ob],
            "ob_haut": ob_haut,
            "ob_bas": ob_bas,
            "rr_tp1": rr,
            "depassement_d1": jour.depassement,
            "recuperation_d1": jour.recuperation,
        },
    )


def resample(m5: pd.DataFrame, regle: str) -> pd.DataFrame:
    """M5 → M15 / H1 / D1, convention OHLC standard.

    Fourni ici pour que l'appelant n'ait pas à réinventer l'agrégation, et
    surtout pour que pretrain et le live la partagent : le dépôt a déjà payé un
    biais H1 calculé différemment en live et en backtest.
    """
    out = m5.resample(regle).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"})
    if "volume" in m5.columns:
        out["volume"] = m5["volume"].resample(regle).sum()
    return out.dropna(subset=["open", "high", "low", "close"])
