"""Interface unique d'accès aux bougies : `get_ohlcv(symbol, timeframe, n_bars)`.

§2 de la spec : MT5 et CCXT doivent être interchangeables. Le point important
est que **l'appelant ne sait jamais d'où viennent les bougies** — c'est ce qui
permettra d'ajouter BTC/USDT sans toucher à structure.py ni à confluence.py.

Choix d'implémentation, et pourquoi il diffère de la spec sur un point :

La spec dit « XAU/USD via MetaTrader5 ». Le paquet MetaTrader5 est **Windows
seulement**, or le déploiement tourne sous Linux (Railway) — un backend MT5 y
serait mort-né. Le backend par défaut est donc `data_provider`, la chaîne déjà
en place dans ce dépôt (Twelve Data → Polygon → Alpha Vantage → yfinance →
synthétique). MT5 reste implémentable derrière la même interface, et le sera
sans rien changer en amont : c'est exactement ce que l'interface unique achète.

Deux garde-fous que ce dépôt a appris à ses dépens :

- **Jamais de repli muet.** Si les bougies sont synthétiques, on le dit dans
  `provider` et `is_synthetic` : un scanner qui alerte sur des données simulées
  serait pire qu'un scanner muet.
- **Bougie en cours exclue par défaut.** §3 : « Aucun calcul ne doit utiliser
  une bougie non clôturée. » C'est appliqué ici, une fois, plutôt que d'espérer
  que chaque module y pense.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_provider  # noqa: E402

# Timeframe → (intervalle demandé nativement au fournisseur, règle de resample).
# On demande le plus gros intervalle NATIF disponible puis on resample : demander
# 3 mois de D1 en bougies M5 ferait des centaines de milliers de lignes pour
# quelques dizaines de bougies utiles.
_TF_SPEC: Dict[str, tuple[str, Optional[str]]] = {
    "M5":  ("5min", None),
    "M15": ("15min", None),
    "H1":  ("1h", None),
    "H4":  ("1h", "240min"),
    "D1":  ("1h", "1440min"),
}

_TF_MINUTES: Dict[str, int] = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}

_AGG = {"open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"}


@dataclass
class Candles:
    """Bougies + leur provenance. La provenance fait partie du résultat : elle
    conditionne le droit d'alerter."""
    df: pd.DataFrame
    symbol: str
    timeframe: str
    provider: str
    is_synthetic: bool

    def __len__(self) -> int:
        return len(self.df)


def timeframe_minutes(timeframe: str) -> int:
    tf = timeframe.upper()
    if tf not in _TF_MINUTES:
        raise ValueError(f"timeframe inconnu : {timeframe!r} "
                         f"(attendu : {', '.join(_TF_MINUTES)})")
    return _TF_MINUTES[tf]


def drop_unclosed(df: pd.DataFrame, timeframe: str,
                  now: Optional[datetime] = None) -> pd.DataFrame:
    """Retire la bougie en cours (§3 : anti look-ahead).

    L'index est daté à la FIN de la période (resample `label="right"`), donc une
    bougie est close quand son horodatage est passé. Sans ce filtre, la dernière
    bougie change de valeur à chaque tick et un BOS peut apparaître puis
    disparaître — le défaut le plus difficile à voir après coup.
    """
    if df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
        df = df.copy()
        df.index = idx
    return df[idx <= pd.Timestamp(now)]


def _bars_to_days(timeframe: str, n_bars: int) -> int:
    """Combien de jours calendaires demander pour obtenir n_bars clôturées.

    ×2 puis un plancher : le marché ferme la nuit et le week-end, donc une
    fenêtre calculée sur du temps continu rend systématiquement trop peu de
    bougies — et un manque silencieux de bougies est exactement ce qui a produit
    l'« EMA200 sur 43 bougies » du bot principal.
    """
    minutes = timeframe_minutes(timeframe) * n_bars
    return max(5, int(minutes / (60 * 24) * 2) + 2)


def get_ohlcv(symbol: str, timeframe: str, n_bars: int = 500,
              now: Optional[datetime] = None,
              include_unclosed: bool = False) -> Candles:
    """Les `n_bars` dernières bougies CLÔTURÉES de `symbol` en `timeframe`.

    `include_unclosed=True` n'est là que pour l'affichage (le prix courant sur un
    graphique) : aucune détection ne doit l'utiliser.
    """
    tf = timeframe.upper()
    if tf not in _TF_SPEC:
        raise ValueError(f"timeframe inconnu : {timeframe!r} "
                         f"(attendu : {', '.join(_TF_SPEC)})")
    interval, rule = _TF_SPEC[tf]

    days = _bars_to_days(tf, n_bars)
    end = (now or datetime.now(timezone.utc)).date() + timedelta(days=1)
    start = end - timedelta(days=days)

    raw, provider = data_provider.get_m5(
        start=start.isoformat(), end=end.isoformat(),
        symbol=symbol, interval=interval,
    )
    if rule:
        raw = raw.resample(rule, label="right", closed="right").agg(_AGG).dropna()

    if not include_unclosed:
        raw = drop_unclosed(raw, tf, now=now)

    return Candles(
        df=raw.tail(n_bars),
        symbol=symbol,
        timeframe=tf,
        provider=provider,
        is_synthetic=(provider == "synthetic"),
    )
