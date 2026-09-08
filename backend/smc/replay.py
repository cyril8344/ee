"""Replay : rejouer l'historique M15 et compter les signaux (§11.3).

La spec fixe elle-même le critère d'acceptation, et c'est la meilleure chose
qu'elle contienne :

    « rejouer 6 mois de M15 et compter les signaux générés (ordre de grandeur
      attendu : 2–6 signaux/semaine ; >15/semaine = filtres trop laxistes,
      corriger avant de continuer). »

C'est une prédiction falsifiable posée AVANT de voir le résultat. Ce module ne
fait que la mesurer — il ne cherche pas à la satisfaire, et il ne contient aucun
réglage qu'on pourrait tourner jusqu'à ce que le chiffre tombe juste. Si le
compte sort hors bornes, la correction se fait dans confluence.py ou zones.py,
en connaissance de cause, pas ici.

Le replay avance bougie par bougie et ne passe à `evaluate()` que les données
disponibles à cet instant (`.iloc[:i+1]` sur chaque timeframe) : c'est le même
anti look-ahead que dans structure.py, appliqué au niveau au-dessus. Sans ça, le
compte serait flatteur et ne se reproduirait jamais en live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from smc import config
from smc.confluence import Signal, evaluate, mark_dead_zones
from smc.zones import active_zones

# Bornes de la spec (§11.3). Elles ne sont pas des réglages : ce sont les
# hypothèses posées avant la mesure.
SIGNALS_PER_WEEK_MIN = 2.0
SIGNALS_PER_WEEK_MAX = 6.0
SIGNALS_PER_WEEK_ALARM = 15.0


@dataclass
class ReplayResult:
    signals: List[Signal] = field(default_factory=list)
    bars: int = 0
    days: float = 0.0
    weeks: float = 0.0

    @property
    def per_week(self) -> float:
        return len(self.signals) / self.weeks if self.weeks > 0 else 0.0

    @property
    def sweep_rate(self) -> Optional[float]:
        """Part des signaux taggés sweep. La statistique pour laquelle le tag
        existe (§5) : elle dira un jour si le sweep vaut d'être exigé."""
        if not self.signals:
            return None
        return sum(1 for s in self.signals if s.sweep) / len(self.signals) * 100

    @property
    def verdict(self) -> str:
        """Le jugement de la spec, pas le mien."""
        if self.weeks <= 0:
            return "indéterminé : période trop courte"
        r = self.per_week
        if r > SIGNALS_PER_WEEK_ALARM:
            return (f"FILTRES TROP LAXISTES ({r:.1f}/sem > {SIGNALS_PER_WEEK_ALARM}) "
                    f"— corriger avant de continuer (§11.3)")
        if r > SIGNALS_PER_WEEK_MAX:
            return f"au-dessus de l'attendu ({r:.1f}/sem, attendu 2–6)"
        if r < SIGNALS_PER_WEEK_MIN:
            return f"en dessous de l'attendu ({r:.1f}/sem, attendu 2–6)"
        return f"conforme à l'attendu ({r:.1f}/sem)"

    def summary(self) -> Dict[str, Any]:
        par_direction: Dict[str, int] = {}
        par_zone: Dict[str, int] = {}
        for s in self.signals:
            par_direction[s.direction] = par_direction.get(s.direction, 0) + 1
            par_zone[s.zone.kind] = par_zone.get(s.zone.kind, 0) + 1
        return {"n": len(self.signals), "bars": self.bars,
                "days": round(self.days, 1), "weeks": round(self.weeks, 2),
                "per_week": round(self.per_week, 2),
                "sweep_rate_pct": (round(self.sweep_rate, 1)
                                   if self.sweep_rate is not None else None),
                "by_direction": par_direction, "by_zone_kind": par_zone,
                "verdict": self.verdict}


def _slice_upto(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Les bougies d'un timeframe supérieur CLÔTURÉES à l'instant `ts`.

    Comparaison stricte : une bougie H1 datée 10:00 (fin de période) n'est close
    qu'à 10:00, donc à 09:45 en M15 elle n'existe pas encore. L'inclure ferait
    lire au scanner une bougie H1 en cours de formation — exactement le défaut
    que `drop_unclosed` évite en live, et qu'un replay naïf réintroduit.
    """
    return df[df.index <= ts]


def run_replay(m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame,
               d1: Optional[pd.DataFrame] = None, symbol: str = "XAUUSD",
               warmup: int = 200, progress_every: int = 0,
               on_progress: Optional[Any] = None) -> ReplayResult:
    """Rejoue la série M15 et collecte les signaux.

    `warmup` laisse de quoi calculer l'ATR(14) et les premiers pivots ; sans lui
    les premières centaines de bougies produiraient des non-signaux qui
    fausseraient le compte hebdomadaire vers le bas.

    `on_progress(done, total, n_signaux)` est appelé périodiquement. Un replay de
    6 mois prend plusieurs minutes : sans ce rappel, l'interface resterait muette
    assez longtemps pour qu'on la croie plantée.
    """
    res = ReplayResult()
    if len(m15) <= warmup:
        return res

    signalled: Set[Any] = set()
    dead: Set[Any] = set()

    for i in range(warmup, len(m15)):
        ts = m15.index[i]
        h1_i = _slice_upto(h1, ts)
        h4_i = _slice_upto(h4, ts)
        d1_i = _slice_upto(d1, ts) if d1 is not None else None
        if len(h1_i) < 30 or len(h4_i) < 30:
            continue

        zones_h1 = active_zones(h1_i, "H1", h1=h1_i)
        mark_dead_zones(m15, zones_h1, i, dead)

        sig = evaluate(m15.iloc[:i + 1], h1_i, h4_i, d1_i, symbol=symbol,
                       upto=i, dead_zones=dead, signalled_zones=signalled)
        if sig is not None:
            signalled.add(sig.zone_key)
            res.signals.append(sig)

        done, total = i - warmup, len(m15) - warmup
        if progress_every and done % progress_every == 0:
            print(f"  … {done}/{total} bougies, {len(res.signals)} signaux",
                  flush=True)
        if on_progress is not None and done % 25 == 0:
            on_progress(done, total, len(res.signals))

    res.bars = len(m15) - warmup
    span = m15.index[-1] - m15.index[warmup]
    res.days = span.total_seconds() / 86400
    res.weeks = res.days / 7
    return res


def main() -> None:  # pragma: no cover - utilitaire de ligne de commande
    """`python backend/smc/replay.py` — le test d'acceptation du §11.3."""
    import argparse

    from smc import data_feed

    ap = argparse.ArgumentParser(description="Replay SMC (§11.3)")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--m15-bars", type=int, default=6000)
    args = ap.parse_args()

    m15 = data_feed.get_ohlcv(args.symbol, "M15", args.m15_bars)
    h1 = data_feed.get_ohlcv(args.symbol, "H1", args.m15_bars // 4 + 300)
    h4 = data_feed.get_ohlcv(args.symbol, "H4", args.m15_bars // 16 + 200)
    d1 = data_feed.get_ohlcv(args.symbol, "D1", 400)

    if m15.is_synthetic:
        print("⚠️  données SYNTHÉTIQUES — le compte ci-dessous ne dit rien du "
              "marché réel, seulement que la mécanique tourne.")

    res = run_replay(m15.df, h1.df, h4.df, d1.df, symbol=args.symbol,
                     progress_every=500)
    s = res.summary()
    print(f"\n{args.symbol} — {s['bars']} bougies M15 / {s['days']} jours "
          f"({s['weeks']} semaines)")
    print(f"  signaux           : {s['n']}  →  {s['per_week']}/semaine")
    print(f"  par direction     : {s['by_direction']}")
    print(f"  par type de zone  : {s['by_zone_kind']}")
    print(f"  taggés sweep      : {s['sweep_rate_pct']}%")
    print(f"  verdict (§11.3)   : {s['verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
