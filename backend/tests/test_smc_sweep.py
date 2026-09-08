"""Sweep de liquidité — détecté et taggé, jamais exigé (§5).

Le test qui compte ici est `test_sweep_is_never_a_signal_condition` : ce dépôt a
déjà retiré « sweep obligatoire » de la stratégie B après l'avoir testé en
walk-forward. L'exiger de nouveau referait l'erreur, en pire — sans les signaux
sans sweep, on n'aurait même plus de quoi la mesurer.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import config, sweep


def _frame(rows, start="2026-01-05 08:00", freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _calm(price, n):
    return [(price, price + 0.5, price - 0.5, price) for _ in range(n)]


# Profondeur maximale tolérée = 0,5 × ATR. Sur ces fixtures l'ATR vaut ~1.75,
# donc la limite est ~0.87 : une mèche à 0.3 sous le niveau est un sweep, une
# mèche à 50 est une vraie cassure. Mes premières fixtures utilisaient 1.0 et
# étaient donc rejetées — à juste titre par le code, à tort par le test.
def _swept_low(depth=0.3):
    """Un creux net, puis une mèche qui passe dessous et reclôture au-dessus."""
    rows = _calm(100, 16)
    rows += [(100, 100.4, 95, 96)]              # 16 : pivot bas à 95
    rows += _calm(100, 4)                        # confirmé en 18
    rows += [(100, 100.5, 95 - depth, 96)]      # 21 : mèche sous 95, clôture au-dessus
    rows += _calm(100, 3)
    return _frame(rows)


def test_a_wick_below_the_low_that_closes_back_above_is_a_sweep():
    sw = sweep.find_sweeps(_swept_low(depth=0.3))
    hauts = [s for s in sw if s.direction == "bullish"]
    assert hauts
    s = hauts[0]
    assert s.level == pytest.approx(95)
    assert s.extreme == pytest.approx(94.7)
    assert s.depth_atr > 0


def test_a_deep_break_is_not_a_sweep():
    """Au-delà de 0,5 × ATR sous le niveau, c'est une vraie cassure (§5) — la
    différence entre une prise de liquidité et un retournement de structure."""
    profond = sweep.find_sweeps(_swept_low(depth=50.0))
    assert [s for s in profond if s.direction == "bullish"] == []


def test_price_must_reclose_on_the_right_side():
    """Une mèche qui perce et reste sous le niveau n'est pas un sweep : personne
    n'a été piégé, le prix est simplement parti."""
    rows = _calm(100, 16) + [(100, 100.4, 95, 96)] + _calm(100, 4)
    rows += [(100, 100.5, 94, 94.2)]            # perce et clôture SOUS 95
    rows += [(94, 94.5, 93.5, 94)] * 4          # et y reste
    assert [s for s in sweep.find_sweeps(_frame(rows))
            if s.direction == "bullish"] == []


def test_reclose_is_allowed_within_two_bars():
    """§5 : « OU dont le prix reclôture au-dessus dans les 2 bougies suivantes »."""
    rows = _calm(100, 16) + [(100, 100.4, 95, 96)] + _calm(100, 4)
    rows += [(100, 100.5, 94.7, 94.9)]          # perce, clôture sous
    rows += [(94.9, 95.2, 94.8, 94.95)]         # +1 : toujours sous
    rows += [(94.95, 96.5, 94.9, 96)]           # +2 : reclôture au-dessus
    rows += _calm(100, 3)
    sw = [s for s in sweep.find_sweeps(_frame(rows)) if s.direction == "bullish"]
    assert sw
    assert sw[0].reclosed_index == sw[0].index + 2


def test_a_level_is_swept_only_once():
    rows = _calm(100, 16) + [(100, 100.4, 95, 96)] + _calm(100, 4)
    rows += [(100, 100.5, 94.7, 96)]            # premier sweep
    rows += _calm(100, 2)
    rows += [(100, 100.5, 94.8, 96)]            # même niveau re-percé
    rows += _calm(100, 3)
    sw = [s for s in sweep.find_sweeps(_frame(rows)) if s.direction == "bullish"]
    assert len(sw) == 1


def test_bearish_sweep_is_the_mirror():
    rows = _calm(100, 16) + [(100, 105, 99.6, 104)] + _calm(100, 4)
    rows += [(100, 105.3, 99.5, 104)]           # mèche au-dessus de 105, clôture sous
    rows += _calm(100, 3)
    sw = [s for s in sweep.find_sweeps(_frame(rows)) if s.direction == "bearish"]
    assert sw
    assert sw[0].level == pytest.approx(105)
    assert sw[0].extreme == pytest.approx(105.3)


def test_sweep_before_respects_the_eight_bar_window():
    """§5 : le CHoCH doit survenir dans les 8 bougies M15 après le sweep. Au-delà,
    les deux événements ne sont plus liés — les rapprocher produirait un tag qui
    ne veut rien dire, donc une feature ML bruitée."""
    s = sweep.Sweep(index=10, time=None, direction="bullish", level=95,
                    extreme=94, depth_atr=0.3, reclosed_index=10)
    assert sweep.sweep_before([s], 12, "bullish") is s
    assert sweep.sweep_before([s], 18, "bullish") is s         # 8e bougie : encore dans
    assert sweep.sweep_before([s], 19, "bullish") is None      # 9e : hors fenêtre
    assert sweep.sweep_before([s], 12, "bearish") is None      # mauvais sens
    assert sweep.sweep_before([s], 9, "bullish") is None       # CHoCH avant le sweep


def test_no_sweep_is_a_normal_outcome_not_an_error():
    """La majorité des signaux n'auront pas de sweep, et c'est exactement ce
    qu'on veut pouvoir comparer plus tard."""
    assert sweep.sweep_before([], 42, "bullish") is None
    assert sweep.find_sweeps(_frame(_calm(100, 40))) == []


def test_sweep_is_never_a_signal_condition():
    """Garde-fou de conception : le module ne doit exposer aucune fonction qui
    transforme le sweep en prérequis. `OB_REQUIRE_LIQUIDITY` de la stratégie B a
    été laissé à False pour cette raison exacte — trop de confluence exigée =
    trop peu de signaux, sans robustesse démontrée en out-of-sample."""
    interdits = ("require", "must", "mandatory", "obligatoire")
    for nom in dir(sweep):
        assert not any(mot in nom.lower() for mot in interdits)


def test_depth_is_reported_as_a_feature_not_a_verdict():
    """`depth_atr` part dans le dataset ML (§9), il ne doit pas être arrondi ni
    transformé en booléen : c'est la valeur brute qui servira à trancher."""
    sw = sweep.find_sweeps(_swept_low(depth=0.3))
    s = [x for x in sw if x.direction == "bullish"][0]
    assert isinstance(s.depth_atr, float)
    assert 0 < s.depth_atr <= config.SWEEP_MAX_DEPTH_ATR
