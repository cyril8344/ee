"""Zones SMC — OB, FVG, mitigation, fraîcheur, premium/discount (§4).

Ce qui distingue ces OB de ceux du graphique principal : ils sont attachés à une
cassure de structure ET à un déplacement minimum. Sans ces deux conditions on
retombe sur le détecteur qui empile 5-7 zones dont la plupart sont déjà mortes.

Note sur les fixtures : les bougies de swing ont volontairement une mèche
opposée *à l'intérieur* de celles du range calme (high 100.4 < 100.5 pour un
creux, low 99.6 > 99.5 pour un sommet). Sinon une même bougie devient pivot haut
ET pivot bas, et le scénario testé n'est plus celui qu'on croit. Il faut aussi
au moins 15 bougies avant l'événement : sous ce seuil l'ATR(14) est NaN et rien
n'est détecté — pour de bonnes raisons, mais ça ne teste plus rien.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import config, zones


def _frame(rows, start="2026-01-05 08:00", freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _calm(price, n):
    return [(price, price + 0.5, price - 0.5, price) for _ in range(n)]


def _bull_rows():
    """Range calme, un sommet, une bougie rouge, puis une impulsion qui casse le
    sommet en clôture. La bougie rouge (index 20) est l'Order Block."""
    return (_calm(100, 16)
            + [(100, 112, 99.6, 111)]      # 16 : pivot haut, confirmé en 18
            + _calm(100, 3)                # 17-19
            + [(100, 100.4, 95, 96)]       # 20 : bougie ROUGE → l'OB
            + [(96, 130, 96, 128)]         # 21 : impulsion, clôture > 112 → BOS
            + _calm(128, 4))


def _bear_rows():
    return (_calm(100, 16)
            + [(100, 100.4, 88, 89)]       # 16 : pivot bas
            + _calm(100, 3)
            + [(100, 105, 99.6, 104)]      # 20 : bougie VERTE → l'OB
            + [(104, 104, 70, 72)]         # 21 : impulsion → BOS baissier
            + _calm(72, 4))


def _range_h1():
    """Range H1 net : creux à 90, sommet à 110."""
    return _frame(_calm(100, 3) + [(100, 100.4, 90, 91)] + _calm(100, 4)
                  + [(100, 110, 99.6, 109)] + _calm(100, 4), freq="60min")


# --------------------------------------------------------------------------- #
# Order Blocks
# --------------------------------------------------------------------------- #
def test_ob_is_the_last_opposite_candle_before_the_break():
    obs = zones.find_order_blocks(_frame(_bull_rows()), "M15")
    assert len(obs) == 1
    ob = obs[0]
    assert ob.side == "bullish"
    assert ob.index == 20                      # la bougie rouge, pas l'impulsion
    assert ob.low == pytest.approx(95)         # zone = open → low (§4)
    assert ob.high == pytest.approx(100)


def test_ob_is_only_known_from_the_breaking_candle():
    """Anti look-ahead : la bougie rouge ne devient un OB qu'à la clôture qui
    casse la structure. Avant, ce n'est qu'une bougie rouge."""
    rows = _bull_rows()
    ob = zones.find_order_blocks(_frame(rows), "M15")[0]
    assert ob.index == 20 and ob.created_index == 21
    assert zones.find_order_blocks(_frame(rows[:21]), "M15") == []


def test_a_soft_break_produces_no_ob():
    """Déplacement < seuil : une cassure molle ne laisse pas d'ordre
    institutionnel derrière elle."""
    df = _frame(_bull_rows())
    assert len(zones.find_order_blocks(df, "M15", min_displacement_atr=1.5)) == 1
    assert zones.find_order_blocks(df, "M15", min_displacement_atr=99.0) == []


def test_displacement_is_measured_and_reported():
    ob = zones.find_order_blocks(_frame(_bull_rows()), "M15")[0]
    assert ob.displacement_atr >= config.OB_MIN_DISPLACEMENT_ATR


def test_bearish_ob_is_the_exact_mirror():
    obs = zones.find_order_blocks(_frame(_bear_rows()), "M15")
    assert len(obs) == 1
    ob = obs[0]
    assert ob.side == "bearish" and ob.index == 20
    assert ob.low == pytest.approx(100)        # open
    assert ob.high == pytest.approx(105)       # high


def test_no_structure_event_means_no_order_block():
    """Un OB sans cassure de structure n'est qu'une bougie rouge — c'est
    exactement ce qui sépare ce détecteur de celui du graphique principal."""
    assert zones.find_order_blocks(_frame(_calm(100, 40)), "M15") == []


# --------------------------------------------------------------------------- #
# Fair Value Gaps
# --------------------------------------------------------------------------- #
def test_fvg_is_the_gap_between_bar1_high_and_bar3_low():
    rows = _calm(100, 20) + [(100, 101, 99, 100),      # high 101
                             (101, 120, 101, 119),     # impulsion
                             (119, 121, 110, 120)]     # low 110 > 101
    haussiers = [z for z in zones.find_fvgs(_frame(rows), "M15")
                 if z.side == "bullish"]
    assert haussiers
    assert haussiers[-1].low == pytest.approx(101)
    assert haussiers[-1].high == pytest.approx(110)


def test_a_gap_smaller_than_the_minimum_is_ignored():
    rows = _calm(100, 20) + [(100, 101, 99, 100), (101, 103, 100, 102),
                             (102, 104, 101.05, 103)]   # gap de 0.05
    assert zones.find_fvgs(_frame(rows), "M15", min_size_atr=0.3) == []


def test_bearish_fvg_is_detected():
    rows = _calm(100, 20) + [(100, 101, 99, 100),      # low 99
                             (99, 99, 80, 81),
                             (81, 90, 79, 82)]          # high 90 < 99
    baissiers = [z for z in zones.find_fvgs(_frame(rows), "M15")
                 if z.side == "bearish"]
    assert baissiers
    assert baissiers[-1].low == pytest.approx(90)
    assert baissiers[-1].high == pytest.approx(99)


# --------------------------------------------------------------------------- #
# Mitigation / fraîcheur
# --------------------------------------------------------------------------- #
def test_an_ob_is_mitigated_as_soon_as_price_touches_it():
    intacte = _frame(_bull_rows())
    obs = zones.find_order_blocks(intacte, "M15")
    zones.update_states(intacte, obs)
    assert obs[0].state == "active"             # le prix n'est jamais redescendu

    retour = _frame(_bull_rows() + [(128, 128, 97, 98)])   # revient dans 95→100
    obs2 = zones.find_order_blocks(retour, "M15")
    zones.update_states(retour, obs2)
    assert obs2[0].state == "mitigated"
    assert obs2[0].mitigated_index == len(retour) - 1


def test_a_zone_cannot_be_mitigated_by_its_own_candle_or_the_past():
    """La bougie de cassure traverse forcément la zone : la compter rendrait
    toute zone mitigée à la seconde où elle apparaît."""
    df = _frame(_bull_rows())
    obs = zones.find_order_blocks(df, "M15")
    zones.update_states(df, obs, upto=obs[0].created_index)
    assert obs[0].state == "active"


def test_a_zone_older_than_the_max_age_is_expired():
    df = _frame(_bull_rows())
    obs = zones.find_order_blocks(df, "M15")
    zones.update_states(df, obs, max_age_days=4)
    assert obs[0].state == "active"
    zones.update_states(df, obs, max_age_days=0)
    assert obs[0].state == "expired"


def test_a_fvg_half_filled_leaves_the_active_set():
    rows = _calm(100, 20) + [(100, 101, 99, 100), (101, 120, 101, 119),
                             (119, 121, 110, 120),   # FVG 101 → 110
                             (120, 121, 104, 105)]   # redescend à 104 : ~67 %
    df = _frame(rows)
    fvgs = [z for z in zones.find_fvgs(df, "M15") if z.side == "bullish"]
    zones.update_states(df, fvgs)
    assert fvgs[-1].filled_ratio >= config.FVG_FILL_RATIO
    assert fvgs[-1].state == "mitigated"


def test_states_are_recomputed_not_accumulated():
    """update_states() tourne à chaque cycle : elle doit repartir de zéro, sinon
    une zone marquée mitigée le resterait sur une fenêtre où elle ne l'est pas."""
    df = _frame(_bull_rows())
    obs = zones.find_order_blocks(df, "M15")
    zones.update_states(df, obs, max_age_days=0)
    assert obs[0].state == "expired"
    zones.update_states(df, obs, max_age_days=4)
    assert obs[0].state == "active"


# --------------------------------------------------------------------------- #
# Premium / discount
# --------------------------------------------------------------------------- #
def test_h1_range_is_the_last_swing_high_and_low():
    rng = zones.h1_range(_range_h1())
    assert rng["low"] == pytest.approx(90)
    assert rng["high"] == pytest.approx(110)
    assert rng["mid"] == pytest.approx(100)


def test_no_range_returns_none_rather_than_a_default():
    """Sans range, il n'y a ni premium ni discount. Renvoyer un range par défaut
    serait le repli muet qui a déjà coûté cher dans ce dépôt."""
    assert zones.h1_range(_frame(_calm(100, 30), freq="60min")) is None


def test_buys_only_below_the_midpoint_sells_only_above():
    rng = zones.h1_range(_range_h1())
    assert zones.in_valid_half("bullish", 95, rng) is True     # discount
    assert zones.in_valid_half("bullish", 105, rng) is False   # premium
    assert zones.in_valid_half("bearish", 105, rng) is True
    assert zones.in_valid_half("bearish", 95, rng) is False


def test_discount_pct_is_reported_and_the_wrong_half_is_dropped():
    kept = zones.active_zones(_frame(_bull_rows()), "M15", h1=_range_h1())
    for z in kept:
        assert z.discount_pct is not None
        assert (z.discount_pct < 50) if z.side == "bullish" else (z.discount_pct > 50)


def test_the_ob_at_95_100_is_in_discount_and_survives():
    """La zone 95→100 est sous le milieu (100) du range 90→110 : un achat y est
    valide. C'est le filtre qui empêche d'acheter un sommet."""
    kept = zones.active_zones(_frame(_bull_rows()), "M15", h1=_range_h1())
    obs = [z for z in kept if z.kind == "OB"]
    assert obs and obs[0].low == pytest.approx(95)
    assert obs[0].discount_pct == pytest.approx((97.5 - 90) / 20 * 100)


def test_active_zones_excludes_mitigated_and_expired():
    df = _frame(_bull_rows())
    toutes = zones.find_order_blocks(df, "M15") + zones.find_fvgs(df, "M15")
    actives = zones.active_zones(df, "M15")
    assert len(actives) <= len(toutes)
    assert all(z.state == "active" for z in actives)
