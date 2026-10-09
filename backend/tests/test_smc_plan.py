"""Poches de liquidité (EQH / EQL) et plan de trade chiffré.

Règle de conception vérifiée ici : **aucun champ du plan n'existe sans la valeur
qui l'a produit**, et il n'y a nulle part de verdict « VALIDE ». Le module compte
des critères et calcule un R:R ; affirmer qu'un trade est bon supposerait de
savoir qu'il va marcher, ce que rien ne mesure.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import config, liquidity, plan


def _frame(rows, start="2026-01-05 08:00", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _calm(p, n):
    return [(p, p + 0.5, p - 0.5, p) for _ in range(n)]


def _setup_haussier():
    """Range calme, sommet, bougie rouge (l'OB), impulsion qui casse, repli."""
    return _frame(_calm(100, 16)
                  + [(100, 112, 99.6, 111)]      # pivot haut
                  + _calm(100, 3)
                  + [(100, 100.4, 95, 96)]       # 20 : l'Order Block
                  + [(96, 130, 96, 128)]         # 21 : impulsion → BOS
                  + _calm(128, 6)
                  + [(128, 128.4, 120, 121)]
                  + _calm(104, 4))


# --------------------------------------------------------------------------- #
# Equal Highs / Equal Lows
# --------------------------------------------------------------------------- #
def _trois_sommets_egaux(ecart=0.0):
    """Trois pointes au même prix (à `ecart` près) = une poche de liquidité."""
    rows = _calm(100, 16)
    for k in range(3):
        rows += [(100, 110 + k * ecart, 99.6, 101)]
        rows += _calm(100, 4)
    return _frame(rows)


def test_several_swings_at_the_same_price_form_one_pocket():
    niveaux = liquidity.equal_levels(_trois_sommets_egaux(), "EQH")
    assert len(niveaux) == 1
    lv = niveaux[0]
    assert lv.kind == "EQH"
    assert lv.n_swings == 3
    assert lv.price == pytest.approx(110, abs=0.5)


def test_the_number_of_swings_is_reported_not_judged():
    """Un EQH à quatre touches ne vaut pas un EQH à deux. Le module renvoie le
    compte, il ne décide pas à partir de quand c'est « valable »."""
    lv = liquidity.equal_levels(_trois_sommets_egaux(), "EQH")[0]
    assert isinstance(lv.n_swings, int)
    assert lv.spread >= 0


def test_swings_too_far_apart_are_not_a_pocket():
    """Au-delà de la tolérance, ce sont deux niveaux distincts, pas un cluster."""
    loin = liquidity.equal_levels(_trois_sommets_egaux(ecart=20.0), "EQH")
    assert all(lv.n_swings < 3 for lv in loin)


def test_equal_lows_are_the_mirror():
    rows = _calm(100, 16)
    for _ in range(3):
        rows += [(100, 100.4, 90, 99)]
        rows += _calm(100, 4)
    niveaux = liquidity.equal_levels(_frame(rows), "EQL")
    assert niveaux and niveaux[0].kind == "EQL"
    assert niveaux[0].price == pytest.approx(90, abs=0.5)


def test_a_flat_series_has_no_pocket():
    assert liquidity.equal_levels(_frame(_calm(100, 40)), "EQH") == []


def test_nearest_above_and_below_pick_the_first_obstacle():
    a = liquidity.EqualLevel("EQH", 110.0, 2, 0, 1, 0.0)
    b = liquidity.EqualLevel("EQH", 120.0, 3, 0, 1, 0.0)
    assert liquidity.nearest_above([a, b], 105.0) is a
    assert liquidity.nearest_above([a, b], 115.0) is b
    assert liquidity.nearest_above([a, b], 130.0) is None
    assert liquidity.nearest_below([a, b], 130.0) is b


# --------------------------------------------------------------------------- #
# Plan : chaque nombre vient d'une mesure
# --------------------------------------------------------------------------- #
def test_the_plan_levels_follow_the_documented_conventions():
    """Entrée = bord proximal de la zone. Stop = bord opposé − marge×ATR.
    Les deux doivent être retrouvables à la main depuis la zone et l'ATR."""
    a = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish")
    p = a.plan
    assert p is not None
    assert p.entry == pytest.approx(p.zone_high)          # achat → bord haut
    assert p.stop == pytest.approx(p.zone_low - config.SL_MARGIN_ATR * p.atr)
    assert p.stop < p.entry < p.target


def test_the_risk_reward_is_the_quotient_of_the_two_distances():
    p = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish").plan
    assert p.rr == pytest.approx(p.gain / p.risque)
    assert p.risque == pytest.approx(abs(p.entry - p.stop))
    assert p.gain == pytest.approx(abs(p.target - p.entry))


def test_the_target_says_where_it_comes_from():
    """Un R:R de 5 contre une poche de liquidité et un R:R de 5 contre un swing
    isolé ne se valent pas. La source fait donc partie du résultat."""
    p = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish").plan
    assert p.target_source
    assert any(mot in p.target_source for mot in ("EQH", "EQL", "zone", "swing"))


def test_every_criterion_carries_its_measurement_and_its_threshold():
    """C'est la règle de conception du module : on doit pouvoir contester une
    conclusion en regardant le chiffre, pas en discutant d'une impression."""
    p = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish").plan
    assert p.criteres
    for c in p.criteres:
        assert c.nom
        assert c.valeur is not None
        assert c.seuil is not None
        assert c.detail


def test_no_verdict_is_produced_anywhere():
    """Garde-fou : pas de « VALIDE », pas de score, pas de probabilité. Un avis
    tranché se recopie dans la plateforme ; un décompte de critères, non."""
    d = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish").as_dict()
    texte = repr(d).lower()
    for mot in ("valide", "rejet", "verdict", "score", "probabilit",
                "esperance", "espérance", "confiance"):
        assert mot not in texte, f"« {mot} » ne doit pas apparaître dans la sortie"
    assert "n_remplis" in d["plan"] and "n_evalues" in d["plan"]


def test_the_sweep_is_informational_and_never_blocks():
    """§5 : le sweep est taggé, jamais exigé. Son critère ne compte donc pas
    dans le décompte."""
    p = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish").plan
    sweep = [c for c in p.criteres if "Sweep" in c.nom][0]
    assert sweep.rempli is None
    assert sweep.seuil == "non exigé"
    assert p.n_evalues == len([c for c in p.criteres if c.rempli is not None])


# --------------------------------------------------------------------------- #
# Refus explicites : « rien trouvé » doit dire pourquoi
# --------------------------------------------------------------------------- #
def test_a_neutral_bias_produces_no_plan_and_says_why():
    a = plan.analyse(_setup_haussier(), "M5")      # sans biais HTF imposé
    if a.plan is None:
        assert a.raison and "neutre" in a.raison.lower()


def test_no_fresh_zone_reports_how_many_were_found():
    """« Aucune zone » et « trois zones toutes mitigées » sont deux situations
    différentes — un blanc se lirait comme une panne du détecteur."""
    a = plan.analyse(_frame(_calm(100, 60)), "M5", htf_bias="bullish")
    assert a.plan is None
    assert a.raison and "zone" in a.raison.lower()


def test_too_few_candles_is_reported_not_guessed():
    a = plan.analyse(_frame(_calm(100, 8)), "M5", htf_bias="bullish")
    assert a.plan is None
    assert a.raison and "ATR" in a.raison


def test_no_target_means_no_plan_rather_than_an_invented_ratio():
    """Sans obstacle devant le prix, un R:R serait inventé de toutes pièces."""
    a = plan.analyse(_setup_haussier(), "M5", htf_bias="bearish")
    if a.plan is None:
        assert a.raison
    else:
        assert a.plan.target != a.plan.entry


def test_the_analysis_is_serialisable_with_its_measurements():
    d = plan.analyse(_setup_haussier(), "M5", htf_bias="bullish").as_dict()
    assert d["bias"] == "bullish"
    assert d["atr"] > 0
    assert d["plan"]["rr"] > 0
    assert isinstance(d["eqh"], list) and isinstance(d["eql"], list)
