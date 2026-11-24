"""Balayage/récupération de la bougie précédente, et sa cascade.

La règle vient d'une méthode publiée en vidéo, illustrée par **un** exemple
choisi après coup. Ces tests ne vérifient donc pas qu'elle gagne de l'argent —
rien ici ne mesure ça — mais qu'elle est implémentée exactement comme énoncée,
et que les chiffres qui permettront de la juger sont produits honnêtement.

Le test qui compte le plus est le dernier : il interdit au module de présenter
trois conditions corrélées comme trois confirmations indépendantes.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import candle_sweep as cs


def _frame(rows, freq="1D"):
    idx = pd.date_range("2026-01-05", periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


# --------------------------------------------------------------------------- #
# La règle, telle qu'énoncée
# --------------------------------------------------------------------------- #
def test_taking_the_previous_low_and_closing_back_above_is_bullish():
    """Le cas montré dans la vidéo : passé sous le PDL, clôture au-dessus."""
    df = _frame([(100, 105, 95, 103),        # la précédente : low = 95
                 (103, 104, 92, 99)])        # balaie 92 < 95, clôture 99 > 95
    s = cs.sweep_at(df, 1)
    assert s is not None
    assert s.direction == "bullish"
    assert s.niveau == pytest.approx(95)
    assert s.extreme == pytest.approx(92)
    assert s.depassement == pytest.approx(3)
    assert s.recuperation == pytest.approx(4)


def test_the_bearish_case_is_the_exact_mirror():
    df = _frame([(100, 105, 95, 103),
                 (103, 110, 102, 104)])      # dépasse 105, clôture 104 < 105
    s = cs.sweep_at(df, 1)
    assert s is not None and s.direction == "bearish"
    assert s.niveau == pytest.approx(105)
    assert s.depassement == pytest.approx(5)


def test_closing_below_the_level_is_not_a_sweep_but_a_break():
    """C'est toute la différence : sans récupération, le niveau a cédé. Une
    cassure et un piège se ressemblent sur la mèche et s'opposent sur la
    clôture."""
    df = _frame([(100, 105, 95, 103),
                 (103, 104, 92, 93)])        # clôture 93 < 95
    assert cs.sweep_at(df, 1) is None


def test_staying_inside_the_previous_candle_is_not_a_sweep():
    df = _frame([(100, 105, 95, 103),
                 (103, 104, 96, 100)])       # 96 > 95 : rien n'a été balayé
    assert cs.sweep_at(df, 1) is None


def test_an_equal_low_does_not_count():
    """`<` strict, pas `<=` : toucher un niveau au centième près n'est pas le
    balayer, et sur des séries plates le `<=` déclencherait en continu."""
    df = _frame([(100, 105, 95, 103),
                 (103, 104, 95, 100)])
    assert cs.sweep_at(df, 1) is None


def test_the_first_candle_has_no_previous_one():
    df = _frame([(100, 105, 95, 103), (103, 104, 92, 99)])
    assert cs.sweep_at(df, 0) is None


def test_the_direction_can_be_imposed():
    """En cascade on cherche un sens déjà décidé par l'étage du dessus : une
    bougie qui balaie dans l'autre sens ne doit pas être acceptée."""
    df = _frame([(100, 105, 95, 103),
                 (103, 104, 92, 99)])
    assert cs.sweep_at(df, 1, direction="bullish") is not None
    assert cs.sweep_at(df, 1, direction="bearish") is None


# --------------------------------------------------------------------------- #
# Le chiffre qui relativise l'exemple de la vidéo
# --------------------------------------------------------------------------- #
def test_the_frequency_of_the_condition_is_reported():
    """Un exemple bien choisi ne dit rien de la rareté d'une condition. Le taux,
    si."""
    df = _frame([(100, 105, 95, 103),
                 (103, 104, 92, 99),         # balayage haussier
                 (99, 101, 97, 100),         # rien
                 (100, 102, 90, 98)])        # balayage haussier
    assert cs.rate(df) == pytest.approx(2 / 3)
    assert 0.0 <= cs.rate(df, "bearish") <= 1.0


def test_a_series_too_short_has_a_rate_of_zero_rather_than_an_error():
    assert cs.rate(_frame([(100, 105, 95, 103)])) == 0.0


# --------------------------------------------------------------------------- #
# Cascade : direction → setup → entrée
# --------------------------------------------------------------------------- #
def _hausse():
    return _frame([(100, 105, 95, 103), (103, 104, 92, 99)])


def _rien():
    return _frame([(100, 105, 95, 103), (103, 104, 96, 100)])


def _baisse():
    return _frame([(100, 105, 95, 103), (103, 110, 102, 104)])


def test_the_three_stages_aligned_give_a_direction():
    c = cs.cascade({"direction": _hausse(), "setup": _hausse(),
                    "entree": _hausse()})
    assert c.aligne is True
    assert c.direction == "bullish"
    assert c.manquants == []


def test_the_first_stage_decides_the_direction_not_the_last():
    """L'ordre de la méthode : le journalier fixe le sens. Sans ça, une mèche de
    M15 déciderait du biais de la journée."""
    c = cs.cascade({"direction": _baisse(), "setup": _hausse(),
                    "entree": _hausse()})
    assert c.direction == "bearish"
    assert c.aligne is False
    assert c.manquants == ["setup", "entree"]


def test_a_missing_stage_is_named_not_silently_skipped():
    """« Non aligné » et « non aligné parce que le setup manque » sont deux
    informations différentes ; un blanc se lirait comme une panne."""
    c = cs.cascade({"direction": _hausse(), "setup": _rien(),
                    "entree": _hausse()})
    assert c.aligne is False
    assert c.manquants == ["setup"]
    assert c.etages["direction"] is not None
    assert c.etages["setup"] is None


def test_no_direction_at_all_blocks_everything():
    c = cs.cascade({"direction": _rien(), "setup": _hausse(),
                    "entree": _hausse()})
    assert c.direction is None
    assert c.aligne is False
    assert "direction" in c.manquants


def test_an_empty_frame_is_a_missing_stage_not_a_crash():
    c = cs.cascade({"direction": _hausse(), "setup": _frame([]),
                    "entree": _hausse()})
    assert "setup" in c.manquants


def test_the_cascade_is_serialisable_with_its_measurements():
    d = cs.cascade({"direction": _hausse(), "setup": _hausse(),
                    "entree": _hausse()}).as_dict()
    assert d["direction"] == "bullish" and d["aligne"] is True
    assert d["etages"]["setup"]["niveau"] == pytest.approx(95)


# --------------------------------------------------------------------------- #
# Le garde-fou principal
# --------------------------------------------------------------------------- #
def test_the_module_never_claims_the_stages_are_independent_evidence():
    """Trois conditions empilées se présentent comme trois confirmations. Mais
    le balayage H1 qui « confirme » le balayage Daily est souvent le MÊME
    mouvement vu de plus près. Le module doit donner de quoi le mesurer, et ne
    jamais trancher à la place du lecteur."""
    r = cs.independence_report({"direction": _hausse(), "setup": _hausse(),
                                "entree": _hausse()})
    assert set(r["taux_par_etage"]) == {"direction", "setup", "entree"}
    assert 0.0 <= r["produit_si_independants"] <= 1.0
    assert r["note"]

    texte = repr(r).lower()
    for mot in ("probabilit", "espérance", "esperance", "fiable", "valide",
                "confirmé", "confirme"):
        assert mot not in texte, f"« {mot} » ne doit pas apparaître"


def test_no_verdict_is_produced_anywhere():
    d = cs.cascade({"direction": _hausse(), "setup": _hausse(),
                    "entree": _hausse()}).as_dict()
    texte = repr(d).lower()
    for mot in ("valide", "score", "probabilit", "espérance", "esperance",
                "confiance"):
        assert mot not in texte
