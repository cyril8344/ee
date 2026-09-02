"""Géométrie des sorties : constantes atteignables + p(TP2 | TP1).

Contexte. Les huit hypothèses testées jusqu'ici portaient toutes sur l'ENTRÉE, alors
que le walk-forward donne un WR > 50 % avec un PF < 1 sur 2 567 trades : les entrées
sélectionnent correctement, c'est la géométrie des sorties qui perd. Cette partie de
la stratégie était composée de littéraux codés en dur — donc la seule qu'aucun
walk-forward ne pouvait tester.
"""
import os

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import pretrain
import strategy


def test_defaults_preserve_the_previous_hardcoded_behaviour():
    """Le passage en constantes ne doit rien changer : ce sont exactement les valeurs
    qui étaient écrites en dur dans evaluate() et dans les moteurs."""
    assert strategy.TP1_R == 0.7
    assert strategy.TP2_R == 1.8
    assert strategy.TP1_CLOSE_RATIO == 0.5
    assert strategy.BE_AFTER_TP1 is True


def _levels(direction, entry, sl):
    """TP1/TP2 tels que evaluate() les construit, isolés du reste du pipeline."""
    risk = abs(entry - sl)
    if direction == "long":
        return entry + strategy.TP1_R * risk, entry + strategy.TP2_R * risk
    return entry - strategy.TP1_R * risk, entry - strategy.TP2_R * risk


def test_tp_levels_follow_the_constants(monkeypatch):
    monkeypatch.setattr(strategy, "TP1_R", 1.0)
    monkeypatch.setattr(strategy, "TP2_R", 3.0)
    tp1, tp2 = _levels("long", 2000.0, 1990.0)      # risk = 10
    assert tp1 == pytest.approx(2010.0)
    assert tp2 == pytest.approx(2030.0)

    tp1, tp2 = _levels("short", 2000.0, 2010.0)
    assert tp1 == pytest.approx(1990.0)
    assert tp2 == pytest.approx(1970.0)


def _trade(tp1_reached, exit_reason):
    return {"tp1_reached": tp1_reached, "exit_reason": exit_reason}


def test_p_tp2_given_tp1_conditions_on_trades_that_reached_tp1():
    trades = (
        [_trade(True, "tp2") for _ in range(3)]
        + [_trade(True, "sl_after_tp1") for _ in range(7)]
        + [_trade(False, "sl") for _ in range(90)]      # n'atteignent pas TP1 : hors calcul
    )
    d = pretrain._diag_tp1_to_tp2(trades)

    assert d["n_tp1_reached"] == 10
    assert d["n_tp2_reached"] == 3
    assert d["p_tp2_given_tp1"] == pytest.approx(30.0)   # 3/10, pas 3/100


def test_threshold_is_one_over_tp2r_plus_one(monkeypatch):
    """Seuil = 1/(TP2_R+1). Il ne dépend ni de TP1_R ni de la fraction soldée : le
    gain déjà encaissé à TP1 est acquis dans les deux scénarios et s'annule."""
    monkeypatch.setattr(strategy, "TP2_R", 1.8)
    d = pretrain._diag_tp1_to_tp2([_trade(True, "tp2")])
    assert d["seuil_pct"] == pytest.approx(35.7, abs=0.1)

    monkeypatch.setattr(strategy, "TP2_R", 3.0)
    d = pretrain._diag_tp1_to_tp2([_trade(True, "tp2")])
    assert d["seuil_pct"] == pytest.approx(25.0)


def test_verdict_flags_when_removing_be_is_favourable(monkeypatch):
    monkeypatch.setattr(strategy, "TP2_R", 1.8)      # seuil 35.7 %

    below = [_trade(True, "tp2")] + [_trade(True, "sl_after_tp1") for _ in range(3)]  # 25 %
    d = pretrain._diag_tp1_to_tp2(below)
    assert d["retirer_be_favorable"] is False
    assert d["gain_r_sans_be"] < 0

    above = [_trade(True, "tp2") for _ in range(3)] + [_trade(True, "sl_after_tp1") for _ in range(2)]  # 60 %
    d = pretrain._diag_tp1_to_tp2(above)
    assert d["retirer_be_favorable"] is True
    assert d["gain_r_sans_be"] > 0


def test_returns_none_when_no_trade_reaches_tp1():
    assert pretrain._diag_tp1_to_tp2([_trade(False, "sl") for _ in range(20)]) is None
    assert pretrain._diag_tp1_to_tp2([]) is None


def test_timeout_after_tp1_counts_as_having_reached_tp1():
    """exit_reason ne suffit pas : un "timeout" survient aussi bien avant qu'après TP1.
    C'est pourquoi tp1_reached est journalisé explicitement."""
    trades = [_trade(True, "timeout") for _ in range(4)] + [_trade(True, "tp2")]
    d = pretrain._diag_tp1_to_tp2(trades)
    assert d["n_tp1_reached"] == 5
    assert d["p_tp2_given_tp1"] == pytest.approx(20.0)
