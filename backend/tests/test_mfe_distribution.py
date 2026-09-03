"""Distribution du MFE → niveau de TP2 optimal, calculé sans relancer de simulation.

`mfe_r` est le maximum atteint en faveur du trade avant sa sortie, donc `MFE >= x`
équivaut à « le prix a touché x ». La proportion de trades vérifiant ça EST le taux
de touche d'un TP placé à x, à SL identique — d'où une évaluation analytique de TP2.
"""
import os

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import pretrain
import strategy


def _t(mfe_r, tp1_reached=True):
    return {"mfe_r": mfe_r, "tp1_reached": tp1_reached}


def test_returns_none_without_usable_trades():
    assert pretrain._diag_mfe_distribution([]) is None
    # Aucun trade n'atteint TP1 : la seconde moitié n'existe jamais.
    assert pretrain._diag_mfe_distribution([_t(0.4, tp1_reached=False)]) is None


def test_conditions_on_trades_that_reached_tp1():
    trades = [_t(2.0) for _ in range(4)] + [_t(3.0, tp1_reached=False) for _ in range(96)]
    d = pretrain._diag_mfe_distribution(trades)
    assert d["n_tp1_reached"] == 4


def test_hit_rate_is_the_share_of_trades_whose_mfe_reaches_the_level(monkeypatch):
    monkeypatch.setattr(strategy, "TP1_R", 0.7)
    monkeypatch.setattr(strategy, "TP2_R", 1.8)
    # 3 trades sur 10 dépassent 1.8R
    trades = [_t(m) for m in (0.8, 0.9, 1.0, 1.2, 1.5, 1.7, 1.9, 2.1, 2.4, 1.8)]
    d = pretrain._diag_mfe_distribution(trades)

    lv18 = next(l for l in d["niveaux"] if l["tp2_r"] == 1.8)
    assert lv18["p_atteint"] == pytest.approx(40.0)   # 1.8, 1.9, 2.1, 2.4 → >= 1.8
    # E = p×x − (1−p) = 0.4×1.8 − 0.6 = 0.12
    assert lv18["esperance_r"] == pytest.approx(0.12, abs=0.001)


def test_matches_the_tp1_to_tp2_threshold_at_the_current_tp2(monkeypatch):
    """E(TP2 courant) doit redonner exactement le calcul de _diag_tp1_to_tp2(),
    au facteur (1 - fraction soldée) près : les deux décrivent la même chose."""
    monkeypatch.setattr(strategy, "TP1_R", 0.7)
    monkeypatch.setattr(strategy, "TP2_R", 1.8)
    monkeypatch.setattr(strategy, "TP1_CLOSE_RATIO", 0.5)

    trades = [dict(_t(2.0), exit_reason="tp2") for _ in range(4)] + \
             [dict(_t(1.0), exit_reason="sl_after_tp1") for _ in range(6)]

    mfe = pretrain._diag_mfe_distribution(trades)
    seuil = pretrain._diag_tp1_to_tp2(trades)

    lv18 = next(l for l in mfe["niveaux"] if l["tp2_r"] == 1.8)
    assert lv18["p_atteint"] == pytest.approx(seuil["p_tp2_given_tp1"])
    assert lv18["esperance_r"] * 0.5 == pytest.approx(seuil["gain_r_sans_be"], abs=0.001)


def test_levels_at_or_below_tp1_are_excluded(monkeypatch):
    monkeypatch.setattr(strategy, "TP1_R", 1.25)
    d = pretrain._diag_mfe_distribution([_t(2.0)])
    assert all(l["tp2_r"] > 1.25 for l in d["niveaux"])


def test_levels_above_the_run_tp2_are_flagged_unreliable(monkeypatch):
    """Un trade clôturé à TP2 n'enregistre jamais de MFE au-delà : les niveaux
    supérieurs sont des planchers, pas des mesures."""
    monkeypatch.setattr(strategy, "TP2_R", 1.8)
    d = pretrain._diag_mfe_distribution([_t(1.9)])

    assert all(l["fiable"] for l in d["niveaux"] if l["tp2_r"] <= 1.8)
    assert not any(l["fiable"] for l in d["niveaux"] if l["tp2_r"] > 1.8)
    # Le meilleur retenu ne doit jamais être choisi parmi les niveaux censurés.
    assert d["meilleur_tp2_r"] <= 1.8


def test_signals_when_the_optimum_may_lie_beyond_the_censoring(monkeypatch):
    """Si le meilleur TP2 fiable est le plus haut testable, l'optimum est peut-être
    au-delà — il faut alors un run avec TP2_R élevé pour le voir."""
    monkeypatch.setattr(strategy, "TP1_R", 0.7)
    monkeypatch.setattr(strategy, "TP2_R", 1.8)

    # Tous les trades vont loin : l'espérance croît jusqu'au dernier niveau fiable.
    d = pretrain._diag_mfe_distribution([_t(5.0) for _ in range(10)])
    assert d["meilleur_tp2_r"] == 1.8
    assert d["optimum_possiblement_au_dela"] is True

    # Ici l'optimum est intérieur : rien à chercher plus loin.
    d = pretrain._diag_mfe_distribution([_t(1.05) for _ in range(10)])
    assert d["meilleur_tp2_r"] == 1.0
    assert d["optimum_possiblement_au_dela"] is False
