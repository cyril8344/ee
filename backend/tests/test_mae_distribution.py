"""Distribution du MAE — ce qu'un SL plus serré aurait touché.

Motivation : le run H1 a montré un MFE médian qui s'écrase (1.4–1.7R en M5 contre
0.75–0.92R en H1). Le SL s'élargit avec l'ATR, mais le mouvement qui suit ne suit pas
proportionnellement — la règle `SL = 1.8 × ATR` n'a jamais été testée.

Limite assumée et testée ici : ce diagnostic ne produit AUCUNE espérance. MAE et MFE
ne sont pas ordonnés, donc on ne sait pas si un trade tué par un SL resserré avait
déjà encaissé TP1 avant. C'est exactement l'erreur commise sur la distribution du MFE,
dont la formule comptait −1R pour tout trade n'atteignant pas TP2 et inversait la
conclusion.
"""
import os

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import pretrain
import strategy


def _t(mae_r, won=False, mfe_r=1.0):
    return {"mae_r": mae_r, "won": won, "mfe_r": mfe_r}


def test_returns_none_without_usable_trades():
    assert pretrain._diag_mae_distribution([]) is None
    assert pretrain._diag_mae_distribution([{"won": True}]) is None


def test_touch_rate_is_the_share_whose_mae_reaches_the_level():
    """MAE >= f équivaut exactement à « le prix a atteint f×R contre nous » : la part
    des trades vérifiant ça EST le taux de touche d'un SL placé à f."""
    trades = [_t(m) for m in (0.2, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.0, 1.0)]
    d = pretrain._diag_mae_distribution(trades)

    lv = {n["sl_mult"]: n for n in d["niveaux"]}
    assert lv[1.0]["p_touche"] == pytest.approx(30.0)   # trois à 1.0
    assert lv[0.6]["p_touche"] == pytest.approx(70.0)   # 0.65 et au-delà
    assert lv[0.5]["p_touche"] == pytest.approx(80.0)


def test_reports_how_many_current_winners_a_tighter_stop_would_kill():
    """Le chiffre qui porte l'hypothèse : si les gagnants ne descendent presque jamais
    bas, la distance actuelle achète une protection inutile et gonfle le R."""
    trades = ([_t(0.2, won=True) for _ in range(8)]
              + [_t(0.9, won=True) for _ in range(2)]
              + [_t(1.0, won=False) for _ in range(10)])
    d = pretrain._diag_mae_distribution(trades)

    assert d["n_gagnants"] == 10
    lv = {n["sl_mult"]: n for n in d["niveaux"]}
    # Un SL à 0.5R n'aurait tué que les deux gagnants descendus à 0.9.
    assert lv[0.5]["p_touche_gagnants"] == pytest.approx(20.0)
    # À 1.0R (actuel), aucun gagnant n'est touché — ils sont gagnants, justement.
    assert lv[1.0]["p_touche_gagnants"] == pytest.approx(0.0)
    assert d["mae_gagnants_median_r"] == pytest.approx(0.2)


def test_closer_tp1_is_reached_more_often(monkeypatch):
    """Resserrer le SL réduit le R, donc rapproche aussi les TP en prix : un TP1 à
    0.7R d'un SL deux fois plus serré se situe à 0.35R de l'ancien."""
    monkeypatch.setattr(strategy, "TP1_R", 0.7)
    trades = [_t(0.5, mfe_r=m) for m in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0)]
    d = pretrain._diag_mae_distribution(trades)
    lv = {n["sl_mult"]: n for n in d["niveaux"]}

    # SL actuel : TP1 à 0.7R -> 0.8/1.0/1.2/1.4/1.6/2.0 soit 6 trades sur 10
    assert lv[1.0]["p_tp1_atteint"] == pytest.approx(60.0)
    # SL à 0.5 : TP1 à 0.35R -> bien plus de trades l'atteignent
    assert lv[0.5]["p_tp1_atteint"] > lv[1.0]["p_tp1_atteint"]


def test_reports_no_expectancy():
    """Garde-fou explicite : aucune clé d'espérance ne doit apparaître. MAE et MFE
    n'étant pas ordonnés, toute formule bâtie dessus serait fausse — et une valeur
    affichée serait prise pour une prédiction."""
    d = pretrain._diag_mae_distribution([_t(0.5, won=True), _t(1.0)])
    for niveau in d["niveaux"]:
        assert not any("esperance" in k or "gain" in k for k in niveau)
    assert not any("esperance" in k or "gain" in k for k in d)


def test_handles_a_sample_without_any_winner():
    d = pretrain._diag_mae_distribution([_t(1.0), _t(0.9)])
    assert d["n_gagnants"] == 0
    assert d["mae_gagnants_median_r"] is None
    assert all(n["p_touche_gagnants"] is None for n in d["niveaux"])
