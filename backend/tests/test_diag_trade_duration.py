"""Tests du diagnostic de durée de vie des trades réels (pretrain.diag_real_trades_duration).

Répond à « combien de temps un trade reste-t-il ouvert en live ? » — la valeur
théorique du plafond ne suffit pas, la gestion de position étant suspendue pendant
un repli sur données synthétiques (main.py), ce qui laisse des trades dépasser
MAX_TRADE_MINUTES de la durée de la coupure.
"""
import pytest

import pretrain
import strategy


def _trade(duration_min, exit_reason="tp1"):
    return {"duration_min": duration_min, "exit_reason": exit_reason}


def test_returns_empty_result_without_trades():
    r = pretrain.diag_real_trades_duration(_trades=[])
    assert r["n"] == 0
    assert r["by_exit_reason"] == []
    assert "note" in r


def test_ignores_trades_without_recorded_duration():
    """duration_min est NULL sur les lignes ré-insérées ou anciennes : les compter
    comme 0 minute écraserait la médiane vers le bas."""
    trades = [_trade(30.0), _trade(None), {"exit_reason": "sl"}]
    r = pretrain.diag_real_trades_duration(_trades=trades)
    assert r["n"] == 1
    assert r["median_min"] == 30.0


def test_aggregates_global_stats():
    trades = [_trade(d) for d in (10.0, 20.0, 30.0, 40.0, 50.0)]
    r = pretrain.diag_real_trades_duration(_trades=trades)
    assert r["n"] == 5
    assert r["avg_min"] == pytest.approx(30.0)
    assert r["median_min"] == pytest.approx(30.0)
    assert r["max_min"] == pytest.approx(50.0)
    # p90 interpolé sur 5 points : 40 + (50-40)*0.6 = 46.0 — surtout PAS le max,
    # sinon p90 et max seraient toujours identiques sur un petit échantillon.
    assert r["p90_min"] == pytest.approx(46.0)


def test_splits_by_exit_reason_ordered_by_frequency():
    trades = ([_trade(20.0, "sl") for _ in range(6)]
              + [_trade(75.0, "timeout") for _ in range(2)]
              + [_trade(10.0, "early_exit") for _ in range(4)])
    r = pretrain.diag_real_trades_duration(_trades=trades)

    reasons = [row["exit_reason"] for row in r["by_exit_reason"]]
    assert reasons == ["sl", "early_exit", "timeout"]  # 6, 4, 2

    sl = r["by_exit_reason"][0]
    assert sl["n"] == 6
    assert sl["pct"] == pytest.approx(50.0)
    assert sl["avg_min"] == pytest.approx(20.0)
    assert sl["max_min"] == pytest.approx(20.0)


def test_buckets_missing_exit_reason_as_unknown():
    r = pretrain.diag_real_trades_duration(_trades=[_trade(15.0, None)])
    assert r["by_exit_reason"][0]["exit_reason"] == "inconnu"


def test_counts_trades_that_outlived_the_cap(monkeypatch):
    """Le cas qui motive ce diagnostic : la boucle suspend la gestion de position sur
    données synthétiques, donc un trade peut rester ouvert bien au-delà du plafond."""
    monkeypatch.setattr(strategy, "MAX_TRADE_MINUTES", 75)
    trades = [_trade(30.0), _trade(74.0), _trade(120.0, "timeout"), _trade(200.0, "timeout")]
    r = pretrain.diag_real_trades_duration(_trades=trades)

    assert r["max_trade_minutes"] == 75
    assert r["over_cap_n"] == 2
    assert r["over_cap_pct"] == pytest.approx(50.0)
    assert r["over_cap_max_min"] == pytest.approx(200.0)


def test_does_not_flag_sub_minute_overshoot(monkeypatch):
    """La boucle tourne toutes les ~5 s : un dépassement de quelques secondes est
    normal et ne doit pas être signalé comme une anomalie."""
    monkeypatch.setattr(strategy, "MAX_TRADE_MINUTES", 75)
    r = pretrain.diag_real_trades_duration(_trades=[_trade(75.2), _trade(75.9)])
    assert r["over_cap_n"] == 0
    assert r["over_cap_max_min"] is None
