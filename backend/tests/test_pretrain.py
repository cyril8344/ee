"""Tests for pretrain.py's shared-state isolation between standalone runs and
internal walk-forward/Optuna windows (write_to_db=False)."""
import pytest

import database as db
import pretrain
import strategy

db.init_db()


def test_write_to_db_false_does_not_pollute_shared_pretrain_state():
    """A walk-forward window (write_to_db=False) must never overwrite the result
    shown in the standalone "Pré-entraînement" panel — only a real, user-triggered
    pretrain run (write_to_db=True) should update that shared state."""
    r1 = pretrain.run_pretrain("2024-01-01", "2024-01-10", symbol="XAUUSD",
                                reset=True, write_to_db=True)
    before = pretrain.get_progress()["last_result"]
    assert before is not None
    assert before["period"] == r1["period"]
    assert pretrain.get_last_results()["A"]["period"] == r1["period"]

    r2 = pretrain.run_pretrain("2024-02-01", "2024-02-10", symbol="XAUUSD",
                                reset=True, write_to_db=False)
    assert r2["period"] != r1["period"]

    after = pretrain.get_progress()["last_result"]
    assert after["period"] == before["period"]
    assert pretrain.get_last_results()["A"]["period"] == r1["period"]


def _fake_window(pf, n_trades, adx, sl_pct, wr, pct_long=50.0, rejections=None):
    return {
        "n_trades": n_trades,
        "profit_factor": pf,
        "win_rate": wr / 100,  # stocké en ratio 0-1 dans le vrai code, comme pretrain.py
        "sl_direct_pct": sl_pct,
        "regime_signature": {
            "avg_adx_h1": adx, "avg_atr": 4.0, "avg_rsi_h1": 50.0,
            "pct_long": pct_long, "wr_long": wr, "wr_short": wr,
            "efficiency_ratio_h1": 0.05,
        },
        "rejection_counts": rejections or {"session": 700, "timing": 150, "adx": 20},
    }


def test_compare_window_regimes_ranks_biggest_differences_first():
    windows = [
        _fake_window(pf=1.5, n_trades=40, adx=35.0, sl_pct=15.0, wr=60.0),
        _fake_window(pf=1.3, n_trades=30, adx=34.0, sl_pct=18.0, wr=58.0),
        _fake_window(pf=0.6, n_trades=25, adx=25.0, sl_pct=45.0, wr=40.0),
        _fake_window(pf=0.7, n_trades=35, adx=26.0, sl_pct=42.0, wr=42.0),
    ]
    cmp = pretrain._compare_window_regimes(windows)
    assert cmp is not None
    assert cmp["n_fenetres_gagnantes"] == 2
    assert cmp["n_fenetres_perdantes"] == 2
    # SL direct % has by far the largest gap (≈15-18 vs 42-45) — must rank first.
    assert cmp["facteurs"][0]["factor"] == "SL direct %"
    assert cmp["facteurs"][0]["diff"] < 0  # winning windows have LOWER SL direct %

    wr_factor = next(f for f in cmp["facteurs"] if f["factor"] == "Win rate %")
    # win_rate is stored as a 0-1 ratio (e.g. 0.59 for windows 1+2 averaged) — must be
    # displayed on the same 0-100 scale as the other percentage factors, not 0.59.
    assert wr_factor["avg_fenetres_gagnantes"] == 59.0
    assert wr_factor["avg_fenetres_perdantes"] == 41.0


def test_compare_window_regimes_returns_none_with_too_few_windows():
    windows = [_fake_window(pf=1.5, n_trades=40, adx=35.0, sl_pct=15.0, wr=60.0)]
    assert pretrain._compare_window_regimes(windows) is None


def test_compare_window_regimes_returns_none_when_all_same_side():
    windows = [
        _fake_window(pf=1.5, n_trades=40, adx=35.0, sl_pct=15.0, wr=60.0),
        _fake_window(pf=1.3, n_trades=30, adx=34.0, sl_pct=18.0, wr=58.0),
        _fake_window(pf=1.2, n_trades=25, adx=25.0, sl_pct=20.0, wr=55.0),
        _fake_window(pf=1.4, n_trades=35, adx=26.0, sl_pct=19.0, wr=57.0),
    ]
    assert pretrain._compare_window_regimes(windows) is None


def _fake_trade(direction, h4_bias, won, pnl=10.0):
    return {"direction": direction, "h4_bias": h4_bias, "won": won, "pnl": pnl}


def test_diag_by_direction_regime_splits_long_by_h4_alignment():
    trades = (
        [_fake_trade("long", 1, True, 20.0) for _ in range(6)]      # LONG, H4 aligné, tous gagnants
        + [_fake_trade("long", 1, False, -10.0) for _ in range(2)]
        + [_fake_trade("long", -1, False, -15.0) for _ in range(5)]  # LONG, H4 PAS aligné, tous perdants
        + [_fake_trade("long", 0, False, -12.0)]
    )
    result = pretrain._diag_by_direction_regime(trades)

    assert result["long_h4_aligné"]["n"] == 8
    assert result["long_h4_aligné"]["wr"] == pytest.approx(75.0)
    assert result["long_h4_non_aligné"]["n"] == 6
    assert result["long_h4_non_aligné"]["wr"] == pytest.approx(0.0)


def test_diag_by_direction_regime_matches_short_alignment_on_h4_minus_one():
    trades = (
        [_fake_trade("short", -1, True, 15.0) for _ in range(4)]
        + [_fake_trade("short", 1, False, -8.0) for _ in range(3)]
    )
    result = pretrain._diag_by_direction_regime(trades)

    assert result["short_h4_aligné"]["n"] == 4
    assert result["short_h4_aligné"]["wr"] == pytest.approx(100.0)
    assert result["short_h4_non_aligné"]["n"] == 3
    assert result["short_h4_non_aligné"]["wr"] == pytest.approx(0.0)


def test_diag_by_direction_regime_drops_groups_under_min_sample():
    trades = [_fake_trade("long", 1, True) for _ in range(2)]  # n=2, sous le seuil de 3
    result = pretrain._diag_by_direction_regime(trades)
    assert "long_h4_aligné" not in result


def _fake_exit_trade(direction, exit_reason, won, pnl):
    return {"direction": direction, "exit_reason": exit_reason, "won": won, "pnl": pnl}


def test_diag_by_direction_exit_reason_splits_by_bucket_and_direction():
    trades = (
        [_fake_exit_trade("long", "tp1", True, 5.0) for _ in range(4)]
        + [_fake_exit_trade("long", "sl", False, -10.0) for _ in range(6)]
        + [_fake_exit_trade("short", "tp2", True, 20.0) for _ in range(5)]
        + [_fake_exit_trade("short", "sl", False, -10.0) for _ in range(3)]
    )
    result = pretrain._diag_by_direction_exit_reason(trades)

    assert result["long_tp1"]["n"] == 4
    assert result["long_tp1"]["pct"] == pytest.approx(40.0)  # 4/10 trades LONG
    assert result["long_tp1"]["avg_pnl"] == pytest.approx(5.0)
    assert result["long_sl"]["n"] == 6
    assert result["long_sl"]["pct"] == pytest.approx(60.0)
    assert result["short_tp2"]["n"] == 5
    assert result["short_tp2"]["pct"] == pytest.approx(62.5)  # 5/8 trades SHORT
    assert result["short_sl"]["n"] == 3
    assert result["short_sl"]["pct"] == pytest.approx(37.5)


def test_diag_by_direction_exit_reason_buckets_unknown_reasons_as_other():
    trades = (
        [_fake_exit_trade("long", "sl_realtime", False, -10.0) for _ in range(3)]
        + [_fake_exit_trade("long", "tp1", True, 5.0) for _ in range(3)]
    )
    result = pretrain._diag_by_direction_exit_reason(trades)
    assert result["long_other"]["n"] == 3
    assert "long_sl_realtime" not in result


def test_diag_by_direction_exit_reason_drops_groups_under_min_sample():
    trades = [_fake_exit_trade("long", "tp1", True, 5.0) for _ in range(2)]  # n=2
    result = pretrain._diag_by_direction_exit_reason(trades)
    assert "long_tp1" not in result


def test_atr_min_override_actually_reaches_evaluate(monkeypatch):
    """L'override ATR_MIN (champ "ATR minimum" du panel walk-forward) doit atteindre
    evaluate(). Il ne l'atteignait pas : evaluate() reçoit atr_min en argument explicite,
    et run_pretrain le calculait depuis un 3.0 codé en dur AVANT d'appliquer les
    overrides — un setattr(strategy, "ATR_MIN", 4.5) restait donc sans effet, et le
    walk-forward renvoyait exactement les mêmes trades qu'un run de contrôle."""
    seen = []

    def _spy_evaluate(*args, **kwargs):
        seen.append(kwargs.get("atr_min"))
        return None  # aucun signal : on ne teste que la valeur transmise

    monkeypatch.setattr(pretrain, "evaluate", _spy_evaluate)
    pretrain.run_pretrain("2024-01-01", "2024-01-10", symbol="XAUUSD",
                          reset=True, write_to_db=False,
                          extra_overrides={"ATR_MIN": 4.5})

    assert seen, "evaluate() n'a jamais été appelée — test inopérant"
    assert set(seen) == {4.5}


def test_atr_min_defaults_to_live_setting_when_not_overridden(monkeypatch):
    """Sans override, le walk-forward doit utiliser le réglage live (strategy.ATR_MIN),
    pas une valeur de calibration divergente — sinon il ne valide pas ce qui tourne."""
    seen = []

    def _spy_evaluate(*args, **kwargs):
        seen.append(kwargs.get("atr_min"))
        return None

    monkeypatch.setattr(pretrain, "evaluate", _spy_evaluate)
    monkeypatch.setattr(strategy, "ATR_MIN", 3.0)
    pretrain.run_pretrain("2024-01-01", "2024-01-10", symbol="XAUUSD",
                          reset=True, write_to_db=False)

    assert seen
    assert set(seen) == {3.0}
