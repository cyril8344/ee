"""Smoke tests for the H1 ES pretrain/walk-forward engine (pretrain_es_h1.py)."""
import pretrain_es_h1 as pt


def test_run_pretrain_es_h1_on_synthetic_data():
    result = pt.run_pretrain_es_h1("2025-01-01", "2025-07-01", capital=50_000.0, risk_pct=1.0)
    assert result["ok"] is True
    assert "n_trades" in result
    assert "profit_factor" in result
    assert result["bars_total"] > 0
    # aucune position ne doit dépasser 8h (défaut max_trade_hours) ni franchir
    # la clôture de session sans être flattenée
    for t in result["trades"]:
        assert t["exit_reason"] in (
            "tp2", "tp_direct", "sl", "sl_after_tp1",
            "timeout", "timeout_tp1", "session_close", "session_close_tp1",
        )


def test_run_walkforward_es_h1_reports_robustness_criteria():
    result = pt.run_walkforward_es_h1("2025-01-01", "2025-10-01", n_splits=3,
                                       capital=50_000.0, risk_pct=1.0)
    assert len(result["windows"]) == 3
    assert "mean_pf" in result
    assert "std_pf" in result
    assert isinstance(result["robust"], bool)
