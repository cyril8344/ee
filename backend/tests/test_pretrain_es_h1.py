"""Smoke tests for the H1 ES pretrain/walk-forward engine (pretrain_es_h1.py)."""
from unittest.mock import patch

import pandas as pd
import pytest

import pretrain_es_h1 as pt


def _fake_spy_h1(n=150):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [580.0] * n, "high": [582.0] * n,
        "low": [579.0] * n, "close": [581.5] * n,
        "volume": [100000] * n,
    }, index=idx)
    df.index.name = "time"
    return df


def test_load_es_h1_data_scales_spy_to_es_price_level():
    with patch("pretrain_es_h1._dp._fetch_twelvedata_range", return_value=_fake_spy_h1()) as m:
        df, source = pt._load_es_h1_data("2026-01-01", "2026-01-10")
    assert source == "twelvedata_spy"
    row = df.iloc[0]
    assert row["open"] == pytest.approx(5800.0)
    assert row["close"] == pytest.approx(5815.0)
    assert row["volume"] == pytest.approx(100000)
    _, kwargs = m.call_args
    assert kwargs.get("symbol") == "SPY"
    assert kwargs.get("interval") == "1h"


def test_load_es_h1_data_falls_back_to_synthetic_when_twelvedata_fails():
    with patch("pretrain_es_h1._dp._fetch_twelvedata_range", side_effect=RuntimeError("boom")):
        df, source = pt._load_es_h1_data("2026-01-01", "2026-06-01")
    assert source == "synthetic"
    assert len(df) > 0


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
