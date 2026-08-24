"""Tests for pretrain.diag_real_trades_by_day_volatility() — the diagnostic that
cross-references REAL executed trades (not a pretrain simulation) with the real
M5 ATR at each entry, grouped by calendar day. Added to test the hypothesis that
quiet (low-ATR) days produce disproportionately more early_exit outcomes, instead
of eyeballing a handful of examples in a screenshot.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

import pretrain


def _m5_two_days():
    """Day 1 (2024-06-03) : ATR constant bas (2.0). Day 2 (2024-06-04) : ATR
    constant haut (8.0). 3 bougies M5 par jour suffit pour le lookup."""
    idx = pd.to_datetime([
        "2024-06-03T09:00:00Z", "2024-06-03T09:05:00Z", "2024-06-03T09:10:00Z",
        "2024-06-04T09:00:00Z", "2024-06-04T09:05:00Z", "2024-06-04T09:10:00Z",
    ], utc=True)
    atr = [2.0, 2.0, 2.0, 8.0, 8.0, 8.0]
    return pd.DataFrame({"atr": atr}, index=idx)


def _m5_three_days():
    """3 jours d'ATR croissant (2.0 / 5.0 / 8.0) — nécessaire pour la corrélation,
    qui exige au moins 3 jours valides (voir diag_real_trades_by_day_volatility)."""
    idx = pd.to_datetime([
        "2024-06-03T09:00:00Z", "2024-06-03T09:05:00Z", "2024-06-03T09:10:00Z",
        "2024-06-04T09:00:00Z", "2024-06-04T09:05:00Z", "2024-06-04T09:10:00Z",
        "2024-06-05T09:00:00Z", "2024-06-05T09:05:00Z", "2024-06-05T09:10:00Z",
    ], utc=True)
    atr = [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0]
    return pd.DataFrame({"atr": atr}, index=idx)


def _trade(entry_iso, date_cet, exit_reason):
    return {"entry_ts_utc": entry_iso, "entry_time": entry_iso,
            "date_cet": date_cet, "exit_reason": exit_reason}


def test_groups_by_day_and_flags_early_exit_pct():
    m5 = _m5_two_days()
    trades = (
        # jour calme (ATR=2.0) : surtout des early_exit
        [_trade("2024-06-03T09:06:00Z", "2024-06-03", "early_exit") for _ in range(3)]
        + [_trade("2024-06-03T09:07:00Z", "2024-06-03", "tp1")]
        # jour volatil (ATR=8.0) : aucun early_exit
        + [_trade("2024-06-04T09:06:00Z", "2024-06-04", "tp2") for _ in range(3)]
        + [_trade("2024-06-04T09:07:00Z", "2024-06-04", "sl")]
    )
    result = pretrain.diag_real_trades_by_day_volatility(_trades=trades, _m5=m5)

    days = {d["day"]: d for d in result["days"]}
    assert days["2024-06-03"]["n"] == 4
    assert days["2024-06-03"]["early_exit_pct"] == pytest.approx(75.0)
    assert days["2024-06-03"]["avg_atr"] == pytest.approx(2.0)

    assert days["2024-06-04"]["n"] == 4
    assert days["2024-06-04"]["early_exit_pct"] == pytest.approx(0.0)
    assert days["2024-06-04"]["avg_atr"] == pytest.approx(8.0)
    assert result["data_provider_used"] == "injected"


def _raw_ohlcv_two_days():
    """Données OHLCV brutes (pas encore indicateurs) — ce que load_m5_data()
    renvoie réellement, avant add_indicators()."""
    idx = pd.to_datetime([
        "2024-06-03T09:00:00Z", "2024-06-03T09:05:00Z", "2024-06-03T09:10:00Z",
        "2024-06-04T09:00:00Z", "2024-06-04T09:05:00Z", "2024-06-04T09:10:00Z",
    ], utc=True)
    return pd.DataFrame({
        "open":  [2000.0] * 6, "high": [2001.0] * 6, "low": [1999.0] * 6,
        "close": [2000.5] * 6, "volume": [100.0] * 6,
    }, index=idx)


def test_flags_synthetic_fallback_so_the_correlation_isnt_trusted_blindly(monkeypatch):
    """load_m5_data() retombe silencieusement sur des données synthétiques si le
    vrai fournisseur échoue (voir CLAUDE.md). Le diagnostic doit exposer quelle
    source a été utilisée, pas juste calculer une corrélation sur des prix
    potentiellement inventés sans le signaler."""
    raw = _raw_ohlcv_two_days()
    raw.attrs["provider"] = "synthetic"

    def _fake_load_m5_data(start, end, symbol="XAUUSD", **kwargs):
        return raw

    monkeypatch.setattr(pretrain, "load_m5_data", _fake_load_m5_data)
    monkeypatch.setattr(pretrain._dp, "has_backtest_key", lambda: False)
    monkeypatch.setattr(pretrain._dp, "get_last_errors",
                         lambda: {"XAUUSD:twelvedata_range": "429 Too Many Requests"})
    trades = [_trade("2024-06-03T09:06:00Z", "2024-06-03", "tp1")]
    result = pretrain.diag_real_trades_by_day_volatility(symbol="XAUUSD", _trades=trades)
    assert result["data_provider_used"] == "synthetic"
    assert result["data_provider_debug"]["backtest_key_configured"] is False
    assert "XAUUSD:twelvedata_range" in result["data_provider_debug"]["errors"]


def test_no_debug_info_when_real_provider_succeeds(monkeypatch):
    raw = _raw_ohlcv_two_days()
    raw.attrs["provider"] = "twelvedata"

    monkeypatch.setattr(pretrain, "load_m5_data", lambda start, end, symbol="XAUUSD", **kw: raw)
    trades = [_trade("2024-06-03T09:06:00Z", "2024-06-03", "tp1")]
    result = pretrain.diag_real_trades_by_day_volatility(symbol="XAUUSD", _trades=trades)
    assert result["data_provider_used"] == "twelvedata"
    assert result["data_provider_debug"] is None


def test_never_requests_market_data_beyond_now(monkeypatch):
    """Demander une marge en avant sur la dernière entrée fait sortir la plage
    dans le futur quand un trade a eu lieu aujourd'hui — le marché n'a pas
    produit ces bougies, donc la couverture était structurellement déficitaire
    (49j obtenus / 51j demandés) et le résultat basculait sur le synthétique."""
    raw = _raw_ohlcv_two_days()
    raw.attrs["provider"] = "twelvedata"
    captured = {}

    def _fake_load_m5_data(start, end, symbol="XAUUSD", **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured["min_coverage"] = kwargs.get("min_coverage")
        return raw

    monkeypatch.setattr(pretrain, "load_m5_data", _fake_load_m5_data)

    now = datetime.now(timezone.utc)
    today_entry = now.isoformat()
    trades = [_trade(today_entry, now.date().isoformat(), "tp1")]
    pretrain.diag_real_trades_by_day_volatility(symbol="XAUUSD", _trades=trades)

    assert captured["end"] <= now.date().isoformat()
    # Seuil assoupli transmis : ce diagnostic tolère les trous (lookups ponctuels).
    assert captured["min_coverage"] is not None
    assert captured["min_coverage"] < 0.99


def test_flags_real_provider_when_fetch_succeeds(monkeypatch):
    raw = _raw_ohlcv_two_days()
    raw.attrs["provider"] = "twelvedata"

    def _fake_load_m5_data(start, end, symbol="XAUUSD", **kwargs):
        return raw

    monkeypatch.setattr(pretrain, "load_m5_data", _fake_load_m5_data)
    trades = [_trade("2024-06-03T09:06:00Z", "2024-06-03", "tp1")]
    result = pretrain.diag_real_trades_by_day_volatility(symbol="XAUUSD", _trades=trades)
    assert result["data_provider_used"] == "twelvedata"


def test_correlation_is_negative_when_low_atr_means_more_early_exit():
    m5 = _m5_three_days()
    trades = (
        # jour 1 (ATR=2.0, le plus calme) : early_exit_pct le plus haut
        [_trade("2024-06-03T09:06:00Z", "2024-06-03", "early_exit") for _ in range(4)]
        # jour 2 (ATR=5.0) : early_exit_pct intermédiaire
        + [_trade("2024-06-04T09:06:00Z", "2024-06-04", "early_exit") for _ in range(2)]
        + [_trade("2024-06-04T09:07:00Z", "2024-06-04", "tp1") for _ in range(2)]
        # jour 3 (ATR=8.0, le plus volatil) : aucun early_exit
        + [_trade("2024-06-05T09:06:00Z", "2024-06-05", "tp2") for _ in range(4)]
    )
    result = pretrain.diag_real_trades_by_day_volatility(_trades=trades, _m5=m5)
    assert result["correlation_atr_vs_early_exit_pct"] is not None
    assert result["correlation_atr_vs_early_exit_pct"] < 0


def test_returns_note_when_no_trades():
    result = pretrain.diag_real_trades_by_day_volatility(_trades=[], _m5=_m5_two_days())
    assert result["days"] == []
    assert result["correlation_atr_vs_early_exit_pct"] is None
    assert "note" in result


def test_picks_last_bar_at_or_before_entry_time():
    """Une entrée juste après la dernière bougie du jour 1 doit prendre l'ATR de
    cette dernière bougie (dernière bougie <= entry_time), pas déborder sur le jour 2."""
    m5 = _m5_two_days()
    trades = [_trade("2024-06-03T09:11:00Z", "2024-06-03", "tp1")]  # 1 min après la 3e bougie j1
    result = pretrain.diag_real_trades_by_day_volatility(_trades=trades, _m5=m5)
    assert result["days"][0]["avg_atr"] == pytest.approx(2.0)
