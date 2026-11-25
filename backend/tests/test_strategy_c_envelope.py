"""Enveloppe live de la stratégie C : l'ancien paramétrage en stand-by, les
plafonds lus dans le code et pas dans les réglages."""
import os

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")
os.environ.setdefault("ADMIN_USERNAME", "a")
os.environ.setdefault("ADMIN_PASSWORD", "b")

import main  # noqa: E402
import strategy_sweep as ss  # noqa: E402
from risk_manager import RiskManager  # noqa: E402


def test_gold_runs_c_with_the_code_envelope():
    assert main.MARKET_CONFIG["XAUUSD"]["default_strategy"] == "C"
    r = main.state.risk
    assert r.effective_max_trades == ss.LIVE_MAX_TRADES_PER_DAY == 15
    assert r.effective_daily_stop_pct == ss.LIVE_DAILY_STOP_PCT


def test_settings_cannot_silently_bring_back_the_old_cap():
    """sync_from_settings plafonne à 10 et BotState ramène > 10 à 4 : sans
    surcharge en champ séparé, le 15 redeviendrait 4 au premier redémarrage."""
    r = RiskManager()
    main._apply_strategy_c_envelope(r)
    r.sync_from_settings({"max_trades_per_day": 4, "daily_stop_pct": 2.0})
    assert r.effective_max_trades == 15
    assert r.effective_daily_stop_pct == 10.0


def test_the_envelope_never_leaks_to_eur_or_es():
    """_sync_shared_risk_settings recopie les réglages de l'or vers EUR/ES."""
    src = RiskManager()
    main._apply_strategy_c_envelope(src)
    dst = RiskManager()
    main._sync_shared_risk_settings(dst, src)
    assert dst.max_trades_override is None and dst.daily_stop_override is None
    assert dst.effective_max_trades == dst.max_trades_per_day


def test_the_dashboard_form_reads_settings_not_the_override():
    """Le formulaire relit status()["max_trades_per_day"] et le renvoie en
    réglage partagé : y mettre 15 le recopierait vers EUR/ES."""
    r = RiskManager(max_trades_per_day=4)
    main._apply_strategy_c_envelope(r)
    st = r.status()
    assert st["max_trades_per_day"] == 4
    assert st["max_trades_per_day_effective"] == 15


def test_the_override_is_what_blocks_the_day():
    r = RiskManager(capital=1000.0, daily_stop_pct=2.0)
    r.max_trades_override, r.daily_stop_override = 15, 10.0
    r.trades_today = 14
    assert r.can_open_trade(100.0, 99.0, contract_size=1.0).allowed
    r.trades_today = 15
    assert not r.can_open_trade(100.0, 99.0, contract_size=1.0).allowed
    r.trades_today = 0
    r.register_close(-50.0)            # -5 % : l'ancien stop à 2 % aurait bloqué
    assert not r.blocked
    r.register_close(-60.0)            # -11 %
    assert r.blocked


def test_gold_daily_stop_per_instrument_follows_the_envelope():
    ms = main.state.market_states["XAUUSD"]
    assert main._inst_daily_stop_pct(ms) == ss.LIVE_DAILY_STOP_PCT
    eur = main.state.market_states.get("EURUSD")
    if eur is not None:
        assert main._inst_daily_stop_pct(eur) == eur.inst_settings.get("daily_stop_pct", 2.0)


def test_without_c_the_envelope_is_removed():
    r = RiskManager()
    old = main.MARKET_CONFIG["XAUUSD"]["default_strategy"]
    try:
        main.MARKET_CONFIG["XAUUSD"]["default_strategy"] = "A"
        main._apply_strategy_c_envelope(r)
        assert r.max_trades_override is None and r.daily_stop_override is None
    finally:
        main.MARKET_CONFIG["XAUUSD"]["default_strategy"] = old
