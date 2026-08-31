"""Le contexte live doit être assez long pour que l'EMA200 H1 soit réellement une
moyenne 200 périodes.

Bug d'origine : build_context() demandait 500 bougies M5, soit ~43 bougies H1 une fois
resamplé. ema() est un `ewm(span=200, adjust=False)` sans min_periods — il renvoie donc
toujours un nombre, même avec 43 bougies, mais ~65 % de ce nombre n'est que la valeur
d'amorce (le prix d'il y a deux jours). Or le biais H1 (EMA50 vs EMA200) et
TREND_BIAS_DISTANCE, qui décident LONG/SHORT, en dépendent directement — alors que
pretrain/walk-forward resample toute la période et obtient des milliers de bougies H1.
Live et backtest ne calculaient pas le même biais.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import broker as broker_mod
import main as main_module
import strategy


def _m5_frame(n_bars: int) -> pd.DataFrame:
    """Série M5 continue avec une vraie tendance — une dérive rend l'écart entre une
    EMA200 amorcée et une EMA200 convergée mesurable (sur du bruit pur, les deux
    tournent autour de la même moyenne et le bug reste invisible)."""
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="5min", tz="UTC")
    close = np.linspace(2000.0, 2100.0, n_bars)
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n_bars, 100.0),
    }, index=idx)


class _StubBroker:
    """Reproduit le contrat de MarketData : une longue série en cache, dont
    get_rates_m5(bars) ne renvoie que la queue demandée."""

    def __init__(self, available_bars=20_000):
        self._full = _m5_frame(available_bars)
        self.last_requested = None

    def get_rates_m5(self, bars=500):
        self.last_requested = bars
        return self._full.tail(bars).copy()


def test_build_context_requests_the_long_window():
    br = _StubBroker()
    main_module.build_context(br)
    assert br.last_requested == main_module.CONTEXT_BARS_M5
    assert main_module.CONTEXT_BARS_M5 >= 5000


def test_h1_frame_is_long_enough_for_a_converged_ema200():
    """Poids restant sur la valeur d'amorce après n bougies = (1-alpha)^n, avec
    alpha = 2/(200+1). Il doit rester marginal, pas majoritaire."""
    _, _, h1, _ = main_module.build_context(_StubBroker())
    alpha = 2 / (strategy.EMA_SLOW + 1)
    seed_weight = (1 - alpha) ** len(h1)

    assert len(h1) >= 400, f"seulement {len(h1)} bougies H1"
    assert seed_weight < 0.05, f"amorce encore à {seed_weight:.1%} de l'EMA200 H1"


def test_old_500_bar_window_was_indeed_broken():
    """Verrouille le diagnostic : avec l'ancienne fenêtre, l'EMA200 H1 était
    majoritairement composée de sa valeur d'amorce. Si ce test se met à échouer,
    c'est que la relation fenêtre/convergence a changé et que le raisonnement
    ci-dessus doit être revérifié."""
    h1_old = _StubBroker()._full.tail(500).resample(
        "60min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    alpha = 2 / (strategy.EMA_SLOW + 1)

    assert len(h1_old) < 50
    assert (1 - alpha) ** len(h1_old) > 0.5


def test_long_window_ema200_is_closer_to_the_true_ema200():
    """Le test qui compte : l'EMA200 H1 du contexte live doit approcher celle calculée
    sur l'historique complet, comme le fait le pretrain."""
    br = _StubBroker()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    h1_full = strategy.add_indicators(
        br._full.resample("60min", label="right", closed="right").agg(agg).dropna())
    truth = float(h1_full["ema200"].iloc[-1])

    _, _, h1_new, _ = main_module.build_context(br)
    h1_old = strategy.add_indicators(
        br._full.tail(500).resample("60min", label="right", closed="right").agg(agg).dropna())

    err_new = abs(float(h1_new["ema200"].iloc[-1]) - truth)
    err_old = abs(float(h1_old["ema200"].iloc[-1]) - truth)

    assert err_new < err_old / 10, f"nouvelle erreur {err_new:.2f} vs ancienne {err_old:.2f}"


def test_vwap_is_unaffected_by_the_window_length():
    """La fenêtre s'allonge, mais le VWAP est groupé par jour : la valeur courante ne
    doit pas bouger, sinon on aurait changé un filtre live au passage."""
    br = _StubBroker()
    m5_long = strategy.add_indicators(br._full.tail(main_module.CONTEXT_BARS_M5))
    m5_short = strategy.add_indicators(br._full.tail(500))

    assert float(m5_long["vwap"].iloc[-1]) == pytest.approx(float(m5_short["vwap"].iloc[-1]))


def test_feed_fetches_enough_bars_to_serve_the_context():
    """Le cache alimenté par MarketData._fetch() doit pouvoir servir CONTEXT_BARS_M5 :
    sinon build_context redemanderait plus que ce que le feed détient et retomberait
    silencieusement sur une série courte."""
    assert broker_mod.LIVE_FETCH_BARS >= main_module.CONTEXT_BARS_M5


def test_yfinance_fallback_requests_enough_history(monkeypatch):
    """Le repli yfinance utilisait period="5d" quel que soit `bars` : le contexte live
    retombait alors à ~1400 bougies M5, soit le bug de fenêtre courte réintroduit dès
    que Twelve Data échouait."""
    import data_provider

    seen = {}

    class _FakeYF:
        @staticmethod
        def download(sym, period=None, interval=None, **kw):
            seen["period"] = period
            idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
            return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                                 "Close": 1.0, "Volume": 1.0}, index=idx)

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _FakeYF)
    data_provider._fetch_yfinance(None, None, bars=5000, symbol="XAUUSD")

    days = int(seen["period"].rstrip("d"))
    # 5000 bougies M5 ≈ 18 jours de cotation ≈ 25 jours calendaires, et jamais au-delà
    # des 60 jours que yfinance autorise en intraday 5 minutes.
    assert 20 <= days <= 60


def test_yfinance_fallback_stays_within_the_60_day_intraday_limit(monkeypatch):
    import data_provider

    seen = {}

    class _FakeYF:
        @staticmethod
        def download(sym, period=None, interval=None, **kw):
            seen["period"] = period
            idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
            return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                                 "Close": 1.0, "Volume": 1.0}, index=idx)

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _FakeYF)
    data_provider._fetch_yfinance(None, None, bars=500_000, symbol="XAUUSD")
    assert seen["period"] == "60d"
