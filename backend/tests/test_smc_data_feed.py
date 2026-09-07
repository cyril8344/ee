"""Accès aux bougies — l'interface unique, et les deux garde-fous.

Ce dépôt s'est déjà fait avoir deux fois par la même famille de bugs : un repli
silencieux sur des données simulées, et un calcul fait sur une bougie non
clôturée. Les deux sont traités ici, une fois, pour que les modules en aval
n'aient pas à y penser.
"""
import os
from datetime import datetime, timezone

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import data_feed


def _frame(n, freq="15min", start="2026-01-05 08:00"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                       "volume": 10.0}, index=idx)
    return df


def test_unknown_timeframe_raises_instead_of_falling_back():
    """Un timeframe inconnu doit lever. Le repli muet sur XAU/USD de
    `market_symbol` avait déjà fait passer des données d'or pour de l'ES."""
    with pytest.raises(ValueError):
        data_feed.get_ohlcv("XAUUSD", "M3", 10)
    with pytest.raises(ValueError):
        data_feed.timeframe_minutes("M3")


def test_the_unclosed_bar_is_dropped():
    """§3 : aucun calcul ne doit utiliser une bougie non clôturée. L'index est
    daté à la fin de période, donc une bougie est close quand son horodatage est
    passé."""
    df = _frame(5)                       # 08:00 → 09:00, pas de 15 min
    now = df.index[3].to_pydatetime()    # la 5e bougie est encore en cours
    out = data_feed.drop_unclosed(df, "M15", now=now)
    assert len(out) == 4
    assert out.index[-1] == df.index[3]


def test_drop_unclosed_handles_a_naive_index():
    df = _frame(4)
    df.index = df.index.tz_localize(None)
    out = data_feed.drop_unclosed(df, "M15", now=datetime(2026, 1, 5, 8, 30,
                                                          tzinfo=timezone.utc))
    assert len(out) == 3


def test_drop_unclosed_on_an_empty_frame_is_a_noop():
    vide = _frame(0)
    assert data_feed.drop_unclosed(vide, "M15").empty


def test_timeframe_minutes_are_the_spec_ones():
    assert data_feed.timeframe_minutes("M15") == 15
    assert data_feed.timeframe_minutes("H4") == 240
    assert data_feed.timeframe_minutes("D1") == 1440


def test_window_asked_for_covers_weekends():
    """Le marché ferme la nuit et le week-end : une fenêtre calculée sur du temps
    continu rend systématiquement trop peu de bougies — exactement ce qui a
    produit l'« EMA200 sur 43 bougies » du bot principal."""
    jours = data_feed._bars_to_days("H4", 200)
    continu = 240 * 200 / (60 * 24)      # ~33 jours si le marché ne fermait jamais
    assert jours > continu


def test_synthetic_data_is_flagged_not_hidden():
    """Un scanner qui alerte sur des bougies simulées est pire qu'un scanner
    muet : il faut que l'appelant puisse le savoir."""
    c = data_feed.get_ohlcv("XAUUSD", "M15", 50)
    assert c.provider == "synthetic"
    assert c.is_synthetic is True


def test_get_ohlcv_returns_the_requested_timeframe_and_count():
    c = data_feed.get_ohlcv("XAUUSD", "M15", 60)
    assert c.timeframe == "M15"
    assert c.symbol == "XAUUSD"
    assert 0 < len(c) <= 60
    assert list(c.df.columns[:4]) == ["open", "high", "low", "close"]
    ecarts = c.df.index.to_series().diff().dropna().unique()
    assert all(pd.Timedelta(e) >= pd.Timedelta(minutes=15) for e in ecarts)


def test_higher_timeframes_are_resampled_from_a_native_interval():
    """H4 n'existe nativement chez aucun fournisseur : il est resamplé depuis H1.
    Le demander en M5 ferait des centaines de milliers de lignes pour quelques
    dizaines de bougies utiles."""
    h4 = data_feed.get_ohlcv("XAUUSD", "H4", 30)
    assert len(h4) > 0
    ecarts = h4.df.index.to_series().diff().dropna()
    assert all(e >= pd.Timedelta(hours=4) for e in ecarts)


def test_ohlc_stays_coherent_after_resampling():
    """§10 : les données sont validées à chaque cycle. Un high sous le low
    signalerait une agrégation cassée."""
    for tf in ("M15", "H1", "H4"):
        df = data_feed.get_ohlcv("XAUUSD", tf, 40).df
        assert not df.isna().any().any()
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
