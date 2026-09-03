"""Sonde H1 : la stratégie décalée d'un cran, sans changement de logique.

Pourquoi. Le coût d'un trade est fixe (~0.30 $ aller-retour sur l'or) alors que la
cible croît en racine du temps : la friction passe de ~4 % du R en M5 à ~1 % en H1.
Mais il y a 12 fois moins de trades, donc le passage n'est gagnant que si le signal
est intrinsèquement meilleur — ce que seul un walk-forward H1 peut dire.

Or il était impossible à lancer : tout le chargeur demandait du 5 minutes, et
l'historique M5 du fournisseur s'arrête vers 2022.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import backtest
import data_provider
import pretrain
import strategy


def test_m5_remains_the_default():
    """Aucun appelant existant ne doit changer de comportement."""
    assert pretrain.TIMEFRAME_SETS["M5"]["base"] == "5min"
    assert pretrain.TIMEFRAME_SETS["M5"]["scale"] == 1


@pytest.mark.parametrize("interval,expected", [
    ("5min", pd.Timedelta(minutes=5)),
    ("1h", pd.Timedelta(hours=1)),
    ("4h", pd.Timedelta(hours=4)),
])
def test_provider_returns_the_requested_bar_size(interval, expected):
    df, _ = data_provider.get_m5(start="2024-02-01", end="2024-05-01",
                                 symbol="XAUUSD", interval=interval)
    assert df.index[1] - df.index[0] == expected


def test_synthetic_last_resort_honours_the_interval():
    """Le repli de dernier recours renvoyait toujours du 5 minutes : une demande
    horaire recevait des bougies M5 étiquetées H1, et tout le contexte multi-TF
    resamplé par-dessus était faux sans le moindre signe."""
    df, provider = data_provider._fetch_synthetic("2024-01-01", "2024-03-01",
                                                  5000, "XAUUSD", "1h"), "synthetic"
    assert df.index[1] - df.index[0] == pd.Timedelta(hours=1)


def test_cache_key_separates_intervals():
    """Sans l'intervalle dans la clé, une plage H1 écraserait la même plage M5 et un
    walk-forward M5 relirait ensuite des bougies horaires."""
    k5 = data_provider._cache_key("XAUUSD", "2024-01-01", "2024-06-30", "5min")
    k1h = data_provider._cache_key("XAUUSD", "2024-01-01", "2024-06-30", "1h")
    assert k5 != k1h
    # La forme historique est préservée pour M5 : le cache déjà constitué reste valide.
    assert k5 == data_provider._cache_key("XAUUSD", "2024-01-01", "2024-06-30")


def test_coverage_tolerates_the_weekend_edge():
    """Une plage demandée du lundi au dimanche ne peut pas être couverte à 100 % :
    la dernière bougie tombe le vendredi. 179j/181j = 98.9 % faisait rejeter un fetch
    pourtant complet et basculer en silence sur le repli synthétique."""
    assert data_provider._coverage_ok(179, 181, 0.99) is True
    # Une vraie troncature (rate limit) coûte des semaines et reste rejetée.
    assert data_provider._coverage_ok(120, 181, 0.99) is False


def test_load_data_at_h1_gives_hourly_bars():
    df = backtest.load_m5_data("2024-01-01", "2024-06-30", symbol="XAUUSD", interval="1h")
    assert df.index[1] - df.index[0] == pd.Timedelta(hours=1)
    # 6 mois ouvrés en horaire ≈ 3000 bougies ; du M5 en donnerait ~37 000.
    assert 2000 < len(df) < 5000


def test_h1_run_scales_time_constants_and_restores_them():
    """Le timeout et l'early exit sont en minutes : sans mise à l'échelle, une bougie
    et demie suffirait à couper chaque trade H1. Et sans restauration, le module
    strategy garderait un timeout de 900 min pour le live qui suit."""
    before = (strategy.MAX_TRADE_MINUTES, strategy.EARLY_EXIT_MINUTES, strategy.ATR_MIN)
    seen = {}

    original = pretrain.evaluate

    def _spy(*args, **kwargs):
        seen["atr_min"] = kwargs.get("atr_min")
        seen["max_trade_minutes"] = strategy.MAX_TRADE_MINUTES
        seen["early_exit_minutes"] = strategy.EARLY_EXIT_MINUTES
        return None

    pretrain.evaluate = _spy
    try:
        pretrain.run_pretrain("2024-01-01", "2024-06-30", symbol="XAUUSD",
                              reset=True, write_to_db=False, timeframe_set="H1")
    finally:
        pretrain.evaluate = original

    assert seen["max_trade_minutes"] == before[0] * 12
    assert seen["early_exit_minutes"] == before[1] * 12
    # L'ATR suit la racine du temps, pas le facteur plein.
    assert seen["atr_min"] == pytest.approx(before[2] * (12 ** 0.5), abs=0.01)

    assert (strategy.MAX_TRADE_MINUTES, strategy.EARLY_EXIT_MINUTES, strategy.ATR_MIN) == before


def test_early_exit_counts_bars_at_the_right_duration():
    """_try_exit comptait les bougies à 300 s en dur : en H1 le compteur aurait tourné
    12 fois trop vite."""
    from datetime import datetime, timezone
    entry = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    trade = {
        "direction": "long", "entry": 2000.0, "stop_loss": 1990.0,
        "tp1": 2007.0, "tp2": 2018.0, "tp1_done": False, "be_after_tp1": True,
        "remaining": 0.1, "volume": 0.1, "realised": 0.0, "risk": 10.0,
        "mae": 0.0, "mfe": 0.0, "entry_time": entry,
        "max_exit_time": entry.replace(hour=23),
        "bar_minutes": 60,
    }
    bar = {"high": 2000.5, "low": 1999.5, "close": 2000.0, "atr": 5.0}
    # 30 minutes après l'entrée : moins d'une bougie H1, l'early exit (15 min × 12
    # = 180 min) ne doit pas se déclencher.
    ts = pd.Timestamp(entry) + pd.Timedelta(minutes=30)
    assert backtest._try_exit(trade, bar, ts, 0.01, 100.0) is None
