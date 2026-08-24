"""Tests for the unified data provider (synthetic / fallback path)."""
from unittest.mock import patch, MagicMock

import pandas as pd

import data_provider
import database as db


def test_synthetic_always_returns_data():
    df = data_provider._fetch_synthetic("2024-01-01", "2024-02-01", 500)
    assert len(df) > 0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None


def test_synthetic_is_deterministic():
    a = data_provider._fetch_synthetic("2024-03-01", "2024-03-15", 500)
    b = data_provider._fetch_synthetic("2024-03-01", "2024-03-15", 500)
    pd.testing.assert_frame_equal(a, b)


def test_get_m5_returns_dataframe_and_provider():
    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-01-20", bars=500)
    assert len(df) > 0
    assert provider in ("twelvedata", "polygon", "alphavantage", "yfinance", "synthetic")


def test_available_providers_includes_keyless():
    avail = data_provider.available_providers()
    assert "yfinance" in avail
    assert "synthetic" in avail


def test_ohlc_integrity():
    df = data_provider._fetch_synthetic("2024-01-01", "2024-01-10", 500)
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()


def test_has_backtest_key_reflects_env(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API_KEY_BACKTEST", raising=False)
    monkeypatch.delenv("TWELVEDATA_API_KEY_BACKTEST_2", raising=False)
    assert data_provider.has_backtest_key() is False

    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    assert data_provider.has_backtest_key() is True


def test_get_m5_records_error_when_paginated_range_fetch_raises(monkeypatch):
    """Avant ce correctif, une exception dans le fetch paginé (backtest range)
    était avalée silencieusement (except Exception: pass) — get_last_errors()
    n'avait alors jamais la vraie cause du fallback synthétique, rendant le
    diagnostic "jours calmes vs early exit" impossible à déboguer depuis
    l'extérieur (voir diag_real_trades_by_day_volatility::data_provider_debug).

    Symbole unique + cache vidé explicitement : "synthetic" est juste une entrée
    de plus dans _AUTO_ORDER (le fallback ultime n'est qu'un filet de sécurité
    théorique) — un run réussi met donc bien le résultat synthétique en cache
    disque partagé entre invocations pytest (XAU_DB_PATH pointe vers un fichier
    temp fixe, pas recréé à chaque run). Sans ce nettoyage, relancer ce test
    après un premier succès lit le cache et ne touche plus aux mocks du tout,
    laissant get_last_errors() vide silencieusement."""
    symbol = "XAUUSD_TESTRANGEERR"
    db.init_db()
    db.ohlcv_cache_clear(symbol)
    monkeypatch.setenv("XAU_DATA_PROVIDER", "auto")
    # order = [...] n'inclut "twelvedata" que si la clé LIVE (pas juste backtest)
    # est présente (voir get_m5 : _KEY_ENV["twelvedata"] = TWELVEDATA_API_KEY) —
    # cohérent avec la config réelle de l'utilisateur (les deux existent).
    monkeypatch.setenv("TWELVEDATA_API_KEY", "live_key")
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    data_provider._last_errors.clear()

    def _raise_range(*a, **k):
        raise RuntimeError("429 Too Many Requests (key rotated)")

    def _raise_single(*a, **k):
        raise RuntimeError("no data")

    def _raise_yf(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(data_provider, "_fetch_twelvedata_range", _raise_range)
    monkeypatch.setattr(data_provider, "_fetch_twelvedata", _raise_single)
    # _PROVIDERS est un dict figé au chargement du module — patcher juste
    # _fetch_yfinance ne suffit pas, la boucle providers appelle _PROVIDERS[name].
    monkeypatch.setitem(data_provider._PROVIDERS, "yfinance", _raise_yf)

    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-03-01", symbol=symbol)
    assert provider == "synthetic"
    assert len(df) > 0

    errs = data_provider.get_last_errors()
    assert f"{symbol}:twelvedata_range" in errs
    assert "429" in errs[f"{symbol}:twelvedata_range"]


def test_get_m5_records_insufficient_coverage_reason(monkeypatch):
    symbol = "XAUUSD_TESTCOVERAGE"
    db.init_db()
    db.ohlcv_cache_clear(symbol)
    monkeypatch.setenv("XAU_DATA_PROVIDER", "auto")
    monkeypatch.setenv("TWELVEDATA_API_KEY", "live_key")
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    data_provider._last_errors.clear()

    # Ne couvre que 5 jours sur les 2 mois demandés -> couverture insuffisante
    short_idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    partial_df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    }, index=short_idx)

    monkeypatch.setattr(data_provider, "_fetch_twelvedata_range", lambda *a, **k: partial_df)
    monkeypatch.setattr(data_provider, "_fetch_twelvedata", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    monkeypatch.setitem(data_provider._PROVIDERS, "yfinance",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))

    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-03-01", symbol=symbol)
    assert provider == "synthetic"

    errs = data_provider.get_last_errors()
    assert f"{symbol}:twelvedata_range" in errs
    assert "couverture insuffisante" in errs[f"{symbol}:twelvedata_range"]


def test_synthetic_fallback_is_never_cached(monkeypatch):
    """Le TTL du cache OHLCV (7 jours) suppose des données de marché réelles et
    immuables — "synthetic" traverse la même boucle providers que les vrais
    fournisseurs (voir get_m5) et se faisait donc mettre en cache comme eux. Un
    échec transitoire (clé invalide, rate limit) restait alors invisible pendant
    7 jours même après correction de la cause réelle, puisque get_m5() sert le
    cache avant même de retenter un fetch. Ce test vérifie qu'un résultat
    "synthetic" ne laisse plus aucune trace dans le cache disque."""
    symbol = "XAUUSD_TESTNOCACHE"
    db.init_db()
    db.ohlcv_cache_clear(symbol)
    monkeypatch.setenv("XAU_DATA_PROVIDER", "auto")
    monkeypatch.setenv("TWELVEDATA_API_KEY", "live_key")
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    data_provider._last_errors.clear()

    monkeypatch.setattr(data_provider, "_fetch_twelvedata_range",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")))
    monkeypatch.setattr(data_provider, "_fetch_twelvedata",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    monkeypatch.setitem(data_provider._PROVIDERS, "yfinance",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))

    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-03-01", symbol=symbol)
    assert provider == "synthetic"

    cache_key = data_provider._cache_key(symbol, "2024-01-01", "2024-03-01")
    assert db.ohlcv_cache_load(cache_key) is None


def _partial_range_df(days: int):
    idx = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    }, index=idx)


def _setup_partial_fetch(monkeypatch, symbol, days):
    db.init_db()
    db.ohlcv_cache_clear(symbol)
    monkeypatch.setenv("XAU_DATA_PROVIDER", "auto")
    monkeypatch.setenv("TWELVEDATA_API_KEY", "live_key")
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    data_provider._last_errors.clear()
    monkeypatch.setattr(data_provider, "_fetch_twelvedata_range",
                         lambda *a, **k: _partial_range_df(days))
    monkeypatch.setitem(data_provider._PROVIDERS, "yfinance",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))


def test_relaxed_min_coverage_accepts_partial_range(monkeypatch):
    """Une plage de quelques semaines perd facilement 2 jours aux bornes
    (week-end, ou marge demandée au-delà de maintenant) — 96% de couverture, ce
    qui échouait le seuil strict de 99% et faisait basculer sur le synthétique
    alors que les données réelles étaient là. Un appelant qui fait des lookups
    ponctuels doit pouvoir accepter ça explicitement."""
    symbol = "XAUUSD_TESTRELAXED"
    # 49 jours de données réelles sur les 51 demandés ≈ 96% — cas réel observé.
    _setup_partial_fetch(monkeypatch, symbol, days=49)

    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-02-21",
                                         symbol=symbol, min_coverage=0.85)
    assert provider == "twelvedata"
    assert len(df) > 0


def test_strict_min_coverage_still_rejects_same_partial_range(monkeypatch):
    symbol = "XAUUSD_TESTSTRICT"
    _setup_partial_fetch(monkeypatch, symbol, days=49)

    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-02-21", symbol=symbol)
    assert provider == "synthetic"  # défaut 0.99 inchangé pour le walk-forward


def test_relaxed_acceptance_is_never_written_to_cache(monkeypatch):
    """Le cache est partagé par clé symbol+start+end : une entrée partielle
    acceptée via min_coverage assoupli serait relue telle quelle par un
    walk-forward, qui exige 99%."""
    symbol = "XAUUSD_TESTRELAXEDCACHE"
    _setup_partial_fetch(monkeypatch, symbol, days=49)

    df, provider = data_provider.get_m5(start="2024-01-01", end="2024-02-21",
                                         symbol=symbol, min_coverage=0.85)
    assert provider == "twelvedata"

    cache_key = data_provider._cache_key(symbol, "2024-01-01", "2024-02-21")
    assert db.ohlcv_cache_load(cache_key) is None


def test_fetch_twelvedata_range_rotates_key_on_429(monkeypatch):
    """A 429 on the first backtest key must roll over to the second one
    (TWELVEDATA_API_KEY_BACKTEST_2) instead of just re-hitting the same key."""
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST", "key_a")
    monkeypatch.setenv("TWELVEDATA_API_KEY_BACKTEST_2", "key_b")
    data_provider._td_key_index["backtest"][0] = 0

    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="5min", tz="UTC")
    values = [
        {"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "open": "100", "high": "101",
         "low": "99", "close": "100", "volume": "10"}
        for t in idx
    ]

    resp_429 = MagicMock(status_code=429)
    resp_ok = MagicMock(status_code=200)
    resp_ok.json.return_value = {"status": "ok", "values": values}
    resp_ok.raise_for_status.return_value = None

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["apikey"])
        return resp_429 if params["apikey"] == "key_a" else resp_ok

    with patch("data_provider.requests.get", side_effect=fake_get), \
         patch("data_provider._td_throttle", return_value=None):
        df = data_provider._fetch_twelvedata_range("2024-01-01", "2024-01-03", "XAUUSD")

    assert len(df) > 0
    assert calls[0] == "key_a"   # tente d'abord la clé initiale
    assert "key_b" in calls      # tourne vers la 2e clé au lieu de rester bloqué sur key_a
