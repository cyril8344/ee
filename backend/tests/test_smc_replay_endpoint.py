"""Endpoint du replay SMC (§11.3) et proxy de données du trainer RL.

Le replay est le seul test d'acceptation que la spec pose elle-même. Il doit
tourner là où les clés de données réelles existent — d'où un endpoint plutôt
qu'un script local : sur des bougies synthétiques le comptage ne dit rien, une
marche aléatoire n'ayant pas de structure de marché.
"""
import os
import time

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")
os.environ.setdefault("ADMIN_USERNAME", "a")
os.environ.setdefault("ADMIN_PASSWORD", "b")

from fastapi.testclient import TestClient

import main
from rl_trainer import RLTrainer


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture(scope="module")
def auth(client):
    tok = client.post("/api/login",
                      json={"username": "a", "password": "b"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _wait_done(client, auth, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get("/api/smc/replay", headers=auth).json()
        if not s["running"]:
            return s
        time.sleep(0.5)
    raise AssertionError("le replay n'a pas terminé dans le délai imparti")


def test_replay_runs_and_returns_the_spec_verdict(client, auth):
    r = client.post("/api/smc/replay",
                    json={"symbol": "XAUUSD", "m15_bars": 400}, headers=auth)
    assert r.json()["ok"] is True
    s = _wait_done(client, auth)
    assert s["error"] is None
    assert s["result"] is not None
    assert "verdict" in s["result"]
    assert "per_week" in s["result"]


def test_the_provenance_of_the_data_is_reported(client, auth):
    """Un verdict calculé sur des bougies simulées serait pris pour une mesure du
    marché réel. La provenance fait donc partie du résultat, pas d'un log."""
    client.post("/api/smc/replay",
                json={"symbol": "XAUUSD", "m15_bars": 400}, headers=auth)
    s = _wait_done(client, auth)
    assert s["provider"] == "synthetic"
    assert s["synthetic"] is True


def test_two_replays_do_not_run_at_once(client, auth):
    """Même garde que le walk-forward : deux replays concurrents se disputeraient
    le quota de l'API de données."""
    main._smc_replay_state["running"] = True
    try:
        r = client.post("/api/smc/replay", json={"symbol": "XAUUSD"}, headers=auth)
        assert r.json()["ok"] is False
        assert "déjà en cours" in r.json()["message"]
    finally:
        main._smc_replay_state["running"] = False


def test_progress_is_exposed_while_running(client, auth):
    """Un replay de 6 mois prend plusieurs minutes : sans progression affichée,
    l'interface serait indistinguable d'une panne."""
    s = client.get("/api/smc/replay", headers=auth).json()
    for clef in ("running", "done", "total", "signals", "provider", "symbol"):
        assert clef in s


def test_the_endpoint_requires_authentication(client):
    assert client.post("/api/smc/replay", json={"symbol": "XAUUSD"}).status_code == 401
    assert client.get("/api/smc/replay").status_code == 401


# --------------------------------------------------------------------------- #
# Régression : le trainer RL demandait "ES" au fournisseur de données
# --------------------------------------------------------------------------- #
def test_es_rl_trainer_goes_through_the_spy_proxy():
    """« ES » n'existe pas chez le fournisseur. Avant que market_symbol() lève sur
    un symbole inconnu, il renvoyait SILENCIEUSEMENT des données XAU/USD : l'agent
    RL « ES » s'entraînait sur de l'or. Il passe maintenant par le proxy SPY ×10,
    comme la boucle live et pretrain_es."""
    cfg = main.MARKET_CONFIG["ES"]
    t = RLTrainer(symbol="ES", data_symbol=cfg.get("data_symbol"),
                  price_scale=cfg.get("price_scale", 1.0))
    assert t.symbol == "ES"
    assert t.data_symbol == "SPY"
    assert t.price_scale == 10.0


def test_other_markets_keep_their_own_symbol_unchanged():
    for sym in ("XAUUSD", "EURUSD"):
        cfg = main.MARKET_CONFIG[sym]
        t = RLTrainer(symbol=sym, data_symbol=cfg.get("data_symbol"),
                      price_scale=cfg.get("price_scale", 1.0))
        assert t.data_symbol == sym
        assert t.price_scale == 1.0


def test_the_rl_trainer_never_silently_falls_back_to_gold():
    """Garde-fou : demander « ES » sans proxy doit lever, pas rendre de l'or."""
    import data_provider
    with pytest.raises(KeyError):
        data_provider.market_symbol("ES")
