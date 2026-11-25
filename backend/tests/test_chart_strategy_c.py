"""/api/chart sur l'or en stratégie C : le graphique montre ce que la boucle
live calcule, avec la même fonction et les mêmes données."""
import os

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")
os.environ.setdefault("ADMIN_USERNAME", "a")
os.environ.setdefault("ADMIN_PASSWORD", "b")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture(scope="module")
def auth(client):
    tok = client.post("/api/login",
                      json={"username": "a", "password": "b"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.mark.parametrize("tf", ["M5", "M15", "H1"])
def test_gold_chart_carries_the_strategy_c_trace(client, auth, tf):
    assert main.MARKET_CONFIG["XAUUSD"]["default_strategy"] == "C"
    d = client.get(f"/api/chart?tf={tf}&symbol=XAUUSD", headers=auth).json()
    tr = d["sweep_c"]
    assert isinstance(tr, dict)
    # Soit toutes les étapes passent, soit l'étape bloquante est nommée.
    assert tr.get("signal") is True or tr.get("bloque_a")
    # Le seul Order Block affiché est celui de la C (jamais le détecteur A).
    assert len(d["order_blocks"]) <= 1
    if tr.get("order_block"):
        assert d["order_blocks"][0]["high"] == pytest.approx(tr["order_block"]["haut"], abs=1e-4)
    times = {c["time"] for c in d["candles"]}
    for m in d["markers"]:
        if m["type"] == "sweep":
            assert m["time"] in times


def test_other_markets_have_no_strategy_c_trace(client, auth):
    d = client.get("/api/chart?tf=M5&symbol=EURUSD", headers=auth).json()
    assert d["sweep_c"] is None
