"""Endpoint capture d'écran → Order Blocks.

Le test qui compte : la calibration prend DEUX POINTS (position + prix), pas deux
prix. Une première version acceptait « prix du haut / prix du bas » en supposant
qu'ils tombaient sur les extrémités des bougies. C'est faux — ce sont les
positions des étiquettes de l'axe, qui ne coïncident pas avec les bougies :
vérifié contre une capture réelle, la zone du haut sortait à 79 282 alors qu'elle
est à ~78 840. Des prix plausibles mais faux sont pires que pas de prix.
"""
import io
import os

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")
os.environ.setdefault("ADMIN_USERNAME", "a")
os.environ.setdefault("ADMIN_PASSWORD", "b")

pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402
from tests.test_smc_chart_image import _png, _render, _serie  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture(scope="module")
def auth(client):
    tok = client.post("/api/login",
                      json={"username": "a", "password": "b"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture(scope="module")
def capture():
    return _png(_render(_serie(80), height=600))


def _post(client, auth, data_img, **form):
    return client.post("/api/smc/chart-image",
                       files={"file": ("chart.png", data_img, "image/png")},
                       data={k: str(v) for k, v in form.items()},
                       headers=auth)


def test_zones_are_detected_and_the_image_comes_back_annotated(client, auth, capture):
    d = _post(client, auth, capture).json()
    assert d["ok"] is True
    assert d["candles"] > 50
    assert d["image"].startswith("data:image/png;base64,")
    assert isinstance(d["zones"], list)


def test_without_calibration_zones_stay_in_pixels_and_say_so(client, auth, capture):
    """Ne jamais présenter des pixels comme des prix : c'est la seule chose qui
    rendrait la sortie dangereuse plutôt qu'inutile."""
    d = _post(client, auth, capture).json()
    assert d["calibrated"] is False
    assert d["note"]
    for z in d["zones"]:
        assert "low_px" in z and "high_px" in z
        assert "low" not in z and "high" not in z


def test_calibration_needs_positions_not_just_prices(client, auth, capture):
    """Deux prix sans leurs positions ne suffisent pas — c'est l'erreur qui
    sortait 79 282 pour une zone réellement à 78 840."""
    r = _post(client, auth, capture, cal_price1=100.0, cal_price2=50.0)
    assert r.status_code == 400
    assert "incomplète" in r.json()["detail"].lower()


def test_two_calibration_points_give_prices(client, auth, capture):
    d = _post(client, auth, capture,
              cal_y1=100, cal_price1=2000.0, cal_y2=500, cal_price2=1000.0).json()
    assert d["calibrated"] is True
    assert d["note"] is None
    for z in d["zones"]:
        assert z["high"] >= z["low"]


def test_calibration_is_the_linear_map_through_the_two_points(client, auth, capture):
    """Les prix rendus doivent être ceux de la droite passant par les deux
    points — vérifiable sans connaître le détecteur."""
    from smc import chart_image as ci

    ext = ci.extract(capture)
    d = _post(client, auth, capture,
              cal_y1=100, cal_price1=2000.0, cal_y2=500, cal_price2=1000.0).json()
    to_price = ci.calibrate(100, 2000.0, 500, 1000.0, ext.height)
    for z, brut in zip(d["zones"], d["zones"]):
        attendu_bas = to_price(brut["low_px"])
        attendu_haut = to_price(brut["high_px"])
        assert z["low"] == pytest.approx(min(attendu_bas, attendu_haut), abs=0.02)
        assert z["high"] == pytest.approx(max(attendu_bas, attendu_haut), abs=0.02)


def test_two_points_on_the_same_row_are_refused(client, auth, capture):
    r = _post(client, auth, capture,
              cal_y1=300, cal_price1=2000.0, cal_y2=300, cal_price2=1000.0)
    assert r.status_code == 400


def test_an_image_without_candles_is_refused_explicitly(client, auth):
    """« Je n'ai pas su lire cette image » plutôt que des zones inventées."""
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (255, 255, 255)).save(buf, format="PNG")
    d = _post(client, auth, buf.getvalue()).json()
    assert d["ok"] is False
    assert "bougies" in d["error"]


def test_an_empty_upload_is_refused(client, auth):
    assert _post(client, auth, b"").status_code == 400


def test_the_endpoint_requires_authentication(client, capture):
    r = client.post("/api/smc/chart-image",
                    files={"file": ("c.png", capture, "image/png")})
    assert r.status_code == 401
