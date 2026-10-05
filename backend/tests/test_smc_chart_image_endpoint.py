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


def test_a_missing_server_dependency_is_not_reported_as_a_bad_image(
        client, auth, capture, monkeypatch):
    """Vécu en production : Pillow n'était déclaré que dans le requirements.txt
    de la racine, que le build Railway n'installe pas. L'utilisateur voyait
    « Lecture impossible : No module named 'PIL' » et cherchait le défaut du côté
    de sa capture. Une dépendance absente doit se dire comme telle."""
    from smc import chart_image as ci

    def _boom(_data):
        raise ImportError("No module named 'PIL'")

    monkeypatch.setattr(ci, "extract", _boom)
    r = _post(client, auth, capture)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "Dépendance manquante" in detail
    assert "ton image" in detail


def _capture_avec_ob():
    """Une capture contenant un vrai Order Block : range calme, sommet, bougie
    rouge, impulsion qui casse la structure, puis repli vers la zone."""
    def calm(p, n):
        # Corps réellement visible : une bougie dont open == close se dessine
        # sur 1 px de haut, ce qu'aucun graphique réel ne produit, et le
        # détecteur la traite alors comme un trait et non comme une bougie.
        return [(p - 0.4, p + 0.8, p - 0.8, p + 0.4) for _ in range(n)]
    rows = calm(100, 40) + [(100, 112, 99.6, 111)] + calm(100, 3)
    rows += [(100, 100.4, 95, 96)] + [(96, 130, 96, 128)] + calm(128, 10)
    rows += [(128, 128.4, 120, 121)] + calm(104, 10)
    return _png(_render(rows, height=700))


def test_the_response_carries_a_measured_analysis(client, auth):
    """Tout ce que l'endpoint affirme doit venir d'un nombre : c'est la
    contrainte posée par l'utilisateur."""
    d = _post(client, auth, _capture_avec_ob(), htf_bias="bullish").json()
    assert d["ok"] is True
    a = d["analyse"]
    assert a["bias"] == "bullish"
    assert a["atr"] > 0
    assert "eqh" in a and "eql" in a
    assert a["plan"] is not None or a["raison"]


def test_every_criterion_of_the_plan_shows_its_value_and_threshold(client, auth):
    a = _post(client, auth, _capture_avec_ob(), htf_bias="bullish").json()["analyse"]
    if a["plan"] is None:
        pytest.skip(f"pas de plan sur cette capture : {a['raison']}")
    for c in a["plan"]["criteres"]:
        assert c["valeur"] is not None
        assert c["seuil"] is not None
        assert c["detail"]


def test_the_plan_is_in_pixels_until_the_chart_is_calibrated(client, auth):
    """Mêmes garde-fous que pour les zones : des pixels ne doivent jamais être
    présentés comme des prix recopiables."""
    brut = _post(client, auth, _capture_avec_ob(), htf_bias="bullish").json()["analyse"]
    assert brut["en_prix"] is False

    cal = _post(client, auth, _capture_avec_ob(), htf_bias="bullish",
                cal_y1=100, cal_price1=4700.0,
                cal_y2=600, cal_price2=4300.0).json()["analyse"]
    assert cal["en_prix"] is True
    if cal["plan"]:
        assert 4200 < cal["plan"]["entry"] < 4800


def test_the_ratio_survives_the_change_of_scale(client, auth):
    """Le R:R est un rapport de deux distances : il ne doit pas bouger quand on
    passe des pixels aux prix, sinon l'une des deux lectures est fausse."""
    brut = _post(client, auth, _capture_avec_ob(), htf_bias="bullish").json()["analyse"]
    cal = _post(client, auth, _capture_avec_ob(), htf_bias="bullish",
                cal_y1=100, cal_price1=4700.0,
                cal_y2=600, cal_price2=4300.0).json()["analyse"]
    if brut["plan"] and cal["plan"]:
        assert brut["plan"]["rr"] == pytest.approx(cal["plan"]["rr"], rel=0.01)


def test_the_response_never_contains_a_verdict(client, auth):
    """Un « VALIDE » se recopie dans la plateforme ; un décompte de critères,
    non. Rien dans ce dépôt ne mesure si un trade va marcher."""
    d = _post(client, auth, _capture_avec_ob(), htf_bias="bullish").json()
    texte = repr(d["analyse"]).lower()
    for mot in ("valide", "verdict", "probabilit", "espérance", "esperance"):
        assert mot not in texte
