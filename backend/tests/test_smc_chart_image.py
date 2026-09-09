"""Lecture d'un graphique en bougies depuis une image.

Les images de test sont **générées** ici plutôt que stockées : un PNG figé dans
le dépôt ne dit pas ce qu'il est censé contenir, alors qu'un rendu programmé
porte la vérité terrain avec lui — on sait exactement quelles bougies on a
dessinées, donc on peut vérifier qu'on les relit.

Les trois cas de rejet testés ci-dessous ne sont pas théoriques : ce sont les
trois défauts trouvés en regardant le rendu sur une vraie capture MT5 mobile.
"""
import io
import os

import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

pytest.importorskip("PIL")

from PIL import Image, ImageDraw  # noqa: E402

from smc import chart_image as ci  # noqa: E402

UP = (56, 158, 177)        # teal, mesuré sur la capture de référence
DOWN = (218, 83, 76)
BG = (255, 255, 255)


def _render(ohlc, *, pitch=9, body=5, top=40, height=400, width=None,
            left=10, bg=BG):
    """Dessine des bougies. `ohlc` en unités de prix croissant vers le haut."""
    n = len(ohlc)
    width = width or left + n * pitch + 60
    im = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    lo = min(c[2] for c in ohlc)
    hi = max(c[1] for c in ohlc)
    span = max(1e-9, hi - lo)
    plot_h = height - top - 40

    def y(p):
        return int(top + (hi - p) / span * plot_h)

    for k, (o, h, l, c) in enumerate(ohlc):
        cx = left + k * pitch + pitch // 2
        col = UP if c >= o else DOWN
        d.line([(cx, y(h)), (cx, y(l))], fill=col, width=1)
        x0 = cx - body // 2
        y0, y1 = sorted((y(o), y(c)))
        d.rectangle([x0, y0, x0 + body - 1, max(y1, y0 + 1)], fill=col)
    return im


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _serie(n=60):
    """Une série qui monte, redescend et remonte — de vraies structures."""
    out = []
    p = 100.0
    for k in range(n):
        if k < n // 3:
            p += 1.2
        elif k < 2 * n // 3:
            p -= 1.5
        else:
            p += 1.0
        o = p - 0.6 if k % 2 else p + 0.6
        c = p
        h = max(o, c) + 0.8
        l = min(o, c) - 0.8
        out.append((o, h, l, c))
    return out


# --------------------------------------------------------------------------- #
# Extraction nominale
# --------------------------------------------------------------------------- #
def test_candles_are_recovered_from_the_pixels():
    serie = _serie(60)
    ext = ci.extract(_png(_render(serie)))
    assert ext is not None
    assert abs(ext.n_candles - len(serie)) <= 2      # bords à ±1 créneau
    assert ext.pitch == pytest.approx(9, abs=1)
    assert ext.periodicity > ci.MIN_PERIODICITY


def test_the_recovered_ohlc_is_internally_coherent():
    ext = ci.extract(_png(_render(_serie(60))))
    df = ext.df
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_the_shape_of_the_move_survives_the_round_trip():
    """On ne peut pas exiger les prix exacts — le rendu quantifie en pixels.
    Mais la FORME doit tenir : le sommet et le creux doivent tomber au bon
    endroit, sinon la structure lue serait celle d'un autre graphique."""
    serie = _serie(60)
    ext = ci.extract(_png(_render(serie, height=600)))
    df = ext.df.reset_index(drop=True)
    k_haut_attendu = max(range(len(serie)), key=lambda i: serie[i][1])
    k_bas_attendu = min(range(len(serie)), key=lambda i: serie[i][2])
    tol = 3
    assert abs(int(df["high"].idxmax()) - k_haut_attendu) <= tol
    assert abs(int(df["low"].idxmin()) - k_bas_attendu) <= tol


def test_candle_colour_decides_the_direction():
    """Une bougie verte a close > open, une rouge l'inverse. Sans ça, open et
    close seraient intervertis et toute la structure serait lue à l'envers."""
    hausse = [(100, 106, 99, 105)] * 30
    baisse = [(105, 106, 99, 100)] * 30
    eh = ci.extract(_png(_render(hausse)))
    eb = ci.extract(_png(_render(baisse)))
    assert (eh.df["close"] >= eh.df["open"]).mean() > 0.9
    assert (eb.df["close"] <= eb.df["open"]).mean() > 0.9


# --------------------------------------------------------------------------- #
# Les trois défauts trouvés sur une vraie capture
# --------------------------------------------------------------------------- #
def test_a_full_width_price_line_is_not_read_as_a_high():
    """Défaut n°1. La ligne de prix courante de MT5 est de la même couleur que
    les bougies : elle était lue comme le plus haut de CHACUNE d'elles, donnant
    un `high` identique sur des dizaines de bougies d'affilée."""
    im = _render(_serie(60))
    d = ImageDraw.Draw(im)
    d.line([(0, 120), (im.width, 120)], fill=UP, width=2)
    ext = ci.extract(_png(im))
    assert ext is not None
    # Sans le filtre, le high serait constant sur toutes les bougies traversées.
    assert ext.df["high"].nunique() > len(ext.df) // 3


def test_the_price_tag_block_is_excluded_from_the_candle_span():
    """Défaut n°2. L'étiquette de prix est un rectangle plein de la couleur des
    bougies, collé à droite : les zones tracées débordaient dessus, c'est-à-dire
    sur les chiffres qu'on veut lire."""
    serie = _serie(60)
    im = _render(serie, width=10 + 60 * 9 + 260)
    d = ImageDraw.Draw(im)
    x_tag = 10 + 60 * 9 + 60
    d.rectangle([x_tag, 150, x_tag + 180, 220], fill=UP)   # bloc plein
    ext = ci.extract(_png(im))
    assert ext is not None
    assert ext.x_end < x_tag
    assert ext.x_plot_end <= x_tag


def test_a_non_periodic_indicator_panel_is_not_mistaken_for_candles():
    """Défaut n°3, le plus vicieux : un sous-panneau d'indicateurs utilise les
    MÊMES couleurs. Rien ne le distingue par la couleur — c'est la périodicité
    qui tranche, les bougies ayant un pas fixe et une courbe non."""
    # Deux panneaux SÉPARÉS, comme sur une capture MT5 réelle : le graphique en
    # haut, l'indicateur en bas, un blanc entre les deux.
    serie = _serie(60)
    chart = _render(serie, height=420)
    im = Image.new("RGB", (chart.width, 760), BG)
    im.paste(chart, (0, 0))
    d = ImageDraw.Draw(im)
    pts = [(10 + k * 9, 560 + int(90 * ((k * 7) % 13) / 13)) for k in range(60)]
    d.line(pts, fill=UP, width=3)
    pts2 = [(10 + k * 9, 620 + int(80 * ((k * 5) % 11) / 11)) for k in range(60)]
    d.line(pts2, fill=DOWN, width=3)

    ext = ci.extract(_png(im))
    assert ext is not None
    y0, y1 = ext.band
    assert y1 < 520, f"la bande {ext.band} déborde sur le panneau d'indicateurs"
    assert ext.n_candles >= 50, "le graphique du haut doit rester lisible"


# --------------------------------------------------------------------------- #
# Refus explicites
# --------------------------------------------------------------------------- #
def test_an_image_without_a_chart_returns_none():
    """Mieux vaut « je n'ai pas su lire cette image » que des bougies inventées
    — une lecture fausse produirait des zones qui ont l'air sérieuses."""
    im = Image.new("RGB", (400, 300), BG)
    assert ci.extract(_png(im)) is None


def test_too_few_candles_returns_none():
    assert ci.extract(_png(_render(_serie(6)))) is None


# --------------------------------------------------------------------------- #
# Calibration prix ↔ pixels
# --------------------------------------------------------------------------- #
def test_calibration_is_linear_and_exact_on_its_two_points():
    """Deux points suffisent, et sont exacts par construction — contrairement à
    un OCR de l'axe, qui se trompe d'un chiffre sans le signaler."""
    to_price = ci.calibrate(y_top=100, price_top=2000.0,
                            y_bottom=500, price_bottom=1000.0, height=600)
    assert to_price(600 - 100) == pytest.approx(2000.0)
    assert to_price(600 - 500) == pytest.approx(1000.0)
    assert to_price(600 - 300) == pytest.approx(1500.0)


def test_calibration_refuses_two_points_on_the_same_row():
    with pytest.raises(ValueError):
        ci.calibrate(100, 2000.0, 100, 1000.0, 600)


# --------------------------------------------------------------------------- #
# Rendu
# --------------------------------------------------------------------------- #
def test_annotation_returns_a_png_of_the_same_size():
    from smc import zones
    from smc.structure import find_events

    data = _png(_render(_serie(80), height=600))
    ext = ci.extract(data)
    ev = find_events(ext.df, 2)
    obs = zones.find_order_blocks(ext.df, "M15", events=ev)
    zones.update_states(ext.df, obs)

    out = ci.annotate(data, ext, obs)
    im = Image.open(io.BytesIO(out))
    assert im.format == "PNG"
    assert im.size == (ext.width, ext.height)


def test_annotation_never_paints_over_the_price_axis():
    """Le rectangle s'arrête au bord de la zone de tracé : recouvrir l'axe
    reviendrait à masquer les seuls chiffres utilisables."""
    from smc import zones
    from smc.structure import find_events

    serie = _serie(80)
    im0 = _render(serie, height=600, width=10 + 80 * 9 + 260)
    d = ImageDraw.Draw(im0)
    x_tag = 10 + 80 * 9 + 60
    d.rectangle([x_tag, 150, x_tag + 180, 220], fill=UP)
    data = _png(im0)

    ext = ci.extract(data)
    ev = find_events(ext.df, 2)
    obs = zones.find_order_blocks(ext.df, "M15", events=ev)
    zones.update_states(ext.df, obs)
    assert ext.x_plot_end <= x_tag
