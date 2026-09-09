"""Lire un graphique en bougies depuis une capture d'écran, et y tracer les OB.

Le but : on envoie une capture MT5 (téléphone compris), on récupère la même
image avec les Order Blocks dessinés dessus. Aucune donnée de marché n'est
consultée — tout vient des pixels.

Comment ça marche, et pourquoi ça marche :

1. **Les bougies sont saturées, le fond ne l'est pas.** Sur la capture de
   référence, haussier = (56,158,177), baissier = (218,83,76), fond blanc. Un
   simple seuil de saturation les sépare, quel que soit le thème clair/sombre.

2. **Les bougies sont PÉRIODIQUES, les indicateurs ne le sont pas.** C'est le
   point clé. Une capture MT5 mobile contient souvent un sous-panneau
   d'indicateurs qui utilise exactement les mêmes couleurs (un ZigZag vert et
   rouge, par exemple). Impossible de les distinguer par la couleur. Mais un
   graphique en bougies a un pas constant — mesuré à 9 px sur la référence, avec
   ses harmoniques à 18, 27, 36 — alors qu'une ligne d'indicateur n'a aucune
   périodicité. On choisit donc la bande horizontale dont le profil de colonnes
   a la plus forte autocorrélation.

3. **Corps et mèches se lisent dans la colonne.** À l'intérieur d'un pas, les
   colonnes centrales portent le corps (run vertical large), les colonnes du
   milieu portent la mèche. Le haut et le bas de l'ensemble donnent high/low ;
   les bords du corps donnent open/close, l'ordre venant de la couleur.

Ce que ce module NE fait PAS : convertir les pixels en prix. Il faudrait lire
l'axe, or il n'y a pas d'OCR ici — et un OCR sur un axe de graphique se trompe
d'un chiffre sans prévenir. La calibration se fait donc en donnant deux prix et
leurs positions (cf. `calibrate`), ce qui est exact par construction. Sans
calibration, les zones sont rendues en pixels : suffisant pour les voir
dessinées, insuffisant pour les recopier en chiffres — et c'est dit comme tel.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Seuil de saturation au-dessus duquel un pixel est « de la couleur d'une
# bougie ». 60 sépare largement les bougies mesurées du fond et du gris des
# grilles, sans dépendre du thème.
SAT_MIN = 60
# Écart minimal entre canaux pour trancher haussier/baissier.
CHANNEL_MARGIN = 25
# Une bande de moins de 60 px de haut ne peut pas porter un graphique lisible.
MIN_BAND_HEIGHT = 60
# Pas de bougie plausible, en pixels.
PITCH_MIN, PITCH_MAX = 4, 60
# En dessous, le motif n'est pas assez périodique pour être un graphique.
MIN_PERIODICITY = 0.25
# Une rangée colorée sur plus de cette fraction de la largeur est une LIGNE
# HORIZONTALE, pas des bougies. Au pas mesuré (corps de 5 px tous les 9), une
# rangée traversée par tous les corps plafonne à ~55 % ; la ligne de prix
# courante de MT5, elle, va d'un bord à l'autre. Sans ce filtre elle est lue
# comme le plus haut de CHAQUE bougie qu'elle traverse — constaté sur la capture
# de référence : `high` identique sur des dizaines de bougies d'affilée.
HLINE_COVERAGE = 0.70


@dataclass
class Extraction:
    """Bougies reconstituées + de quoi rendre la lecture vérifiable.

    `periodicity` et `n_candles` sont renvoyés pour que l'appelant puisse dire
    « je n'ai pas su lire cette image » au lieu de rendre des bougies inventées.
    """
    df: pd.DataFrame                  # OHLC en pixels (y inversé : haut = grand)
    band: Tuple[int, int]             # bande verticale retenue
    pitch: float
    periodicity: float
    n_candles: int
    width: int
    height: int
    x_start: int = 0            # bord gauche des bougies
    x_end: int = 0              # dernière bougie
    x_plot_end: int = 0         # bord de la zone de tracé (avant l'axe des prix)


def load_rgb(data: bytes) -> np.ndarray:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(im).astype(int)


def candle_masks(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tout, haussier, baissier) — par saturation puis dominance de canal.

    « Haussier » = le vert domine (vert, teal, cyan selon les thèmes).
    « Baissier » = le rouge domine. Les thèmes noir/blanc ne sont pas gérés :
    sans couleur, rien ne distingue une bougie d'une grille.
    """
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    sat = a.max(2) - a.min(2)
    colored = sat > SAT_MIN
    up = colored & (g > r + CHANNEL_MARGIN)
    down = colored & (r > g + CHANNEL_MARGIN) & (r > b + CHANNEL_MARGIN)
    return (up | down), up, down


def strip_horizontal_lines(mask: np.ndarray, *masks: np.ndarray
                           ) -> Tuple[np.ndarray, ...]:
    """Efface les rangées traversées de part en part : lignes de prix, niveaux.

    Appliqué à tous les masques d'un coup pour qu'ils restent cohérents entre
    eux — sinon une rangée disparaîtrait du masque global mais resterait dans
    le masque haussier, et la couleur de la bougie serait faussée.
    """
    xs = np.nonzero(mask.any(0))[0]
    if len(xs) == 0:
        return (mask,) + masks
    span = xs.max() - xs.min() + 1
    lines = mask[:, xs.min(): xs.max() + 1].sum(1) > HLINE_COVERAGE * span
    out = []
    for m in (mask,) + masks:
        m = m.copy()
        m[lines] = False
        out.append(m)
    return tuple(out)


def _bands(mask: np.ndarray, min_px_per_row: int = 3) -> List[Tuple[int, int]]:
    """Bandes horizontales contenant de la couleur, séparées par du vide."""
    rows = mask.sum(1)
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for y, n in enumerate(rows):
        if n > min_px_per_row and start is None:
            start = y
        elif n <= min_px_per_row and start is not None:
            if y - start >= MIN_BAND_HEIGHT:
                out.append((start, y))
            start = None
    if start is not None and len(rows) - start >= MIN_BAND_HEIGHT:
        out.append((start, len(rows)))
    return out


def _pitch_of(mask: np.ndarray, y0: int, y1: int) -> Tuple[float, float]:
    """Pas dominant et force de périodicité d'une bande, par autocorrélation.

    C'est la mesure qui distingue un graphique en bougies d'un panneau
    d'indicateurs : les bougies se répètent à intervalle fixe, une courbe non.
    """
    prof = mask[y0:y1].sum(0).astype(float)
    xs = np.nonzero(prof)[0]
    if len(xs) < PITCH_MIN * 4:
        return 0.0, 0.0
    p = prof[xs.min(): xs.max() + 1]
    p = p - p.mean()
    if not np.any(p):
        return 0.0, 0.0
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    best_lag, best = 0.0, 0.0
    for lag in range(PITCH_MIN, min(PITCH_MAX, len(ac) - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > best:
            best_lag, best = float(lag), float(ac[lag])
    return best_lag, best


def grow_band(mask: np.ndarray, y0: int, y1: int,
              tolerance: int = 4) -> Tuple[int, int]:
    """Étend la bande aux extrêmes du graphique, une fois qu'elle est identifiée.

    La sélection par périodicité a besoin du cœur dense de la bande, mais ce
    cœur s'arrête là où peu de bougies montent. Les extrêmes du graphique —
    précisément les plus hauts et les plus bas, donc ce qui définit la structure
    — n'y sont représentés que par une ou deux bougies et tombent sous le seuil.
    Constaté sur la capture de référence : la bande s'arrêtait 100 px trop bas et
    tronquait le sommet. On repart donc du cœur et on grimpe tant qu'on trouve de
    la couleur, en tolérant quelques rangées vides (les mèches sont fines).
    """
    rows = mask.sum(1)
    n = len(rows)
    top = y0
    gap = 0
    y = y0 - 1
    while y >= 0 and gap <= tolerance:
        if rows[y] > 0:
            top, gap = y, 0
        else:
            gap += 1
        y -= 1
    bot = y1
    gap = 0
    y = y1
    while y < n and gap <= tolerance:
        if rows[y] > 0:
            bot, gap = y + 1, 0
        else:
            gap += 1
        y += 1
    return top, bot


def pick_chart_band(mask: np.ndarray) -> Optional[Tuple[int, int, float, float]]:
    """La bande la plus périodique = le graphique en bougies.

    Renvoie None quand aucune bande n'atteint MIN_PERIODICITY : mieux vaut dire
    « je n'ai pas trouvé de graphique » que rendre les oscillations d'un
    indicateur en les appelant des bougies.
    """
    best = None
    for y0, y1 in _bands(mask):
        pitch, strength = _pitch_of(mask, y0, y1)
        if pitch and strength >= MIN_PERIODICITY:
            if best is None or strength > best[3]:
                best = (y0, y1, pitch, strength)
    return best


SOLID_FILL = 0.85


def trim_to_candles(band_mask: np.ndarray) -> Tuple[int, int, int]:
    """Bornes horizontales des vraies bougies, blocs pleins exclus.

    L'étiquette de prix courante de MT5 est un rectangle teal collé à droite. Sans
    filtre, elle est prise pour des bougies et les zones dessinées débordent sur
    les chiffres de l'axe — précisément ce qu'on veut pouvoir lire.

    Deux discriminants ont été essayés et rejetés sur la capture de référence :
    « chercher un creux régulier » (le profil retombe à zéro APRÈS l'étiquette,
    ce qui compte à tort comme un creux), puis « les groupes larges ne sont pas
    des bougies » (un rallye de bougies qui se touchent forme un groupe large, et
    la fin du graphique était tronquée à x=883 au lieu de ~1180).

    Le bon critère est le remplissage : une étiquette est un rectangle **plein**,
    alors qu'un paquet de bougies, même collées, laisse du fond au-dessus et en
    dessous des corps — les mèches ne remplissent pas leur boîte englobante.
    """
    prof = band_mask.sum(0)
    xs = np.nonzero(prof)[0]
    if len(xs) == 0:
        return 0, 0, 0

    groupes: List[Tuple[int, int]] = []
    debut = None
    for x in range(len(prof)):
        if prof[x] > 0 and debut is None:
            debut = x
        elif prof[x] == 0 and debut is not None:
            groupes.append((debut, x - 1))
            debut = None
    if debut is not None:
        groupes.append((debut, len(prof) - 1))

    gardes: List[Tuple[int, int]] = []
    rejets: List[Tuple[int, int]] = []
    for s, e in groupes:
        bloc = band_mask[:, s:e + 1]
        ys = np.nonzero(bloc.any(1))[0]
        if len(ys) == 0:
            continue
        boite = (ys.max() - ys.min() + 1) * (e - s + 1)
        etroit = (e - s + 1) <= 2 * max(1, int(round(band_mask.shape[0] / 100)))
        if boite and (bloc.sum() / boite < SOLID_FILL or etroit):
            gardes.append((s, e))
        else:
            rejets.append((s, e))
    if not gardes:
        return int(xs.min()), int(xs.max()), int(xs.max())

    x0, x1 = gardes[0][0], gardes[-1][1]
    # Bord de la zone de TRACÉ : les bougies s'arrêtent avant la marge droite de
    # MT5. Prolonger les zones jusqu'à la marge est plus parlant — c'est là qu'on
    # attend le retest — mais s'arrêter avant l'étiquette de prix, sinon on
    # recouvre les chiffres. Le premier bloc plein à droite marque cet axe.
    apres = [s for s, _ in rejets if s > x1]
    plot_end = (min(apres) - 8) if apres else x1
    return x0, x1, max(x1, plot_end)


def extract(data: bytes) -> Optional[Extraction]:
    """Bougies d'une capture, en coordonnées PIXELS (y inversé).

    L'axe y de l'image descend, les prix montent : on renvoie donc `hauteur - y`,
    pour que « plus grand » veuille dire « plus haut » comme partout ailleurs.
    """
    a = load_rgb(data)
    h, w, _ = a.shape
    allm, up, down = candle_masks(a)
    allm, up, down = strip_horizontal_lines(allm, up, down)
    band = pick_chart_band(allm)
    if band is None:
        return None
    y0, y1, pitch, strength = band
    y0, y1 = grow_band(allm, y0, y1)

    sub_all = allm[y0:y1]
    sub_up = up[y0:y1]
    sub_down = down[y0:y1]
    prof = sub_all.sum(0)
    xs = np.nonzero(prof)[0]
    if len(xs) == 0:
        return None
    x_start, x_end, x_plot_end = trim_to_candles(sub_all)
    if x_end <= x_start:
        return None

    rows: List[Dict[str, Any]] = []
    step = pitch
    n_slots = int((x_end - x_start) / step) + 1
    for k in range(n_slots):
        cx0 = int(round(x_start + k * step))
        cx1 = int(round(x_start + (k + 1) * step))
        cx1 = min(cx1, x_end + 1)
        if cx1 <= cx0:
            continue
        cols_all = sub_all[:, cx0:cx1]
        if cols_all.sum() == 0:
            continue

        ys = np.nonzero(cols_all.any(1))[0]
        top, bot = int(ys.min()), int(ys.max())        # mèches

        # Corps = les lignes où au moins la moitié des colonnes du créneau sont
        # colorées. Une mèche n'occupe qu'une ou deux colonnes ; un corps les
        # occupe presque toutes.
        dense = cols_all.sum(1) >= max(2, int(0.5 * (cx1 - cx0)))
        dys = np.nonzero(dense)[0]
        if len(dys):
            b_top, b_bot = int(dys.min()), int(dys.max())
        else:
            b_top, b_bot = top, bot

        n_up = int(sub_up[:, cx0:cx1].sum())
        n_down = int(sub_down[:, cx0:cx1].sum())
        bullish = n_up >= n_down

        # Repasser en « prix » : y descend dans l'image, le prix monte.
        high = float(h - (y0 + top))
        low = float(h - (y0 + bot))
        body_hi = float(h - (y0 + b_top))
        body_lo = float(h - (y0 + b_bot))
        open_, close = (body_lo, body_hi) if bullish else (body_hi, body_lo)

        rows.append({"x": (cx0 + cx1) / 2 + 0.0, "open": open_, "high": high,
                     "low": low, "close": close, "volume": 0.0,
                     "bullish": bullish})

    if len(rows) < 10:
        return None

    df = pd.DataFrame(rows)
    # Un index temporel factice : les modules en aval datent leurs objets, et un
    # index entier suffit tant qu'on ne prétend pas connaître les vraies heures.
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="5min", tz="UTC")
    return Extraction(df=df[["open", "high", "low", "close", "volume"]].copy(),
                      band=(y0, y1), pitch=pitch, periodicity=strength,
                      n_candles=len(df), width=w, height=h,
                      x_start=x_start, x_end=x_end, x_plot_end=x_plot_end)


# --------------------------------------------------------------------------- #
# Calibration prix ↔ pixels
# --------------------------------------------------------------------------- #
def calibrate(y_top: float, price_top: float,
              y_bottom: float, price_bottom: float, height: int):
    """Deux points suffisent : l'axe d'un graphique est linéaire.

    Volontairement pas d'OCR de l'axe. Il n'y en a pas sur ce serveur, et surtout
    un OCR se trompe d'un chiffre sans le signaler — sur un prix, ça donne un
    niveau faux qui a l'air juste. Deux valeurs saisies sont exactes.
    """
    py_top, py_bottom = height - y_top, height - y_bottom
    if py_top == py_bottom:
        raise ValueError("les deux points de calibration sont sur la même ligne")
    k = (price_top - price_bottom) / (py_top - py_bottom)

    def to_price(py: float) -> float:
        return price_bottom + (py - py_bottom) * k

    return to_price


# --------------------------------------------------------------------------- #
# Rendu
# --------------------------------------------------------------------------- #
def annotate(data: bytes, ext: Extraction, zones: List[Any]) -> bytes:
    """Renvoie la capture avec un rectangle par zone, en PNG.

    Deux règles de tracé, apprises en regardant le premier rendu :

    - **On s'arrête au bord droit des bougies**, pas au bord de l'image. Sinon
      les rectangles recouvrent l'axe des prix, c'est-à-dire précisément les
      chiffres qu'on veut lire.
    - **Une zone déjà mitigée est dessinée en pointillé pâle.** Elle a une valeur
      historique — elle explique le mouvement — mais il n'y a plus de retest à en
      attendre, et la montrer comme une zone vivante induirait en erreur.
    """
    from PIL import Image, ImageDraw

    im = Image.open(io.BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    h = ext.height
    right = ext.x_plot_end or ext.x_end or im.width

    for z in zones:
        y_hi = int(round(h - z.high))     # retour en coordonnées image
        y_lo = int(round(h - z.low))
        if y_lo < y_hi:
            y_hi, y_lo = y_lo, y_hi
        if y_lo - y_hi < 2:               # zone d'un pixel : illisible
            y_lo = y_hi + 2

        haussier = z.side == "bullish"
        vive = z.state == "active"
        base = (0, 170, 120) if haussier else (220, 60, 50)
        outline = base + (255 if vive else 130,)
        fill = base + (70 if vive else 25,)

        px0 = int(round(_x_of(ext, min(z.index, ext.n_candles - 1))))
        draw.rectangle([px0, y_hi, right, y_lo],
                       fill=fill, outline=outline, width=4 if vive else 2)

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _x_of(ext: Extraction, k: int) -> float:
    """Colonne image du k-ième créneau de bougie."""
    return ext.x_start + ext.pitch * k + ext.pitch / 2
