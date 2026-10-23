"""Les définitions, pas les réglages.

Ce fichier ne teste pas des seuils : il teste **ce qu'on appelle un pivot et ce
qu'on appelle un Order Block**. C'est le seul endroit du scanner où une erreur
ne se voit pas — un seuil mal choisi donne trop ou trop peu de zones, une
définition divergente donne des zones *ailleurs*, et le graphique de
l'utilisateur et le mien racontent alors deux histoires différentes sans que ni
l'un ni l'autre ait l'air en panne.

Trois divergences étaient en place, et elles allaient toutes dans le même sens
— plus de pivots, donc plus de cassures, donc plus de zones que ce qui est
réellement dessiné à l'écran.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import config, structure, zones


def _frame(rows, start="2026-01-05 08:00", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _calm(p, n, amp=0.5):
    return [(p, p + amp, p - amp, p) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Pivots : confirmation à droite seulement, et alternance stricte
# --------------------------------------------------------------------------- #
def _escalier_haussier():
    """Une montée en paliers suivie d'un repli, avec des creux intermédiaires :
    de quoi produire plusieurs pivots des deux côtés."""
    rows = _calm(100, 6)
    for p in (104, 108, 112):
        rows += [(p - 2, p, p - 3, p - 0.5)]
        rows += _calm(p - 2, 2)
    return _frame(rows + _calm(100, 8))


def _marche_aleatoire(seed, n=800):
    """Une marche aléatoire reproductible, avec de vraies bougies (open =
    clôture précédente). Les séries jouets de ce fichier isolent une règle ;
    celle-ci sert à mesurer un écart d'ensemble entre deux définitions."""
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    op = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(op, close) + np.abs(rng.normal(0, 0.8, n))
    low = np.minimum(op, close) - np.abs(rng.normal(0, 0.8, n))
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": op, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


def test_the_rolling_mode_never_produces_more_pivots_than_the_fractal():
    """Le chiffre annoncé dans `config`, tenu par un test — sinon personne ne
    saura qu'il a cessé d'être vrai.

    Piège de protocole, payé une fois : sur une série où `open == close`, les
    deux modes s'équivalent et la mesure conclut à tort qu'ils se valent. Il
    faut des bougies avec corps, d'où `_marche_aleatoire`.
    """
    ecarts = []
    for seed in (1, 7, 13, 42, 99, 123, 404):
        df = _marche_aleatoire(seed)
        sym = len(structure.find_pivots(df, 3, mode="symmetric"))
        roll = len(structure.find_pivots(df, 3, mode="rolling"))
        assert roll <= sym, f"seed {seed} : {roll} > {sym}"
        ecarts.append((sym - roll) / sym)
    assert max(ecarts) < 0.25, "réduction bien plus forte qu'annoncé : à remesurer"
    assert max(ecarts) > 0.05, "plus aucune réduction : à remesurer"


def test_the_two_order_block_definitions_point_at_different_candles():
    """La mesure qui justifie le changement. Si les deux définitions tombaient
    presque toujours sur la même bougie, s'aligner serait cosmétique. Elles
    divergent sur la majorité des cassures — le scanner et le graphique
    montraient donc deux zones à deux endroits, sans qu'aucun des deux ait
    l'air en panne."""
    memes = differentes = 0
    for seed in (1, 7, 13, 42, 99):
        df = _marche_aleatoire(seed)
        ev = structure.find_events(df, 3)
        par_cassure = {z.created_index: z.index for z in
                       zones.find_order_blocks(df, "H1", events=ev,
                                               style="last_opposite",
                                               max_height_atr=None)}
        for z in zones.find_order_blocks(df, "H1", events=ev,
                                         style="leg_extreme",
                                         max_height_atr=None):
            if z.created_index in par_cassure:
                if par_cassure[z.created_index] == z.index:
                    memes += 1
                else:
                    differentes += 1
    total = memes + differentes
    assert total > 50, "échantillon trop petit pour conclure"
    assert differentes / total > 0.5


def test_rolling_pivots_alternate_strictly():
    """Après un swing haut, le pivot suivant ne peut être qu'un swing bas : une
    jambe se termine là où l'autre commence."""
    pivots = structure.find_pivots(_escalier_haussier(), 2, mode="rolling")
    genres = [p.kind for p in pivots]
    assert genres, "aucun pivot détecté : le test ne prouverait rien"
    for a, b in zip(genres, genres[1:]):
        assert a != b


def test_a_rolling_pivot_needs_nothing_on_its_left():
    """La différence de fond : une bougie peut être un swing haut sans aucune
    bougie plus basse à sa gauche. Un sommet atteint dès l'ouverture en est un.
    """
    df = _frame([(100, 120, 99, 119)] + _calm(100, 10))
    roulant = structure.find_pivots(df, 2, mode="rolling")
    assert any(p.index == 0 and p.kind == "high" for p in roulant)
    # La fractale symétrique, elle, ne peut rien dire de la première bougie.
    assert all(p.index != 0 for p in structure.find_pivots(df, 2, mode="symmetric"))


def test_a_rolling_pivot_is_confirmed_n_bars_after_the_candidate():
    for p in structure.find_pivots(_escalier_haussier(), 2, mode="rolling"):
        assert p.confirmed_index == p.index + 2


def test_rolling_pivots_do_not_look_ahead():
    """La seule preuve qui vaille : calculer sur df[:k] doit rendre exactement
    le préfixe du calcul sur df. Sans ça, le backtest voit les retournements
    avant qu'ils existent et ne se reproduit jamais en live."""
    df = _escalier_haussier()
    complet = structure.find_pivots(df, 2, mode="rolling")
    for k in range(6, len(df) + 1):
        partiel = structure.find_pivots(df.iloc[:k], 2, mode="rolling")
        attendu = [p for p in complet if p.confirmed_index < k]
        assert [(p.index, p.kind, p.price) for p in partiel] == \
               [(p.index, p.kind, p.price) for p in attendu], f"divergence à k={k}"


def test_an_unknown_pivot_mode_raises_rather_than_falling_back():
    """Un repli muet sur « symmetric » rendrait une faute de frappe invisible —
    exactement le défaut que `market_symbol()` a coûté à ce dépôt."""
    with pytest.raises(ValueError):
        structure.find_pivots(_escalier_haussier(), 2, mode="fractal")


# --------------------------------------------------------------------------- #
# Deux échelles de structure
# --------------------------------------------------------------------------- #
def test_the_short_scale_sees_more_breaks_than_the_long_one():
    df = _frame(_calm(100, 10)
                + [(100, 112, 99, 111)] + _calm(104, 6)
                + [(104, 105, 92, 93)] + _calm(97, 6)
                + [(97, 120, 96, 119)] + _calm(110, 10))
    deux = structure.dual_structure(df, internal=3, swing=12)
    assert len(deux["internal"]) >= len(deux["swing"])


def test_a_series_too_short_for_the_long_scale_says_so_with_an_empty_list():
    """Pas d'échelle rabotée en silence : une série trop courte pour le swing
    rend une liste vide, et l'appelant décide quoi en faire."""
    deux = structure.dual_structure(_frame(_calm(100, 40)), internal=5, swing=50)
    assert deux["swing"] == []


# --------------------------------------------------------------------------- #
# Order Block : l'extrême de la jambe, la bougie entière
# --------------------------------------------------------------------------- #
def _jambe_dont_le_plus_bas_nest_pas_la_derniere_rouge():
    """Descente en deux temps : la bougie la plus basse (93) est suivie d'une
    autre bougie rouge, plus haute (94.2), juste avant l'impulsion."""
    return _frame(_calm(100, 16)
                  + [(100, 112, 99.6, 111)]        # 16 : le sommet cassé plus tard
                  + _calm(100, 2)
                  + [(100, 100.2, 93, 94)]         # 19 : le vrai plus bas
                  + [(94, 95.5, 93.8, 95.2)]
                  + [(95.2, 95.4, 94.2, 94.4)]     # 21 : dernière bougie rouge
                  + [(94.4, 130, 94.4, 128)]       # 22 : impulsion → BOS haussier
                  + _calm(120, 6))


def test_the_order_block_is_the_extreme_bar_of_the_leg_not_the_last_red_one():
    df = _jambe_dont_le_plus_bas_nest_pas_la_derniere_rouge()
    ev = structure.find_events(df, 2)
    hausse = [e for e in ev if e.direction == "bullish"]
    assert hausse, "pas de cassure haussière : le test ne prouverait rien"

    extreme = zones.find_order_blocks(df, "M5", events=hausse, style="leg_extreme",
                                      max_height_atr=None)
    derniere = zones.find_order_blocks(df, "M5", events=hausse, style="last_opposite",
                                       max_height_atr=None)
    assert extreme and derniere
    assert extreme[0].low == pytest.approx(93.0)
    assert derniere[0].low == pytest.approx(94.2)


def test_the_zone_spans_the_whole_bar_wick_included():
    """La mèche fait partie de l'empreinte : c'est elle que le prix revient
    chercher. Une zone open→extrême manque le haut de la bougie."""
    df = _jambe_dont_le_plus_bas_nest_pas_la_derniere_rouge()
    z = zones.find_order_blocks(df, "M5", style="leg_extreme",
                                max_height_atr=None)[0]
    assert z.low == pytest.approx(93.0)
    assert z.high == pytest.approx(100.2)     # le high de la bougie, pas son open


def test_an_abnormally_wide_bar_is_not_chosen_as_the_order_block():
    """Une bougie de nouvelle imprime un extrême qui n'est l'empreinte d'aucun
    ordre, seulement d'un trou de liquidité."""
    normal = _frame(_calm(100, 16, amp=1.0)
                    + [(100, 112, 99.6, 111)]
                    + [(100, 100.4, 97, 98)]
                    + [(98, 98.4, 96, 97)]
                    + [(97, 97.4, 95, 96)]
                    + [(96, 130, 96, 128)]
                    + _calm(120, 6))
    avec_anomalie = normal.copy()
    avec_anomalie.iloc[18, :4] = [100.0, 101.0, 70.0, 98.0]   # mèche monstrueuse

    sans = zones.find_order_blocks(normal, "M5", style="leg_extreme")
    avec = zones.find_order_blocks(avec_anomalie, "M5", style="leg_extreme")
    assert sans and avec
    assert avec[0].low > 70.0
    assert avec[0].low == pytest.approx(sans[0].low)


def test_the_exclusion_does_not_depend_on_the_order_of_the_bars():
    """La référence est l'ATR, pas l'amplitude du candidat courant : sinon la
    bougie monstrueuse serait écartée quand elle arrive en second et retenue
    quand elle arrive en premier — ce n'est pas une définition, c'est un effet
    de bord du sens de parcours."""
    base = _calm(100, 16, amp=1.0) + [(100, 112, 99.6, 111)]
    petites = [(100, 100.4, 97, 98), (98, 98.4, 96, 97), (97, 97.4, 95, 96)]
    monstre = (100.0, 101.0, 70.0, 98.0)
    fin = [(96, 130, 96, 128)] + _calm(120, 6)

    debut = zones.find_order_blocks(_frame(base + [monstre] + petites + fin),
                                    "M5", style="leg_extreme")
    tard = zones.find_order_blocks(_frame(base + petites + [monstre] + fin),
                                   "M5", style="leg_extreme")
    assert debut and tard
    assert debut[0].low > 70.0 and tard[0].low > 70.0


def test_a_zone_taller_than_the_cap_is_refused():
    """Le filtre qui manquait. L'entrée se prend au bord proximal et le stop
    sous le bord opposé : une zone de 4 ATR impose un risque de 4 ATR pour une
    cible qui, elle, ne grandit pas avec la zone. Le R:R affiché serait faux.
    """
    df = _jambe_dont_le_plus_bas_nest_pas_la_derniere_rouge()
    sans_plafond = zones.find_order_blocks(df, "M5", max_height_atr=None)
    assert sans_plafond
    atr_series = zones.atr(df).values
    hauteur = sans_plafond[0].height
    atr_val = float(atr_series[sans_plafond[0].created_index])

    serre = hauteur / atr_val * 0.9
    for z in zones.find_order_blocks(df, "M5", max_height_atr=serre):
        assert z.height <= serre * float(atr_series[z.created_index]) + 1e-9
    # Et la zone trop haute, elle, a bien disparu.
    assert all(z.height < hauteur
               for z in zones.find_order_blocks(df, "M5", max_height_atr=serre))


def test_an_unknown_order_block_style_raises():
    with pytest.raises(ValueError):
        zones.find_order_blocks(_jambe_dont_le_plus_bas_nest_pas_la_derniere_rouge(),
                                "M5", style="institutionnel")


# --------------------------------------------------------------------------- #
# Mitigation : mèche ou clôture
# --------------------------------------------------------------------------- #
def _zone_puis_bougie_qui_leffleure():
    """Un OB, puis une bougie dont la MÈCHE entre dans la zone mais dont la
    clôture reste au-dessus. La bougie sonde est construite à partir de la zone
    trouvée, et non devinée : sinon le test passerait ou non selon le détecteur.
    """
    df = _jambe_dont_le_plus_bas_nest_pas_la_derniere_rouge()
    z = zones.find_order_blocks(df, "M5", max_height_atr=None)
    assert z, "pas de zone : le test ne prouverait rien"
    zone = z[0]
    sonde = (zone.high + 5, zone.high + 6, zone.mid, zone.high + 5)
    suite = _frame([sonde] + _calm(zone.high + 5, 2),
                   start=df.index[-1] + pd.Timedelta("5min"))
    return pd.concat([df, suite]), zone


def test_a_wick_mitigates_in_wick_mode_but_not_in_close_mode():
    df, zone = _zone_puis_bougie_qui_leffleure()

    meche = zones.update_states(df, [zone], max_age_days=10_000, mitigation="wick")
    assert meche[0].state == "mitigated"

    cloture = zones.update_states(df, [zone], max_age_days=10_000, mitigation="close")
    assert cloture[0].state == "active"


def test_an_unknown_mitigation_mode_raises():
    df, zone = _zone_puis_bougie_qui_leffleure()
    with pytest.raises(ValueError):
        zones.update_states(df, [zone], mitigation="touch")


# --------------------------------------------------------------------------- #
# Les réglages par défaut sont bien ceux que le scanner applique
# --------------------------------------------------------------------------- #
def test_the_defaults_are_the_screen_aligned_ones():
    """Si ces valeurs changent, les zones du scanner cessent de tomber au même
    endroit que celles de l'indicateur — sans aucun message d'erreur."""
    assert config.PIVOT_MODE == "rolling"
    assert config.OB_STYLE == "leg_extreme"
    assert config.OB_MAX_HEIGHT_ATR is not None
