"""Niveaux à recopier dans MT5 — lisibilité d'abord.

Le panneau OB du graphique empile 5-7 zones parce que `strategy.find_order_blocks`
ne teste pas la mitigation. Ces tests verrouillent ce qui rend la nouvelle liste
utilisable : un plafond dur, l'exclusion des zones que le prix a dépassées, et un
canal qui s'annonce non fiable quand ses bords ne sont pas vraiment touchés.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import levels
import strategy
import strategy_ict


def _frame(rows, start="2026-01-05 08:00", freq="5min"):
    """rows = [(open, high, low, close), ...]"""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _trend(n, start=2000.0, slope=1.0, amp=3.0, seed=0):
    """Série montante avec une oscillation régulière — un vrai canal, donc des
    bords réellement touchés à plusieurs reprises."""
    rows = []
    for i in range(n):
        mid = start + slope * i
        osc = amp * np.sin(i / 4.0)
        c = mid + osc
        o = mid + amp * np.sin((i - 1) / 4.0)
        rows.append((o, max(o, c) + 0.4, min(o, c) - 0.4, c))
    return _frame(rows)


# --------------------------------------------------------------------------- #
# Order Blocks
# --------------------------------------------------------------------------- #
def test_order_blocks_are_capped_per_side(monkeypatch):
    """Le plafond est la raison d'être du panneau : une liste qu'on ne peut pas
    tracer en une minute ne sert à rien."""
    fake = [{"low": 1990.0 - i, "high": 1992.0 - i, "ts": None} for i in range(8)]
    monkeypatch.setattr(strategy_ict, "_find_order_blocks",
                        lambda df, direction, atr, require_bos=None:
                        fake if direction == "LONG" else [])
    df = strategy.add_indicators(_trend(60))
    obs, _ = levels.order_blocks(df, price=2000.0, max_per_side=2)
    assert len(obs) == 2
    # Et ce sont les plus proches du prix, pas les deux premières trouvées.
    assert obs[0]["high"] == pytest.approx(1992.0)


def test_order_block_already_passed_by_price_is_dropped(monkeypatch):
    """Un OB haussier au-dessus du prix n'a plus de retest à offrir : le prix est
    déjà passé dessous."""
    monkeypatch.setattr(strategy_ict, "_find_order_blocks",
                        lambda df, direction, atr, require_bos=None:
                        [{"low": 2050.0, "high": 2052.0, "ts": None}]
                        if direction == "LONG" else [])
    df = strategy.add_indicators(_trend(60))
    assert levels.order_blocks(df, price=2000.0)[0] == []


def test_panel_asks_for_bos_while_the_live_path_keeps_its_default(monkeypatch):
    """Le panneau est visuel : un filtre trop strict n'y coûte que des niveaux en
    moins. La stratégie B, elle, garde OB_REQUIRE_BOS=False."""
    seen = []
    monkeypatch.setattr(strategy_ict, "_find_order_blocks",
                        lambda df, direction, atr, require_bos=None:
                        (seen.append(require_bos), [])[1])
    levels.order_blocks(strategy.add_indicators(_trend(60)), price=2000.0)
    assert seen and all(v is True for v in seen)
    assert strategy_ict.OB_REQUIRE_BOS is False


def test_require_bos_override_does_not_change_the_default_call():
    """La surcharge ne doit rien changer quand on ne la passe pas — le chemin de
    trading appelle toujours sans l'argument."""
    df = strategy.add_indicators(_trend(80))
    atr_val = float(df["atr"].iloc[-1])
    assert (strategy_ict._find_order_blocks(df, "LONG", atr_val)
            == strategy_ict._find_order_blocks(df, "LONG", atr_val, require_bos=None))


# --------------------------------------------------------------------------- #
# Supports / résistances
# --------------------------------------------------------------------------- #
def test_sr_keeps_only_the_nearest_level_on_each_side():
    df = strategy.add_indicators(_trend(160))
    price = float(df["close"].iloc[-1])
    out, _ = levels.sr_levels(df, price)
    kinds = [lv["kind"] for lv in out]
    assert len(kinds) == len(set(kinds))          # au plus un de chaque
    for lv in out:
        assert lv["distance"] >= 0
        assert lv["touches"] >= levels.SR_MIN_TOUCHES
        if lv["kind"] == "support":
            assert lv["price"] <= price
        else:
            assert lv["price"] >= price


def test_touch_counts_a_candle_that_straddles_the_level():
    """Compte par intervalle, pas par distance à l'extrême : une bougie qui
    traverse le niveau de part en part le touche."""
    df = _frame([(10.0, 20.0, 0.0, 15.0), (30.0, 31.0, 29.0, 30.0)])
    assert levels.count_touches(df, 10.0, tol=0.0) == 1
    assert levels.count_touches(df, 29.5, tol=0.0) == 1
    assert levels.count_touches(df, 25.0, tol=0.0) == 0


def test_a_long_stay_inside_the_band_is_one_touch_not_thirty():
    """La tolérance vaut 0.5×ATR : dans un range, des dizaines de bougies
    stationnent dans la bande. Les compter une par une affichait « 28 touches »
    pour un niveau visité une fois — un chiffre qui se lit comme une force."""
    rows = [(10.0, 10.4, 9.6, 10.0) for _ in range(30)]
    rows += [(50.0, 50.4, 49.6, 50.0) for _ in range(10)]
    rows += [(10.0, 10.4, 9.6, 10.0) for _ in range(5)]
    assert levels.count_touches(_frame(rows), 10.0, tol=0.0) == 2


def test_a_level_the_price_is_standing_on_is_not_offered():
    """« Support 2012.06 » alors que le prix est à 2012.29 : rien à tracer."""
    df = strategy.add_indicators(_trend(160))
    price = float(df["close"].iloc[-1])
    tol = levels.SR_TOL_ATR * levels._tol_unit(df)
    for lv in levels.sr_levels(df, price)[0]:
        assert lv["distance"] >= tol


# --------------------------------------------------------------------------- #
# Canal
# --------------------------------------------------------------------------- #
def test_channel_anchors_are_the_three_mt5_points():
    ch = levels.parallel_channel(strategy.add_indicators(_trend(120)))
    assert ch is not None
    p1, p2, p3 = ch["mt5"]["point1"], ch["mt5"]["point2"], ch["mt5"]["point3"]
    # point1/point2 = la droite du bas, point3 = la parallèle, au même instant que
    # point1 et exactement une largeur au-dessus.
    assert p3["time"] == p1["time"]
    assert p3["price"] - p1["price"] == pytest.approx(ch["width"])
    assert p2["price"] == pytest.approx(ch["bottom_now"])
    assert p1["time"] < p2["time"]


def test_channel_contains_every_candle_of_the_window():
    df = strategy.add_indicators(_trend(120))
    ch = levels.parallel_channel(df)
    sub = df.tail(ch["bars"])
    slope = ch["slope_per_bar"]
    n = ch["bars"]
    bottom0 = ch["bottom_now"] - slope * (n - 1)
    for i in range(n):
        assert float(sub["low"].iloc[i]) >= bottom0 + slope * i - 1e-6
        assert float(sub["high"].iloc[i]) <= bottom0 + slope * i + ch["width"] + 1e-6


def test_channel_is_flagged_unreliable_when_a_rail_is_barely_touched():
    """Deux parallèles calées sur les extrêmes encadrent le prix PAR CONSTRUCTION.
    Ce sont les touches qui disent si le canal existe — sinon c'est une enveloppe,
    et l'afficher sans réserve la ferait prendre pour un niveau."""
    rows = []
    for i in range(120):
        c = 2000.0 + 0.02 * i
        rows.append((c, c + 0.2, c - 0.2, c))
    rows[60] = (2001.2, 2060.0, 2001.0, 2001.4)   # un unique pic isolé en haut
    ch = levels.parallel_channel(strategy.add_indicators(_frame(rows)))
    assert ch["touches_high"] < levels.CHANNEL_MIN_TOUCHES
    assert ch["fiable"] is False


def test_a_real_oscillation_touches_both_rails_repeatedly():
    ch = levels.parallel_channel(strategy.add_indicators(_trend(120, amp=6.0)))
    assert ch["touches_low"] >= levels.CHANNEL_MIN_TOUCHES
    assert ch["touches_high"] >= levels.CHANNEL_MIN_TOUCHES
    assert ch["fiable"] is True
    assert ch["direction"] == "haussier"


def test_adjacent_bars_on_the_same_low_count_as_one_touch():
    assert levels._distinct_touches([10, 11, 12]) == 1
    assert levels._distinct_touches([10, 11, 12, 40, 41]) == 2
    assert levels._distinct_touches([]) == 0


def test_channel_needs_a_minimum_window():
    assert levels.parallel_channel(strategy.add_indicators(_trend(15))) is None


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def test_build_levels_rounds_prices_to_the_symbol_digits():
    """EUR/USD se lit à 5 décimales ; 2 décimales écraseraient tout à 1.10."""
    df = strategy.add_indicators(_trend(120, start=1.1000, slope=0.00002, amp=0.00030))
    data = levels.build_levels(df, symbol="EURUSD", timeframe="M5")
    assert data["digits"] == 5
    assert data["channel"]["width"] == round(data["channel"]["width"], 5)
    assert data["channel"]["width"] > 0
    xau = levels.build_levels(strategy.add_indicators(_trend(120)),
                              symbol="XAUUSD", timeframe="M5")
    assert xau["digits"] == 2


def test_build_levels_survives_a_short_frame():
    data = levels.build_levels(strategy.add_indicators(_trend(5)),
                               symbol="XAUUSD", timeframe="H1")
    assert data["order_blocks"] == [] and data["sr"] == [] and data["channel"] is None
    assert "note" in data


def test_text_rendering_is_copy_pasteable():
    df = strategy.add_indicators(_trend(120, amp=6.0))
    txt = levels.as_text(levels.build_levels(df, symbol="XAUUSD", timeframe="M5"))
    assert "XAUUSD M5" in txt
    assert "Canal" in txt and "OBJ_CHANNEL" in txt
    assert "point1" in txt and "point3" in txt


def test_text_warns_when_the_channel_is_not_reliable():
    rows = [(2000.0 + 0.02 * i, 2000.2 + 0.02 * i, 1999.8 + 0.02 * i, 2000.0 + 0.02 * i)
            for i in range(120)]
    rows[60] = (2001.2, 2060.0, 2001.0, 2001.4)
    txt = levels.as_text(levels.build_levels(strategy.add_indicators(_frame(rows)),
                                             symbol="XAUUSD", timeframe="M5"))
    assert "peu fiable" in txt


# --------------------------------------------------------------------------- #
# Fenêtres découplées et motifs d'absence
# --------------------------------------------------------------------------- #
def test_sr_window_is_longer_than_the_channel_window():
    """Le canal décrit le mouvement en cours, un support reste un support à 300
    bougies. Les deux partageaient 120 bougies : quand le prix atteignait
    l'extrême de la fenêtre, il n'y avait plus rien d'un côté par construction —
    et le panneau se taisait exactement quand on en avait besoin."""
    assert levels.SR_LOOKBACK > levels.CHANNEL_LOOKBACK


def test_a_level_outside_the_channel_window_is_still_found():
    """Le prix casse sous tout son range récent : le support est loin derrière,
    hors des 120 bougies du canal. Il doit quand même sortir."""
    rows = [(2000.0, 2001.0, 1999.0, 2000.0) for _ in range(6)]   # socle testé 2×
    rows += [(2000.0, 2001.0, 1999.0, 2000.0) for _ in range(6)]
    rows += [(2020.0 + 0.05 * i, 2021.0 + 0.05 * i, 2019.0 + 0.05 * i,
              2020.0 + 0.05 * i) for i in range(180)]             # loin au-dessus
    rows += [(2005.0 - 0.5 * i, 2006.0 - 0.5 * i, 2004.0 - 0.5 * i,
              2005.0 - 0.5 * i) for i in range(8)]                # chute vers 2001
    df = strategy.add_indicators(_frame(rows))
    price = float(df["close"].iloc[-1])
    long_, _ = levels.sr_levels(df, price, lookback=400)
    court, notes = levels.sr_levels(df, price, lookback=120)
    # Sur la fenêtre courte le socle est hors champ : aucun support, et le motif
    # est justement celui que le panneau doit maintenant expliquer.
    assert not any(lv["kind"] == "support" for lv in court)
    assert any("plus bas" in n for n in notes)
    assert any(lv["kind"] == "support" for lv in long_)


def test_an_empty_side_says_why():
    """Un blanc se lit comme une panne. « Le prix est au plus bas des N bougies »
    est une information — c'est même celle qui compte."""
    rows = [(2000.0 - i, 2001.0 - i, 1999.0 - i, 2000.0 - i) for i in range(60)]
    df = strategy.add_indicators(_frame(rows))
    out, notes = levels.sr_levels(df, float(df["close"].iloc[-1]))
    assert not any(lv["kind"] == "support" for lv in out)
    assert any("plus bas" in n for n in notes)


def test_order_blocks_explain_an_empty_result(monkeypatch):
    """« Aucune zone détectée » et « toutes déjà dépassées par le prix » sont deux
    situations opposées ; les afficher pareil rend le panneau illisible."""
    monkeypatch.setattr(strategy_ict, "_find_order_blocks",
                        lambda df, direction, atr, require_bos=None: [])
    _, notes = levels.order_blocks(strategy.add_indicators(_trend(60)), price=2000.0)
    assert any("Aucune zone non mitiguée" in n for n in notes)

    monkeypatch.setattr(strategy_ict, "_find_order_blocks",
                        lambda df, direction, atr, require_bos=None:
                        [{"low": 2050.0, "high": 2052.0, "ts": None}]
                        if direction == "LONG" else [])
    _, notes = levels.order_blocks(strategy.add_indicators(_trend(60)), price=2000.0)
    assert any("déjà dépassées" in n for n in notes)


# --------------------------------------------------------------------------- #
# Fibonacci
# --------------------------------------------------------------------------- #
def _leg(up=True, n=60):
    """Une jambe nette de 2000 à 2100 (ou l'inverse), puis un retracement."""
    rows = []
    for i in range(n):
        p = 2000.0 + (100.0 * i / (n - 1) if up else -100.0 * i / (n - 1))
        rows.append((p, p + 0.5, p - 0.5, p))
    for i in range(1, 21):                       # retour de 30 % vers l'origine
        p = (2100.0 - 1.5 * i) if up else (1900.0 + 1.5 * i)
        rows.append((p, p + 0.5, p - 0.5, p))
    return _frame(rows)


def test_fib_anchors_on_the_extremes_in_chronological_order():
    fib = levels.fibonacci(strategy.add_indicators(_leg(up=True)))
    assert fib["direction"] == "haussier"
    assert fib["depart"]["price"] == pytest.approx(1999.5)   # mèche du plus bas
    assert fib["arrivee"]["price"] == pytest.approx(2100.5)  # mèche du plus haut
    assert fib["depart"]["time"] < fib["arrivee"]["time"]

    down = levels.fibonacci(strategy.add_indicators(_leg(up=False)))
    assert down["direction"] == "baissier"
    assert down["depart"]["price"] > down["arrivee"]["price"]


def test_fib_uses_one_formula_matching_the_mt5_tool():
    """prix(r) = fin − r × (fin − départ). r=1 retombe sur le départ, r>1 le
    dépasse — exactement les niveaux > 100 % de l'outil Fibonacci de MT5."""
    fib = levels.fibonacci(strategy.add_indicators(_leg(up=True)))
    start = fib["depart"]["price"]
    end = fib["arrivee"]["price"]
    lv = {n["ratio"]: n["price"] for n in fib["niveaux"]}
    assert lv[0.5] == pytest.approx((start + end) / 2, abs=0.01)
    assert lv[0.618] == pytest.approx(end - 0.618 * (end - start), abs=0.01)
    # 127.2 % dépasse l'origine de la jambe, donc passe SOUS le départ.
    assert lv[1.272] < start


def test_fib_refuses_a_leg_that_is_only_range_noise():
    """Sous 1.5×ATR, les sept niveaux seraient séparés par moins d'une mèche."""
    rows = [(2000.0 + 0.01 * (i % 3), 2000.2, 1999.8, 2000.0) for i in range(60)]
    assert levels.fibonacci(strategy.add_indicators(_frame(rows))) is None


def test_fib_reports_where_the_retracement_currently_stands():
    """0 % = le prix est encore sur l'extrémité, 100 % = revenu à l'origine."""
    fib = levels.fibonacci(strategy.add_indicators(_leg(up=True)))
    assert 20 < fib["retracement_pct"] < 45     # on a reculé de ~30 points sur 100
    assert fib["bars_since"] == 20


def test_fib_touches_are_measured_only_after_the_leg_ended():
    """Le seul chiffre mesuré du bloc. Compter les contacts pendant la jambe
    ferait passer chaque niveau traversé une fois en montant pour un niveau
    respecté — l'inverse de ce qu'on cherche à savoir."""
    fib = levels.fibonacci(strategy.add_indicators(_leg(up=True)))
    par_ratio = {n["ratio"]: n["touches"] for n in fib["niveaux"]}
    # Le retracement s'arrête vers 2070 : il a touché 23.6 %, jamais 78.6 %.
    assert par_ratio[0.236] >= 1
    assert par_ratio[0.786] == 0


def test_fib_claims_no_predictive_value():
    """Garde-fou : rien dans ce dépôt n'a validé Fibonacci sur l'or. Aucune clé
    d'espérance ou de probabilité ne doit apparaître — une valeur affichée serait
    prise pour une prédiction (l'erreur commise sur la distribution du MFE)."""
    fib = levels.fibonacci(strategy.add_indicators(_leg(up=True)))
    interdits = ("esperance", "expectancy", "proba", "score", "gain", "edge")
    assert not any(k in interdits for k in fib)
    for niveau in fib["niveaux"]:
        assert not any(k in interdits for k in niveau)


def test_fib_is_included_in_the_text_block():
    txt = levels.as_text(levels.build_levels(
        strategy.add_indicators(_leg(up=True)), symbol="XAUUSD", timeframe="M5"))
    assert "Fibonacci" in txt and "OBJ_FIBO" in txt
    assert "61.8%" in txt
