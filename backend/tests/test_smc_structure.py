"""Structure SMC — pivots, BOS, CHoCH, et surtout l'absence de look-ahead.

Le test qui porte tout le module est `test_no_look_ahead_*` : calculer sur les k
premières bougies doit rendre exactement le préfixe du calcul sur la série
entière. Un détecteur qui repère les pivots sur toute la série puis « constate »
les cassures passe tous les autres tests et échoue celui-là — et produit un
backtest flatteur qui ne se reproduit jamais en live.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import structure


def _frame(rows, start="2026-01-05 08:00", freq="15min"):
    """rows = [(open, high, low, close), ...]"""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _bar(high, low, close=None):
    close = close if close is not None else (high + low) / 2
    return ((high + low) / 2, high, low, close)


def _flat(price, n):
    return [_bar(price + 0.5, price - 0.5, price) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Pivots
# --------------------------------------------------------------------------- #
def test_pivot_high_needs_n_bars_on_each_side():
    # Sommet isolé en position 3, n=2 → confirmé en position 5.
    rows = [_bar(10, 9), _bar(11, 10), _bar(12, 11), _bar(20, 19),
            _bar(12, 11), _bar(11, 10), _bar(10, 9)]
    pivots = structure.find_pivots(_frame(rows), n=2)
    highs = [p for p in pivots if p.kind == "high"]
    assert len(highs) == 1
    assert highs[0].index == 3
    assert highs[0].price == pytest.approx(20)
    assert highs[0].confirmed_index == 5


def test_last_n_bars_can_never_hold_a_confirmed_pivot():
    """Il manque les bougies de droite : c'est la définition même d'une fractale,
    et c'est ce qui interdit de « voir » un retournement en avance."""
    rows = [_bar(10 + i, 9 + i) for i in range(10)] + [_bar(50, 49)]
    df = _frame(rows)
    for p in structure.find_pivots(df, n=3):
        assert p.index <= len(df) - 1 - 3


def test_a_plateau_does_not_produce_several_pivots():
    """Avec `>=` au lieu de `>`, trois bougies au même sommet donneraient trois
    pivots au même prix, et autant de faux niveaux à casser."""
    rows = [_bar(10, 9), _bar(11, 10), _bar(20, 19), _bar(20, 19), _bar(20, 19),
            _bar(11, 10), _bar(10, 9)]
    assert [p for p in structure.find_pivots(_frame(rows), n=2)
            if p.kind == "high"] == []


def test_pivot_low_is_detected_symmetrically():
    rows = [_bar(20, 19), _bar(19, 18), _bar(18, 17), _bar(10, 9),
            _bar(18, 17), _bar(19, 18), _bar(20, 19)]
    lows = [p for p in structure.find_pivots(_frame(rows), n=2) if p.kind == "low"]
    assert len(lows) == 1
    assert lows[0].index == 3 and lows[0].price == pytest.approx(9)


# --------------------------------------------------------------------------- #
# BOS / CHoCH
# --------------------------------------------------------------------------- #
def _bull_then_bear():
    """Monte, fait un sommet, casse ce sommet (BOS haussier), puis fait un creux
    et le casse à la baisse (CHoCH baissier)."""
    rows = []
    rows += _flat(100, 3)
    rows += [_bar(112, 110, 111)]          # pivot haut en 3 (prix 112)
    rows += _flat(100, 3)                  # confirmé en 6
    rows += [_bar(115, 113, 114)]          # clôture 114 > 112 → BOS haussier
    rows += _flat(120, 3)
    rows += [_bar(96, 94, 95)]             # pivot bas en 12 (prix 94)
    rows += _flat(120, 3)                  # confirmé en 15
    rows += [_bar(93, 90, 91)]             # clôture 91 < 94 → CHoCH baissier
    return _frame(rows)


def test_first_break_is_a_bos_and_the_counter_break_is_a_choch():
    events = structure.find_events(_bull_then_bear(), n=3)
    kinds = [(e.kind, e.direction) for e in events]
    assert kinds[0] == ("BOS", "bullish")
    assert ("CHoCH", "bearish") in kinds


def test_break_is_validated_on_close_never_on_the_wick():
    """Une mèche qui dépasse puis revient n'est pas une cassure — c'est même
    souvent un sweep, donc l'inverse d'un signal de continuation."""
    rows = _flat(100, 3) + [_bar(112, 110, 111)] + _flat(100, 3)
    rows += [_bar(120, 105, 106)]          # mèche à 120 au-dessus de 112, clôture 106
    df = _frame(rows)
    assert structure.find_events(df, n=3) == []

    rows[-1] = _bar(120, 105, 113)         # même mèche, mais clôture 113 > 112
    assert len(structure.find_events(_frame(rows), n=3)) == 1


def test_a_level_can_only_be_broken_once():
    rows = _flat(100, 3) + [_bar(112, 110, 111)] + _flat(100, 3)
    rows += [_bar(115, 113, 114), _bar(116, 114, 115), _bar(117, 115, 116)]
    events = structure.find_events(_frame(rows), n=3)
    assert len(events) == 1
    assert events[0].level == pytest.approx(112)


# --------------------------------------------------------------------------- #
# Anti look-ahead — le test qui porte le module
# --------------------------------------------------------------------------- #
def test_no_look_ahead_events_are_stable_as_bars_arrive():
    df = _bull_then_bear()
    complet = [e.as_dict() for e in structure.find_events(df, n=3)]
    for k in range(5, len(df) + 1):
        partiel = [e.as_dict() for e in structure.find_events(df.iloc[:k], n=3)]
        attendu = [e for e in complet if e["index"] < k]
        assert partiel == attendu, f"divergence à k={k}"


def test_no_look_ahead_pivots_are_stable_once_confirmed():
    df = _bull_then_bear()
    complet = {p.index: p.as_dict() for p in structure.find_pivots(df, n=3)}
    for k in range(7, len(df) + 1):
        for p in structure.find_pivots(df.iloc[:k], n=3):
            assert p.as_dict() == complet[p.index]


def test_last_swings_ignores_a_pivot_not_yet_confirmed():
    df = _bull_then_bear()
    # En 4, le pivot haut de 3 existe mais n'est pas confirmé (il l'est en 6).
    assert structure.last_swings(df, n=3, upto=4)["high"] is None
    assert structure.last_swings(df, n=3, upto=6)["high"].index == 3


# --------------------------------------------------------------------------- #
# Biais
# --------------------------------------------------------------------------- #
def test_bias_follows_the_last_structural_event():
    events = structure.find_events(_bull_then_bear(), n=3)
    premiers = [e for e in events if e.kind == "BOS" and e.direction == "bullish"]
    assert structure.current_bias(premiers) == "bullish"


def test_a_terminal_choch_gives_a_neutral_bias():
    """§6 : un CHoCH non confirmé par un BOS → biais neutre → aucun signal, dans
    aucun sens. Un CHoCH est le premier signe d'un retournement, pas sa preuve."""
    events = structure.find_events(_bull_then_bear(), n=3)
    assert events[-1].kind == "CHoCH"
    assert structure.current_bias(events) is None


def test_no_event_gives_no_bias():
    assert structure.current_bias([]) is None
