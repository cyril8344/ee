"""Confluence H4 → H1 → M15 (§6), et le replay qui la juge (§11.3).

Le test central est `test_the_three_conditions_together_produce_a_signal` : sans
une fixture qui tire réellement, « 0 signal » sur des données de marché serait
indistinguable d'un pipeline cassé. Chaque test négatif qui suit retire **une
seule** condition de cette même fixture et vérifie que le signal disparaît —
c'est ce qui attribue le zéro à la bonne cause.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

from smc import config, confluence, replay
from smc.structure import find_events


def _F(rows, start, freq):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def _calm(p, n):
    return [(p, p + 0.5, p - 0.5, p) for _ in range(n)]


def _h4_bullish():
    """Dernier événement structurel H4 = BOS haussier → biais haussier."""
    return _F(_calm(100, 16) + [(100, 112, 99.6, 111)] + _calm(100, 3)
              + [(100, 130, 99.6, 128)] + _calm(128, 4),
              "2026-01-01 00:00", "240min")


def _h1_with_demand_zone():
    """OB haussier 95→100, en discount du range 95–130, jamais retesté en H1."""
    return _F(_calm(100, 16) + [(100, 112, 99.6, 111)] + _calm(100, 3)
              + [(100, 100.4, 95, 96)] + [(96, 130, 96, 128)] + _calm(128, 6),
              "2026-01-05 00:00", "60min")


def _m15_pullback_then_choch():
    """Le prix redescend dans la zone H1, puis un CHoCH haussier M15 se forme sur
    la dernière bougie. C'est le déclencheur du §6."""
    return _F(_calm(128, 6)
              + [(128, 128.4, 120, 121)]      # 6  : pivot bas
              + _calm(128, 3)
              + [(128, 128.5, 110, 112)]      # 10 : BOS baissier
              + [(112, 112.5, 98, 99)]        # 11 : entre dans la zone 95–100
              + _calm(99, 2)
              + [(99, 105, 98.6, 104)]        # 14 : pivot haut
              + _calm(99, 2)
              + [(99, 110, 98.6, 108)],       # 17 : CHoCH haussier → déclencheur
              "2026-01-05 03:15", "15min")


# --------------------------------------------------------------------------- #
# Le chemin nominal
# --------------------------------------------------------------------------- #
def test_the_three_conditions_together_produce_a_signal():
    """Sans cette preuve, « 0 signal » en replay serait indistinguable d'un
    pipeline cassé — et on corrigerait les mauvais seuils."""
    s = confluence.evaluate(_m15_pullback_then_choch(), _h1_with_demand_zone(),
                            _h4_bullish(), None)
    assert s is not None
    assert s.direction == "bullish" and s.bias_h4 == "bullish"
    assert s.zone.kind == "OB"
    assert (s.zone.low, s.zone.high) == (95.0, 100.0)
    assert s.choch.kind == "CHoCH" and s.choch.direction == "bullish"


def test_the_signal_carries_the_features_the_ml_dataset_needs():
    """§9 : chaque signal est journalisé avec ses features au moment T."""
    d = confluence.evaluate(_m15_pullback_then_choch(), _h1_with_demand_zone(),
                            _h4_bullish(), None).as_dict()
    for clef in ("direction", "bias_h4", "d1_aligned", "zone_type", "zone_tf",
                 "discount_pct", "sweep", "sweep_depth_atr", "atr_m15", "atr_h1"):
        assert clef in d


# --------------------------------------------------------------------------- #
# Chaque condition retirée doit faire disparaître le signal
# --------------------------------------------------------------------------- #
def test_a_neutral_h4_bias_blocks_every_direction():
    """§6 : CHoCH H4 non confirmé par un BOS → biais neutre → aucun signal, dans
    aucun sens. Un CHoCH est le premier signe d'un retournement, pas sa preuve."""
    h4 = _F(_calm(100, 16) + [(100, 112, 99.6, 111)] + _calm(100, 3)
            + [(100, 130, 99.6, 128)] + _calm(128, 3)
            + [(128, 128.4, 90, 91)] + _calm(128, 3)
            + [(128, 128.5, 80, 82)],          # CHoCH baissier terminal
            "2026-01-01 00:00", "240min")
    assert find_events(h4, config.FRACTAL_N["H4"])[-1].kind == "CHoCH"
    assert confluence.evaluate(_m15_pullback_then_choch(),
                               _h1_with_demand_zone(), h4, None) is None


def test_no_h1_zone_means_no_signal():
    h1_plat = _F(_calm(100, 40), "2026-01-05 00:00", "60min")
    assert confluence.evaluate(_m15_pullback_then_choch(), h1_plat,
                               _h4_bullish(), None) is None


def test_a_zone_the_price_never_reached_gives_no_signal():
    """La zone existe et est fraîche, mais le prix n'est jamais venu la chercher
    dans les 8 dernières bougies M15."""
    m15 = _m15_pullback_then_choch()
    loin = m15.copy()
    for col in ("open", "high", "low", "close"):
        loin[col] = loin[col] + 500        # même structure, très au-dessus de la zone
    assert confluence.evaluate(loin, _h1_with_demand_zone(),
                               _h4_bullish(), None) is None


def test_a_bos_trigger_is_not_a_choch_trigger():
    """§6 exige un CHoCH M15, pas n'importe quelle cassure : le déclencheur est un
    retournement local dans le sens du biais, pas une continuation."""
    m15 = _m15_pullback_then_choch()
    ev = find_events(m15, config.FRACTAL_N["M15"])
    assert ev[-1].kind == "CHoCH"
    tronque = m15.iloc[:11]                 # s'arrête sur le BOS baissier
    assert find_events(tronque, config.FRACTAL_N["M15"])[-1].kind == "BOS"
    assert confluence.evaluate(tronque, _h1_with_demand_zone(),
                               _h4_bullish(), None) is None


def test_the_trigger_must_land_on_the_current_bar():
    """Sinon le même signal serait ré-émis à chaque bougie suivante — l'anti-spam
    du §8 ne rattraperait pas une émission par bougie."""
    m15 = _m15_pullback_then_choch()
    prolonge = pd.concat([m15, _F(_calm(108, 3), "2026-01-05 07:45", "15min")])
    assert confluence.evaluate(prolonge, _h1_with_demand_zone(),
                               _h4_bullish(), None) is None


def test_one_signal_per_zone_maximum():
    """§6 : 1 signal max par zone."""
    m15, h1, h4 = (_m15_pullback_then_choch(), _h1_with_demand_zone(), _h4_bullish())
    s = confluence.evaluate(m15, h1, h4, None)
    assert s is not None
    assert confluence.evaluate(m15, h1, h4, None,
                               signalled_zones={s.zone_key}) is None


def test_a_zone_traversed_by_the_price_is_dead():
    """§6 : zone traversée sans déclencheur = dead, supprimée."""
    m15, h1, h4 = (_m15_pullback_then_choch(), _h1_with_demand_zone(), _h4_bullish())
    s = confluence.evaluate(m15, h1, h4, None)
    assert confluence.evaluate(m15, h1, h4, None, dead_zones={s.zone_key}) is None


# --------------------------------------------------------------------------- #
# Sweep : taggé, jamais exigé
# --------------------------------------------------------------------------- #
def test_a_signal_without_sweep_is_still_a_signal():
    """§5 : le sweep n'est pas une condition. Ce dépôt a déjà retiré « sweep
    obligatoire » de la stratégie B ; l'exiger ici referait l'erreur, et
    priverait en plus le dataset des signaux sans sweep qui servent de témoin."""
    s = confluence.evaluate(_m15_pullback_then_choch(), _h1_with_demand_zone(),
                            _h4_bullish(), None)
    assert s is not None
    assert s.sweep is False
    assert s.sweep_obj is None


# --------------------------------------------------------------------------- #
# D1 : information seule
# --------------------------------------------------------------------------- #
def test_d1_never_blocks_a_signal_in_v1():
    """§6 : la structure D1 est mentionnée dans l'alerte à titre d'information,
    sans bloquer le signal en v1. Le tester évite qu'elle devienne un filtre par
    inadvertance."""
    d1_contraire = _F(_calm(100, 16) + [(100, 100.4, 88, 89)] + _calm(100, 3)
                      + [(100, 100.5, 70, 72)] + _calm(72, 4),
                      "2025-11-01 00:00", "1440min")
    s = confluence.evaluate(_m15_pullback_then_choch(), _h1_with_demand_zone(),
                            _h4_bullish(), d1_contraire)
    assert s is not None
    assert s.d1_aligned is False          # contraire, mais le signal existe


def test_unknown_d1_is_not_reported_as_contrary():
    """« On ne sait pas » et « contraire » ne doivent pas se confondre dans
    l'alerte."""
    assert confluence.d1_alignment(None, "bullish") is None
    plat = _F(_calm(100, 30), "2025-11-01 00:00", "1440min")
    assert confluence.d1_alignment(plat, "bullish") is None


# --------------------------------------------------------------------------- #
# Replay — le juge du §11.3
# --------------------------------------------------------------------------- #
def test_replay_thresholds_are_the_spec_ones():
    """Ce ne sont pas des réglages : ce sont les hypothèses posées avant mesure."""
    assert replay.SIGNALS_PER_WEEK_MIN == 2.0
    assert replay.SIGNALS_PER_WEEK_MAX == 6.0
    assert replay.SIGNALS_PER_WEEK_ALARM == 15.0


def test_the_verdict_names_the_laxist_case_explicitly():
    r = replay.ReplayResult(bars=100, days=70, weeks=10)
    r.signals = [object()] * 200          # 20/semaine
    assert "LAXISTES" in r.verdict

    r2 = replay.ReplayResult(bars=100, days=70, weeks=10)
    r2.signals = [object()] * 40          # 4/semaine
    assert "conforme" in r2.verdict


def test_replay_only_feeds_closed_higher_timeframe_bars():
    """Une bougie H1 datée 10:00 n'est close qu'à 10:00 : à 09:45 en M15 elle
    n'existe pas encore. L'inclure ferait lire une bougie en formation — le
    défaut que drop_unclosed évite en live et qu'un replay naïf réintroduit."""
    h1 = _h1_with_demand_zone()
    ts = h1.index[5]
    assert replay._slice_upto(h1, ts).index[-1] == ts
    assert len(replay._slice_upto(h1, ts)) == 6


def test_replay_returns_an_empty_result_on_a_too_short_series():
    r = replay.run_replay(_m15_pullback_then_choch(), _h1_with_demand_zone(),
                          _h4_bullish(), None, warmup=200)
    assert r.signals == [] and r.weeks == 0.0
    assert "indéterminé" in r.verdict
