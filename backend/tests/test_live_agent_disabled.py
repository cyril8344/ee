"""L'agent adaptatif ne doit plus ajuster les paramètres de stratégie tout seul.

Motif (voir l'en-tête de live_agent.py) : il évaluait le WR sur 20 trades — erreur-type
±11 points, donc il réagissait au bruit une fois sur cinq — modifiait quatre paramètres
à la fois là où les règles du projet en autorisent trois, sans aucune validation
walk-forward. Et son _load() retenait toujours la valeur la plus permissive entre la
valeur sauvegardée et le défaut module, rendant tout desserrage irréversible :
strategy.ADX_MIN est resté bloqué à 20 pendant des semaines alors que le code
déclarait 28, sans alerte.
"""
import os

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import live_agent
import strategy
from live_agent import LiveAdaptiveAgent


def test_auto_adjust_is_off():
    assert live_agent.AUTO_ADJUST_ENABLED is False


def test_evaluate_and_adjust_is_a_no_op(monkeypatch):
    """Même avec un WR catastrophique sur une fenêtre pleine, aucun paramètre ne bouge."""
    agent = LiveAdaptiveAgent(symbol="XAUUSD")
    agent._trade_log = [{"won": False, "pnl": -10.0} for _ in range(live_agent.WINDOW)]
    before = dict(agent._params)
    n_adj_before = len(agent._adjustments)

    agent._evaluate_and_adjust()

    assert agent._params == before
    assert len(agent._adjustments) == n_adj_before


def test_agent_never_writes_to_the_strategy_module_on_its_own(monkeypatch):
    """_apply_to_strategy() est appelée par _load() au démarrage : elle ne doit plus
    réappliquer l'héritage de l'ancienne boucle par-dessus les valeurs du code."""
    monkeypatch.setattr(strategy, "ADX_MIN", 20.0)
    agent = LiveAdaptiveAgent(symbol="XAUUSD")
    agent._params["ADX_MIN"] = 15.0   # valeur héritée, plus permissive

    agent._apply_to_strategy()

    assert strategy.ADX_MIN == 20.0


def test_manual_force_still_reaches_the_strategy_module(monkeypatch):
    """Le forçage manuel depuis le dashboard est une action humaine explicite et tracée :
    il doit continuer de fonctionner, sinon le seul moyen de reprendre la main disparaît
    avec la boucle."""
    monkeypatch.setattr(strategy, "ADX_MIN", 20.0)
    agent = LiveAdaptiveAgent(symbol="XAUUSD")

    agent.force_params({"ADX_MIN": 28.0})

    assert strategy.ADX_MIN == 28.0
    assert agent._params["ADX_MIN"] == 28.0


def test_trade_counting_and_bootstrap_exit_survive(monkeypatch):
    """Ce qui n'a rien à voir avec l'optimisation reste en place : le comptage affiché
    au dashboard et la sortie unique de BOOTSTRAP_MODE."""
    monkeypatch.setattr(strategy, "BOOTSTRAP_MODE", True)
    agent = LiveAdaptiveAgent(symbol="XAUUSD")
    agent._trade_log = []
    agent._total_trades = live_agent.BOOTSTRAP_EXIT_TRADES - 1

    agent.on_trade_closed(won=True, pnl=5.0)

    assert agent._total_trades == live_agent.BOOTSTRAP_EXIT_TRADES
    assert strategy.BOOTSTRAP_MODE is False
    assert agent.consume_bootstrap_exit() is True
    assert agent.consume_bootstrap_exit() is False   # une seule fois


def test_strategy_adx_min_matches_what_live_was_actually_running():
    """Le PR ne doit changer AUCUN comportement de trading : la valeur déclarée doit
    être celle que l'agent appliquait réellement (20), pas le 28 que le code affichait
    sans l'appliquer. 20 vs 28 est une décision séparée, à trancher en walk-forward."""
    assert strategy.ADX_MIN == 20.0
