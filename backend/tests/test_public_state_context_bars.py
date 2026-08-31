"""Regression : _public_state() construit le payload marché par liste blanche de clés.

Une clé ajoutée à ms.last_snapshot mais oubliée ici n'atteint jamais le dashboard —
sans erreur, sans log : le panneau reste simplement vide. C'est ce qui est arrivé à
context_bars/h1_context_ok, ajoutés au snapshot pour rendre visible la longueur du
contexte H1 (dont dépend la fiabilité de l'EMA200 H1, donc du biais).
"""
import os

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import main as main_module


def _market_payload(snapshot_extra):
    ms = next(iter(main_module.state.market_states.values()))
    original = ms.last_snapshot
    ms.last_snapshot = {**(original or {}), **snapshot_extra}
    try:
        payload = main_module._public_state()
    finally:
        ms.last_snapshot = original
    return payload["markets"][ms.symbol]


def test_context_bars_reach_the_dashboard_payload():
    mkt = _market_payload({
        "context_bars": {"m5": 5000, "m15": 1666, "h1": 421, "h4": 109},
        "h1_context_ok": True,
    })
    assert mkt["context_bars"] == {"m5": 5000, "m15": 1666, "h1": 421, "h4": 109}
    assert mkt["h1_context_ok"] is True


def test_short_h1_context_is_forwarded_as_not_ok():
    """Le cas qui compte : un contexte dégradé doit être visible côté dashboard,
    pas silencieusement absent."""
    mkt = _market_payload({
        "context_bars": {"m5": 500, "m15": 168, "h1": 43, "h4": 12},
        "h1_context_ok": False,
    })
    assert mkt["context_bars"]["h1"] == 43
    assert mkt["h1_context_ok"] is False


def test_payload_tolerates_a_snapshot_without_context_bars():
    """Avant le premier tick, last_snapshot ne porte pas encore ces clés : le payload
    doit rester constructible (le dashboard masque la ligne quand la valeur est nulle)."""
    mkt = _market_payload({})
    assert "context_bars" in mkt
    assert mkt["context_bars"] is None
