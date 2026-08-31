"""Regression: le compte à rebours affiché sur le trade actif doit se baser sur le
MÊME plafond que celui réellement appliqué par la boucle de trading.

La boucle ferme sur `pos.meta.get("max_duration_min", strategy.MAX_TRADE_MINUTES)`,
tandis que _position_payload() lisait directement strategy.MAX_TRADE_MINUTES (75) —
ES stocke son propre plafond (45 min) dans pos.meta, donc le dashboard annonçait
30 minutes de sursis qui n'existaient pas.
"""
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import main as main_module
import strategy
from broker import Position


class _FakeBroker:
    contract_size = 100.0

    def get_price(self):
        return 2000.0


class _FakeMS:
    def __init__(self, pos):
        self.position = pos
        self.broker = _FakeBroker()


def _remaining_min(meta, age_min):
    pos = Position(
        ticket=1, direction="long", entry=2000.0, volume=0.1,
        stop_loss=1990.0, take_profit1=2007.0, take_profit2=2018.0,
        open_time=datetime.now(timezone.utc) - timedelta(minutes=age_min),
        risk_amount=100.0, session="London", meta=meta,
    )
    payload = main_module._position_payload(_FakeMS(pos))
    return payload["remaining_seconds"] / 60.0


def test_countdown_uses_position_specific_cap(monkeypatch):
    monkeypatch.setattr(strategy, "MAX_TRADE_MINUTES", 75)
    # Position ES : plafond propre à 45 min, ouverte depuis 30 → il reste 15, pas 45.
    assert _remaining_min({"max_duration_min": 45}, age_min=30) == 15.0


def test_countdown_falls_back_to_global_cap_when_meta_absent(monkeypatch):
    monkeypatch.setattr(strategy, "MAX_TRADE_MINUTES", 75)
    # XAUUSD/EURUSD ne stockent pas max_duration_min dans meta (cf. main.py::_open_trade).
    assert _remaining_min({}, age_min=30) == 45.0


def test_countdown_never_goes_negative(monkeypatch):
    monkeypatch.setattr(strategy, "MAX_TRADE_MINUTES", 75)
    assert _remaining_min({"max_duration_min": 45}, age_min=200) == 0.0
