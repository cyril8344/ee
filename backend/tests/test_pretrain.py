"""Tests for pretrain.py's shared-state isolation between standalone runs and
internal walk-forward/Optuna windows (write_to_db=False)."""
import database as db
import pretrain

db.init_db()


def test_write_to_db_false_does_not_pollute_shared_pretrain_state():
    """A walk-forward window (write_to_db=False) must never overwrite the result
    shown in the standalone "Pré-entraînement" panel — only a real, user-triggered
    pretrain run (write_to_db=True) should update that shared state."""
    r1 = pretrain.run_pretrain("2024-01-01", "2024-01-10", symbol="XAUUSD",
                                reset=True, write_to_db=True)
    before = pretrain.get_progress()["last_result"]
    assert before is not None
    assert before["period"] == r1["period"]
    assert pretrain.get_last_results()["A"]["period"] == r1["period"]

    r2 = pretrain.run_pretrain("2024-02-01", "2024-02-10", symbol="XAUUSD",
                                reset=True, write_to_db=False)
    assert r2["period"] != r1["period"]

    after = pretrain.get_progress()["last_result"]
    assert after["period"] == before["period"]
    assert pretrain.get_last_results()["A"]["period"] == r1["period"]
