"""Tests for pretrain.diag_real_trades_early_exit_by_hour() — follow-up to
diag_real_trades_by_day_volatility (PR #340/#342): the ATR-of-the-day correlation
came back essentially zero (0.019) on real twelvedata-sourced data, so this
diagnostic checks a different angle — does the early_exit rate depend on the
precise hour of entry instead of the day's average volatility?
"""
import pretrain


def _trade(hour_cet, exit_reason):
    return {"hour_cet": hour_cet, "exit_reason": exit_reason}


def test_groups_by_hour_and_computes_early_exit_pct():
    trades = (
        [_trade(8, "early_exit") for _ in range(3)]
        + [_trade(8, "tp1")]
        + [_trade(15, "tp2") for _ in range(4)]
    )
    result = pretrain.diag_real_trades_early_exit_by_hour(_trades=trades)

    hours = {h["hour"]: h for h in result["hours"]}
    assert hours[8]["n"] == 4
    assert hours[8]["early_exit_pct"] == 75.0
    assert hours[15]["n"] == 4
    assert hours[15]["early_exit_pct"] == 0.0


def test_hours_are_sorted_ascending():
    trades = [_trade(17, "tp1"), _trade(8, "tp1"), _trade(11, "tp1")]
    result = pretrain.diag_real_trades_early_exit_by_hour(_trades=trades)
    assert [h["hour"] for h in result["hours"]] == [8, 11, 17]


def test_skips_trades_with_no_hour_cet():
    trades = [_trade(None, "tp1"), _trade(9, "early_exit")]
    result = pretrain.diag_real_trades_early_exit_by_hour(_trades=trades)
    assert len(result["hours"]) == 1
    assert result["hours"][0]["hour"] == 9


def test_returns_note_when_no_trades():
    result = pretrain.diag_real_trades_early_exit_by_hour(_trades=[])
    assert result["hours"] == []
    assert "note" in result
