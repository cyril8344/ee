"""Tests for the liquidity-pocket (Equal Highs/Lows) filter on Strategy B Order Blocks."""
import numpy as np
import pandas as pd

import strategy_ict as ict


def _flat_df_with_ob(n=60):
    """OHLC series: mostly flat, with one clean SHORT Order Block (green candle at
    index 20 followed by a strong bearish impulse) and no nearby liquidity pocket."""
    idx = pd.date_range("2026-07-01 06:00", periods=n, freq="5min", tz="UTC")
    base = 1.1000
    open_ = np.full(n, base)
    close = np.full(n, base)
    high = np.full(n, base + 0.0003)
    low = np.full(n, base - 0.0003)

    open_[20], close[20], high[20], low[20] = 1.1000, 1.1006, 1.1007, 1.0999
    for j, lo in zip([21, 22, 23], [1.0985, 1.0970, 1.0960]):
        open_[j] = 1.1000
        close[j] = lo + 0.0003
        high[j] = open_[j] + 0.0002
        low[j] = lo

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_find_equal_levels_detects_clustered_swings():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-07-01 08:00", periods=40, freq="5min", tz="UTC")
    close = 1.1000 + rng.normal(0, 0.00005, 40).cumsum()
    high = close + np.abs(rng.normal(0.0001, 0.00002, 40))
    low = close - np.abs(rng.normal(0.0001, 0.00002, 40))

    # Deux swing highs quasi égaux et isolés (voisins nettement plus bas)
    high[9], high[10], high[11] = 1.1010, 1.1030, 1.1010
    high[24], high[25], high[26] = 1.1010, 1.10295, 1.1010

    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    levels = ict._find_equal_levels(df, "SHORT", atr_val=0.0005)
    assert any(abs(lv - 1.10298) < 0.001 for lv in levels)


def test_ob_require_liquidity_filters_out_ob_without_nearby_pocket():
    """An OB is only accepted with OB_REQUIRE_LIQUIDITY=True if it sits near a
    detected Equal High/Low — must never find MORE OBs than with it disabled."""
    df = _flat_df_with_ob()
    atr_val = 0.0008

    obs_off = ict._find_order_blocks(df, "SHORT", atr_val)
    assert len(obs_off) == 1

    ict.OB_REQUIRE_LIQUIDITY = True
    try:
        obs_on = ict._find_order_blocks(df, "SHORT", atr_val)
    finally:
        ict.OB_REQUIRE_LIQUIDITY = False

    assert len(obs_on) == 0
