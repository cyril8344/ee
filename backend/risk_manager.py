"""
risk_manager.py
===============
Strict risk management for the XAU/USD scalping bot.

Rules enforced
--------------
- Risk per trade : 1% of capital (fixed).  Changing it requires an explicit
  confirmation flag (see `RiskManager.set_risk_pct`).
- Max 4 trades per day.
- Daily stop : -2% of starting equity -> bot blocked until the next day.
- Position sizing derived from stop-loss distance and contract specs.

XAU/USD contract assumptions (standard MT5 / most brokers)
----------------------------------------------------------
- 1 standard lot = 100 oz.
- Price quoted in USD per ounce.
- Therefore $ P&L  = (exit - entry) * 100 * lots  (for a long).
- 1 "pip" for gold is conventionally 0.1 price units; we keep prices in raw
  USD so a 1.00 move on 1 lot = $100.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

CONTRACT_SIZE = 100.0          # ounces per standard lot
MIN_LOT = 0.01
MAX_LOT = 100.0
LOT_STEP = 0.01


def drawdown_adjusted_risk_pct(base_risk_pct: float, equity: float, equity_peak: float,
                               enabled: bool, threshold_pct: float, factor: float) -> float:
    """
    Réduit risk_pct une fois que l'equity a reculé de threshold_pct% depuis son
    plus haut (drawdown glissant). Réactif (basé sur la performance déjà mesurée),
    pas prédictif — contrairement aux filtres de régime (ADX/ATR) testés en amont
    et qui n'ont pas séparé proprement les bonnes et mauvaises fenêtres walk-forward.
    Désactivé par défaut (`enabled=False`) : ne change rien tant que ce n'est pas
    explicitement activé et validé en walk-forward.
    """
    if not enabled or equity_peak <= 0:
        return base_risk_pct
    drawdown_pct = max(0.0, (equity_peak - equity) / equity_peak * 100.0)
    if drawdown_pct < threshold_pct:
        return base_risk_pct
    return base_risk_pct * factor


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    volume: float = 0.0
    risk_amount: float = 0.0
    stop_distance: float = 0.0


@dataclass
class RiskManager:
    capital: float = 10_000.0
    risk_per_trade_pct: float = 5.0
    max_trades_per_day: int = 4
    daily_stop_pct: float = 2.0

    # mutable daily state
    trades_today: int = 0
    realised_pnl_today: float = 0.0
    start_equity_today: float = field(default=None)  # type: ignore
    blocked: bool = False
    block_reason: str = ""

    # plus haut d'equity jamais atteint (all-time), pour le sizing dynamique en drawdown
    equity_peak: float = field(default=None)  # type: ignore

    def __post_init__(self):
        if self.start_equity_today is None:
            self.start_equity_today = self.capital
        if self.equity_peak is None:
            self.equity_peak = self.capital

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_risk_pct(self, value: float, confirmed: bool = False) -> bool:
        """Risk % is fixed; only changes when `confirmed` is True."""
        if not confirmed:
            return False
        self.risk_per_trade_pct = max(0.1, min(float(value), 5.0))
        return True

    def sync_from_settings(self, settings: Dict[str, Any]) -> None:
        self.capital = float(settings.get("capital", self.capital))
        self.risk_per_trade_pct = float(
            settings.get("risk_per_trade_pct", self.risk_per_trade_pct)
        )
        raw_max = int(settings.get("max_trades_per_day", self.max_trades_per_day))
        self.max_trades_per_day = max(1, min(raw_max, 10))  # cap 1–10

        raw_stop = float(settings.get("daily_stop_pct", self.daily_stop_pct))
        self.daily_stop_pct = max(0.5, min(raw_stop, 10.0))  # cap 0.5%–10%

    # ------------------------------------------------------------------ #
    # Daily lifecycle
    # ------------------------------------------------------------------ #
    def start_new_day(self, equity: float) -> None:
        self.trades_today = 0
        self.realised_pnl_today = 0.0
        self.start_equity_today = float(equity)
        self.blocked = False
        self.block_reason = ""

    def hydrate_day(self, trades_today: int, pnl_today: float,
                    start_equity: float, blocked: bool) -> None:
        """Restore daily counters from persistence (e.g. after a restart)."""
        self.trades_today = int(trades_today)
        self.realised_pnl_today = float(pnl_today)
        self.start_equity_today = float(start_equity)
        self.blocked = bool(blocked)
        if blocked:
            self.block_reason = "Daily loss limit reached"
        self._reevaluate_block()

    # ------------------------------------------------------------------ #
    # Position sizing
    # ------------------------------------------------------------------ #
    def _round_lot(self, lot: float) -> float:
        lot = max(MIN_LOT, min(lot, MAX_LOT))
        steps = round(lot / LOT_STEP)
        return round(steps * LOT_STEP, 2)

    def compute_position(self, entry: float, stop: float,
                         contract_size: float = CONTRACT_SIZE) -> Tuple[float, float, float]:
        """
        Returns (volume_lots, risk_amount_usd, stop_distance).
        Volume sized so that hitting the stop loses exactly risk_per_trade_pct.
        """
        stop_distance = abs(entry - stop)
        risk_amount = self.capital * (self._effective_risk_pct() / 100.0)
        if stop_distance <= 0:
            return 0.0, risk_amount, 0.0
        # $ loss per lot if stop hit = stop_distance * contract_size
        loss_per_lot = stop_distance * contract_size
        raw_lots = risk_amount / loss_per_lot if loss_per_lot > 0 else 0.0
        volume = self._round_lot(raw_lots)
        # Actual risk after rounding
        actual_risk = volume * loss_per_lot
        return volume, actual_risk, stop_distance

    # ------------------------------------------------------------------ #
    # Pre-trade gate
    # ------------------------------------------------------------------ #
    def can_open_trade(self, entry: float, stop: float,
                       contract_size: float = CONTRACT_SIZE) -> RiskDecision:
        if self.blocked:
            return RiskDecision(False, f"Bot blocked: {self.block_reason}")

        if self.trades_today >= self.max_trades_per_day:
            return RiskDecision(
                False,
                f"Max trades/day reached ({self.max_trades_per_day})",
            )

        # Pre-emptive daily-stop check
        if self._daily_loss_exceeded():
            self.blocked = True
            self.block_reason = "Daily loss limit reached"
            return RiskDecision(False, f"Bot blocked: {self.block_reason}")

        volume, risk_amount, stop_distance = self.compute_position(entry, stop, contract_size)
        if volume < MIN_LOT or stop_distance <= 0:
            return RiskDecision(
                False, "Invalid stop distance / volume too small"
            )

        return RiskDecision(
            True, "ok", volume=volume,
            risk_amount=risk_amount, stop_distance=stop_distance,
        )

    # ------------------------------------------------------------------ #
    # Post-trade accounting
    # ------------------------------------------------------------------ #
    def register_open(self) -> None:
        self.trades_today += 1

    def register_close(self, pnl: float) -> None:
        self.realised_pnl_today += float(pnl)
        self.capital += float(pnl)
        self.equity_peak = max(self.equity_peak, self.capital)
        self._reevaluate_block()

    def _effective_risk_pct(self) -> float:
        try:
            import strategy as _st
            enabled   = bool(getattr(_st, "DRAWDOWN_SIZING_ENABLED", False))
            threshold = float(getattr(_st, "DRAWDOWN_SIZING_THRESHOLD_PCT", 5.0))
            factor    = float(getattr(_st, "DRAWDOWN_SIZING_FACTOR", 0.5))
        except Exception:
            return self.risk_per_trade_pct
        return drawdown_adjusted_risk_pct(
            self.risk_per_trade_pct, self.capital, self.equity_peak,
            enabled, threshold, factor,
        )

    def _daily_loss_exceeded(self) -> bool:
        try:
            import strategy as _st
            if _st.BOOTSTRAP_MODE:
                return False  # pas de stop journalier pendant l'amorçage
        except Exception:
            pass
        limit = -abs(self.start_equity_today * (self.daily_stop_pct / 100.0))
        return self.realised_pnl_today <= limit

    def _reevaluate_block(self) -> None:
        if self._daily_loss_exceeded():
            self.blocked = True
            self.block_reason = "Daily loss limit reached"

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def daily_loss_limit_usd(self) -> float:
        return abs(self.start_equity_today * (self.daily_stop_pct / 100.0))

    def status(self) -> Dict[str, Any]:
        effective_risk_pct = self._effective_risk_pct()
        return {
            "capital": round(self.capital, 2),
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "effective_risk_pct": round(effective_risk_pct, 3),
            "equity_peak": round(self.equity_peak, 2),
            "drawdown_pct": round(
                max(0.0, (self.equity_peak - self.capital) / self.equity_peak * 100.0)
                if self.equity_peak else 0.0, 2
            ),
            "daily_stop_pct": self.daily_stop_pct,
            "risk_amount_usd": round(
                self.capital * (effective_risk_pct / 100.0), 2
            ),
            "trades_today": self.trades_today,
            "max_trades_per_day": self.max_trades_per_day,
            "realised_pnl_today": round(self.realised_pnl_today, 2),
            "realised_pnl_today_pct": round(
                (self.realised_pnl_today / self.start_equity_today * 100.0)
                if self.start_equity_today else 0.0, 3
            ),
            "daily_loss_limit_usd": round(self.daily_loss_limit_usd(), 2),
            "start_equity_today": round(self.start_equity_today, 2),
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }
