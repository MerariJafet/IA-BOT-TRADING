"""Translate routed signals into concrete trading actions with risk controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DecisionConfig:
    max_position: float = 0.01
    stop_loss_pct: float = 0.003
    take_profit_pct: float = 0.006


class TradeDecision:
    """Hold minimal state to decide entries, exits, and holds."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.cfg = config or DecisionConfig()
        self.position: float = 0.0
        self.entry_price: Optional[float] = None

    def decide(self, routed_signal: Dict[str, float], price: float) -> Dict[str, float | str]:
        signal = routed_signal.get("signal", "NONE")
        prob_up = routed_signal.get("prob_up", 0.0)
        prob_down = routed_signal.get("prob_down", 0.0)

        if signal == "BUY" and self.position <= 0:
            return self._enter_trade("BUY", price, prob_up)
        if signal == "SELL" and self.position >= 0:
            return self._enter_trade("SELL", price, prob_down)

        return self._manage_open_position(price)

    def _enter_trade(self, side: str, price: float, confidence: float) -> Dict[str, float | str]:
        size = self.cfg.max_position
        self.position = size if side == "BUY" else -size
        self.entry_price = price
        return {
            "action": side,
            "size": size,
            "entry_price": price,
            "confidence": confidence,
        }

    def _manage_open_position(self, price: float) -> Dict[str, float | str]:
        if self.position == 0 or self.entry_price is None:
            return {"action": "HOLD", "reason": "No position"}

        direction = 1 if self.position > 0 else -1
        pnl_pct = direction * ((price - self.entry_price) / self.entry_price)

        if pnl_pct <= -self.cfg.stop_loss_pct:
            return self._exit_trade("STOP_LOSS", price)
        if pnl_pct >= self.cfg.take_profit_pct:
            return self._exit_trade("TAKE_PROFIT", price)

        return {"action": "HOLD", "reason": "Monitoring position"}

    def _exit_trade(self, reason: str, price: float) -> Dict[str, float | str]:
        self.position = 0.0
        entry = self.entry_price or price
        pnl_pct = (price - entry) / entry if entry else 0.0
        self.entry_price = None
        return {
            "action": "EXIT",
            "reason": reason,
            "exit_price": price,
            "pnl_pct": pnl_pct,
        }
