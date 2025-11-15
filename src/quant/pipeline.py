"""Quant pipeline orchestrating signal routing and trade execution decisions."""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import pandas as pd

from src.quant.filters.filters import FilterConfig
from src.quant.micro.microstructure import MicrostructureConfig
from src.quant.signal_router import SignalRouter
from src.quant.signals.signal_engine import SignalThresholds
from src.quant.trade_decision import DecisionConfig, TradeDecision


class MarketDataBuffer:
    """Maintain a sliding window of microstructure-friendly ticks."""

    def __init__(self, maxlen: int = 240) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._buffer: deque[Dict[str, float]] = deque(maxlen=maxlen)

    def append(
        self,
        *,
        mid: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume_buy: float = 1.0,
        volume_sell: float = 1.0,
    ) -> pd.DataFrame:
        if mid <= 0:
            raise ValueError("mid price must be positive")

        if bid is None or ask is None:
            half_spread = max(mid * 0.0001, 1e-6)
            bid = mid - half_spread
            ask = mid + half_spread

        row = {
            "mid": float(mid),
            "bid": float(bid),
            "ask": float(ask),
            "volume_buy": float(max(volume_buy, 1e-9)),
            "volume_sell": float(max(volume_sell, 1e-9)),
        }
        self._buffer.append(row)
        return pd.DataFrame(self._buffer)

    def to_frame(self) -> pd.DataFrame:
        """Return current buffer contents as DataFrame."""
        return pd.DataFrame(self._buffer)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


class QuantPipeline:
    """Tie together signal routing, quantitative filters, and trade decisions."""

    def __init__(
        self,
        thresholds: Optional[SignalThresholds] = None,
        micro_config: Optional[MicrostructureConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        decision_config: Optional[DecisionConfig] = None,
        max_buffer: int = 240,
    ) -> None:
        self.router = SignalRouter(
            thresholds=thresholds,
            micro_config=micro_config,
            filter_config=filter_config,
        )
        self.trader = TradeDecision(decision_config)
        self.buffer = MarketDataBuffer(maxlen=max_buffer)
        thresholds = thresholds or SignalThresholds()
        micro_config = micro_config or MicrostructureConfig()
        self.min_window = max(thresholds.vol_window, micro_config.vol_window * 2)

    def add_tick(
        self,
        *,
        mid: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume_buy: float = 1.0,
        volume_sell: float = 1.0,
    ) -> pd.DataFrame:
        """Append a tick to the buffer and return the updated frame."""
        return self.buffer.append(
            mid=mid,
            bid=bid,
            ask=ask,
            volume_buy=volume_buy,
            volume_sell=volume_sell,
        )

    def process(self, market_frame: pd.DataFrame, price: Optional[float] = None) -> Dict[str, object]:
        """Run the quant pipeline using the provided market frame."""
        if market_frame.empty:
            return {
                "router": None,
                "decision": {"action": "HOLD", "reason": "NO_DATA"},
            }

        if len(market_frame) < self.min_window:
            return {
                "router": None,
                "decision": {
                    "action": "HOLD",
                    "reason": f"INSUFFICIENT_DATA_{len(market_frame)}",
                },
            }

        router_result = self.router.route(market_frame)
        router_dict = router_result.as_dict()
        decision = self.trader.decide(router_dict, price or float(market_frame["mid"].iloc[-1]))
        return {
            "router": router_dict,
            "decision": decision,
        }
