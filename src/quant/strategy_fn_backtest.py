from __future__ import annotations

from typing import List

import pandas as pd

from src.quant.filters.filters import FilterConfig
from src.quant.micro.microstructure import MicrostructureConfig
from src.quant.signal_router import SignalRouter
from src.quant.signals.signal_engine import SignalThresholds

thresholds = SignalThresholds()
micro_config = MicrostructureConfig()
filter_config = FilterConfig()
router = SignalRouter(
    thresholds=thresholds,
    micro_config=micro_config,
    filter_config=filter_config,
)

_history: List[dict] = []


def _append_history(row: pd.Series) -> None:
    mid = float(row["close"])
    bid = mid * (1 - 0.00005)
    ask = mid * (1 + 0.00005)
    volume = float(row.get("volume", 0))
    buy_volume = volume * 0.55
    sell_volume = max(volume - buy_volume, 0.0)

    _history.append(
        {
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "volume_buy": buy_volume,
            "volume_sell": sell_volume,
        }
    )

    if len(_history) > 300:
        _history.pop(0)


def _to_router_frame() -> pd.DataFrame:
    return pd.DataFrame(_history)


def strategy_fn(row: pd.Series) -> str:
    _append_history(row)
    frame = _to_router_frame()

    if frame.empty:
        return "NONE"

    result = router.route(frame)
    return result.signal
