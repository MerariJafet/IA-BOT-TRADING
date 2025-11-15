import pandas as pd

from src.quant.signal_router import SignalRouter


def test_router_output_format():
    router = SignalRouter()

    ticks = []
    base_mid = 50000.0
    for idx in range(120):
        spread = 0.5 + (idx % 5) * 0.1
        mid_price = base_mid + idx * 0.2
        ticks.append(
            {
                "mid": mid_price,
                "bid": mid_price - spread,
                "ask": mid_price + spread,
                "volume_buy": 1.0 + (idx % 3) * 0.5,
                "volume_sell": 1.0 + ((idx + 1) % 3) * 0.4,
            }
        )

    frame = pd.DataFrame(ticks)
    result = router.route(frame)

    assert hasattr(result, "signal")
    assert hasattr(result, "prob_up")
    assert hasattr(result, "prob_down")
    assert hasattr(result, "filters_passed")
    assert hasattr(result, "filters_reason")
