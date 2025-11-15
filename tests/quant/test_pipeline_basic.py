import pandas as pd

from src.quant.pipeline import QuantPipeline


def test_pipeline_minimum_structure():
    pipeline = QuantPipeline()
    frame = pipeline.add_tick(
        mid=50000.0,
        bid=49999.5,
        ask=50000.5,
        volume_buy=1.2,
        volume_sell=0.8,
    )
    result = pipeline.process(frame, price=50000.0)

    assert "decision" in result
    assert result["decision"]["action"] in {"HOLD", "BUY", "SELL", "EXIT"}
    assert "router" in result
