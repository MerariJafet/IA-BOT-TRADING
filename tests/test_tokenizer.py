import pandas as pd

from src.core.tokenizer import dollar_bars, imbalance_bars


def _sample_df():
    return pd.DataFrame(
        {
            "price": [100, 102, 101, 103],
            "volume": [10, 20, 15, 5],
            "side": ["buy", "sell", "buy", "sell"],
        }
    )


def test_dollar_bars_basic():
    df = _sample_df()
    result = dollar_bars(df, threshold=1000)
    assert "bar_id" in result.columns
    assert not result.empty
    assert result["bar_id"].iloc[0] == 0


def test_imbalance_bars_basic():
    df = _sample_df()
    result = imbalance_bars(df, imbalance_threshold=500)
    assert "bar_id" in result.columns
    assert not result.empty
    assert set(result["side"]) <= {"buy", "sell"}
