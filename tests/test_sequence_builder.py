from pathlib import Path

import pandas as pd

from src.core.sequence_builder import generate_sequences


def test_generate_sequences(tmp_path):
    data_dir = tmp_path / "data" / "tokens"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="S"),
            "price": 100 + pd.Series(range(200)) * 0.1,
            "volume": 1.0,
        }
    )
    df.to_parquet(data_dir / "BTCUSDT_dollar_tokens.parquet")

    output_dir = tmp_path / "out"
    results = generate_sequences(
        "BTCUSDT",
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        windows=["2s", "1m"],
    )

    assert set(results.keys()) == {"2s", "1m"}
    assert all("price_mean" in res.columns for res in results.values())
    assert all("volume_sum" in res.columns for res in results.values())
    assert all(len(res) > 0 for res in results.values())
    for window in results:
        expected_path = output_dir / f"BTCUSDT_seq_{window}.parquet"
        assert expected_path.exists()
