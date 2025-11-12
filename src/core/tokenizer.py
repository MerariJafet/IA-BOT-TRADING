"""Tokenizadores dinámicos (Dollar/Imbalance bars) para datos de Binance."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal

import pandas as pd

from src.core.logger import get_logger

TokenMethod = Literal["dollar", "imbalance"]

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"price", "volume"}
IMBALANCE_COLUMNS = REQUIRED_COLUMNS | {"side"}


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {', '.join(sorted(missing))}")


def _generate_bars(
    df: pd.DataFrame,
    metric_fn: Callable[[Dict[str, object]], float],
    threshold: float,
    use_absolute: bool = False,
) -> pd.DataFrame:
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    records: List[Dict[str, object]] = []
    chunk: List[Dict[str, object]] = []
    bar_id = 0
    accumulator = 0.0

    for row in df.to_dict(orient="records"):
        value = metric_fn(row)
        accumulator = accumulator + value
        row["bar_id"] = bar_id
        chunk.append(row)

        compare_value = abs(accumulator) if use_absolute else accumulator
        if compare_value >= threshold:
            records.extend(chunk)
            chunk = []
            accumulator = 0.0
            bar_id += 1

    if chunk:
        records.extend(chunk)

    if not records:
        # Empty DataFrame preserving expected schema.
        return pd.DataFrame(columns=list(df.columns) + ["bar_id"])

    return pd.DataFrame.from_records(records)


def dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Agrupa trades hasta acumular un volumen en dólares igual al threshold."""
    _validate_columns(df, REQUIRED_COLUMNS)
    work_df = df.copy()
    work_df["dollar_value"] = work_df["price"] * work_df["volume"]
    bars = _generate_bars(
        work_df,
        metric_fn=lambda row: row["dollar_value"],
        threshold=threshold,
        use_absolute=False,
    )
    return bars


def imbalance_bars(df: pd.DataFrame, imbalance_threshold: float) -> pd.DataFrame:
    """Agrupa trades basándose en el desequilibrio entre órdenes de compra y venta."""
    _validate_columns(df, IMBALANCE_COLUMNS)
    work_df = df.copy()
    side_sign = (
        work_df["side"].astype(str).str.lower().map({"buy": 1, "sell": -1}).fillna(0).astype(float)
    )
    work_df["imbalance"] = work_df["volume"] * work_df["price"] * side_sign
    bars = _generate_bars(
        work_df,
        metric_fn=lambda row: row["imbalance"],
        threshold=imbalance_threshold,
        use_absolute=True,
    )
    return bars


def tokenize_symbol(
    symbol: str,
    method: TokenMethod = "dollar",
    threshold: float = 10_000.0,
    input_dir: str = "data/historical_1y_parquet",
    output_dir: str = "data/tokens",
) -> None:
    """Genera tokens dinámicos y guarda los resultados en Parquet."""
    token_method = method.lower()
    input_path = Path(input_dir) / f"{symbol}.parquet"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")

    df = pd.read_parquet(input_path)

    if token_method == "dollar":
        token_df = dollar_bars(df, threshold=threshold)
    elif token_method == "imbalance":
        token_df = imbalance_bars(df, imbalance_threshold=threshold)
    else:
        raise ValueError("method must be either 'dollar' or 'imbalance'")

    out_path = output_dir_path / f"{symbol}_{token_method}_tokens.parquet"
    token_df.to_parquet(out_path, index=False)
    logger.info("Tokenización completada: %s (%s tokens)", out_path, len(token_df))


__all__ = ["dollar_bars", "imbalance_bars", "tokenize_symbol"]
