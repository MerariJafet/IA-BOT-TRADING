"""Generador de secuencias multi-escala basado en tokens de Binance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WINDOWS = ["2s", "1min", "5min", "15min", "1h", "1d"]


def compute_features(df: pd.DataFrame) -> pd.Series:
    """Calcula estadísticas básicas y momentum para una ventana."""
    if df.empty:
        return pd.Series(dtype=float)

    price = df["price"]
    volume = df["volume"]
    returns = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    momentum = 0.0
    if price.iloc[0] != 0:
        momentum = (price.iloc[-1] - price.iloc[0]) / price.iloc[0]

    features = {
        "price_mean": float(price.mean()),
        "price_std": float(price.std(ddof=0)),
        "price_skew": float(price.skew()),
        "price_kurt": float(price.kurt()),
        "volume_sum": float(volume.sum()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=0)),
        "momentum": float(momentum),
    }
    return pd.Series(features)


def generate_sequences(
    symbol: str,
    input_dir: str = "data/tokens/",
    output_dir: str = "data/sequences/",
    windows: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Combina tokens en secuencias multi-escala y extrae features estadísticos."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if windows is None:
        windows = DEFAULT_WINDOWS

    path = Path(input_dir) / f"{symbol}_dollar_tokens.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de tokens: {path}")

    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("El DataFrame debe contener columna 'timestamp' para resampling.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    results: Dict[str, pd.DataFrame] = {}
    for window in windows:
        grouped = df.resample(window)
        res = grouped.apply(compute_features).dropna(how="all")
        if not res.empty:
            res.index.name = "timestamp"
            out_path = Path(output_dir) / f"{symbol}_seq_{window}.parquet"
            res.to_parquet(out_path)
            logger.info("✅ Secuencia %s generada con %s filas → %s", window, len(res), out_path)
        else:
            logger.warning("⚠️ Secuencia %s no generó datos (dataset insuficiente).", window)
        results[window] = res

    return results


__all__ = ["compute_features", "generate_sequences"]
