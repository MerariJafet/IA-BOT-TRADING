"""Construcción de librería persistente de patrones derivados de clusters."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


def build_pattern_library(
    cluster_path: str = "data/patterns/pattern_clusters.parquet",
    out_path: str = "data/patterns/pattern_library.parquet",
) -> pd.DataFrame:
    """Calcula centroides y métricas por cluster y las persiste."""
    df = pd.read_parquet(cluster_path)
    if "cluster" not in df.columns:
        raise ValueError("El archivo de clusters debe contener la columna 'cluster'.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    grouped = df.groupby("cluster")
    summary = grouped[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(map(str, col)).strip() for col in summary.columns]
    summary["cluster_size"] = grouped.size()
    summary["feature_count"] = len([col for col in df.columns if col != "cluster"])
    summary = summary.reset_index()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path, index=False)
    logger.info("✅ Pattern library creada con %s patrones → %s", len(summary), out_path)
    return summary


if __name__ == "__main__":
    build_pattern_library()
