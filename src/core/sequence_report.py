"""Reporte de métricas agregadas sobre secuencias multi-escala."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


def summarize_sequences(
    path: str = "data/sequences/",
    out: str = "reports/sequence_stats.json",
) -> Dict[str, dict]:
    """Recorre los Parquets generados y sintetiza métricas básicas."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, dict] = {}

    for file_path in Path(path).glob("*_seq_*.parquet"):
        df = pd.read_parquet(file_path)
        stats[file_path.name] = {
            "rows": len(df),
            "cols": list(df.columns),
            "price_mean_avg": float(df["price_mean"].mean()) if "price_mean" in df.columns else 0.0,
            "volume_total_avg": (
                float(df["volume_sum"].mean()) if "volume_sum" in df.columns else 0.0
            ),
        }

    with open(out, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    logger.info("✅ Reporte generado en %s", out)
    return stats


if __name__ == "__main__":
    summarize_sequences()
