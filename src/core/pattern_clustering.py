"""Agrupamiento de embeddings para descubrir patrones de mercado."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.core.logger import get_logger

logger = get_logger(__name__)


def cluster_embeddings(
    path: str = "data/embeddings/feature_embeddings.parquet",
    method: str = "kmeans",
    n_clusters: int = 8,
) -> Dict[str, float]:
    """Agrupa embeddings y genera reporte de calidad."""
    df = pd.read_parquet(path)
    numeric = df.select_dtypes("number")
    if numeric.empty:
        raise ValueError("No numeric columns available for clustering.")

    if method.lower() == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    else:
        model = DBSCAN(eps=0.8, min_samples=5)

    labels = model.fit_predict(numeric)
    df["cluster"] = labels

    silhouette = -1.0
    unique_labels = set(labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        try:
            silhouette = float(silhouette_score(numeric, labels))
        except Exception:
            silhouette = -1.0

    pattern_dir = Path("data/patterns")
    pattern_dir.mkdir(parents=True, exist_ok=True)
    out_file = pattern_dir / "pattern_clusters.parquet"
    df.to_parquet(out_file)

    report = {
        "method": method,
        "n_clusters": int(getattr(model, "n_clusters", len(unique_labels))),
        "silhouette": silhouette,
        "rows": int(len(df)),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "pattern_clustering_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    logger.info("✅ Clustering %s completado (silhouette=%.3f) → %s", method, silhouette, out_file)
    return report


if __name__ == "__main__":
    cluster_embeddings()
