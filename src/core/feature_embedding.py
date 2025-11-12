"""Construcción de embeddings PCA a partir de secuencias multi-escala."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.logger import get_logger

logger = get_logger(__name__)


def build_feature_matrix(seq_dir: str = "data/sequences/", max_components: int = 10) -> pd.DataFrame:
    """Lee Parquets de secuencias y construye embeddings PCA."""
    files = sorted(Path(seq_dir).glob("*_seq_*.parquet"))
    if not files:
        raise FileNotFoundError("No sequence files found in data/sequences/")

    frames = []
    for file_path in files:
        df = pd.read_parquet(file_path)
        df = df.reset_index(drop=True)
        df["source_file"] = file_path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True).dropna()
    numeric = data.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric columns found for embeddings.")

    logger.info("Features shape before scaling: %s", numeric.shape)
    X = StandardScaler().fit_transform(numeric)

    n_components = min(max_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    emb = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(X_pca.shape[1])])
    emb["source_file"] = data["source_file"].values

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_embeddings.parquet"
    emb.to_parquet(out_path)
    logger.info("✅ Embedding guardado en %s con shape %s", out_path, emb.shape)
    return emb


if __name__ == "__main__":
    build_feature_matrix()
