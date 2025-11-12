from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.feature_embedding import build_feature_matrix
from src.core.pattern_clustering import cluster_embeddings


def test_embedding_and_clustering(tmp_path, monkeypatch):
    seq_dir = tmp_path / "data" / "sequences"
    seq_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "price_mean": [100, 101, 102],
            "price_std": [1, 2, 3],
            "price_skew": [0.1, 0.2, 0.3],
            "price_kurt": [3, 3, 3],
            "volume_sum": [10, 11, 12],
            "return_mean": [0.01, 0.02, 0.03],
            "return_std": [0.1, 0.1, 0.1],
            "momentum": [0.01, 0.02, 0.03],
        }
    )
    df.to_parquet(seq_dir / "BTCUSDT_seq_1m.parquet")

    emb = build_feature_matrix(seq_dir=str(seq_dir))
    assert "pca_1" in emb.columns

    report = cluster_embeddings(
        path="data/embeddings/feature_embeddings.parquet",
        method="kmeans",
        n_clusters=2,
    )
    assert "silhouette" in report
