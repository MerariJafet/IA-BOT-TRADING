import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.pattern_library import build_pattern_library
from src.core.reinforcement_adapter import ReinforcementAdapter


def test_reinforcement_flow(tmp_path):
    patterns_dir = tmp_path / "data" / "patterns"
    patterns_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1],
            "pca_1": [0.1, 0.2, 0.3, 0.4],
            "pca_2": [0.5, 0.6, 0.7, 0.8],
        }
    )
    clusters_path = patterns_dir / "pattern_clusters.parquet"
    df.to_parquet(clusters_path)

    library_path = patterns_dir / "pattern_library.parquet"
    library = build_pattern_library(cluster_path=str(clusters_path), out_path=str(library_path))
    assert len(library) == 2

    adapter = ReinforcementAdapter(library_path=str(library_path))
    adapter.simulate_rewards()
    out_path = adapter.update_strength()
    assert Path(out_path).exists()
