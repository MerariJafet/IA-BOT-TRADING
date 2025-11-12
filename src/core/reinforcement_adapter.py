"""Adaptador de refuerzo para ajustar la fuerza de patrones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class ReinforcementAdapter:
    """Gestiona recompensas simuladas y fortalece patrones."""

    def __init__(self, library_path: str = "data/patterns/pattern_library.parquet", lr: float = 0.05):
        self.library_path = Path(library_path)
        if not self.library_path.exists():
            raise FileNotFoundError(f"No se encuentra la librería de patrones: {library_path}")
        self.library = pd.read_parquet(self.library_path).copy()
        self.library["reward"] = 0.0
        if "strength" not in self.library.columns:
            self.library["strength"] = 0.5
        self.lr = lr

    def simulate_rewards(self, seed: int = 42) -> None:
        np.random.seed(seed)
        simulated = np.random.normal(loc=0.0, scale=1.0, size=len(self.library))
        self.library["reward"] = simulated
        logger.info("🎯 Recompensas simuladas generadas.")

    def update_strength(self) -> Path:
        self.library["strength"] += self.lr * self.library["reward"]
        self.library["strength"] = self.library["strength"].clip(0, 1)

        out_dir = Path("data/reinforcement")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pattern_strengths.parquet"
        self.library.to_parquet(out_path, index=False)
        logger.info("✅ Fuerzas de patrones actualizadas y guardadas en %s", out_path)
        return out_path


if __name__ == "__main__":
    adapter = ReinforcementAdapter()
    adapter.simulate_rewards()
    adapter.update_strength()
