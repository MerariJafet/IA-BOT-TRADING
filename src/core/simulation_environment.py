"""
Simulation Environment - Entorno de simulación para evaluar políticas.

Este módulo implementa un simulador de mercado que ejecuta políticas de trading
y calcula métricas de desempeño como PnL, Sharpe ratio y drawdown máximo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class MarketSimulator:
    """Simulador de mercado para evaluar políticas de trading."""

    def __init__(
        self,
        sequence_path: str = "data/sequences/BTCUSDT_seq_1m.parquet",
        policy: Optional[Callable] = None,
        capital: float = 10000.0,
    ):
        """
        Inicializa el simulador de mercado.

        Args:
            sequence_path: Ruta al archivo parquet con secuencias de mercado
            policy: Función que toma una observación y retorna decisión (-1, 0, 1)
            capital: Capital inicial para simulación
        """
        self.sequence_path = Path(sequence_path)
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Secuencia no encontrada: {sequence_path}")

        self.df = pd.read_parquet(self.sequence_path)
        self.policy = policy
        self.capital = capital
        self.trades = []

        logger.info(
            f"📊 MarketSimulator inicializado con {len(self.df)} observaciones"
        )

    def run(self) -> dict:
        """
        Ejecuta la simulación de mercado.

        Returns:
            Diccionario con métricas de desempeño
        """
        logger.info("🚀 Iniciando simulación de mercado...")

        for i in range(1, len(self.df)):
            # Obtener decisión de la política
            decision = self.policy(self.df.iloc[i - 1]) if self.policy else 0

            # Calcular retorno
            ret = self.df["return_mean"].iloc[i]

            # Calcular PnL del trade
            pnl = decision * ret * self.capital
            self.trades.append(pnl)

        # Calcular métricas
        total_pnl = float(np.sum(self.trades))
        sharpe = float(np.mean(self.trades) / (np.std(self.trades) + 1e-8))
        cumulative = np.cumsum(self.trades)
        max_drawdown = float(np.min(cumulative) if len(cumulative) > 0 else 0.0)

        metrics = {
            "pnl_total": total_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": len(self.trades),
        }

        # Guardar reporte
        report_path = Path("reports")
        report_path.mkdir(exist_ok=True)
        report_file = report_path / "simulation_metrics.json"

        with open(report_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"✅ Simulación completada. PnL={total_pnl:.2f}, "
            f"Sharpe={sharpe:.3f}, DD={max_drawdown:.2f}"
        )

        return metrics


if __name__ == "__main__":
    # Simulación básica con política aleatoria
    def random_policy(obs):
        return np.random.choice([-1, 0, 1])

    sim = MarketSimulator(policy=random_policy)
    metrics = sim.run()
    print(json.dumps(metrics, indent=2))
