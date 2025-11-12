"""
Backtester - Sistema de backtesting con datos reales.

Este módulo ejecuta estrategias de trading sobre datos históricos reales
y calcula métricas detalladas de rendimiento para evaluación.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.core.policy_agent import PolicyAgent

logger = get_logger(__name__)


class Backtester:
    """Sistema de backtesting para estrategias de trading."""

    def __init__(
        self,
        data_path: str = "data/real/BTCUSDT_1m.parquet",
        pattern_strength_path: str = "data/reinforcement/pattern_strengths.parquet",
    ):
        """
        Inicializa el backtester.

        Args:
            data_path: Ruta a datos históricos OHLCV
            pattern_strength_path: Ruta a pattern strengths
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Datos no encontrados: {data_path}")

        self.df = pd.read_parquet(self.data_path)

        # Inicializar agente
        try:
            self.agent = PolicyAgent(pattern_strength_path=pattern_strength_path)
        except FileNotFoundError:
            logger.warning(
                "⚠️ Pattern strengths no encontrado, usando agente básico"
            )
            self.agent = PolicyAgent()

        logger.info(f"📊 Backtester inicializado con {len(self.df)} velas")

    def run(self) -> dict:
        """
        Ejecuta el backtest completo.

        Returns:
            Diccionario con métricas de rendimiento
        """
        logger.info("🚀 Iniciando backtesting...")

        trades = []
        returns = []

        for i in range(1, len(self.df)):
            # Obtener observación previa
            obs = self.df.iloc[i - 1]

            # Obtener decisión del agente
            decision = self.agent.decide(obs)

            # Calcular cambio de precio
            price_change = (
                self.df["close"].iloc[i] - obs["close"]
            ) / obs["close"]

            # Calcular retorno del trade
            trade_return = decision * price_change

            returns.append(trade_return)
            trades.append(
                {
                    "index": i,
                    "timestamp": self.df["open_time"].iloc[i]
                    if "open_time" in self.df.columns
                    else i,
                    "decision": decision,
                    "price_change": price_change,
                    "return": trade_return,
                }
            )

        # Calcular métricas
        total_pnl = float(np.sum(returns))
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9))
        cumulative_returns = np.cumsum(returns)
        max_drawdown = float(np.min(cumulative_returns))

        win_trades = sum(1 for r in returns if r > 0)
        lose_trades = sum(1 for r in returns if r < 0)
        win_rate = win_trades / len(returns) if returns else 0

        report = {
            "pnl_total": total_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": len(returns),
            "win_trades": win_trades,
            "lose_trades": lose_trades,
            "win_rate": float(win_rate),
        }

        # Guardar trades detallados
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        trades_df = pd.DataFrame(trades)
        trades_path = reports_dir / "backtest_trades.parquet"
        trades_df.to_parquet(trades_path, index=False)

        # Guardar métricas
        metrics_path = reports_dir / "backtest_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"✅ Backtest completo: PnL={total_pnl:.4f}, "
            f"Sharpe={sharpe:.3f}, DD={max_drawdown:.3f}, "
            f"WinRate={win_rate:.2%}"
        )
        logger.info(f"📄 Trades guardados en {trades_path}")
        logger.info(f"📄 Métricas guardadas en {metrics_path}")

        return report


if __name__ == "__main__":
    backtester = Backtester()
    metrics = backtester.run()

    print("\n📊 Resultados del Backtesting:")
    print(f"  PnL Total: {metrics['pnl_total']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.3f}")
    print(f"  Trades: {metrics['num_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
