"""
Self-Learning Agent - Agente con capacidad de aprendizaje adaptativo.

Este módulo extiende PolicyAgent para permitir refinamiento automático
de parámetros basado en resultados de backtesting.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.logger import get_logger
from src.core.policy_agent import PolicyAgent

logger = get_logger(__name__)


class SelfLearningAgent(PolicyAgent):
    """Agente de política con capacidad de auto-refinamiento."""

    def __init__(
        self,
        pattern_strength_path: str = "data/reinforcement/pattern_strengths.parquet",
        threshold: float = 0.55,
        learning_rate: float = 0.02,
    ):
        """
        Inicializa el agente de aprendizaje.

        Args:
            pattern_strength_path: Ruta a pattern strengths
            threshold: Umbral inicial para decisiones
            learning_rate: Tasa de ajuste del threshold
        """
        super().__init__(pattern_strength_path, threshold)
        self.learning_rate = learning_rate
        self.refinement_history = []

        logger.info(
            f"🧠 SelfLearningAgent inicializado con threshold={threshold:.2f}, "
            f"lr={learning_rate:.3f}"
        )

    def refine(self, metrics_path: str = "reports/backtest_metrics.json") -> None:
        """
        Refina la política basándose en métricas de backtest.

        Args:
            metrics_path: Ruta al archivo JSON con métricas
        """
        metrics_file = Path(metrics_path)

        if not metrics_file.exists():
            logger.warning(
                f"⚠️ No se encontró {metrics_path} para refinar la política"
            )
            return

        # Cargar métricas
        with open(metrics_file) as f:
            metrics = json.load(f)

        old_threshold = self.threshold

        # Estrategia de refinamiento:
        # Si PnL positivo → aumentar threshold (más conservador)
        # Si PnL negativo → disminuir threshold (más agresivo)
        if metrics["pnl_total"] > 0:
            # Backtest exitoso: hacer el agente más selectivo
            self.threshold = min(self.threshold + self.learning_rate, 0.9)
            logger.info("✅ Backtest positivo: incrementando threshold")
        else:
            # Backtest negativo: hacer el agente más agresivo
            self.threshold = max(self.threshold - self.learning_rate, 0.1)
            logger.info("⚠️ Backtest negativo: reduciendo threshold")

        # Considerar también Sharpe ratio
        if "sharpe" in metrics:
            sharpe = metrics["sharpe"]
            if sharpe > 1.0:
                # Excelente Sharpe: pequeño ajuste positivo
                self.threshold = min(self.threshold + 0.01, 0.9)
            elif sharpe < 0:
                # Sharpe negativo: ajuste correctivo
                self.threshold = max(self.threshold - 0.01, 0.1)

        # Guardar historial de refinamiento
        refinement_record = {
            "old_threshold": float(old_threshold),
            "new_threshold": float(self.threshold),
            "pnl": float(metrics["pnl_total"]),
            "sharpe": float(metrics.get("sharpe", 0)),
            "adjustment": float(self.threshold - old_threshold),
        }
        self.refinement_history.append(refinement_record)

        logger.info(
            f"🔄 Refinamiento de política: "
            f"Threshold {old_threshold:.3f} → {self.threshold:.3f} "
            f"(Δ={self.threshold - old_threshold:+.3f})"
        )

        # Guardar historial de refinamiento
        history_path = Path("reports/refinement_history.json")
        with open(history_path, "w") as f:
            json.dump(self.refinement_history, f, indent=2)

        logger.info(f"📄 Historial de refinamiento guardado en {history_path}")

    def get_refinement_summary(self) -> dict:
        """
        Obtiene un resumen del proceso de refinamiento.

        Returns:
            Diccionario con estadísticas de refinamiento
        """
        if not self.refinement_history:
            return {"refinements": 0, "total_adjustment": 0.0}

        adjustments = [r["adjustment"] for r in self.refinement_history]

        return {
            "refinements": len(self.refinement_history),
            "total_adjustment": sum(adjustments),
            "avg_adjustment": sum(adjustments) / len(adjustments),
            "current_threshold": self.threshold,
        }


if __name__ == "__main__":
    # Demo de refinamiento
    agent = SelfLearningAgent(threshold=0.55, learning_rate=0.02)

    print("🧠 Agente de aprendizaje inicializado")
    print(f"Threshold inicial: {agent.threshold:.3f}")

    # Intentar refinar basado en métricas previas
    agent.refine()

    summary = agent.get_refinement_summary()
    print("\n📊 Resumen de refinamiento:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
