"""
Policy Agent - Agente de política basado en patrones reforzados.

Este módulo implementa un agente que utiliza la biblioteca de patrones
y sus fuerzas para tomar decisiones de trading (BUY/SELL/NEUTRAL).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class PolicyAgent:
    """Agente de política que usa pattern strengths para decisiones."""

    def __init__(
        self,
        pattern_strength_path: str = "data/reinforcement/pattern_strengths.parquet",
        threshold: float = 0.55,
    ):
        """
        Inicializa el agente de política.

        Args:
            pattern_strength_path: Ruta al archivo con fuerzas de patrones
            threshold: Umbral para decisiones BUY (strength > threshold)
        """
        self.pattern_strength_path = Path(pattern_strength_path)
        if not self.pattern_strength_path.exists():
            raise FileNotFoundError(
                f"Pattern strengths no encontrado: {pattern_strength_path}"
            )

        self.patterns = pd.read_parquet(self.pattern_strength_path)
        self.threshold = threshold

        logger.info(
            f"🤖 PolicyAgent inicializado con {len(self.patterns)} patrones"
        )

    def decide(self, observation) -> int:
        """
        Toma una decisión basada en la observación actual.

        Args:
            observation: Observación del mercado (puede ser una fila de DataFrame)

        Returns:
            1 para BUY, -1 para SELL, 0 para NEUTRAL
        """
        # Seleccionar patrón aleatorio (en producción sería matching real)
        idx = np.random.randint(0, len(self.patterns))
        strength = self.patterns["strength"].iloc[idx]

        if strength > self.threshold:
            logger.debug(f"BUY decision (strength={strength:.2f})")
            return 1
        elif strength < (1 - self.threshold):
            logger.debug(f"SELL decision (strength={strength:.2f})")
            return -1
        else:
            logger.debug(f"NEUTRAL decision (strength={strength:.2f})")
            return 0


if __name__ == "__main__":
    agent = PolicyAgent()
    # Simulación de decisiones
    for i in range(10):
        decision = agent.decide(None)
        action = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[decision]
        print(f"Decision {i+1}: {action}")
