"""
Hybrid Agent - Agente híbrido que combina predicciones y patrones.

Este módulo implementa un agente que integra predicciones LSTM con
fuerzas de patrones de refuerzo para tomar decisiones optimizadas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.core.policy_agent import PolicyAgent
from src.core.predictive_model import PricePredictor

logger = get_logger(__name__)


class HybridAgent(PolicyAgent):
    """Agente híbrido que combina predicciones LSTM y patrones de refuerzo."""

    def __init__(
        self,
        pattern_strength_path: str = "data/reinforcement/pattern_strengths.parquet",
        model_path: str = "models/lstm_forecaster.pt",
        weight_prediction: float = 0.5,
        weight_pattern: float = 0.5,
        threshold: float = 0.55,
    ):
        """
        Inicializa el agente híbrido.

        Args:
            pattern_strength_path: Ruta a pattern strengths
            model_path: Ruta al modelo LSTM
            weight_prediction: Peso de las predicciones (0-1)
            weight_pattern: Peso de los patrones (0-1)
            threshold: Umbral para decisiones
        """
        # Inicializar agente base
        super().__init__(pattern_strength_path, threshold)

        # Inicializar predictor
        try:
            self.predictor = PricePredictor(model_path=model_path)
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando predictor: {e}")
            self.predictor = None

        # Pesos de combinación
        total_weight = weight_prediction + weight_pattern
        self.w_pred = weight_prediction / total_weight
        self.w_pat = weight_pattern / total_weight

        logger.info(
            f"🔀 HybridAgent inicializado: "
            f"w_pred={self.w_pred:.2f}, w_pat={self.w_pat:.2f}"
        )

    def decide(self, observation) -> int:
        """
        Toma una decisión híbrida combinando predicción y patrones.

        Args:
            observation: Observación del mercado (fila de DataFrame o dict)

        Returns:
            1 para BUY, -1 para SELL, 0 para HOLD
        """
        # Convertir a dict si es Series
        if isinstance(observation, pd.Series):
            obs_dict = observation.to_dict()
        else:
            obs_dict = observation

        # 1. Señal de predicción
        pred_signal = 0
        if "pred_next" in obs_dict and "close" in obs_dict:
            pred_next = obs_dict["pred_next"]
            current_price = obs_dict["close"]

            if not np.isnan(pred_next) and not np.isnan(current_price):
                # Calcular dirección predicha
                price_change = (pred_next - current_price) / current_price
                pred_signal = np.sign(price_change)

                # Amplificar señal si el cambio es significativo
                if abs(price_change) > 0.01:  # >1% de cambio
                    pred_signal *= 1.5

        # 2. Señal de patrones (usando PolicyAgent)
        pattern_signal = super().decide(observation)

        # 3. Combinar señales
        hybrid_score = self.w_pred * pred_signal + self.w_pat * pattern_signal

        # 4. Aplicar umbral de confianza
        if abs(hybrid_score) < 0.3:  # Señal débil
            decision = 0
        else:
            decision = int(np.sign(hybrid_score))

        logger.debug(
            f"🔀 Hybrid: pred={pred_signal:.2f}, "
            f"pattern={pattern_signal}, "
            f"score={hybrid_score:.2f} → decision={decision}"
        )

        return decision

    def update_weights(
        self, weight_prediction: float, weight_pattern: float
    ) -> None:
        """
        Actualiza los pesos de combinación.

        Args:
            weight_prediction: Nuevo peso para predicciones
            weight_pattern: Nuevo peso para patrones
        """
        total = weight_prediction + weight_pattern
        self.w_pred = weight_prediction / total
        self.w_pat = weight_pattern / total

        logger.info(
            f"🔄 Pesos actualizados: "
            f"w_pred={self.w_pred:.2f}, w_pat={self.w_pat:.2f}"
        )


if __name__ == "__main__":
    # Demo del agente híbrido
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Crear datos simulados
        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.6, 0.7, 0.8]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        # Crear agente híbrido
        agent = HybridAgent(
            pattern_strength_path=str(strengths_path),
            weight_prediction=0.6,
            weight_pattern=0.4,
        )

        # Simulación de observaciones
        observations = [
            {"close": 100.0, "pred_next": 102.0, "volume": 1000},
            {"close": 101.0, "pred_next": 100.5, "volume": 1100},
            {"close": 102.0, "pred_next": 103.0, "volume": 1200},
        ]

        print("\n🎯 Decisiones del agente híbrido:")
        for i, obs in enumerate(observations):
            decision = agent.decide(obs)
            action = {1: "BUY", -1: "SELL", 0: "HOLD"}[decision]
            print(
                f"  Obs {i+1}: close={obs['close']:.2f}, "
                f"pred={obs['pred_next']:.2f} → {action}"
            )
