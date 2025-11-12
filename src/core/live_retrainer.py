"""
Live Retrainer - Módulo de aprendizaje adaptativo continuo.

Este módulo monitorea resultados de trading en vivo, calcula PnL acumulado,
y ajusta los pesos de la política híbrida (w_pred, w_pat) para maximizar retornos.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class LiveRetrainer:
    """Reentrenador adaptativo basado en resultados en vivo."""

    LIVE_TRADES_PATH = "data/live_trades.parquet"
    POLICY_WEIGHTS_PATH = "data/policy_weights.json"
    RETRAINING_LOG_PATH = "data/retraining_log.json"

    def __init__(
        self,
        learning_rate: float = 0.1,
        min_trades_threshold: int = 10,
        initial_w_pred: float = 0.6,
        initial_w_pat: float = 0.4,
    ):
        """
        Inicializa el reentrenador.

        Args:
            learning_rate: Tasa de aprendizaje para ajustar pesos
            min_trades_threshold: Mínimo de trades antes de reentrenar
            initial_w_pred: Peso inicial para predicción LSTM
            initial_w_pat: Peso inicial para patrones
        """
        self.learning_rate = learning_rate
        self.min_trades_threshold = min_trades_threshold

        # Cargar o inicializar pesos
        self.weights = self._load_weights(initial_w_pred, initial_w_pat)

        logger.info("🧠 LiveRetrainer inicializado")
        logger.info(
            f"  Pesos actuales: w_pred={self.weights['w_pred']:.3f}, w_pat={self.weights['w_pat']:.3f}"
        )

    def _load_weights(
        self, initial_w_pred: float, initial_w_pat: float
    ) -> Dict[str, float]:
        """
        Carga pesos desde archivo o usa valores iniciales.

        Args:
            initial_w_pred: Peso inicial predicción
            initial_w_pat: Peso inicial patrones

        Returns:
            Diccionario con pesos
        """
        weights_path = Path(self.POLICY_WEIGHTS_PATH)

        if weights_path.exists():
            with open(weights_path, "r") as f:
                weights = json.load(f)
            logger.info(f"📂 Pesos cargados desde {weights_path}")
            return weights

        # Valores por defecto
        weights = {"w_pred": initial_w_pred, "w_pat": initial_w_pat}

        # Crear archivo
        self._save_weights(weights)

        return weights

    def _save_weights(self, weights: Dict[str, float]) -> None:
        """
        Guarda pesos en archivo.

        Args:
            weights: Pesos a guardar
        """
        weights_path = Path(self.POLICY_WEIGHTS_PATH)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)

        logger.info(f"💾 Pesos guardados en {weights_path}")

    def get_live_trades(self) -> pd.DataFrame:
        """
        Obtiene trades en vivo registrados.

        Returns:
            DataFrame con trades
        """
        trades_path = Path(self.LIVE_TRADES_PATH)

        if not trades_path.exists() or trades_path.stat().st_size == 0:
            logger.warning("⚠️ No hay trades registrados aún")
            return pd.DataFrame()

        return pd.read_parquet(trades_path)

    def calculate_pnl_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula métricas de PnL a partir de trades.

        Args:
            trades_df: DataFrame con trades

        Returns:
            Diccionario con métricas
        """
        if len(trades_df) == 0:
            return {
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

        # Filtrar solo trades completados
        filled_trades = trades_df[trades_df["status"] == "FILLED"]

        if len(filled_trades) == 0:
            return {
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

        # Calcular PnL total
        total_pnl = filled_trades["pnl"].sum()
        avg_pnl = filled_trades["pnl"].mean()

        # Win rate
        winning_trades = (filled_trades["pnl"] > 0).sum()
        win_rate = (winning_trades / len(filled_trades)) * 100

        return {
            "total_pnl": float(total_pnl),
            "avg_pnl_per_trade": float(avg_pnl),
            "win_rate": float(win_rate),
            "total_trades": len(filled_trades),
        }

    def adjust_weights(self, pnl_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Ajusta pesos de política basándose en métricas de PnL.

        Estrategia:
        - Si PnL positivo → incrementar w_pred (modelo predictivo funcionando)
        - Si PnL negativo → incrementar w_pat (volver a patrones)
        - Mantener suma w_pred + w_pat = 1

        Args:
            pnl_metrics: Métricas de PnL

        Returns:
            Pesos actualizados
        """
        total_pnl = pnl_metrics["total_pnl"]
        win_rate = pnl_metrics["win_rate"]

        # Calcular factor de ajuste basado en PnL
        if total_pnl > 0:
            # PnL positivo → favorecer predicción
            adjustment = self.learning_rate * np.tanh(total_pnl / 1000)
            self.weights["w_pred"] += adjustment
            self.weights["w_pat"] -= adjustment
        else:
            # PnL negativo → favorecer patrones
            adjustment = self.learning_rate * np.tanh(abs(total_pnl) / 1000)
            self.weights["w_pred"] -= adjustment
            self.weights["w_pat"] += adjustment

        # Ajuste adicional basado en win_rate
        if win_rate < 40:
            # Win rate bajo → aumentar conservadurismo (patrones)
            self.weights["w_pat"] += 0.05
            self.weights["w_pred"] -= 0.05

        # Normalizar para que sumen 1
        total = self.weights["w_pred"] + self.weights["w_pat"]
        self.weights["w_pred"] /= total
        self.weights["w_pat"] /= total

        # Clamping para evitar extremos
        self.weights["w_pred"] = np.clip(self.weights["w_pred"], 0.2, 0.8)
        self.weights["w_pat"] = np.clip(self.weights["w_pat"], 0.2, 0.8)

        # Renormalizar después de clamp
        total = self.weights["w_pred"] + self.weights["w_pat"]
        self.weights["w_pred"] /= total
        self.weights["w_pat"] /= total

        return self.weights

    def retrain(self) -> Tuple[bool, str]:
        """
        Ejecuta ciclo de reentrenamiento.

        Returns:
            Tupla (success, message)
        """
        logger.info("🔄 Iniciando ciclo de reentrenamiento...")

        # Obtener trades
        trades = self.get_live_trades()

        if len(trades) == 0:
            msg = "No hay trades para reentrenar"
            logger.warning(f"⚠️ {msg}")
            return False, msg

        # Verificar threshold mínimo
        if len(trades) < self.min_trades_threshold:
            msg = f"Insuficientes trades: {len(trades)}/{self.min_trades_threshold}"
            logger.warning(f"⚠️ {msg}")
            return False, msg

        # Calcular métricas
        pnl_metrics = self.calculate_pnl_metrics(trades)

        logger.info("📊 Métricas actuales:")
        logger.info(f"  Total PnL: ${pnl_metrics['total_pnl']:.2f}")
        logger.info(f"  Avg PnL/trade: ${pnl_metrics['avg_pnl_per_trade']:.2f}")
        logger.info(f"  Win Rate: {pnl_metrics['win_rate']:.1f}%")
        logger.info(f"  Total Trades: {pnl_metrics['total_trades']}")

        # Guardar pesos anteriores
        old_weights = self.weights.copy()

        # Ajustar pesos
        new_weights = self.adjust_weights(pnl_metrics)

        logger.info("🎯 Pesos actualizados:")
        logger.info(
            f"  w_pred: {old_weights['w_pred']:.3f} → {new_weights['w_pred']:.3f}"
        )
        logger.info(
            f"  w_pat: {old_weights['w_pat']:.3f} → {new_weights['w_pat']:.3f}"
        )

        # Guardar nuevos pesos
        self._save_weights(new_weights)

        # Log de reentrenamiento
        self._log_retraining_event(old_weights, new_weights, pnl_metrics)

        msg = "Reentrenamiento completado exitosamente"
        logger.info(f"✅ {msg}")

        return True, msg

    def _log_retraining_event(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        pnl_metrics: Dict[str, float],
    ) -> None:
        """
        Registra evento de reentrenamiento.

        Args:
            old_weights: Pesos anteriores
            new_weights: Pesos nuevos
            pnl_metrics: Métricas de PnL
        """
        log_path = Path(self.RETRAINING_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Cargar log existente
        if log_path.exists():
            with open(log_path, "r") as f:
                log_data = json.load(f)
        else:
            log_data = {"events": []}

        # Agregar evento
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "old_weights": old_weights,
            "new_weights": new_weights,
            "pnl_metrics": pnl_metrics,
        }

        log_data["events"].append(event)

        # Guardar
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"📝 Evento de reentrenamiento registrado en {log_path}")

    def get_current_weights(self) -> Dict[str, float]:
        """
        Obtiene pesos actuales de política.

        Returns:
            Diccionario con pesos
        """
        return self.weights.copy()

    def reset_weights(self, w_pred: float = 0.6, w_pat: float = 0.4) -> None:
        """
        Resetea pesos a valores específicos.

        Args:
            w_pred: Nuevo peso predicción
            w_pat: Nuevo peso patrones
        """
        self.weights = {"w_pred": w_pred, "w_pat": w_pat}
        self._save_weights(self.weights)
        logger.info(f"🔄 Pesos reseteados: w_pred={w_pred}, w_pat={w_pat}")


def run_retraining_cycle() -> None:
    """Ejecuta un ciclo completo de reentrenamiento."""
    print("=" * 60)
    print("🧠 LIVE RETRAINER - ADAPTIVE POLICY OPTIMIZATION")
    print("=" * 60)

    retrainer = LiveRetrainer(
        learning_rate=0.1, min_trades_threshold=5  # Threshold bajo para demo
    )

    success, message = retrainer.retrain()

    print(f"\n📋 Resultado: {message}")

    if success:
        weights = retrainer.get_current_weights()
        print(f"\n🎯 Pesos optimizados:")
        print(f"  w_pred (LSTM): {weights['w_pred']:.3f}")
        print(f"  w_pat (Patterns): {weights['w_pat']:.3f}")


if __name__ == "__main__":
    run_retraining_cycle()
