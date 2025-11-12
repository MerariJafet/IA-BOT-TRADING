"""
Predictive Model - Modelo LSTM para pronóstico de precios.

Este módulo implementa un forecaster basado en LSTM para predecir
precios futuros de criptomonedas usando series temporales OHLCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.core.logger import get_logger

logger = get_logger(__name__)


class LSTMForecaster(nn.Module):
    """Red neuronal LSTM para pronóstico de series temporales."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        output_size: int = 1,
        num_layers: int = 2,
    ):
        """
        Inicializa el modelo LSTM.

        Args:
            input_size: Número de features de entrada (OHLCV = 5)
            hidden_size: Tamaño de la capa oculta
            output_size: Número de valores a predecir
            num_layers: Número de capas LSTM
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo."""
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Tomar última salida temporal
        return self.fc(out[:, -1, :])


class PricePredictor:
    """Predictor de precios usando LSTM."""

    def __init__(self, model_path: str = "models/lstm_forecaster.pt"):
        """
        Inicializa el predictor.

        Args:
            model_path: Ruta donde guardar/cargar el modelo
        """
        self.model_path = Path(model_path)
        self.model = LSTMForecaster()
        self.sequence_length = 10  # Ventana de observación

        # Cargar modelo si existe
        if self.model_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                logger.info(f"📥 Modelo LSTM cargado desde {self.model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando modelo: {e}")

        self.model.eval()
        logger.info("🧠 PricePredictor inicializado")

    def train(self, df: pd.DataFrame, epochs: int = 10, lr: float = 0.001) -> None:
        """
        Entrena el modelo LSTM.

        Args:
            df: DataFrame con columnas OHLCV
            epochs: Número de épocas de entrenamiento
            lr: Learning rate
        """
        logger.info(f"🚀 Iniciando entrenamiento LSTM: {epochs} épocas")

        X, y = self._prepare_data(df)

        if len(X) == 0:
            logger.warning("⚠️ No hay suficientes datos para entrenamiento")
            return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={loss.item():.6f}")

        # Guardar modelo
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"✅ Modelo guardado en {self.model_path}")

        self.model.eval()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera predicciones para el próximo precio.

        Args:
            df: DataFrame con datos OHLCV

        Returns:
            DataFrame con columna 'pred_next' agregada
        """
        logger.info("🔮 Generando predicciones...")

        X, _ = self._prepare_data(df)

        if len(X) == 0:
            logger.warning("⚠️ No hay suficientes datos para predicción")
            df["pred_next"] = df["close"]
            return df

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).numpy().flatten()

        # Agregar predicciones al DataFrame
        # Las primeras sequence_length filas no tienen predicción
        df = df.copy()
        df["pred_next"] = np.nan
        df.iloc[self.sequence_length :, df.columns.get_loc("pred_next")] = predictions

        # Rellenar NaN con el precio actual
        df["pred_next"].fillna(df["close"], inplace=True)

        # Guardar predicciones
        pred_dir = Path("data/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / "latest_predictions.parquet"
        df.to_parquet(pred_path, index=False)

        logger.info(f"✅ Predicciones guardadas en {pred_path}")

        return df

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepara datos para entrenamiento/predicción.

        Args:
            df: DataFrame con datos OHLCV

        Returns:
            Tupla (X, y) de tensores
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame debe contener columnas: {required_cols}")

        # Extraer features OHLCV
        data = df[required_cols].values.astype(np.float32)

        if len(data) <= self.sequence_length:
            return torch.tensor([]), torch.tensor([])

        # Crear secuencias
        X_list = []
        y_list = []

        for i in range(self.sequence_length, len(data)):
            X_list.append(data[i - self.sequence_length : i])
            y_list.append(data[i, 3:4])  # close price

        X = torch.tensor(np.array(X_list))
        y = torch.tensor(np.array(y_list))

        return X, y


if __name__ == "__main__":
    # Demo de entrenamiento y predicción
    logger.info("🎯 Demo de PricePredictor")

    # Crear datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "open_time": dates,
            "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(100) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(100) * 0.5),
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "volume": np.random.uniform(1000, 2000, 100),
        }
    )

    # Entrenar modelo
    predictor = PricePredictor()
    predictor.train(df, epochs=5)

    # Generar predicciones
    df_with_preds = predictor.predict(df)

    print("\n📊 Primeras predicciones:")
    print(df_with_preds[["close", "pred_next"]].tail(10))
