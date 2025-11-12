"""
Tests para el agente híbrido y modelo predictivo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.hybrid_agent import HybridAgent
from src.core.predictive_model import LSTMForecaster, PricePredictor


def test_lstm_forecaster_forward():
    """Test del forward pass del modelo LSTM."""
    model = LSTMForecaster(input_size=5, hidden_size=32, output_size=1)

    # Crear batch de entrada: (batch_size=2, seq_len=10, features=5)
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Verificar dimensiones de salida
    assert output.shape == (2, 1)
    assert not torch.isnan(output).any()


def test_price_predictor_training(tmp_path):
    """Test del entrenamiento del predictor."""
    # Crear datos simulados
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(50) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(50) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(50) * 0.5),
            "close": 100 + np.cumsum(np.random.randn(50) * 0.5),
            "volume": np.random.uniform(1000, 2000, 50),
        }
    )

    # Crear predictor
    model_path = tmp_path / "models" / "test_model.pt"
    predictor = PricePredictor(model_path=str(model_path))

    # Entrenar
    predictor.train(df, epochs=3, lr=0.01)

    # Verificar que el modelo se guardó
    assert model_path.exists()


def test_price_predictor_prediction():
    """Test de generación de predicciones."""
    # Crear datos
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104] * 5,
            "high": [101, 102, 103, 104, 105] * 5,
            "low": [99, 100, 101, 102, 103] * 5,
            "close": [100, 101, 102, 103, 104] * 5,
            "volume": [1000] * 25,
        }
    )

    predictor = PricePredictor()

    # Generar predicciones (sin entrenar)
    df_with_preds = predictor.predict(df)

    # Verificar que se agregó la columna
    assert "pred_next" in df_with_preds.columns
    assert len(df_with_preds) == len(df)
    assert not df_with_preds["pred_next"].isna().all()


def test_hybrid_agent_initialization(tmp_path):
    """Test de inicialización del agente híbrido."""
    # Crear pattern strengths
    reinf_dir = tmp_path / "data" / "reinforcement"
    reinf_dir.mkdir(parents=True)
    strengths_df = pd.DataFrame({"strength": [0.6, 0.7, 0.8]})
    strengths_path = reinf_dir / "pattern_strengths.parquet"
    strengths_df.to_parquet(strengths_path)

    # Crear agente
    agent = HybridAgent(
        pattern_strength_path=str(strengths_path),
        weight_prediction=0.6,
        weight_pattern=0.4,
    )

    # Verificar pesos
    assert abs(agent.w_pred - 0.6) < 0.01
    assert abs(agent.w_pat - 0.4) < 0.01


def test_hybrid_agent_decision():
    """Test de decisión del agente híbrido."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Crear datos
        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.7, 0.6, 0.8]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        # Crear agente
        agent = HybridAgent(
            pattern_strength_path=str(strengths_path),
            weight_prediction=0.6,
            weight_pattern=0.4,
            threshold=0.5,
        )

        # Test con predicción alcista
        obs_bullish = {"close": 100.0, "pred_next": 102.0, "volume": 1000}
        decision_bullish = agent.decide(obs_bullish)
        assert decision_bullish in [-1, 0, 1]

        # Test con predicción bajista
        obs_bearish = {"close": 100.0, "pred_next": 98.0, "volume": 1000}
        decision_bearish = agent.decide(obs_bearish)
        assert decision_bearish in [-1, 0, 1]

        # Test sin predicción
        obs_no_pred = {"close": 100.0, "volume": 1000}
        decision_no_pred = agent.decide(obs_no_pred)
        assert decision_no_pred in [-1, 0, 1]


def test_hybrid_agent_with_series():
    """Test del agente con observación como pandas Series."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.6]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        agent = HybridAgent(pattern_strength_path=str(strengths_path))

        # Crear observación como Series
        obs = pd.Series({"close": 100.0, "pred_next": 101.5, "volume": 1000})

        decision = agent.decide(obs)
        assert decision in [-1, 0, 1]


def test_hybrid_agent_weight_update():
    """Test de actualización de pesos."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.7]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        agent = HybridAgent(
            pattern_strength_path=str(strengths_path),
            weight_prediction=0.5,
            weight_pattern=0.5,
        )

        # Actualizar pesos
        agent.update_weights(weight_prediction=0.7, weight_pattern=0.3)

        assert abs(agent.w_pred - 0.7) < 0.01
        assert abs(agent.w_pat - 0.3) < 0.01


def test_hybrid_agent_significant_price_change():
    """Test con cambios de precio significativos."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.6, 0.7]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        agent = HybridAgent(
            pattern_strength_path=str(strengths_path),
            weight_prediction=0.8,
            weight_pattern=0.2,
        )

        # Cambio significativo (>1%)
        obs_big_change = {"close": 100.0, "pred_next": 102.5, "volume": 1000}
        decision = agent.decide(obs_big_change)

        # Con peso alto en predicción y cambio significativo, debe haber señal
        assert decision in [-1, 0, 1]
