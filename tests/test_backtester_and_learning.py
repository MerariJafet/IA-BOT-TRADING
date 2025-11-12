"""
Tests para backtesting y aprendizaje adaptativo.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.backtester import Backtester
from src.core.self_learning_agent import SelfLearningAgent


def test_backtester_basic(tmp_path):
    """Test básico del backtester."""
    # Crear datos reales simulados
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=20, freq="min"),
            "close": [100, 101, 99, 102, 103, 102, 104, 103, 105, 106] * 2,
            "open": [100] * 20,
            "high": [102] * 20,
            "low": [99] * 20,
            "volume": [1000] * 20,
        }
    )
    data_path = data_dir / "BTCUSDT_1m.parquet"
    df.to_parquet(data_path)

    # Crear pattern strengths
    reinf_dir = tmp_path / "data" / "reinforcement"
    reinf_dir.mkdir(parents=True, exist_ok=True)
    strengths_df = pd.DataFrame({"strength": [0.7, 0.6, 0.8]})
    strengths_path = reinf_dir / "pattern_strengths.parquet"
    strengths_df.to_parquet(strengths_path)

    # Ejecutar backtest
    backtester = Backtester(
        data_path=str(data_path), pattern_strength_path=str(strengths_path)
    )
    metrics = backtester.run()

    # Validaciones
    assert "pnl_total" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "num_trades" in metrics
    assert "win_rate" in metrics

    assert metrics["num_trades"] == 19  # len(df) - 1

    # Verificar archivos generados
    assert Path("reports/backtest_trades.parquet").exists()
    assert Path("reports/backtest_metrics.json").exists()


def test_self_learning_agent_refinement():
    """Test del refinamiento del agente de aprendizaje."""
    # Crear métricas de backtest simuladas
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    metrics = {
        "pnl_total": 150.5,
        "sharpe": 1.2,
        "max_drawdown": -20.0,
        "num_trades": 100,
        "win_rate": 0.65,
    }

    metrics_path = reports_dir / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    # Crear pattern strengths simulados
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.6, 0.7]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        # Inicializar agente
        agent = SelfLearningAgent(
            pattern_strength_path=str(strengths_path),
            threshold=0.55,
            learning_rate=0.02,
        )

        initial_threshold = agent.threshold

        # Refinar con backtest positivo
        agent.refine()

        # Validaciones
        assert agent.threshold != initial_threshold
        assert 0.1 <= agent.threshold <= 0.9
        assert len(agent.refinement_history) == 1

        # Verificar ajuste en dirección correcta (PnL positivo → threshold sube)
        assert agent.threshold > initial_threshold


def test_self_learning_agent_negative_backtest():
    """Test del refinamiento con backtest negativo."""
    # Crear métricas negativas
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    metrics = {
        "pnl_total": -50.0,
        "sharpe": -0.5,
        "max_drawdown": -100.0,
        "num_trades": 50,
        "win_rate": 0.3,
    }

    metrics_path = reports_dir / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.5]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        agent = SelfLearningAgent(
            pattern_strength_path=str(strengths_path), threshold=0.55
        )
        initial_threshold = agent.threshold

        agent.refine()

        # Con backtest negativo, threshold debe bajar
        assert agent.threshold < initial_threshold


def test_full_backtest_and_learning_flow():
    """Test del flujo completo: backtest → learning → refinement."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Preparar datos
        data_dir = tmp_path / "data" / "real"
        data_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2024-01-01", periods=50, freq="min"),
                "close": [100 + i * 0.5 for i in range(50)],
                "open": [100] * 50,
                "high": [101] * 50,
                "low": [99] * 50,
                "volume": [1000] * 50,
            }
        )
        data_path = data_dir / "TEST.parquet"
        df.to_parquet(data_path)

        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.8, 0.6]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        # Paso 1: Ejecutar backtest
        backtester = Backtester(
            data_path=str(data_path), pattern_strength_path=str(strengths_path)
        )
        metrics = backtester.run()

        assert "pnl_total" in metrics

        # Paso 2: Refinar agente basado en resultados
        agent = SelfLearningAgent(pattern_strength_path=str(strengths_path))
        agent.refine()

        # Validar refinamiento
        assert len(agent.refinement_history) > 0

        # Verificar archivos generados
        assert Path("reports/backtest_metrics.json").exists()
        assert Path("reports/refinement_history.json").exists()
