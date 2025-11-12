"""
Tests para el entorno de simulación y el agente de política.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.policy_agent import PolicyAgent
from src.core.simulation_environment import MarketSimulator


def test_simulation_flow(tmp_path):
    """Test del flujo completo de simulación con agente de política."""
    # Crear directorio de secuencias
    seq_dir = tmp_path / "data" / "sequences"
    seq_dir.mkdir(parents=True)

    # Crear datos de secuencia simulados
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="min"),
            "price": [100 + i for i in range(100)],
            "volume": [1000] * 100,
            "return_mean": [0.001] * 100,
        }
    )
    seq_path = seq_dir / "BTCUSDT_seq_1m.parquet"
    df.to_parquet(seq_path)

    # Crear datos de reinforcement strengths
    reinf_dir = tmp_path / "data" / "reinforcement"
    reinf_dir.mkdir(parents=True)
    strengths_df = pd.DataFrame({"strength": [0.6, 0.3, 0.8]})
    strengths_path = reinf_dir / "pattern_strengths.parquet"
    strengths_df.to_parquet(strengths_path)

    # Crear agente
    agent = PolicyAgent(pattern_strength_path=str(strengths_path))

    # Ejecutar simulación
    sim = MarketSimulator(
        sequence_path=str(seq_path), policy=agent.decide, capital=10000
    )
    metrics = sim.run()

    # Validar métricas
    assert "pnl_total" in metrics
    assert isinstance(metrics["pnl_total"], float)
    assert "sharpe" in metrics
    assert isinstance(metrics["sharpe"], float)
    assert "max_drawdown" in metrics
    assert isinstance(metrics["max_drawdown"], float)
    assert "num_trades" in metrics
    assert metrics["num_trades"] == 99  # len(df) - 1


def test_policy_agent_decisions():
    """Test de las decisiones del agente de política."""
    # Crear datos temporales
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)

        # Crear patrones con diferentes fuerzas
        strengths_df = pd.DataFrame(
            {"strength": [0.9, 0.1, 0.5]}  # BUY  # SELL  # NEUTRAL
        )
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        agent = PolicyAgent(pattern_strength_path=str(strengths_path), threshold=0.55)

        # Tomar varias decisiones
        decisions = [agent.decide(None) for _ in range(30)]

        # Validar que hay diversidad de decisiones
        assert len(set(decisions)) > 1
        assert all(d in [-1, 0, 1] for d in decisions)


def test_market_simulator_no_policy():
    """Test del simulador sin política (todas las decisiones = 0)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seq_dir = tmp_path / "data" / "sequences"
        seq_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="min"),
                "price": [100] * 50,
                "volume": [1000] * 50,
                "return_mean": [0.001] * 50,
            }
        )
        seq_path = seq_dir / "BTCUSDT_seq_1m.parquet"
        df.to_parquet(seq_path)

        # Simulación sin política (decisión siempre 0)
        sim = MarketSimulator(sequence_path=str(seq_path), policy=None, capital=10000)
        metrics = sim.run()

        # Sin trades activos, PnL debe ser 0
        assert metrics["pnl_total"] == 0.0
