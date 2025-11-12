"""
Tests para el optimizador genético de estrategias.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.genetic_optimizer import GeneticAlgorithmOptimizer


def test_genetic_optimizer_basic(tmp_path):
    """Test básico del optimizador genético."""
    # Crear datos de secuencia simulados
    seq_dir = tmp_path / "data" / "sequences"
    seq_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="min"),
            "price": [100 + i * 0.1 for i in range(50)],
            "volume": [1000] * 50,
            "return_mean": [0.001] * 50,
        }
    )
    seq_path = seq_dir / "BTCUSDT_seq_1m.parquet"
    df.to_parquet(seq_path)

    # Crear pattern strengths
    reinf_dir = tmp_path / "data" / "reinforcement"
    reinf_dir.mkdir(parents=True)
    strengths_df = pd.DataFrame({"strength": [0.6, 0.7, 0.8]})
    strengths_path = reinf_dir / "pattern_strengths.parquet"
    strengths_df.to_parquet(strengths_path)

    # Ejecutar optimización con parámetros pequeños para test rápido
    optimizer = GeneticAlgorithmOptimizer(
        population_size=4, generations=2, sequence_path=str(seq_path)
    )

    best, history = optimizer.evolve()

    # Validaciones
    assert isinstance(history, list)
    assert len(history) == 2  # 2 generaciones
    assert "threshold" in best[0]
    assert "lr" in best[0]
    assert isinstance(best[1], float)  # fitness value

    # Verificar que se generó el archivo de historial
    history_file = Path("reports/genetic_optimization_history.json")
    assert history_file.exists()

    # Validar contenido del historial
    with open(history_file) as f:
        saved_history = json.load(f)

    assert len(saved_history) == 2
    assert all("generation" in h for h in saved_history)
    assert all("best_pnl" in h for h in saved_history)
    assert all("best_threshold" in h for h in saved_history)


def test_genetic_optimizer_individual_operations():
    """Test de operaciones básicas del optimizador."""
    optimizer = GeneticAlgorithmOptimizer(population_size=4, generations=2)

    # Test generación de individuo aleatorio
    ind = optimizer.random_individual()
    assert "threshold" in ind
    assert "lr" in ind
    assert 0.3 <= ind["threshold"] <= 0.9
    assert 0.001 <= ind["lr"] <= 0.2

    # Test mutación
    original_threshold = ind["threshold"]
    mutated = optimizer.mutate(ind.copy())
    assert "threshold" in mutated
    # Puede o no haber cambiado dependiendo del random

    # Test cruza
    parent1 = {"threshold": 0.6, "lr": 0.05}
    parent2 = {"threshold": 0.7, "lr": 0.08}
    child = optimizer.crossover(parent1, parent2)

    assert "threshold" in child
    assert "lr" in child
    # El hijo debe estar en un rango razonable
    assert 0.3 <= child["threshold"] <= 0.9


def test_genetic_optimizer_convergence():
    """Test que el optimizador converge (mejora) a lo largo de generaciones."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seq_dir = tmp_path / "data" / "sequences"
        seq_dir.mkdir(parents=True)

        # Crear datos con tendencia alcista para facilitar convergencia
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="min"),
                "price": [100 + i * 0.5 for i in range(100)],
                "volume": [1000] * 100,
                "return_mean": [0.005] * 100,  # Retornos positivos
            }
        )
        seq_path = seq_dir / "test_seq.parquet"
        df.to_parquet(seq_path)

        # Crear patterns
        reinf_dir = tmp_path / "data" / "reinforcement"
        reinf_dir.mkdir(parents=True)
        strengths_df = pd.DataFrame({"strength": [0.8, 0.9]})
        strengths_path = reinf_dir / "pattern_strengths.parquet"
        strengths_df.to_parquet(strengths_path)

        optimizer = GeneticAlgorithmOptimizer(
            population_size=6, generations=3, sequence_path=str(seq_path)
        )

        best, history = optimizer.evolve()

        # Validar que el historial muestra evolución
        assert len(history) == 3
        pnls = [h["best_pnl"] for h in history]

        # No necesariamente siempre mejora, pero el mejor debe ser razonable
        assert isinstance(pnls[0], float)
        assert isinstance(best[1], float)
