"""
Tests para ProfitabilityAnalyzer.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.profitability_analyzer import ProfitabilityAnalyzer


@pytest.fixture
def sample_trades_df():
    """Crea un DataFrame de ejemplo con trades."""
    np.random.seed(42)

    trades = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "action": np.random.choice(["BUY", "SELL"], 100),
        "price": np.random.uniform(40000, 50000, 100),
        "return": np.random.normal(0.001, 0.02, 100),  # Media positiva
    }

    return pd.DataFrame(trades)


@pytest.fixture
def analyzer(tmp_path):
    """Crea un ProfitabilityAnalyzer con paths temporales."""
    trades_path = tmp_path / "backtest_trades.parquet"
    metrics_path = tmp_path / "backtest_metrics.json"

    analyzer = ProfitabilityAnalyzer(
        trades_path=str(trades_path),
        metrics_path=str(metrics_path),
        initial_capital=10000.0,
    )

    return analyzer


def test_calculate_roi(analyzer, sample_trades_df):
    """Test cálculo de ROI."""
    roi = analyzer.calculate_roi(sample_trades_df)

    assert isinstance(roi, float)
    assert -100 <= roi <= 1000  # Rango razonable


def test_calculate_profit_factor(analyzer, sample_trades_df):
    """Test cálculo de Profit Factor."""
    profit_factor = analyzer.calculate_profit_factor(sample_trades_df)

    assert isinstance(profit_factor, float)
    assert profit_factor >= 0


def test_profit_factor_edge_cases(analyzer):
    """Test casos extremos del Profit Factor."""
    # Solo ganancias
    df_wins = pd.DataFrame({"return": [0.01, 0.02, 0.03]})
    pf = analyzer.calculate_profit_factor(df_wins)
    assert pf > 0

    # Solo pérdidas
    df_losses = pd.DataFrame({"return": [-0.01, -0.02, -0.03]})
    pf = analyzer.calculate_profit_factor(df_losses)
    assert pf == 0.0

    # Sin trades
    df_empty = pd.DataFrame({"return": []})
    pf = analyzer.calculate_profit_factor(df_empty)
    assert pf == 0.0


def test_calculate_sharpe_ratio(analyzer, sample_trades_df):
    """Test cálculo de Sharpe Ratio."""
    returns = sample_trades_df["return"]
    sharpe = analyzer.calculate_sharpe_ratio(returns)

    assert isinstance(sharpe, float)
    assert -3 <= sharpe <= 5  # Rango típico


def test_calculate_sortino_ratio(analyzer, sample_trades_df):
    """Test cálculo de Sortino Ratio."""
    returns = sample_trades_df["return"]
    sortino = analyzer.calculate_sortino_ratio(returns)

    assert isinstance(sortino, float)
    assert -5 <= sortino <= 10  # Rango ampliado para sortino


def test_calculate_drawdown(analyzer):
    """Test cálculo de Drawdown."""
    # Equity curve con drawdown conocido
    equity = pd.Series([10000, 11000, 10500, 9500, 10000, 11500])
    dd_metrics = analyzer.calculate_drawdown(equity)

    assert "max_drawdown" in dd_metrics
    assert "max_drawdown_pct" in dd_metrics
    assert "recovery_time" in dd_metrics

    assert dd_metrics["max_drawdown"] <= 0
    assert dd_metrics["max_drawdown_pct"] <= 0


def test_calculate_stability_score(analyzer, sample_trades_df):
    """Test cálculo de Stability Score."""
    returns = sample_trades_df["return"]
    stability = analyzer.calculate_stability_score(returns)

    assert isinstance(stability, float)
    assert 0 <= stability <= 1


def test_generate_profitability_report(analyzer, sample_trades_df, tmp_path):
    """Test generación de reporte completo."""
    # Guardar trades de muestra
    trades_path = Path(analyzer.trades_path)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    sample_trades_df.to_parquet(trades_path)

    # Generar reporte
    report = analyzer.generate_profitability_report()

    # Validar estructura
    assert "summary" in report
    assert "drawdown" in report
    assert "trade_stats" in report
    assert "capital" in report

    # Validar métricas en summary
    assert "roi_pct" in report["summary"]
    assert "profit_factor" in report["summary"]
    assert "sharpe_ratio" in report["summary"]
    assert "sortino_ratio" in report["summary"]
    assert "stability_score" in report["summary"]

    # Validar que el archivo se creó
    report_path = trades_path.parent / "profitability_report.json"
    assert report_path.exists()

    # Validar ranking
    ranking_path = trades_path.parent / "strategy_ranking.csv"
    assert ranking_path.exists()


def test_empty_report(analyzer):
    """Test generación de reporte cuando no hay datos."""
    report = analyzer.generate_profitability_report()

    assert report["summary"]["roi_pct"] == 0.0
    assert report["summary"]["profit_factor"] == 0.0
    assert report["trade_stats"]["total_trades"] == 0


def test_report_metrics_validity(analyzer, sample_trades_df, tmp_path):
    """Test que las métricas del reporte sean válidas."""
    trades_path = Path(analyzer.trades_path)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    sample_trades_df.to_parquet(trades_path)

    report = analyzer.generate_profitability_report()

    # ROI debe ser numérico
    assert isinstance(report["summary"]["roi_pct"], (int, float))

    # Profit Factor >= 0
    assert report["summary"]["profit_factor"] >= 0

    # Sharpe ratio debe estar en un rango razonable
    assert -5 <= report["summary"]["sharpe_ratio"] <= 10

    # Win rate entre 0 y 100
    assert 0 <= report["trade_stats"]["win_rate_pct"] <= 100

    # Capital final >= 0
    assert report["capital"]["final"] >= 0


def test_ranking_csv_structure(analyzer, sample_trades_df, tmp_path):
    """Test estructura del CSV de ranking."""
    trades_path = Path(analyzer.trades_path)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    sample_trades_df.to_parquet(trades_path)

    analyzer.generate_profitability_report()

    ranking_path = trades_path.parent / "strategy_ranking.csv"
    ranking_df = pd.read_csv(ranking_path)

    assert "Metric" in ranking_df.columns
    assert "Value" in ranking_df.columns
    assert len(ranking_df) == 7  # 7 métricas principales
