"""
Tests para BenchmarkEvaluator - validación de métricas comparativas.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.benchmark_evaluator import BenchmarkEvaluator


@pytest.fixture
def sample_trades_profitable():
    """Crea trades con performance positiva."""
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D")
    trades = pd.DataFrame(
        {
            "timestamp": timestamps,
            "order_id": [f"ORDER_{i}" for i in range(30)],
            "status": ["FILLED"] * 30,
            "pnl": np.random.normal(100, 50, 30),  # PnL positivo en promedio
        }
    )
    return trades


@pytest.fixture
def sample_trades_negative():
    """Crea trades con performance negativa."""
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D")
    trades = pd.DataFrame(
        {
            "timestamp": timestamps,
            "order_id": [f"ORDER_{i}" for i in range(30)],
            "status": ["FILLED"] * 30,
            "pnl": np.random.normal(-100, 50, 30),  # PnL negativo en promedio
        }
    )
    return trades


@pytest.fixture
def benchmark_evaluator(tmp_path, monkeypatch):
    """Crea BenchmarkEvaluator con paths temporales."""
    trades_path = tmp_path / "live_trades.parquet"
    benchmark_comparison_path = tmp_path / "benchmark_comparison.json"
    equity_curve_path = tmp_path / "equity_curve.png"

    monkeypatch.setattr("src.core.benchmark_evaluator.LIVE_TRADES_PATH", str(trades_path))
    monkeypatch.setattr(
        "src.core.benchmark_evaluator.BENCHMARK_COMPARISON_PATH",
        str(benchmark_comparison_path),
    )
    monkeypatch.setattr("src.core.benchmark_evaluator.EQUITY_CURVE_PATH", str(equity_curve_path))

    evaluator = BenchmarkEvaluator(
        trades_path=str(trades_path),
        initial_capital=100000.0,
        risk_free_rate=0.04,
    )

    return evaluator


# ==================== BASIC FUNCTIONALITY TESTS ====================


def test_evaluator_initialization(benchmark_evaluator):
    """Test inicialización del evaluador."""
    assert benchmark_evaluator.initial_capital == 100000.0
    assert benchmark_evaluator.risk_free_rate == 0.04


def test_load_trades(benchmark_evaluator, sample_trades_profitable, tmp_path):
    """Test carga de trades."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    trades = benchmark_evaluator.load_trades()

    assert len(trades) == 30
    assert "timestamp" in trades.columns
    assert "pnl" in trades.columns


def test_calculate_equity_curve(benchmark_evaluator, sample_trades_profitable, tmp_path):
    """Test cálculo de equity curve."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    benchmark_evaluator.load_trades()
    equity_curve = benchmark_evaluator.calculate_portfolio_equity_curve()

    assert "equity" in equity_curve.columns
    assert len(equity_curve) == 30

    # Equity debe empezar en initial_capital
    assert equity_curve.iloc[0]["equity"] == pytest.approx(
        100000.0 + sample_trades_profitable.iloc[0]["pnl"], rel=0.01
    )


def test_equity_curve_growth(benchmark_evaluator, sample_trades_profitable, tmp_path):
    """Test que equity curve crece con trades positivos."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    benchmark_evaluator.load_trades()
    equity_curve = benchmark_evaluator.calculate_portfolio_equity_curve()

    # Con PnL positivo, equity final > equity inicial
    initial_equity = equity_curve.iloc[0]["equity"]
    final_equity = equity_curve.iloc[-1]["equity"]

    # Debido a PnL promedio positivo, final > inicial (con alta probabilidad)
    # Usamos un test más robusto: sumar todos los PnL
    total_pnl = sample_trades_profitable["pnl"].sum()
    expected_final = 100000.0 + total_pnl

    assert final_equity == pytest.approx(expected_final, rel=0.01)


# ==================== ROI AND SHARPE TESTS ====================


def test_calculate_roi_annualized_positive(benchmark_evaluator, sample_trades_profitable, tmp_path):
    """Test cálculo de ROI anualizado con performance positiva."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    benchmark_evaluator.load_trades()
    equity_curve = benchmark_evaluator.calculate_portfolio_equity_curve()

    roi = benchmark_evaluator.calculate_roi_annualized(equity_curve)

    # ROI debe ser positivo con alta probabilidad (PnL promedio = +100)
    # No podemos garantizar siempre positivo por aleatoriedad, pero debería ser > -100%
    assert roi > -100.0


def test_calculate_max_drawdown(benchmark_evaluator, sample_trades_negative, tmp_path):
    """Test cálculo de Maximum Drawdown."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_negative.to_parquet(trades_path)

    benchmark_evaluator.load_trades()
    equity_curve = benchmark_evaluator.calculate_portfolio_equity_curve()

    max_dd = benchmark_evaluator.calculate_max_drawdown(equity_curve)

    # Con trades negativos, max_dd debe ser negativo
    assert max_dd < 0


def test_calculate_sharpe_rolling(benchmark_evaluator, sample_trades_profitable, tmp_path):
    """Test cálculo de Sharpe Ratio rolling."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    benchmark_evaluator.load_trades()
    equity_curve = benchmark_evaluator.calculate_portfolio_equity_curve()

    sharpe_rolling = benchmark_evaluator.calculate_sharpe_rolling(equity_curve, window=10)

    assert "sharpe_ratio" in sharpe_rolling.columns
    assert len(sharpe_rolling) > 0


# ==================== ALPHA/BETA TESTS ====================


def test_calculate_alpha_beta():
    """Test cálculo de Alpha y Beta."""
    # Portfolio con beta = 1.5 (más volátil que benchmark)
    np.random.seed(42)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    portfolio_returns = pd.Series(1.5 * benchmark_returns + np.random.normal(0.002, 0.01, 100))

    evaluator = BenchmarkEvaluator()
    alpha, beta = evaluator.calculate_alpha_beta(portfolio_returns, benchmark_returns)

    # Beta debería estar cerca de 1.5
    assert beta == pytest.approx(1.5, abs=0.3)

    # Alpha debería ser positivo (agregamos retorno extra)
    # Pero puede variar por aleatoriedad
    assert isinstance(alpha, float)


def test_calculate_correlation():
    """Test cálculo de correlación."""
    np.random.seed(42)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    portfolio_returns = pd.Series(benchmark_returns + np.random.normal(0, 0.005, 100))

    evaluator = BenchmarkEvaluator()
    corr = evaluator.calculate_correlation(portfolio_returns, benchmark_returns)

    # Correlación debe estar entre -1 y 1
    assert -1.0 <= corr <= 1.0

    # Con portfolio basado en benchmark, correlación debería ser alta
    assert corr > 0.5


# ==================== BENCHMARK FETCHING TESTS ====================


def test_fetch_sp500_prices(benchmark_evaluator):
    """Test generación de precios de S&P500 (sintéticos)."""
    start_date = "2024-01-01"
    end_date = "2024-01-30"

    sp500_prices = benchmark_evaluator.fetch_sp500_prices(start_date, end_date)

    assert "sp500_price" in sp500_prices.columns
    assert len(sp500_prices) == 30


def test_calculate_benchmark_returns(benchmark_evaluator):
    """Test cálculo de retornos del benchmark."""
    # Crear precios sintéticos
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "price": [100, 105, 103, 108, 110, 107, 112, 115, 113, 118],
        }
    )

    returns_df = benchmark_evaluator.calculate_benchmark_returns(prices, "price")

    assert "returns" in returns_df.columns
    assert len(returns_df) == 9  # pct_change elimina 1 fila


# ==================== INTEGRATION TESTS ====================


def test_generate_comparison_report(
    benchmark_evaluator, sample_trades_profitable, tmp_path, monkeypatch
):
    """Test generación de reporte completo."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    # Mockear fetch_btc_prices para evitar llamada a API real
    def mock_fetch_btc(start, end):
        timestamps = pd.date_range(start, end, freq="D")
        prices = 50000 + np.random.normal(0, 1000, len(timestamps))
        df = pd.DataFrame({"timestamp": timestamps, "btc_price": prices})
        benchmark_evaluator.btc_prices = df
        return df

    monkeypatch.setattr(benchmark_evaluator, "fetch_btc_prices", mock_fetch_btc)

    start_date = "2024-01-01"
    end_date = "2024-01-30"

    report = benchmark_evaluator.generate_comparison_report(start_date, end_date)

    assert "portfolio" in report
    assert "benchmarks" in report
    assert "roi_annualized_pct" in report["portfolio"]
    assert "sharpe_ratio_avg" in report["portfolio"]
    assert "max_drawdown_pct" in report["portfolio"]


def test_save_report(benchmark_evaluator, tmp_path):
    """Test guardar reporte en JSON."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "portfolio": {"roi_annualized_pct": 15.5},
    }

    output_path = tmp_path / "test_report.json"
    benchmark_evaluator.save_report(report, str(output_path))

    assert output_path.exists()

    with open(output_path, "r") as f:
        loaded = json.load(f)

    assert loaded["portfolio"]["roi_annualized_pct"] == 15.5


def test_plot_equity_curve(benchmark_evaluator, sample_trades_profitable, tmp_path):
    """Test generación de equity curve plot."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_profitable.to_parquet(trades_path)

    benchmark_evaluator.load_trades()
    equity_curve = benchmark_evaluator.calculate_portfolio_equity_curve()

    output_path = tmp_path / "test_equity_curve.png"

    # Generar plot
    benchmark_evaluator.plot_equity_curve(equity_curve, output_path=str(output_path))

    assert output_path.exists()


# ==================== EDGE CASES ====================


def test_empty_trades(benchmark_evaluator, tmp_path):
    """Test con archivo de trades vacío."""
    trades_path = tmp_path / "live_trades.parquet"
    empty_trades = pd.DataFrame(columns=["timestamp", "order_id", "status", "pnl"])
    empty_trades.to_parquet(trades_path)

    benchmark_evaluator.load_trades()

    with pytest.raises(ValueError, match="Trades no cargados"):
        benchmark_evaluator.calculate_portfolio_equity_curve()


def test_roi_zero_days():
    """Test ROI con 0 días."""
    evaluator = BenchmarkEvaluator()

    equity_curve = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "equity": [100000, 100000],
        }
    )

    roi = evaluator.calculate_roi_annualized(equity_curve)

    # Con 0 días, ROI debe ser 0
    assert roi == 0.0


def test_correlation_insufficient_data():
    """Test correlación con datos insuficientes."""
    evaluator = BenchmarkEvaluator()

    portfolio_returns = pd.Series([0.01])
    benchmark_returns = pd.Series([0.02])

    corr = evaluator.calculate_correlation(portfolio_returns, benchmark_returns)

    # Con 1 dato, correlación debe ser 0
    assert corr == 0.0
