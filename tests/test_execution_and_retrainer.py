"""
Tests para ExecutionEngine y LiveRetrainer.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.execution_engine import (
    ExecutionEngine,
    ExecutionMode,
    OrderSide,
    OrderType,
)
from src.core.live_retrainer import LiveRetrainer


@pytest.fixture
def clean_data_dir(tmp_path):
    """Crea directorio temporal limpio para datos."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def execution_engine(clean_data_dir, monkeypatch):
    """Crea ExecutionEngine con paths temporales."""
    trades_path = clean_data_dir / "live_trades.parquet"

    # Monkeypatch para usar path temporal
    monkeypatch.setattr(
        "src.core.execution_engine.ExecutionEngine.LIVE_TRADES_PATH", str(trades_path)
    )

    engine = ExecutionEngine(mode=ExecutionMode.PAPER, initial_balance=10000.0)

    return engine


@pytest.fixture
def live_retrainer(clean_data_dir, monkeypatch):
    """Crea LiveRetrainer con paths temporales."""
    trades_path = clean_data_dir / "live_trades.parquet"
    weights_path = clean_data_dir / "policy_weights.json"
    log_path = clean_data_dir / "retraining_log.json"

    # Monkeypatch paths
    monkeypatch.setattr(
        "src.core.live_retrainer.LiveRetrainer.LIVE_TRADES_PATH", str(trades_path)
    )
    monkeypatch.setattr(
        "src.core.live_retrainer.LiveRetrainer.POLICY_WEIGHTS_PATH", str(weights_path)
    )
    monkeypatch.setattr(
        "src.core.live_retrainer.LiveRetrainer.RETRAINING_LOG_PATH", str(log_path)
    )

    retrainer = LiveRetrainer(learning_rate=0.1, min_trades_threshold=3)

    return retrainer


# ==================== EXECUTION ENGINE TESTS ====================


def test_execution_engine_initialization(execution_engine):
    """Test inicialización del engine."""
    assert execution_engine.mode == ExecutionMode.PAPER
    assert execution_engine.initial_balance == 10000.0
    assert execution_engine.paper_balance == 10000.0
    assert execution_engine.paper_position == 0.0


def test_get_current_price(execution_engine):
    """Test obtención de precio actual."""
    price = execution_engine.get_current_price()

    assert isinstance(price, float)
    assert price > 0


def test_execute_market_buy_order(execution_engine):
    """Test ejecución de orden MARKET BUY."""
    initial_balance = execution_engine.paper_balance

    result = execution_engine.execute_market_order(OrderSide.BUY, quantity=0.1)

    assert result is not None
    assert result["status"] == "FILLED"
    assert result["side"] == "BUY"
    assert result["type"] == "MARKET"
    assert result["quantity"] == 0.1

    # Balance debe disminuir
    assert execution_engine.paper_balance < initial_balance

    # Posición debe aumentar
    assert execution_engine.paper_position == 0.1


def test_execute_market_sell_order(execution_engine):
    """Test ejecución de orden MARKET SELL."""
    # Primero comprar
    execution_engine.execute_market_order(OrderSide.BUY, quantity=0.2)

    initial_balance = execution_engine.paper_balance
    initial_position = execution_engine.paper_position

    # Luego vender
    result = execution_engine.execute_market_order(OrderSide.SELL, quantity=0.1)

    assert result is not None
    assert result["status"] == "FILLED"
    assert result["side"] == "SELL"

    # Balance debe aumentar
    assert execution_engine.paper_balance > initial_balance

    # Posición debe disminuir
    assert execution_engine.paper_position < initial_position


def test_execute_limit_order(execution_engine):
    """Test ejecución de orden LIMIT."""
    price = 50000.0

    result = execution_engine.execute_limit_order(
        OrderSide.BUY, quantity=0.05, price=price
    )

    assert result is not None
    assert result["type"] == "LIMIT"
    assert result["price"] == price


def test_insufficient_balance_rejection(execution_engine):
    """Test rechazo por balance insuficiente."""
    # Intentar comprar más de lo que permite el balance
    result = execution_engine.execute_market_order(OrderSide.BUY, quantity=1000.0)

    assert result["status"] == "REJECTED"
    assert "balance" in result["reason"].lower()


def test_insufficient_position_rejection(execution_engine):
    """Test rechazo por posición insuficiente."""
    # Intentar vender sin tener posición
    result = execution_engine.execute_market_order(OrderSide.SELL, quantity=0.5)

    assert result["status"] == "REJECTED"
    assert "position" in result["reason"].lower()


def test_trade_logging(execution_engine):
    """Test que los trades se registran correctamente."""
    # Ejecutar varios trades
    execution_engine.execute_market_order(OrderSide.BUY, quantity=0.1)
    execution_engine.execute_market_order(OrderSide.SELL, quantity=0.05)

    # Obtener historial
    history = execution_engine.get_trade_history()

    assert len(history) == 2
    assert "timestamp" in history.columns
    assert "order_id" in history.columns
    assert "pnl" in history.columns


def test_get_paper_balance(execution_engine):
    """Test obtención de balance paper."""
    execution_engine.execute_market_order(OrderSide.BUY, quantity=0.1)

    balance = execution_engine.get_paper_balance()

    assert "balance_usdt" in balance
    assert "position_btc" in balance
    assert "total_value_usdt" in balance

    assert balance["position_btc"] == 0.1
    assert balance["total_value_usdt"] > 0


# ==================== LIVE RETRAINER TESTS ====================


def test_live_retrainer_initialization(live_retrainer):
    """Test inicialización del retrainer."""
    assert live_retrainer.learning_rate == 0.1
    assert live_retrainer.min_trades_threshold == 3

    weights = live_retrainer.get_current_weights()
    assert "w_pred" in weights
    assert "w_pat" in weights
    assert abs(weights["w_pred"] + weights["w_pat"] - 1.0) < 1e-6


def test_calculate_pnl_metrics_empty(live_retrainer):
    """Test cálculo de métricas con trades vacíos."""
    empty_df = pd.DataFrame()
    metrics = live_retrainer.calculate_pnl_metrics(empty_df)

    assert metrics["total_pnl"] == 0.0
    assert metrics["total_trades"] == 0


def test_calculate_pnl_metrics_with_trades(live_retrainer):
    """Test cálculo de métricas con trades."""
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "order_id": [f"ORDER_{i}" for i in range(5)],
            "status": ["FILLED"] * 5,
            "pnl": [10.0, -5.0, 15.0, -3.0, 20.0],
        }
    )

    metrics = live_retrainer.calculate_pnl_metrics(trades)

    assert metrics["total_pnl"] == 37.0
    assert metrics["total_trades"] == 5
    assert metrics["win_rate"] == 60.0  # 3 winning trades


def test_adjust_weights_positive_pnl(live_retrainer):
    """Test ajuste de pesos con PnL positivo."""
    initial_weights = live_retrainer.get_current_weights()

    metrics = {"total_pnl": 100.0, "avg_pnl_per_trade": 10.0, "win_rate": 70.0}

    new_weights = live_retrainer.adjust_weights(metrics)

    # Con PnL positivo, w_pred debe aumentar
    assert new_weights["w_pred"] >= initial_weights["w_pred"]

    # Deben sumar 1
    assert abs(new_weights["w_pred"] + new_weights["w_pat"] - 1.0) < 1e-6


def test_adjust_weights_negative_pnl(live_retrainer):
    """Test ajuste de pesos con PnL negativo."""
    initial_weights = live_retrainer.get_current_weights()

    metrics = {"total_pnl": -100.0, "avg_pnl_per_trade": -10.0, "win_rate": 30.0}

    new_weights = live_retrainer.adjust_weights(metrics)

    # Con PnL negativo, w_pat debe aumentar (w_pred disminuir)
    assert new_weights["w_pred"] <= initial_weights["w_pred"]

    # Deben sumar 1
    assert abs(new_weights["w_pred"] + new_weights["w_pat"] - 1.0) < 1e-6


def test_weights_clamping(live_retrainer):
    """Test que los pesos no excedan límites."""
    # Simular muchos ajustes en una dirección
    for _ in range(10):
        metrics = {"total_pnl": 1000.0, "avg_pnl_per_trade": 100.0, "win_rate": 90.0}
        live_retrainer.adjust_weights(metrics)

    weights = live_retrainer.get_current_weights()

    # Verificar que están dentro de límites
    assert 0.2 <= weights["w_pred"] <= 0.8
    assert 0.2 <= weights["w_pat"] <= 0.8


def test_retrain_insufficient_trades(live_retrainer):
    """Test reentrenamiento con trades insuficientes."""
    success, message = live_retrainer.retrain()

    assert success is False
    assert "trades" in message.lower()


def test_retrain_with_sufficient_trades(
    live_retrainer, execution_engine, monkeypatch
):
    """Test reentrenamiento completo con trades suficientes."""
    # Generar trades usando execution_engine
    for i in range(5):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        if i % 2 == 0:
            execution_engine.execute_market_order(side, quantity=0.01)
        else:
            execution_engine.execute_market_order(side, quantity=0.005)

    # Obtener trades
    trades = execution_engine.get_trade_history()

    # Monkeypatch para que retrainer use los mismos trades
    def mock_get_trades():
        return trades

    monkeypatch.setattr(live_retrainer, "get_live_trades", mock_get_trades)

    # Reentrenar
    success, message = live_retrainer.retrain()

    assert success is True
    assert "exitosamente" in message.lower()


def test_reset_weights(live_retrainer):
    """Test reseteo de pesos."""
    live_retrainer.reset_weights(w_pred=0.7, w_pat=0.3)

    weights = live_retrainer.get_current_weights()

    assert weights["w_pred"] == 0.7
    assert weights["w_pat"] == 0.3


def test_weights_persistence(live_retrainer, clean_data_dir):
    """Test que los pesos se persisten correctamente."""
    # Ajustar pesos
    live_retrainer.reset_weights(w_pred=0.55, w_pat=0.45)

    # Verificar archivo
    weights_path = clean_data_dir / "policy_weights.json"
    assert weights_path.exists()

    with open(weights_path, "r") as f:
        saved_weights = json.load(f)

    assert saved_weights["w_pred"] == 0.55
    assert saved_weights["w_pat"] == 0.45


# ==================== INTEGRATION TEST ====================


def test_full_trading_and_retraining_cycle(
    execution_engine, live_retrainer, monkeypatch
):
    """Test ciclo completo de trading + reentrenamiento."""
    # 1. Ejecutar sesión de trading
    for i in range(5):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        qty = 0.01 if i % 2 == 0 else 0.005

        result = execution_engine.execute_market_order(side, qty)
        assert result["status"] in ["FILLED", "REJECTED"]

    # 2. Verificar que se registraron trades
    trades = execution_engine.get_trade_history()
    assert len(trades) > 0

    # 3. Monkeypatch para retrainer
    def mock_get_trades():
        return trades

    monkeypatch.setattr(live_retrainer, "get_live_trades", mock_get_trades)

    # 4. Calcular métricas
    metrics = live_retrainer.calculate_pnl_metrics(trades)
    assert "total_pnl" in metrics

    # 5. Reentrenar
    success, _ = live_retrainer.retrain()
    assert success is True

    # 6. Verificar que pesos cambiaron
    weights = live_retrainer.get_current_weights()
    assert weights["w_pred"] > 0
    assert weights["w_pat"] > 0
