"""
Tests para PortfolioManager.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.portfolio_manager import PortfolioManager


@pytest.fixture
def portfolio_manager(tmp_path, monkeypatch):
    """Crea PortfolioManager con paths temporales."""
    state_path = tmp_path / "portfolio_state.json"
    risk_path = tmp_path / "risk_metrics.json"
    weights_path = tmp_path / "allocation_weights.json"

    # Monkeypatch paths
    monkeypatch.setattr(
        "src.core.portfolio_manager.PortfolioManager.PORTFOLIO_STATE_PATH",
        str(state_path),
    )
    monkeypatch.setattr(
        "src.core.portfolio_manager.PortfolioManager.RISK_METRICS_PATH",
        str(risk_path),
    )
    monkeypatch.setattr(
        "src.core.portfolio_manager.PortfolioManager.ALLOCATION_WEIGHTS_PATH",
        str(weights_path),
    )

    pm = PortfolioManager(
        initial_capital=100000.0,
        max_exposure_per_symbol=0.3,
        max_total_exposure=1.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
    )

    return pm


def test_portfolio_manager_initialization(portfolio_manager):
    """Test inicialización del portfolio manager."""
    assert portfolio_manager.initial_capital == 100000.0
    assert portfolio_manager.current_capital == 100000.0
    assert portfolio_manager.max_exposure_per_symbol == 0.3
    assert portfolio_manager.max_total_exposure == 1.0
    assert portfolio_manager.stop_loss_pct == 0.05
    assert portfolio_manager.take_profit_pct == 0.15


def test_register_strategy(portfolio_manager):
    """Test registro de estrategias."""
    portfolio_manager.register_strategy("test_strategy", allocation=0.5)

    assert "test_strategy" in portfolio_manager.strategies
    assert portfolio_manager.strategies["test_strategy"]["allocation"] == 0.5
    assert portfolio_manager.strategies["test_strategy"]["capital"] == 50000.0
    assert portfolio_manager.strategies["test_strategy"]["pnl"] == 0.0


def test_open_position_success(portfolio_manager):
    """Test apertura exitosa de posición."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    assert success is True
    assert pos_id in portfolio_manager.positions

    position = portfolio_manager.positions[pos_id]
    assert position["symbol"] == "BTCUSDT"
    assert position["side"] == "BUY"
    assert position["quantity"] == 0.5
    assert position["entry_price"] == 50000.0
    assert position["status"] == "OPEN"


def test_open_position_unregistered_strategy(portfolio_manager):
    """Test apertura de posición con estrategia no registrada."""
    success, msg = portfolio_manager.open_position(
        "unknown_strategy", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    assert success is False
    assert "no registrada" in msg.lower()


def test_open_position_exposure_limit(portfolio_manager):
    """Test límite de exposición por símbolo."""
    portfolio_manager.register_strategy("strategy1", allocation=0.5)

    # Primera posición OK
    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )
    assert success is True

    # Segunda posición excede límite
    success, msg = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=1.0, entry_price=50000.0
    )

    assert success is False
    assert "exposición" in msg.lower()


def test_update_position_price(portfolio_manager):
    """Test actualización de precio de posición."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    # Actualizar precio
    new_price = 51000.0
    portfolio_manager.update_position_price(pos_id, new_price)

    position = portfolio_manager.positions[pos_id]
    assert position["current_price"] == 51000.0
    assert position["pnl"] == 500.0  # (51000 - 50000) * 0.5
    assert position["pnl_pct"] > 0


def test_close_position(portfolio_manager):
    """Test cierre de posición."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    # Actualizar precio y cerrar
    portfolio_manager.update_position_price(pos_id, 52000.0)
    closed_position = portfolio_manager.close_position(pos_id, reason="MANUAL")

    assert closed_position is not None
    assert closed_position["status"] == "CLOSED"
    assert closed_position["close_reason"] == "MANUAL"
    assert closed_position["pnl"] == 1000.0  # (52000 - 50000) * 0.5


def test_stop_loss_trigger(portfolio_manager):
    """Test activación de stop-loss."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    # Bajar precio por debajo de stop-loss (5%)
    stop_loss_price = 50000.0 * (1 - 0.05)  # 47500
    portfolio_manager.update_position_price(pos_id, stop_loss_price - 100)

    position = portfolio_manager.positions[pos_id]
    assert position["status"] == "CLOSED"
    assert position["close_reason"] == "STOP_LOSS"


def test_take_profit_trigger(portfolio_manager):
    """Test activación de take-profit."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    # Subir precio por encima de take-profit (15%)
    take_profit_price = 50000.0 * (1 + 0.15)  # 57500
    portfolio_manager.update_position_price(pos_id, take_profit_price + 100)

    position = portfolio_manager.positions[pos_id]
    assert position["status"] == "CLOSED"
    assert position["close_reason"] == "TAKE_PROFIT"


def test_calculate_var(portfolio_manager):
    """Test cálculo de Value-at-Risk."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    # Abrir posiciones
    success, pos_id1 = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=1.0, entry_price=50000.0
    )

    # Actualizar precios
    portfolio_manager.update_position_price(pos_id1, 51000.0)

    # Calcular VaR
    var_95 = portfolio_manager.calculate_var(confidence_level=0.95)

    assert isinstance(var_95, float)
    # VaR debe ser negativo o cero
    assert var_95 <= 0


def test_calculate_beta(portfolio_manager):
    """Test cálculo de Beta."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)

    # Abrir posiciones
    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=1.0, entry_price=50000.0
    )

    portfolio_manager.update_position_price(pos_id, 51000.0)

    # Calcular Beta
    beta = portfolio_manager.calculate_beta()

    assert isinstance(beta, float)
    # Beta típicamente entre 0 y 2
    assert -2 <= beta <= 3


def test_get_portfolio_value(portfolio_manager):
    """Test cálculo de valor del portafolio."""
    portfolio_manager.register_strategy("strategy1", allocation=0.4)
    portfolio_manager.register_strategy("strategy2", allocation=0.6)

    initial_value = portfolio_manager.get_portfolio_value()
    assert initial_value == 100000.0

    # Abrir posición
    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    # Actualizar precio
    portfolio_manager.update_position_price(pos_id, 52000.0)

    # Valor debe haber aumentado
    new_value = portfolio_manager.get_portfolio_value()
    assert new_value > initial_value


def test_get_max_drawdown(portfolio_manager):
    """Test cálculo de drawdown máximo."""
    drawdown_dollars, drawdown_pct = portfolio_manager.get_max_drawdown()

    # Sin pérdidas, drawdown debe ser 0 o negativo
    assert drawdown_dollars <= 0
    assert drawdown_pct <= 0


def test_rebalance_portfolio(portfolio_manager):
    """Test rebalanceo de portafolio."""
    portfolio_manager.register_strategy("strategy1", allocation=0.5)
    portfolio_manager.register_strategy("strategy2", allocation=0.5)

    # Cambiar pesos
    portfolio_manager.allocation_weights = {"strategy1": 0.7, "strategy2": 0.3}

    # Rebalancear
    new_allocations = portfolio_manager.rebalance_portfolio()

    assert "strategy1" in new_allocations
    assert "strategy2" in new_allocations

    # Verificar nuevas asignaciones
    total_value = portfolio_manager.get_portfolio_value()
    assert abs(new_allocations["strategy1"] - total_value * 0.7) < 1.0
    assert abs(new_allocations["strategy2"] - total_value * 0.3) < 1.0


def test_calculate_strategy_correlation(portfolio_manager):
    """Test cálculo de correlación entre estrategias."""
    portfolio_manager.register_strategy("strategy1", allocation=0.5)
    portfolio_manager.register_strategy("strategy2", allocation=0.5)

    # Abrir posiciones
    success, pos_id1 = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )

    success, pos_id2 = portfolio_manager.open_position(
        "strategy2", "ETHUSDT", "BUY", quantity=5.0, entry_price=3000.0
    )

    # Actualizar precios
    portfolio_manager.update_position_price(pos_id1, 51000.0)
    portfolio_manager.update_position_price(pos_id2, 3100.0)

    # Calcular correlación
    corr_matrix = portfolio_manager.calculate_strategy_correlation()

    if not corr_matrix.empty:
        # Diagonal debe ser 1 (correlación consigo mismo)
        for strategy in corr_matrix.columns:
            if strategy in corr_matrix.index:
                corr_value = corr_matrix.loc[strategy, strategy]
                # Verificar solo si no es NaN (puede serlo con un solo dato)
                if pd.notna(corr_value):
                    assert abs(corr_value - 1.0) < 0.01


def test_generate_portfolio_state(portfolio_manager, tmp_path):
    """Test generación de estado del portafolio."""
    portfolio_manager.register_strategy("strategy1", allocation=0.5)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=1.0, entry_price=50000.0
    )

    portfolio_manager.update_position_price(pos_id, 51000.0)

    state = portfolio_manager.generate_portfolio_state()

    # Validar estructura
    assert "timestamp" in state
    assert "portfolio_value" in state
    assert "total_pnl" in state
    assert "total_pnl_pct" in state
    assert "strategies" in state

    # Validar valores
    assert state["portfolio_value"] > 0
    assert state["initial_capital"] == 100000.0


def test_generate_risk_metrics(portfolio_manager, tmp_path):
    """Test generación de métricas de riesgo."""
    portfolio_manager.register_strategy("strategy1", allocation=0.5)

    success, pos_id = portfolio_manager.open_position(
        "strategy1", "BTCUSDT", "BUY", quantity=1.0, entry_price=50000.0
    )

    portfolio_manager.update_position_price(pos_id, 51000.0)

    metrics = portfolio_manager.generate_risk_metrics()

    # Validar estructura
    assert "timestamp" in metrics
    assert "value_at_risk" in metrics
    assert "market_exposure" in metrics
    assert "risk_limits" in metrics

    # Validar VaR
    assert "var_95" in metrics["value_at_risk"]
    assert "var_99" in metrics["value_at_risk"]

    # Validar Beta
    assert "beta" in metrics["market_exposure"]


def test_multiple_positions_scenario(portfolio_manager):
    """Test escenario con múltiples posiciones."""
    # Registrar estrategias
    portfolio_manager.register_strategy("lstm", allocation=0.4)
    portfolio_manager.register_strategy("pattern", allocation=0.3)
    portfolio_manager.register_strategy("hybrid", allocation=0.3)

    # Abrir posiciones
    positions = []

    success, pos_id = portfolio_manager.open_position(
        "lstm", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )
    if success:
        positions.append(pos_id)

    success, pos_id = portfolio_manager.open_position(
        "pattern", "ETHUSDT", "BUY", quantity=3.0, entry_price=3000.0
    )
    if success:
        positions.append(pos_id)

    success, pos_id = portfolio_manager.open_position(
        "hybrid", "BTCUSDT", "SELL", quantity=0.2, entry_price=50000.0
    )
    if success:
        positions.append(pos_id)

    # Simular movimientos de precio
    for i, pos_id in enumerate(positions):
        price_changes = [1.03, 0.97, 1.01]
        if i < len(price_changes):
            pos = portfolio_manager.positions[pos_id]
            new_price = pos["entry_price"] * price_changes[i]
            portfolio_manager.update_position_price(pos_id, new_price)

    # Validar estado
    state = portfolio_manager.generate_portfolio_state()

    assert state["portfolio_value"] > 0
    assert len(state["strategies"]) == 3

    # Calcular métricas
    risk_metrics = portfolio_manager.generate_risk_metrics()

    assert "value_at_risk" in risk_metrics
    assert "market_exposure" in risk_metrics
