"""
Portfolio Manager - Gestión de portafolio multi-estrategia con control de riesgo.

Este módulo gestiona la asignación de capital entre múltiples estrategias,
calcula métricas de riesgo (VaR, Beta, correlación) y ejecuta stop-loss/take-profit.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """Gestor de portafolio multi-estrategia con control de riesgo."""

    PORTFOLIO_STATE_PATH = "reports/portfolio_state.json"
    RISK_METRICS_PATH = "reports/risk_metrics.json"
    ALLOCATION_WEIGHTS_PATH = "data/allocation_weights.json"

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_exposure_per_symbol: float = 0.3,
        max_total_exposure: float = 1.0,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
    ):
        """
        Inicializa el gestor de portafolio.

        Args:
            initial_capital: Capital inicial total
            max_exposure_per_symbol: Exposición máxima por símbolo (0-1)
            max_total_exposure: Exposición total máxima (0-1)
            stop_loss_pct: % de pérdida para stop-loss
            take_profit_pct: % de ganancia para take-profit
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_exposure_per_symbol = max_exposure_per_symbol
        self.max_total_exposure = max_total_exposure
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Estrategias y posiciones
        self.strategies: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}

        # Cargar pesos de asignación
        self.allocation_weights = self._load_allocation_weights()

        logger.info("💼 PortfolioManager inicializado")
        logger.info(f"  Capital inicial: ${initial_capital:,.2f}")
        logger.info(f"  Max exposición por símbolo: {max_exposure_per_symbol:.1%}")
        logger.info(f"  Stop-loss: {stop_loss_pct:.1%} | Take-profit: {take_profit_pct:.1%}")

    def _load_allocation_weights(self) -> Dict[str, float]:
        """
        Carga pesos de asignación desde archivo o usa defaults.

        Returns:
            Diccionario con pesos por estrategia
        """
        weights_path = Path(self.ALLOCATION_WEIGHTS_PATH)

        if weights_path.exists():
            with open(weights_path, "r") as f:
                weights = json.load(f)
            logger.info(f"📂 Pesos de asignación cargados desde {weights_path}")
            return weights

        # Pesos por defecto
        default_weights = {
            "lstm_predictor": 0.4,
            "pattern_matcher": 0.3,
            "hybrid_agent": 0.3,
        }

        # Guardar defaults
        self._save_allocation_weights(default_weights)

        return default_weights

    def _save_allocation_weights(self, weights: Dict[str, float]) -> None:
        """Guarda pesos de asignación."""
        weights_path = Path(self.ALLOCATION_WEIGHTS_PATH)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)

        logger.info(f"💾 Pesos guardados en {weights_path}")

    def register_strategy(
        self, strategy_name: str, allocation: Optional[float] = None
    ) -> None:
        """
        Registra una estrategia en el portafolio.

        Args:
            strategy_name: Nombre de la estrategia
            allocation: Asignación de capital (opcional, usa weights si None)
        """
        if allocation is None:
            allocation = self.allocation_weights.get(strategy_name, 0.1)

        allocated_capital = self.initial_capital * allocation

        self.strategies[strategy_name] = {
            "allocation": allocation,
            "capital": allocated_capital,
            "pnl": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
        }

        logger.info(
            f"📊 Estrategia '{strategy_name}' registrada con ${allocated_capital:,.2f} ({allocation:.1%})"
        )

    def open_position(
        self,
        strategy_name: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
    ) -> Tuple[bool, str]:
        """
        Abre una posición.

        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo del activo
            side: BUY o SELL
            quantity: Cantidad
            entry_price: Precio de entrada

        Returns:
            Tupla (success, message)
        """
        # Validar estrategia
        if strategy_name not in self.strategies:
            msg = f"Estrategia '{strategy_name}' no registrada"
            logger.warning(f"⚠️ {msg}")
            return False, msg

        # Calcular exposición
        position_value = quantity * entry_price
        symbol_exposure = self._calculate_symbol_exposure(symbol, position_value)
        total_exposure = self._calculate_total_exposure(position_value)

        # Validar exposición
        if symbol_exposure > self.max_exposure_per_symbol:
            msg = f"Exposición en {symbol} excede límite: {symbol_exposure:.1%} > {self.max_exposure_per_symbol:.1%}"
            logger.warning(f"⚠️ {msg}")
            return False, msg

        if total_exposure > self.max_total_exposure:
            msg = f"Exposición total excede límite: {total_exposure:.1%}"
            logger.warning(f"⚠️ {msg}")
            return False, msg

        # Crear posición
        position_id = f"{strategy_name}_{symbol}_{int(datetime.utcnow().timestamp())}"

        self.positions[position_id] = {
            "strategy": strategy_name,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": entry_price,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "status": "OPEN",
            "opened_at": datetime.utcnow().isoformat(),
            "stop_loss": entry_price * (1 - self.stop_loss_pct)
            if side == "BUY"
            else entry_price * (1 + self.stop_loss_pct),
            "take_profit": entry_price * (1 + self.take_profit_pct)
            if side == "BUY"
            else entry_price * (1 - self.take_profit_pct),
        }

        # Actualizar estrategia
        self.strategies[strategy_name]["capital"] -= position_value
        self.strategies[strategy_name]["trades"] += 1

        logger.info(
            f"✅ Posición abierta: {position_id} | {side} {quantity} {symbol} @ ${entry_price:,.2f}"
        )

        return True, position_id

    def update_position_price(self, position_id: str, current_price: float) -> None:
        """
        Actualiza precio actual de una posición y calcula PnL.

        Args:
            position_id: ID de la posición
            current_price: Precio actual
        """
        if position_id not in self.positions:
            logger.warning(f"⚠️ Posición {position_id} no encontrada")
            return

        position = self.positions[position_id]

        if position["status"] != "OPEN":
            return

        position["current_price"] = current_price

        # Calcular PnL
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        side = position["side"]

        if side == "BUY":
            pnl = (current_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - current_price) * quantity

        position["pnl"] = pnl
        position["pnl_pct"] = (pnl / (entry_price * quantity)) * 100

        # Verificar stop-loss y take-profit
        self._check_exit_conditions(position_id)

    def _check_exit_conditions(self, position_id: str) -> None:
        """
        Verifica condiciones de salida (stop-loss/take-profit).

        Args:
            position_id: ID de la posición
        """
        position = self.positions[position_id]
        current_price = position["current_price"]
        side = position["side"]

        close_reason = None

        if side == "BUY":
            if current_price <= position["stop_loss"]:
                close_reason = "STOP_LOSS"
            elif current_price >= position["take_profit"]:
                close_reason = "TAKE_PROFIT"
        else:  # SELL
            if current_price >= position["stop_loss"]:
                close_reason = "STOP_LOSS"
            elif current_price <= position["take_profit"]:
                close_reason = "TAKE_PROFIT"

        if close_reason:
            self.close_position(position_id, reason=close_reason)

    def close_position(
        self, position_id: str, reason: str = "MANUAL"
    ) -> Optional[Dict]:
        """
        Cierra una posición.

        Args:
            position_id: ID de la posición
            reason: Razón de cierre

        Returns:
            Detalles de la posición cerrada
        """
        if position_id not in self.positions:
            logger.warning(f"⚠️ Posición {position_id} no encontrada")
            return None

        position = self.positions[position_id]

        if position["status"] != "OPEN":
            logger.warning(f"⚠️ Posición {position_id} ya está cerrada")
            return None

        # Cerrar posición
        position["status"] = "CLOSED"
        position["closed_at"] = datetime.utcnow().isoformat()
        position["close_reason"] = reason

        # Actualizar estrategia
        strategy_name = position["strategy"]
        position_value = position["quantity"] * position["current_price"]

        self.strategies[strategy_name]["capital"] += position_value + position["pnl"]
        self.strategies[strategy_name]["pnl"] += position["pnl"]

        if position["pnl"] > 0:
            self.strategies[strategy_name]["wins"] += 1
        else:
            self.strategies[strategy_name]["losses"] += 1

        logger.info(
            f"🔒 Posición cerrada: {position_id} | PnL: ${position['pnl']:,.2f} ({position['pnl_pct']:.2f}%) | Razón: {reason}"
        )

        return position

    def _calculate_symbol_exposure(
        self, symbol: str, additional_value: float
    ) -> float:
        """Calcula exposición actual en un símbolo."""
        current_exposure = sum(
            pos["quantity"] * pos["current_price"]
            for pos in self.positions.values()
            if pos["symbol"] == symbol and pos["status"] == "OPEN"
        )

        total_exposure = current_exposure + additional_value
        return total_exposure / self.initial_capital

    def _calculate_total_exposure(self, additional_value: float = 0.0) -> float:
        """Calcula exposición total del portafolio."""
        current_exposure = sum(
            pos["quantity"] * pos["current_price"]
            for pos in self.positions.values()
            if pos["status"] == "OPEN"
        )

        total_exposure = current_exposure + additional_value
        return total_exposure / self.initial_capital

    def calculate_var(
        self, confidence_level: float = 0.95, time_horizon: int = 1
    ) -> float:
        """
        Calcula Value-at-Risk (VaR) paramétrico.

        Args:
            confidence_level: Nivel de confianza (0.95 = 95%)
            time_horizon: Horizonte temporal en días

        Returns:
            VaR en dólares (valor negativo)
        """
        # Obtener retornos de posiciones
        returns = []
        for position in self.positions.values():
            if position["status"] == "OPEN" and position["pnl_pct"] != 0:
                returns.append(position["pnl_pct"] / 100)

        if len(returns) == 0:
            return 0.0

        returns_array = np.array(returns)

        # Calcular media y desviación estándar
        mean_return = returns_array.mean()
        std_return = returns_array.std()

        # Z-score para el nivel de confianza
        from scipy.stats import norm

        z_score = norm.ppf(1 - confidence_level)

        # VaR = (mean - z * std) * portfolio_value * sqrt(time_horizon)
        portfolio_value = self.get_portfolio_value()
        var = (mean_return + z_score * std_return) * portfolio_value * np.sqrt(
            time_horizon
        )

        return float(var)

    def calculate_beta(self, market_returns: Optional[np.ndarray] = None) -> float:
        """
        Calcula Beta del portafolio vs mercado.

        Args:
            market_returns: Retornos del mercado (opcional, usa simulados si None)

        Returns:
            Beta
        """
        # Obtener retornos del portafolio
        portfolio_returns = []
        for position in self.positions.values():
            if position["pnl_pct"] != 0:
                portfolio_returns.append(position["pnl_pct"] / 100)

        if len(portfolio_returns) == 0:
            return 1.0

        portfolio_returns_array = np.array(portfolio_returns)

        # Simular retornos de mercado si no se proveen
        if market_returns is None:
            market_returns = np.random.normal(0.001, 0.02, len(portfolio_returns))

        # Calcular covarianza y varianza
        covariance = np.cov(portfolio_returns_array, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 1.0

        beta = covariance / market_variance

        return float(beta)

    def calculate_strategy_correlation(self) -> pd.DataFrame:
        """
        Calcula matriz de correlación entre estrategias.

        Returns:
            DataFrame con correlaciones
        """
        # Recopilar retornos por estrategia
        strategy_returns = {name: [] for name in self.strategies.keys()}

        for position in self.positions.values():
            strategy = position["strategy"]
            if position["pnl_pct"] != 0:
                strategy_returns[strategy].append(position["pnl_pct"] / 100)

        # Crear DataFrame
        df = pd.DataFrame(
            dict(
                [(k, pd.Series(v)) for k, v in strategy_returns.items() if len(v) > 0]
            )
        )

        if df.empty:
            return pd.DataFrame()

        # Calcular correlación
        correlation_matrix = df.corr()

        return correlation_matrix

    def get_portfolio_value(self) -> float:
        """
        Calcula valor total del portafolio.

        Returns:
            Valor total en dólares
        """
        # Capital en estrategias
        strategy_capital = sum(s["capital"] for s in self.strategies.values())

        # Valor de posiciones abiertas
        open_positions_value = sum(
            pos["quantity"] * pos["current_price"] + pos["pnl"]
            for pos in self.positions.values()
            if pos["status"] == "OPEN"
        )

        total_value = strategy_capital + open_positions_value

        return float(total_value)

    def get_max_drawdown(self) -> Tuple[float, float]:
        """
        Calcula drawdown máximo del portafolio.

        Returns:
            Tupla (max_drawdown_dollars, max_drawdown_pct)
        """
        current_value = self.get_portfolio_value()
        peak_value = self.initial_capital

        # Actualizar peak si current es mayor
        if current_value > peak_value:
            peak_value = current_value

        # Calcular drawdown
        drawdown_dollars = current_value - peak_value
        drawdown_pct = (drawdown_dollars / peak_value) * 100 if peak_value > 0 else 0.0

        return float(drawdown_dollars), float(drawdown_pct)

    def rebalance_portfolio(self) -> Dict[str, float]:
        """
        Rebalancea el portafolio según pesos de asignación.

        Returns:
            Nuevos capitales por estrategia
        """
        logger.info("⚖️ Rebalanceando portafolio...")

        current_value = self.get_portfolio_value()

        new_allocations = {}

        for strategy_name, weight in self.allocation_weights.items():
            if strategy_name in self.strategies:
                target_capital = current_value * weight
                current_capital = self.strategies[strategy_name]["capital"]

                # Ajustar capital
                self.strategies[strategy_name]["capital"] = target_capital

                new_allocations[strategy_name] = target_capital

                logger.info(
                    f"  {strategy_name}: ${current_capital:,.2f} → ${target_capital:,.2f}"
                )

        logger.info("✅ Rebalanceo completado")

        return new_allocations

    def generate_portfolio_state(self) -> Dict:
        """
        Genera estado actual del portafolio.

        Returns:
            Diccionario con estado completo
        """
        portfolio_value = self.get_portfolio_value()
        total_pnl = portfolio_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        drawdown_dollars, drawdown_pct = self.get_max_drawdown()

        # Posiciones abiertas
        open_positions = [
            pos for pos in self.positions.values() if pos["status"] == "OPEN"
        ]

        # Estadísticas por estrategia
        strategy_stats = {}
        for name, strategy in self.strategies.items():
            win_rate = (
                (strategy["wins"] / strategy["trades"]) * 100
                if strategy["trades"] > 0
                else 0.0
            )

            strategy_stats[name] = {
                "allocation": strategy["allocation"],
                "capital": strategy["capital"],
                "pnl": strategy["pnl"],
                "trades": strategy["trades"],
                "wins": strategy["wins"],
                "losses": strategy["losses"],
                "win_rate": win_rate,
            }

        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio_value": portfolio_value,
            "initial_capital": self.initial_capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "drawdown_dollars": drawdown_dollars,
            "drawdown_pct": drawdown_pct,
            "total_exposure": self._calculate_total_exposure(),
            "open_positions_count": len(open_positions),
            "strategies": strategy_stats,
        }

        # Guardar estado
        state_path = Path(self.PORTFOLIO_STATE_PATH)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"💾 Estado del portafolio guardado en {state_path}")

        return state

    def generate_risk_metrics(self) -> Dict:
        """
        Genera métricas de riesgo del portafolio.

        Returns:
            Diccionario con métricas de riesgo
        """
        var_95 = self.calculate_var(confidence_level=0.95)
        var_99 = self.calculate_var(confidence_level=0.99)
        beta = self.calculate_beta()
        correlation_matrix = self.calculate_strategy_correlation()

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "value_at_risk": {
                "var_95": var_95,
                "var_99": var_99,
                "confidence_95_pct": 95.0,
                "confidence_99_pct": 99.0,
            },
            "market_exposure": {
                "beta": beta,
                "interpretation": "< 1: menos volátil que mercado, > 1: más volátil",
            },
            "strategy_correlation": correlation_matrix.to_dict()
            if not correlation_matrix.empty
            else {},
            "risk_limits": {
                "max_exposure_per_symbol": self.max_exposure_per_symbol,
                "max_total_exposure": self.max_total_exposure,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
            },
        }

        # Guardar métricas
        metrics_path = Path(self.RISK_METRICS_PATH)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"💾 Métricas de riesgo guardadas en {metrics_path}")

        return metrics


def run_portfolio_simulation() -> None:
    """Ejecuta simulación de gestión de portafolio."""
    print("=" * 60)
    print("💼 PORTFOLIO MANAGER - RISK & MULTI-STRATEGY")
    print("=" * 60)

    # Crear portfolio manager
    pm = PortfolioManager(initial_capital=100000.0)

    # Registrar estrategias
    pm.register_strategy("lstm_predictor", allocation=0.4)
    pm.register_strategy("pattern_matcher", allocation=0.3)
    pm.register_strategy("hybrid_agent", allocation=0.3)

    # Simular apertura de posiciones
    print("\n📊 Abriendo posiciones de prueba...")

    positions_opened = []

    # LSTM - BUY
    success, pos_id = pm.open_position(
        "lstm_predictor", "BTCUSDT", "BUY", quantity=0.5, entry_price=50000.0
    )
    if success:
        positions_opened.append(pos_id)

    # Pattern - BUY
    success, pos_id = pm.open_position(
        "pattern_matcher", "ETHUSDT", "BUY", quantity=5.0, entry_price=3000.0
    )
    if success:
        positions_opened.append(pos_id)

    # Hybrid - SELL
    success, pos_id = pm.open_position(
        "hybrid_agent", "BTCUSDT", "SELL", quantity=0.3, entry_price=50000.0
    )
    if success:
        positions_opened.append(pos_id)

    # Simular movimientos de precio
    print("\n📈 Simulando movimientos de precio...")

    # Actualizar precios
    for i, pos_id in enumerate(positions_opened):
        # Simular cambios de precio
        price_changes = [1.02, 0.98, 1.05]  # +2%, -2%, +5%
        if i < len(price_changes):
            pos = pm.positions[pos_id]
            new_price = pos["entry_price"] * price_changes[i]
            pm.update_position_price(pos_id, new_price)

    # Generar estado y métricas
    print("\n💾 Generando reportes...")

    state = pm.generate_portfolio_state()
    risk_metrics = pm.generate_risk_metrics()

    # Mostrar resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DEL PORTAFOLIO")
    print("=" * 60)
    print(f"Valor total: ${state['portfolio_value']:,.2f}")
    print(f"PnL total: ${state['total_pnl']:,.2f} ({state['total_pnl_pct']:.2f}%)")
    print(f"Exposición: {state['total_exposure']:.1%}")
    print(f"Posiciones abiertas: {state['open_positions_count']}")
    print(f"\nVaR (95%): ${risk_metrics['value_at_risk']['var_95']:,.2f}")
    print(f"VaR (99%): ${risk_metrics['value_at_risk']['var_99']:,.2f}")
    print(f"Beta: {risk_metrics['market_exposure']['beta']:.2f}")

    print("\n✅ Simulación completada. Revisar reports/portfolio_state.json")


if __name__ == "__main__":
    run_portfolio_simulation()
