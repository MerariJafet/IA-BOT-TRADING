"""
Execution Engine - Motor de ejecución de trades en Binance TestNet.

Este módulo permite ejecutar órdenes LIMIT y MARKET en dos modos:
- Paper: Simulación local sin conexión real
- TestNet: Ejecución real en Binance TestNet API
"""

from __future__ import annotations

import hashlib
import hmac
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from requests import RequestException

from src.core.logger import get_logger
from src.utils.env_loader import get_binance_credentials

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Modos de ejecución disponibles."""

    PAPER = "paper"  # Simulación local
    TESTNET = "testnet"  # Binance TestNet API


class OrderType(Enum):
    """Tipos de órdenes soportadas."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderSide(Enum):
    """Lado de la orden."""

    BUY = "BUY"
    SELL = "SELL"


class ExecutionEngine:
    """Motor de ejecución de trades en Binance."""

    TESTNET_BASE_URL = "https://testnet.binance.vision"
    LIVE_TRADES_PATH = "data/live_trades.parquet"

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        symbol: str = "BTCUSDT",
        initial_balance: float = 10000.0,
    ):
        """
        Inicializa el motor de ejecución.

        Args:
            mode: Modo de ejecución (paper o testnet)
            symbol: Símbolo del par de trading
            initial_balance: Balance inicial para modo paper
        """
        self.mode = mode
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.paper_balance = initial_balance
        self.paper_position = 0.0  # Cantidad de BTC en paper mode

        # Cargar credenciales si estamos en testnet
        self.api_key = None
        self.api_secret = None

        if self.mode == ExecutionMode.TESTNET:
            try:
                self.api_key, self.api_secret = get_binance_credentials()
                logger.info("🔑 Credenciales de Binance TestNet cargadas")
            except EnvironmentError as e:
                logger.error(f"❌ Error cargando credenciales: {e}")
                raise

        # Inicializar archivo de trades
        self._init_trades_file()

        logger.info(f"🚀 ExecutionEngine inicializado en modo {mode.value}")

    def _init_trades_file(self) -> None:
        """Inicializa el archivo de trades si no existe."""
        trades_path = Path(self.LIVE_TRADES_PATH)
        trades_path.parent.mkdir(parents=True, exist_ok=True)

        if not trades_path.exists():
            # Crear archivo vacío
            df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "order_id",
                    "symbol",
                    "side",
                    "type",
                    "quantity",
                    "price",
                    "status",
                    "mode",
                    "pnl",
                ]
            )
            df.to_parquet(trades_path, index=False)
            logger.info(f"📝 Archivo de trades creado: {trades_path}")

    def _sign_request(self, params: Dict) -> str:
        """
        Firma una petición para Binance API.

        Args:
            params: Parámetros de la petición

        Returns:
            Firma HMAC SHA256
        """
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return signature

    def get_current_price(self) -> float:
        """
        Obtiene el precio actual del símbolo.

        Returns:
            Precio actual
        """
        if self.mode == ExecutionMode.PAPER:
            # En modo paper, usar un precio simulado más bajo para tests
            return 50000.0

        # En testnet, obtener precio real
        try:
            url = f"{self.TESTNET_BASE_URL}/api/v3/ticker/price"
            params = {"symbol": self.symbol}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return float(data["price"])

        except RequestException as e:
            logger.error(f"❌ Error obteniendo precio: {e}")
            return 50000.0  # Fallback

    def execute_market_order(self, side: OrderSide, quantity: float) -> Optional[Dict]:
        """
        Ejecuta una orden MARKET.

        Args:
            side: BUY o SELL
            quantity: Cantidad a operar

        Returns:
            Diccionario con detalles de la orden
        """
        if self.mode == ExecutionMode.PAPER:
            return self._execute_paper_order(side, OrderType.MARKET, quantity, None)
        else:
            return self._execute_testnet_order(side, OrderType.MARKET, quantity, None)

    def execute_limit_order(self, side: OrderSide, quantity: float, price: float) -> Optional[Dict]:
        """
        Ejecuta una orden LIMIT.

        Args:
            side: BUY o SELL
            quantity: Cantidad a operar
            price: Precio límite

        Returns:
            Diccionario con detalles de la orden
        """
        if self.mode == ExecutionMode.PAPER:
            return self._execute_paper_order(side, OrderType.LIMIT, quantity, price)
        else:
            return self._execute_testnet_order(side, OrderType.LIMIT, quantity, price)

    def _execute_paper_order(
        self,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float],
    ) -> Dict:
        """
        Ejecuta una orden en modo paper (simulación).

        Args:
            side: BUY o SELL
            order_type: LIMIT o MARKET
            quantity: Cantidad
            price: Precio (None para MARKET)

        Returns:
            Detalles de la orden simulada
        """
        # Obtener precio de ejecución
        exec_price = price if order_type == OrderType.LIMIT else self.get_current_price()

        # Simular ejecución
        order_id = f"PAPER_{int(time.time() * 1000)}"
        timestamp = pd.Timestamp.utcnow()

        # Calcular costo/ingreso
        total_cost = quantity * exec_price

        # Actualizar balance y posición
        if side == OrderSide.BUY:
            if self.paper_balance < total_cost:
                logger.warning("⚠️ Balance insuficiente para BUY")
                return {
                    "order_id": order_id,
                    "status": "REJECTED",
                    "reason": "Insufficient balance",
                }

            self.paper_balance -= total_cost
            self.paper_position += quantity

        else:  # SELL
            if self.paper_position < quantity:
                logger.warning("⚠️ Posición insuficiente para SELL")
                return {
                    "order_id": order_id,
                    "status": "REJECTED",
                    "reason": "Insufficient position",
                }

            self.paper_balance += total_cost
            self.paper_position -= quantity

        # Calcular PnL aproximado
        pnl = self._calculate_paper_pnl(side, quantity, exec_price)

        order_details = {
            "timestamp": timestamp.isoformat(),
            "order_id": order_id,
            "symbol": self.symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": quantity,
            "price": exec_price,
            "status": "FILLED",
            "mode": "paper",
            "pnl": pnl,
        }

        # Registrar en archivo
        self._log_trade(order_details)

        logger.info(f"✅ Paper {side.value} ejecutado: {quantity} @ {exec_price} | PnL: {pnl:.2f}")

        return order_details

    def _calculate_paper_pnl(self, side: OrderSide, quantity: float, price: float) -> float:
        """Calcula PnL aproximado para paper trading."""
        # PnL simplificado basado en variación de precio
        if side == OrderSide.BUY:
            return -0.1 * quantity * price  # Fee simulado
        else:
            return 0.1 * quantity * price  # Ganancia simulada

    def _execute_testnet_order(
        self,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float],
    ) -> Optional[Dict]:
        """
        Ejecuta una orden en Binance TestNet.

        Args:
            side: BUY o SELL
            order_type: LIMIT o MARKET
            quantity: Cantidad
            price: Precio (requerido para LIMIT)

        Returns:
            Detalles de la orden o None si falla
        """
        try:
            url = f"{self.TESTNET_BASE_URL}/api/v3/order"

            # Construir parámetros
            params = {
                "symbol": self.symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": quantity,
                "timestamp": int(time.time() * 1000),
            }

            if order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for LIMIT orders")
                params["price"] = price
                params["timeInForce"] = "GTC"

            # Firmar petición
            signature = self._sign_request(params)
            params["signature"] = signature

            # Headers
            headers = {"X-MBX-APIKEY": self.api_key}

            # Enviar orden
            response = requests.post(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            order_data = response.json()

            # Construir detalles
            order_details = {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "order_id": str(order_data.get("orderId", "N/A")),
                "symbol": self.symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": quantity,
                "price": float(order_data.get("price", price or 0)),
                "status": order_data.get("status", "UNKNOWN"),
                "mode": "testnet",
                "pnl": 0.0,  # Se calculará después
            }

            # Registrar trade
            self._log_trade(order_details)

            logger.info(f"✅ TestNet {side.value} ejecutado: {quantity} @ {order_details['price']}")

            return order_details

        except RequestException as e:
            logger.error(f"❌ Error ejecutando orden en TestNet: {e}")
            return None

    def _log_trade(self, order_details: Dict) -> None:
        """
        Registra un trade en el archivo de trades.

        Args:
            order_details: Detalles de la orden
        """
        trades_path = Path(self.LIVE_TRADES_PATH)

        # Cargar trades existentes
        if trades_path.exists() and trades_path.stat().st_size > 0:
            df_existing = pd.read_parquet(trades_path)
        else:
            df_existing = pd.DataFrame()

        # Agregar nuevo trade
        df_new = pd.DataFrame([order_details])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)

        if not df_existing.empty and "timestamp" in df_existing.columns:
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"], utc=True)

        # Guardar
        df_combined.to_parquet(trades_path, index=False)

        logger.info(f"📝 Trade registrado en {trades_path}")

    def get_trade_history(self) -> pd.DataFrame:
        """
        Obtiene el historial de trades.

        Returns:
            DataFrame con trades registrados
        """
        trades_path = Path(self.LIVE_TRADES_PATH)

        if not trades_path.exists() or trades_path.stat().st_size == 0:
            return pd.DataFrame()

        return pd.read_parquet(trades_path)

    def get_paper_balance(self) -> Dict[str, float]:
        """
        Obtiene el balance actual en modo paper.

        Returns:
            Diccionario con balance y posición
        """
        return {
            "balance_usdt": self.paper_balance,
            "position_btc": self.paper_position,
            "total_value_usdt": self.paper_balance + (
                self.paper_position * self.get_current_price()
            ),
        }


def simulate_trading_session(num_trades: int = 5) -> None:
    """
    Simula una sesión de trading con múltiples operaciones.

    Args:
        num_trades: Número de trades a ejecutar
    """
    logger.info(f"🎯 Iniciando sesión de trading con {num_trades} trades")

    # Crear engine en modo paper
    engine = ExecutionEngine(mode=ExecutionMode.PAPER, initial_balance=10000.0)

    for i in range(num_trades):
        # Alternar BUY/SELL
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL

        # Cantidad aleatoria pequeña
        quantity = 0.01 + (i * 0.005)

        # Ejecutar orden MARKET
        result = engine.execute_market_order(side, quantity)

        if result and result["status"] == "FILLED":
            logger.info(f"✅ Trade {i + 1}/{num_trades} completado")
        else:
            logger.warning(f"⚠️ Trade {i + 1}/{num_trades} rechazado")

        time.sleep(0.5)  # Pausa entre trades

    # Mostrar resumen
    balance = engine.get_paper_balance()
    logger.info("📊 Sesión completada:")
    logger.info(f"  Balance: ${balance['balance_usdt']:.2f}")
    logger.info(f"  Posición: {balance['position_btc']:.4f} BTC")
    logger.info(f"  Valor total: ${balance['total_value_usdt']:.2f}")

    # Guardar resumen
    trades = engine.get_trade_history()
    logger.info(f"  Total trades: {len(trades)}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 EXECUTION ENGINE - BINANCE TESTNET")
    print("=" * 60)

    # Ejecutar sesión de prueba
    simulate_trading_session(num_trades=5)

    print("\n✅ Sesión completada. Revisar data/live_trades.parquet")
