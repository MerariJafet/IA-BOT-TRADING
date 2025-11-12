"""Data loader para descargar datos de Binance (ejemplo simplificado).

Nota: Para endpoints públicos no se requieren credenciales. Si se pasa
``use_auth=True`` se intentarán leer las credenciales y añadir el header
``X-MBX-APIKEY``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import requests

from src.utils.env_loader import get_binance_credentials

BASE_URL = "https://api.binance.com"  # Spot endpoints


def _request(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    use_auth: bool = False,
) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    headers = {}
    if use_auth:
        api_key, _ = get_binance_credentials()
        headers["X-MBX-APIKEY"] = api_key
    resp = requests.get(url, params=params or {}, headers=headers, timeout=10)
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        raise ValueError("Respuesta no es JSON válido") from None


def download_symbol_price(symbol: str = "BTCUSDT", use_auth: bool = False) -> float:
    """Obtiene el precio promedio actual del símbolo usando endpoint público de Binance.

    Args:
        symbol: par de trading (ej. 'BTCUSDT').
        use_auth: si True, añade cabecera de API Key (no necesaria para este
            endpoint, sólo demostración).
    Returns:
        float: precio promedio.
    """
    data = _request(
        "/api/v3/avgPrice",
        params={"symbol": symbol.upper()},
        use_auth=use_auth,
    )
    return float(data["price"])  # type: ignore


def fetch_agg_trades(
    symbol: str,
    start_time_ms: Optional[int] = None,
    limit: int = 10,
    use_auth: bool = False,
):
    params: Dict[str, Any] = {"symbol": symbol.upper(), "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms
        params["endTime"] = start_time_ms + 60_000  # ejemplo simple
    return _request("/api/v3/aggTrades", params=params, use_auth=use_auth)


def wait_and_download(symbol: str, delay: float = 0.2) -> float:
    time.sleep(delay)
    return download_symbol_price(symbol)


class BinanceDataFetcher:
    """Descarga datos históricos OHLCV de Binance para backtesting."""

    def __init__(
        self, symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 500
    ):
        """
        Inicializa el fetcher de datos.

        Args:
            symbol: Par de trading (ej. 'BTCUSDT')
            interval: Intervalo de tiempo (1m, 5m, 15m, 1h, etc.)
            limit: Número máximo de velas a descargar (max 1000)
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = min(limit, 1000)  # Binance limit

    def fetch(self):
        """
        Descarga datos OHLCV desde Binance y los guarda.

        Returns:
            DataFrame con datos OHLCV
        """
        import pandas as pd
        from pathlib import Path

        from src.core.logger import get_logger

        logger = get_logger(__name__)

        logger.info(
            f"📥 Descargando {self.limit} velas de "
            f"{self.symbol} ({self.interval})..."
        )

        # Obtener datos desde Binance API
        api_key, _ = get_binance_credentials()
        url = f"{BASE_URL}/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit,
        }
        headers = {"X-MBX-APIKEY": api_key}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        # Procesar datos
        data = response.json()
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]

        df = pd.DataFrame(data, columns=columns)

        # Convertir tipos
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        df["num_trades"] = df["num_trades"].astype(int)

        # Guardar datos
        output_dir = Path("data/real")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.symbol}_{self.interval}.parquet"

        df.to_parquet(output_path, index=False)

        logger.info(
            f"✅ Datos guardados: {len(df)} velas → {output_path}"
        )

        return df
