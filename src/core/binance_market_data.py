"""Utility module to fetch OHLCV market data from Binance public REST API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from requests import RequestException


def _to_millis(ts: str | pd.Timestamp) -> int:
    """Convert a datetime-like value to Binance-compatible milliseconds."""
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


@dataclass
class BinanceMarketData:
    """Minimal client for retrieving kline data from Binance."""

    base_url: str = "https://api.binance.com"
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch OHLCV candles between two dates (inclusive start, exclusive end)."""
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)

        klines: list[list] = []
        current = start_ms
        limit = 1000
        url = f"{self.base_url}/api/v3/klines"

        try:
            while current < end_ms:
                params = {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": limit,
                }
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                batch = response.json()

                if not batch:
                    break

                klines.extend(batch)
                last_close_time = batch[-1][6]
                if last_close_time <= current:
                    break
                current = last_close_time + 1

        except RequestException:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        if not klines:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        data = {
            "timestamp": [pd.to_datetime(k[0], unit="ms", utc=True) for k in klines],
            "open": [float(k[1]) for k in klines],
            "high": [float(k[2]) for k in klines],
            "low": [float(k[3]) for k in klines],
            "close": [float(k[4]) for k in klines],
            "volume": [float(k[5]) for k in klines],
        }

        df = pd.DataFrame(data)
        start_boundary = pd.to_datetime(start_ms, unit="ms", utc=True)
        end_boundary = pd.to_datetime(end_ms, unit="ms", utc=True)
        df = df[(df["timestamp"] >= start_boundary) & (df["timestamp"] <= end_boundary)]
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
