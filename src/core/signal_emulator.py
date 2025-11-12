"""
Signal Emulator - Emulador de señales de trading en tiempo real.

Este módulo simula señales de trading en vivo leyendo secuencias históricas
y generando decisiones BUY/SELL/HOLD con timestamps.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.core.logger import get_logger
from src.core.policy_agent import PolicyAgent

logger = get_logger(__name__)


class LiveSignalEmulator:
    """Emulador de señales de trading en tiempo real."""

    def __init__(
        self,
        seq_path: str = "data/sequences/BTCUSDT_seq_1m.parquet",
        interval: float = 1.0,
        pattern_strength_path: Optional[str] = None,
    ):
        """
        Inicializa el emulador de señales.

        Args:
            seq_path: Ruta al archivo parquet con secuencias de mercado
            interval: Intervalo en segundos entre señales (0 para sin delay)
            pattern_strength_path: Ruta opcional para pattern strengths
        """
        self.seq_path = Path(seq_path)
        if not self.seq_path.exists():
            raise FileNotFoundError(f"Secuencia no encontrada: {seq_path}")

        self.df = pd.read_parquet(self.seq_path)
        self.interval = interval

        # Inicializar agente de política
        try:
            if pattern_strength_path:
                self.agent = PolicyAgent(pattern_strength_path=pattern_strength_path)
            else:
                self.agent = PolicyAgent()
        except FileNotFoundError:
            logger.warning(
                "⚠️ Pattern strengths no encontrado, usando decisiones aleatorias"
            )
            self.agent = None

        logger.info(
            f"📡 LiveSignalEmulator inicializado con {len(self.df)} observaciones"
        )

    def stream(self) -> Generator[dict, None, None]:
        """
        Genera señales de trading en tiempo real.

        Yields:
            Diccionario con timestamp, signal y price
        """
        logger.info("🚀 Iniciando stream de señales en vivo...")

        for i, row in self.df.iterrows():
            # Obtener decisión del agente
            if self.agent:
                signal = self.agent.decide(row)
            else:
                # Fallback a decisión aleatoria
                import numpy as np

                signal = np.random.choice([-1, 0, 1])

            # Timestamp actual
            ts = datetime.utcnow().strftime("%H:%M:%S")

            # Obtener precio (intentar diferentes columnas)
            price = None
            for col in ["price", "close", "return_mean"]:
                if col in row.index:
                    price = float(row[col])
                    break

            if price is None:
                price = 0.0

            signal_str = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
            logger.info(f"[{ts}] Signal={signal_str} | Price={price:.2f}")

            yield {"timestamp": ts, "signal": signal, "price": price}

            # Esperar intervalo
            if self.interval > 0:
                time.sleep(self.interval)


if __name__ == "__main__":
    emulator = LiveSignalEmulator()
    count = 0
    max_signals = 10  # Limitar a 10 señales en modo demo

    for signal_data in emulator.stream():
        count += 1
        if count >= max_signals:
            logger.info(f"✅ Demo completado: {count} señales generadas")
            break
