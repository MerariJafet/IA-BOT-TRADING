"""
Tests para el emulador de señales de trading.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.signal_emulator import LiveSignalEmulator


def test_live_signal_emulator(tmp_path):
    """Test del emulador de señales en vivo."""
    # Crear datos de secuencia simulados
    data = pd.DataFrame(
        {
            "price": [100, 101, 102],
            "volume": [1000, 1100, 1200],
            "return_mean": [0.01, 0.02, 0.03],
        }
    )

    seq_dir = tmp_path / "data" / "sequences"
    seq_dir.mkdir(parents=True)
    seq_file = seq_dir / "BTCUSDT_seq_1m.parquet"
    data.to_parquet(seq_file)

    # Crear pattern strengths simulados
    reinf_dir = tmp_path / "data" / "reinforcement"
    reinf_dir.mkdir(parents=True)
    strengths_df = pd.DataFrame({"strength": [0.7, 0.3, 0.9]})
    strengths_path = reinf_dir / "pattern_strengths.parquet"
    strengths_df.to_parquet(strengths_path)

    # Inicializar emulador
    emulator = LiveSignalEmulator(
        seq_path=str(seq_file),
        interval=0,  # Sin delay para tests
        pattern_strength_path=str(strengths_path),
    )

    # Generar señales
    stream = list(emulator.stream())

    # Validaciones
    assert len(stream) == 3
    assert all("signal" in s for s in stream)
    assert all("timestamp" in s for s in stream)
    assert all("price" in s for s in stream)
    assert all(s["signal"] in [-1, 0, 1] for s in stream)


def test_emulator_without_patterns(tmp_path):
    """Test del emulador sin pattern strengths (usa decisiones aleatorias)."""
    # Crear solo datos de secuencia
    data = pd.DataFrame(
        {"price": [100, 101], "volume": [1000, 1100], "return_mean": [0.01, 0.02]}
    )

    seq_dir = tmp_path / "data" / "sequences"
    seq_dir.mkdir(parents=True)
    seq_file = seq_dir / "BTCUSDT_seq_1m.parquet"
    data.to_parquet(seq_file)

    # Inicializar emulador sin pattern strengths
    emulator = LiveSignalEmulator(seq_path=str(seq_file), interval=0)

    # Generar señales
    stream = list(emulator.stream())

    # Debe funcionar incluso sin patterns
    assert len(stream) == 2
    assert all("signal" in s for s in stream)


def test_emulator_signal_values(tmp_path):
    """Test que las señales generadas son válidas."""
    data = pd.DataFrame(
        {"close": [100.5, 101.2, 99.8], "volume": [1000, 1100, 900]}
    )

    seq_dir = tmp_path / "data" / "sequences"
    seq_dir.mkdir(parents=True)
    seq_file = seq_dir / "TEST_seq.parquet"
    data.to_parquet(seq_file)

    # Crear pattern strengths
    reinf_dir = tmp_path / "data" / "reinforcement"
    reinf_dir.mkdir(parents=True)
    strengths_df = pd.DataFrame({"strength": [0.8]})
    strengths_path = reinf_dir / "pattern_strengths.parquet"
    strengths_df.to_parquet(strengths_path)

    emulator = LiveSignalEmulator(
        seq_path=str(seq_file),
        interval=0,
        pattern_strength_path=str(strengths_path),
    )

    stream = list(emulator.stream())

    # Validar que los precios se extrajeron correctamente
    assert stream[0]["price"] == 100.5
    assert stream[1]["price"] == 101.2
    assert stream[2]["price"] == 99.8
