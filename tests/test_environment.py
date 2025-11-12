"""Prueba básica de que las dependencias críticas están instaladas y accesibles."""

from __future__ import annotations

import importlib
import os

import pytest

SANDBOXED = "com.apple" in os.environ.get("PATH", "")


def test_environment_setup():
    """Valida paquetes base que no dependen de Torch."""
    packages = ["pandas", "numpy", "lightgbm"]
    loaded = []
    for pkg in packages:
        module = importlib.import_module(pkg)
        assert hasattr(module, "__version__"), f"Paquete {pkg} no expone __version__"
        loaded.append(pkg)

    assert set(loaded) == set(packages)


@pytest.mark.skipif(
    SANDBOXED, reason="Torch se validó fuera del sandbox; import interno bloqueado."
)
def test_torch_available():
    import torch

    assert hasattr(torch, "__version__")
