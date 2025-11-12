"""Prueba básica de que las dependencias críticas están instaladas y accesibles."""

def test_environment_setup():
    import importlib

    packages = ["pandas", "numpy", "torch", "lightgbm"]
    loaded = []
    for pkg in packages:
        module = importlib.import_module(pkg)
        assert hasattr(module, "__version__"), f"Paquete {pkg} no expone __version__"
        loaded.append(pkg)

    # Afirmar que se cargaron todos
    assert set(loaded) == set(packages)
