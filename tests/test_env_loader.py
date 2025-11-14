from utils.env_loader import get_binance_credentials


def test_env_variables_exist(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
    key, secret = get_binance_credentials()
    assert key == "test_key"
    assert secret == "test_secret"
