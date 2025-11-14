import os
from pathlib import Path

from dotenv import load_dotenv

# Cargar archivo .env en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)


def get_binance_credentials():
    """Obtiene las credenciales de Binance desde variables de entorno.

    Returns:
        tuple[str, str]: (api_key, api_secret)
    Raises:
        EnvironmentError: Si alguna credencial no está definida.
    """
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise EnvironmentError("Binance API credentials not found in environment")
    return api_key, api_secret
