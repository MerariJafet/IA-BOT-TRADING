"""Utility for sending Telegram alerts."""

import os
from typing import Final

import requests

TELEGRAM_TOKEN: Final[str] = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: Final[str] = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_alert(message: str) -> None:
    """Send an alert message to Telegram using simple GET request."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise EnvironmentError("Telegram credentials are not configured")

    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()
