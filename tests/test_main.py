import subprocess
import sys


def test_main_runs():
    result = subprocess.run([sys.executable, "src/main.py"], capture_output=True, text=True)
    assert "IA BOT TRADING" in result.stdout or "Sistema IA BOT TRADING" in result.stdout
