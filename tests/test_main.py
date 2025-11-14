import subprocess
import sys
from pathlib import Path


def test_main_runs():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--check"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(project_root),
    )
    assert result.returncode == 0
    assert "IA BOT TRADING" in result.stdout
