"""Kill switch module providing safety checks for trading operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from src.core.logger import get_logger

KILL_SWITCH_LOG = Path("logs/kill_switch.log")
KILL_SWITCH_FLAG = Path("kill_switch.flag")


def _ensure_directory(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def _attach_file_handler(logger: logging.Logger, log_file: Path) -> None:
	has_file_handler = any(
		isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
		for handler in logger.handlers
	)
	if not has_file_handler:
		handler = logging.FileHandler(log_file, encoding="utf-8")
		handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
		logger.addHandler(handler)
		logger.propagate = False


def _normalise_percent(value: float | int | str | None) -> float:
	"""Devuelve un valor en porcentaje como decimal (0.05 equivale a 5%)."""
	if value is None:
		return 0.0
	try:
		numeric = float(value)
	except (TypeError, ValueError):
		return 0.0
	if abs(numeric) > 1:
		return numeric / 100.0
	return numeric


_LOGGER = get_logger("KillSwitch")
_ensure_directory(KILL_SWITCH_LOG)
_attach_file_handler(_LOGGER, KILL_SWITCH_LOG)


@dataclass
class KillSwitchResult:
	"""Representa el resultado de una evaluación del kill switch."""

	triggered: bool
	reasons: tuple[str, ...] = ()

	def __bool__(self) -> bool:  # pragma: no cover - convenience
		return self.triggered


def _flag_present(flag_path: Path) -> bool:
	try:
		return flag_path.exists()
	except OSError:  # pragma: no cover - filesystem edge case
		return False


def check_kill_conditions(
	engine_state: Optional[Mapping[str, float]] = None,
	pnl: Optional[float] = None,
	errors_last_minute: int = 0,
) -> bool:
	"""Evalúa si se deben detener las operaciones.

	Args:
		engine_state: Mapa con métricas del motor, requiere llaves "pnl_acumulado" y
			"drawdown_relativo" expresadas como porcentaje (por ejemplo -0.08 o -8).
		pnl: PnL de la última operación ejecutada (no imprescindible).
		errors_last_minute: Número de errores críticos registrados en los últimos segundos
			(el umbral esperado es 3 errores en una ventana de 10 segundos).

	Returns:
		bool: ``True`` si se activa el kill switch.
	"""

	engine_state = engine_state or {}

	cumulative_pnl = _normalise_percent(engine_state.get("pnl_acumulado"))
	relative_drawdown = _normalise_percent(engine_state.get("drawdown_relativo"))

	reasons: list[str] = []

	if cumulative_pnl < -0.05:
		reasons.append(f"pnl_acumulado crítico ({cumulative_pnl:.2%})")

	if relative_drawdown < -0.07:
		reasons.append(f"drawdown_relativo crítico ({relative_drawdown:.2%})")

	if errors_last_minute >= 3:
		reasons.append(f"errores_api sucesivos ({errors_last_minute})")

	if _flag_present(KILL_SWITCH_FLAG):
		reasons.append("flag kill_switch.flag detectada")

	triggered = bool(reasons)

	log_context = {
		"cumulative_pnl": cumulative_pnl,
		"relative_drawdown": relative_drawdown,
		"last_pnl": pnl or 0.0,
		"errors_last_minute": errors_last_minute,
	}

	if triggered:
		_LOGGER.error("Kill switch ACTIVADO: %s | contexto=%s", ", ".join(reasons), log_context)
	else:
		_LOGGER.info("Kill switch OK | contexto=%s", log_context)

	return triggered


__all__ = ["check_kill_conditions", "KillSwitchResult", "KILL_SWITCH_FLAG", "KILL_SWITCH_LOG"]
