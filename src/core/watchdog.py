"""Watchdog service responsible for monitoring bot heartbeat and recovery."""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.core.logger import get_logger


def _ensure_directory(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


class Watchdog:
	"""Monitoriza el heartbeat del bot y relanza el proceso si deja de responder."""

	def __init__(
		self,
		heartbeat_file: Path | str = Path("logs/last_heartbeat.txt"),
		log_file: Path | str = Path("logs/watchdog.log"),
		stale_after: timedelta = timedelta(minutes=5),
		restart_command: Optional[Sequence[str]] = None,
	) -> None:
		self.heartbeat_file = Path(heartbeat_file)
		self.log_file = Path(log_file)
		self.stale_after = stale_after
		self.restart_command = list(restart_command) if restart_command else None
		self._last_restart: Optional[datetime] = None
		self._restart_count: int = 0

		_ensure_directory(self.heartbeat_file)
		_ensure_directory(self.log_file)

		self.logger = get_logger("Watchdog")
		self._attach_file_handler(self.logger, self.log_file)

	@staticmethod
	def _attach_file_handler(logger: logging.Logger, log_file: Path) -> None:
		has_file_handler = any(
			isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
			for handler in logger.handlers
		)
		if not has_file_handler:
			file_handler = logging.FileHandler(log_file, encoding="utf-8")
			file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
			logger.addHandler(file_handler)
			logger.propagate = False

	def record_heartbeat(self) -> None:
		"""Escribe timestamp actual en el archivo de heartbeat."""
		now = datetime.utcnow().isoformat()
		self.heartbeat_file.write_text(now, encoding="utf-8")
		self.logger.debug("Heartbeat actualizado: %s", now)

	def _read_last_heartbeat(self) -> Optional[datetime]:
		if not self.heartbeat_file.exists():
			return None
		try:
			content = self.heartbeat_file.read_text(encoding="utf-8").strip()
			if not content:
				return None
			return datetime.fromisoformat(content)
		except ValueError:
			self.logger.warning("Formato inválido en heartbeat: %s", self.heartbeat_file)
			return None

	def is_stale(self) -> bool:
		"""Indica si el heartbeat está vencido."""
		last = self._read_last_heartbeat()
		if last is None:
			return True
		return datetime.utcnow() - last > self.stale_after

	def check_and_restart(self) -> bool:
		"""Verifica el heartbeat y relanza el bot si es necesario."""
		if not self.is_stale():
			return False

		if self._last_restart and datetime.utcnow() - self._last_restart < self.stale_after:
			self.logger.warning("Watchdog detectó inactividad, pero reinicio reciente evita loop infinito.")
			return False

		self.logger.error("Watchdog: sin heartbeat reciente. Reiniciando bot...")
		restarted = self._restart_process()
		if restarted:
			self._last_restart = datetime.utcnow()
			self._restart_count += 1
			self.logger.info("Watchdog reinició el bot | reinicios acumulados=%s", self._restart_count)
		else:
			self.logger.error("Watchdog no pudo reiniciar el bot tras detectar inactividad.")
		return restarted

	def _restart_process(self) -> bool:
		command = self._build_restart_command()
		try:
			subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			self.logger.info("Watchdog lanzó nuevo proceso con comando: %s", " ".join(command))
			return True
		except Exception as exc:  # pragma: no cover - defensive logging
			self.logger.exception("Watchdog no pudo reiniciar el bot: %s", exc)
			return False

	def _build_restart_command(self) -> Sequence[str]:
		if self.restart_command:
			return self.restart_command

		# Comando por defecto relanza main en modo testnet loop
		return [sys.executable, "-m", "src.main", "--mode", "testnet", "--loop"]


def record_heartbeat(heartbeat_file: Path | str = Path("logs/last_heartbeat.txt")) -> None:
	"""Helper global para actualizar heartbeat sin instanciar la clase."""
	Watchdog(heartbeat_file=heartbeat_file).record_heartbeat()
