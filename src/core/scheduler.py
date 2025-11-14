"""Scheduler service orchestrating recurring maintenance tasks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import schedule

from src.core.logger import get_logger

SCHEDULER_LOG = Path("logs/scheduler.log")


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


LOGGER = get_logger("Scheduler")
_ensure_directory(SCHEDULER_LOG)
_attach_file_handler(LOGGER, SCHEDULER_LOG)


class SchedulerService:
	"""Administra el scheduler para tareas periódicas dentro del bot."""

	def __init__(
		self,
		heartbeat_fn: Callable[[], None],
		kill_switch_fn: Callable[[], bool],
		watchdog_check_fn: Callable[[], bool],
		daily_report_fn: Callable[[], None],
		scheduler_instance: Optional[schedule.Scheduler] = None,
	) -> None:
		self._scheduler = scheduler_instance or schedule.Scheduler()
		self._jobs: Dict[str, schedule.Job] = {}
		self._heartbeat_fn = heartbeat_fn
		self._kill_switch_fn = kill_switch_fn
		self._watchdog_check_fn = watchdog_check_fn
		self._daily_report_fn = daily_report_fn
		self._started = False

	def start(self) -> None:
		if self._started:
			return
		self._register_jobs()
		self._started = True
		LOGGER.info("Scheduler inicializado con %s tareas programadas.", len(self._jobs))

	def _register_jobs(self) -> None:
		self._jobs["heartbeat"] = self._scheduler.every(30).seconds.do(
			self._run_job, "heartbeat", self._heartbeat_fn
		)
		self._jobs["kill_switch"] = self._scheduler.every(60).seconds.do(
			self._run_job, "kill_switch", self._kill_switch_fn
		)
		self._jobs["watchdog"] = self._scheduler.every(2).minutes.do(
			self._run_job, "watchdog", self._watchdog_check_fn
		)
		self._jobs["daily_report"] = self._scheduler.every().day.at("23:55").do(
			self._run_job, "daily_report", self._daily_report_fn
		)

	def _run_job(self, name: str, func: Callable[[], Optional[bool]]) -> None:
		try:
			result = func()
			LOGGER.info("Job '%s' ejecutado. resultado=%s", name, result)
		except Exception as exc:  # pragma: no cover - logging only
			LOGGER.exception("Job '%s' falló: %s", name, exc)

	def run_pending(self) -> None:
		"""Ejecuta todas las tareas cuyo horario esté vencido."""
		self._scheduler.run_pending()

	def trigger(self, name: str) -> None:
		"""Dispara manualmente una tarea, usado para pruebas y emergencias."""
		job = self._jobs.get(name)
		if not job:
			raise KeyError(f"Job '{name}' no encontrado")
		job.next_run = datetime.now()
		self.run_pending()

	def jobs(self) -> Dict[str, schedule.Job]:
		return dict(self._jobs)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Scheduler utility module")
	parser.add_argument("--test", action="store_true", help="Ejecuta prueba rápida del scheduler.")
	return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
	args = _parse_args(argv)

	if args.test:
		counters = {"heartbeat": 0, "kill": 0, "watchdog": 0, "report": 0}

		def heartbeat():
			counters["heartbeat"] += 1

		def kill_switch():
			counters["kill"] += 1
			return False

		def watchdog():
			counters["watchdog"] += 1
			return False

		def report():
			counters["report"] += 1

		scheduler = SchedulerService(
			heartbeat_fn=heartbeat,
			kill_switch_fn=kill_switch,
			watchdog_check_fn=watchdog,
			daily_report_fn=report,
		)
		scheduler.start()
		for job in scheduler.jobs().values():
			job.next_run = datetime.now()
		scheduler.run_pending()
		LOGGER.info("Scheduler test completado | counters=%s", counters)
		return

	LOGGER.info("Scheduler listo. Usar desde main para ejecución completa.")


if __name__ == "__main__":  # pragma: no cover
	main()


__all__ = ["SchedulerService", "LOGGER", "SCHEDULER_LOG", "main"]
