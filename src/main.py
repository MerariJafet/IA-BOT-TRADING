"""Entry point for IA BOT TRADING runtime."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Optional, Tuple

from src.core.daily_report import generate_daily_report
from src.core.execution_engine import ExecutionEngine, ExecutionMode, OrderSide
from src.core.kill_switch import check_kill_conditions
from src.core.logger import get_logger
from src.core.scheduler import SchedulerService
from src.core.watchdog import Watchdog

MAIN_LOG = Path("logs/main.log")


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _attach_file_handler(logger, log_file: Path) -> None:
    import logging

    has_file_handler = any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
        for handler in logger.handlers
    )
    if not has_file_handler:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


logger = get_logger("Main")
_ensure_directory(MAIN_LOG)
_attach_file_handler(logger, MAIN_LOG)

BASE_CAPITAL = 10000.0


class ErrorTracker:
    """Mantiene el registro de errores recientes para activar el kill switch."""

    def __init__(self, window_seconds: int = 10) -> None:
        self.window = timedelta(seconds=window_seconds)
        self._events: Deque[datetime] = deque()

    def register_error(self) -> None:
        self._events.append(datetime.utcnow())

    def recent_errors(self) -> int:
        now = datetime.utcnow()
        while self._events and now - self._events[0] > self.window:
            self._events.popleft()
        return len(self._events)


@dataclass
class KillSwitchState:
    triggered: bool = False

    def update(self, active: bool) -> None:
        if active:
            self.triggered = True


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IA BOT TRADING runtime controller")
    parser.add_argument("--mode", choices=[m.value for m in ExecutionMode], default=ExecutionMode.PAPER.value)
    parser.add_argument("--loop", action="store_true", help="Mantiene el bot en ejecución continua.")
    parser.add_argument("--check", action="store_true", help="Ejecuta verificaciones rápidas y termina.")
    parser.add_argument("--interval", type=float, default=15.0, help="Segundos entre ciclos de trading.")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Número máximo de iteraciones (0 = sin límite). Útil para pruebas.",
    )
    return parser.parse_args(argv)


def _gather_engine_metrics(engine: ExecutionEngine) -> Tuple[dict, float]:
    trades = engine.get_trade_history()
    if trades.empty:
        return {"pnl_acumulado": 0.0, "drawdown_relativo": 0.0}, 0.0

    trades = trades.sort_values("timestamp")
    if "pnl" not in trades.columns:
        trades["pnl"] = 0.0
    trades["pnl"] = trades["pnl"].fillna(0.0)
    cumulative_pnl = float(trades["pnl"].sum())
    last_pnl = float(trades["pnl"].iloc[-1])

    pnl_series = trades["pnl"].cumsum()
    equity_curve = BASE_CAPITAL + pnl_series
    running_max = equity_curve.cummax()
    drawdown_series = (equity_curve / running_max) - 1.0
    drawdown_value = float(drawdown_series.min()) if not drawdown_series.empty else 0.0

    engine_state = {
        "pnl_acumulado": cumulative_pnl / BASE_CAPITAL if BASE_CAPITAL else 0.0,
        "drawdown_relativo": drawdown_value,
    }
    return engine_state, last_pnl


def _perform_trading_cycle(engine: ExecutionEngine, iteration: int, dry_run: bool = False) -> float:
    if dry_run:
        price = engine.get_current_price()
        logger.info("Ciclo check-only | precio=%s", price)
        return 0.0

    if engine.mode == ExecutionMode.TESTNET:
        price = engine.get_current_price()
        logger.info("Modo TESTNET - verificación precio actual %s", price)
        return 0.0

    side = OrderSide.BUY if iteration % 2 == 0 else OrderSide.SELL
    result = engine.execute_market_order(side, quantity=0.001)
    pnl = float(result.get("pnl", 0.0)) if result else 0.0
    logger.info("Trade %s ejecutado | side=%s pnl=%.6f", iteration, side.value, pnl)
    return pnl


def _evaluate_kill_switch(
    engine: ExecutionEngine, error_tracker: ErrorTracker, state: KillSwitchState
) -> bool:
    engine_state, last_pnl = _gather_engine_metrics(engine)
    errors = error_tracker.recent_errors()
    triggered = check_kill_conditions(engine_state, last_pnl, errors)
    state.update(triggered)
    return triggered


def _scheduled_kill_switch_check(
    engine: ExecutionEngine, error_tracker: ErrorTracker, state: KillSwitchState
) -> bool:
    logger.debug("Scheduler: verificación de kill switch")
    return _evaluate_kill_switch(engine, error_tracker, state)


def _scheduled_watchdog_check(watchdog: Watchdog) -> bool:
    logger.debug("Scheduler: verificación watchdog")
    return watchdog.check_and_restart()


def run_bot(args: argparse.Namespace) -> None:
    mode = ExecutionMode(args.mode)
    engine = ExecutionEngine(mode=mode, initial_balance=BASE_CAPITAL)
    watchdog = Watchdog()
    error_tracker = ErrorTracker()
    kill_state = KillSwitchState()

    scheduler_service = SchedulerService(
        heartbeat_fn=watchdog.record_heartbeat,
        kill_switch_fn=lambda: _scheduled_kill_switch_check(engine, error_tracker, kill_state),
        watchdog_check_fn=lambda: _scheduled_watchdog_check(watchdog),
        daily_report_fn=lambda: generate_daily_report(base_capital=BASE_CAPITAL),
    )
    scheduler_service.start()

    iteration = 0
    max_iters = args.max_iters if args.max_iters > 0 else None
    loop_forever = args.loop or bool(max_iters is None and not args.check)

    while True:
        iteration += 1

        watchdog.record_heartbeat()
        scheduler_service.run_pending()

        if _evaluate_kill_switch(engine, error_tracker, kill_state):
            logger.error("Kill switch activado. Deteniendo ejecución principal.")
            break

        try:
            dry_run = args.check
            _perform_trading_cycle(engine, iteration, dry_run=dry_run)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Error en ciclo de trading: %s", exc)
            error_tracker.register_error()
        finally:
            if args.check:
                logger.info("Modo check completado. Saliendo.")
                break

        if max_iters and iteration >= max_iters:
            logger.info("Máximo de iteraciones alcanzado (%s).", max_iters)
            break

        if not loop_forever:
            logger.info("Ejecución única completada. Saliendo.")
            break

        time.sleep(max(args.interval, 1.0))


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logger.info(
        "IA BOT TRADING - Inicialización | modo=%s loop=%s check=%s interval=%.1fs",
        args.mode,
        args.loop,
        args.check,
        args.interval,
    )
    run_bot(args)


if __name__ == "__main__":  # pragma: no cover
    main()
