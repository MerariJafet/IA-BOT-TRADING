"""Daily report generator for IA BOT TRADING."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.core.binance_market_data import BinanceMarketData
from src.core.logger import get_logger

DEFAULT_TRADES_PATH = Path("data/live_trades.parquet")
REPORTS_DIR = Path("reports")
DAILY_REPORT_LOG = Path("logs/daily_report.log")


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


LOGGER = get_logger("DailyReport")
_ensure_directory(DAILY_REPORT_LOG)
_attach_file_handler(LOGGER, DAILY_REPORT_LOG)


@dataclass
class DailyMetrics:
	timestamp: str
	date: str
	roi_daily: float
	roi_cumulative: float
	trades_total: int
	win_rate: float
	max_drawdown: float
	btc_roi: float

	def to_dict(self) -> Dict[str, Any]:
		return {
			"timestamp": self.timestamp,
			"date": self.date,
			"roi_daily": self.roi_daily,
			"roi_cumulative": self.roi_cumulative,
			"total_trades": self.trades_total,
			"winrate": self.win_rate,
			"max_drawdown": self.max_drawdown,
			"btc_roi": self.btc_roi,
		}


def _load_trades(trades_path: Path) -> pd.DataFrame:
	if not trades_path.exists() or trades_path.stat().st_size == 0:
		return pd.DataFrame(columns=["timestamp", "pnl", "status"])

	df = pd.read_parquet(trades_path)
	if "timestamp" in df.columns:
		df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
	return df


def _filter_day_trades(trades: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
	if trades.empty:
		return trades
	mask = (trades["timestamp"] >= start) & (trades["timestamp"] <= end)
	return trades.loc[mask].reset_index(drop=True)


def _max_drawdown(pnl_series: pd.Series, base_capital: float) -> float:
	if pnl_series.empty:
		return 0.0

	equity_curve = base_capital + pnl_series.cumsum()
	running_max = equity_curve.cummax()
	drawdowns = (equity_curve / running_max) - 1.0
	return float(drawdowns.min()) if not drawdowns.empty else 0.0


def _btc_roi(client: BinanceMarketData, start: datetime, end: datetime, symbol: str = "BTCUSDT") -> float:
	try:
		df = client.fetch_ohlcv(symbol=symbol, start=start, end=end + timedelta(days=1), interval="1h")
		if df.empty:
			return 0.0
		first_price = df.iloc[0]["close"]
		last_price = df.iloc[-1]["close"]
		if first_price == 0:
			return 0.0
		return float((last_price - first_price) / first_price)
	except Exception:  # pragma: no cover - network errors handled gracefully
		return 0.0


def generate_daily_report(
	report_date: Optional[datetime] = None,
	trades_df: Optional[pd.DataFrame] = None,
	trades_path: Path = DEFAULT_TRADES_PATH,
	base_capital: float = 10000.0,
	market_data_client: Optional[BinanceMarketData] = None,
) -> DailyMetrics:
	"""Genera el reporte diario y lo almacena en ``reports/``."""

	report_date = report_date or datetime.now(tz=timezone.utc)
	start_of_day = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
	end_of_day = start_of_day + timedelta(days=1) - timedelta(seconds=1)

	trades_df = trades_df if trades_df is not None else _load_trades(trades_path)

	day_trades = _filter_day_trades(trades_df, start_of_day, end_of_day)

	daily_pnl = float(day_trades["pnl"].sum()) if not day_trades.empty else 0.0
	cumulative_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0

	roi_daily = daily_pnl / base_capital if base_capital else 0.0
	roi_cumulative = cumulative_pnl / base_capital if base_capital else 0.0

	trades_total = int(len(day_trades))
	wins = int((day_trades["pnl"] > 0).sum()) if not day_trades.empty else 0
	win_rate = (wins / trades_total) if trades_total else 0.0

	max_drawdown = _max_drawdown(day_trades["pnl"], base_capital)

	market_data_client = market_data_client or BinanceMarketData()
	btc_roi = _btc_roi(market_data_client, start_of_day, end_of_day)

	metrics = DailyMetrics(
		timestamp=datetime.now(tz=timezone.utc).isoformat(),
		date=start_of_day.strftime("%Y-%m-%d"),
		roi_daily=roi_daily,
		roi_cumulative=roi_cumulative,
		trades_total=trades_total,
		win_rate=win_rate,
		max_drawdown=max_drawdown,
		btc_roi=btc_roi,
	)

	_write_report(metrics)
	LOGGER.info("Reporte diario generado | fecha=%s trades=%s roi=%.4f btc_roi=%.4f", metrics.date, trades_total, roi_daily, btc_roi)

	return metrics


def _write_report(metrics: DailyMetrics) -> None:
	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	report_filename = REPORTS_DIR / f"daily_report_{metrics.date.replace('-', '')}.json"
	with report_filename.open("w", encoding="utf-8") as handle:
		json.dump(metrics.to_dict(), handle, indent=2)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generador de reportes diarios del bot IA.")
	parser.add_argument("--date", type=str, default=None, help="Fecha en formato YYYY-MM-DD (UTC).")
	parser.add_argument("--base-capital", type=float, default=10000.0, help="Capital base para cálculo de ROI.")
	parser.add_argument("--test", action="store_true", help="Ejecuta una generación de prueba con datos simulados.")
	return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
	args = _parse_args(argv)

	if args.test:
		LOGGER.info("Modo test activado: generando reporte con datos simulados.")
		now = datetime.now(tz=timezone.utc)
		trades = pd.DataFrame(
			{
				"timestamp": pd.date_range(now - timedelta(hours=6), periods=6, freq="h", tz="UTC"),
				"pnl": [10.0, -5.0, 7.5, -2.0, 5.5, 3.0],
				"status": ["FILLED"] * 6,
			}
		)
		generate_daily_report(report_date=now, trades_df=trades, base_capital=args.base_capital)
		return

	report_date = None
	if args.date:
		report_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

	generate_daily_report(report_date=report_date, base_capital=args.base_capital)


if __name__ == "__main__":  # pragma: no cover
	main()


__all__ = ["generate_daily_report", "DailyMetrics", "main"]
