"""
Profitability Analyzer - Análisis completo de rentabilidad de estrategias.

Este módulo calcula métricas financieras avanzadas para evaluar el desempeño
de estrategias de trading: ROI, Profit Factor, Sharpe, Sortino, Drawdown, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class ProfitabilityAnalyzer:
    """Analizador de rentabilidad de estrategias de trading."""

    def __init__(
        self,
        trades_path: str = "reports/backtest_trades.parquet",
        metrics_path: str = "reports/backtest_metrics.json",
        initial_capital: float = 10000.0,
    ):
        """
        Inicializa el analizador de rentabilidad.

        Args:
            trades_path: Ruta al archivo de trades
            metrics_path: Ruta al archivo de métricas
            initial_capital: Capital inicial para cálculos
        """
        self.trades_path = Path(trades_path)
        self.metrics_path = Path(metrics_path)
        self.initial_capital = initial_capital

        logger.info("📊 ProfitabilityAnalyzer inicializado")

    def calculate_roi(self, trades_df: pd.DataFrame) -> float:
        """
        Calcula el Return on Investment (ROI).

        Args:
            trades_df: DataFrame con trades

        Returns:
            ROI como porcentaje
        """
        if "return" not in trades_df.columns or len(trades_df) == 0:
            return 0.0

        total_return = trades_df["return"].sum()
        roi = (total_return / self.initial_capital) * 100

        return float(roi)

    def calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """
        Calcula el Profit Factor (ganancia total / pérdida total).

        Args:
            trades_df: DataFrame con trades

        Returns:
            Profit Factor
        """
        if "return" not in trades_df.columns or len(trades_df) == 0:
            return 0.0

        returns = trades_df["return"]
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return float(gross_profit) if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        return float(profit_factor)

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula el Sharpe Ratio.

        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anualizada

        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0

        # Convertir tasa anual a tasa por periodo
        periods_per_year = 252  # días de trading
        risk_free_period = risk_free_rate / periods_per_year

        excess_returns = returns - risk_free_period
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()

        if std_excess == 0 or np.isnan(std_excess):
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
        return float(sharpe)

    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula el Sortino Ratio (similar a Sharpe pero solo con downside risk).

        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo

        Returns:
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0

        periods_per_year = 252
        risk_free_period = risk_free_rate / periods_per_year

        excess_returns = returns - risk_free_period
        mean_excess = excess_returns.mean()

        # Calcular solo desviación de retornos negativos
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
        return float(sortino)

    def calculate_drawdown(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calcula el Maximum Drawdown y métricas relacionadas.

        Args:
            equity_curve: Serie de equity acumulado

        Returns:
            Diccionario con métricas de drawdown
        """
        if len(equity_curve) == 0:
            return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "recovery_time": 0}

        # Calcular running maximum
        running_max = equity_curve.expanding().max()

        # Calcular drawdown
        drawdown = equity_curve - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max()) * 100 if running_max.max() > 0 else 0.0

        # Tiempo de recuperación (aproximado)
        if max_drawdown < 0:
            dd_idx = drawdown.idxmin()
            recovery_idx = equity_curve[dd_idx:][equity_curve >= running_max[dd_idx]].first_valid_index()
            recovery_time = (recovery_idx - dd_idx) if recovery_idx is not None else len(equity_curve) - dd_idx
        else:
            recovery_time = 0

        return {
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown_pct),
            "recovery_time": int(recovery_time),
        }

    def calculate_stability_score(self, returns: pd.Series) -> float:
        """
        Calcula un score de estabilidad basado en consistencia de retornos.

        Args:
            returns: Serie de retornos

        Returns:
            Stability score (0-1)
        """
        if len(returns) == 0:
            return 0.0

        # Porcentaje de trades positivos
        win_rate = (returns > 0).sum() / len(returns)

        # Consistencia (inverso del coeficiente de variación)
        mean_return = returns.mean()
        std_return = returns.std()

        if mean_return == 0 or std_return == 0:
            consistency = 0.0
        else:
            cv = abs(std_return / mean_return)
            consistency = 1 / (1 + cv)

        # Score combinado
        stability = (win_rate * 0.6 + consistency * 0.4)
        return float(np.clip(stability, 0, 1))

    def generate_profitability_report(self) -> Dict:
        """
        Genera un reporte completo de rentabilidad.

        Returns:
            Diccionario con todas las métricas
        """
        logger.info("🚀 Generando reporte de rentabilidad...")

        # Cargar datos
        if not self.trades_path.exists():
            logger.warning(f"⚠️ No se encontraron trades en {self.trades_path}")
            return self._empty_report()

        trades_df = pd.read_parquet(self.trades_path)

        if len(trades_df) == 0:
            logger.warning("⚠️ DataFrame de trades vacío")
            return self._empty_report()

        # Calcular equity curve
        if "return" in trades_df.columns:
            cumulative_returns = trades_df["return"].cumsum()
            equity_curve = self.initial_capital + (cumulative_returns * self.initial_capital)
        else:
            equity_curve = pd.Series([self.initial_capital] * len(trades_df))

        # Calcular métricas
        roi = self.calculate_roi(trades_df)
        profit_factor = self.calculate_profit_factor(trades_df)

        returns = trades_df["return"] if "return" in trades_df.columns else pd.Series([0])
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)

        drawdown_metrics = self.calculate_drawdown(equity_curve)
        stability = self.calculate_stability_score(returns)

        # Métricas adicionales
        num_trades = len(trades_df)
        win_trades = (returns > 0).sum()
        lose_trades = (returns < 0).sum()
        win_rate = (win_trades / num_trades) * 100 if num_trades > 0 else 0

        avg_win = returns[returns > 0].mean() if win_trades > 0 else 0
        avg_loss = returns[returns < 0].mean() if lose_trades > 0 else 0

        # Construir reporte
        report = {
            "summary": {
                "roi_pct": roi,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "stability_score": stability,
            },
            "drawdown": drawdown_metrics,
            "trade_stats": {
                "total_trades": num_trades,
                "winning_trades": int(win_trades),
                "losing_trades": int(lose_trades),
                "win_rate_pct": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
            },
            "capital": {
                "initial": self.initial_capital,
                "final": float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else self.initial_capital,
                "peak": float(equity_curve.max()) if len(equity_curve) > 0 else self.initial_capital,
            },
        }

        # Guardar reporte
        report_dir = Path(self.trades_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "profitability_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"✅ Reporte guardado en {report_path}")

        # Generar ranking de estrategia
        self._generate_strategy_ranking(report)

        # Log resumen
        logger.info("📈 Resumen de Rentabilidad:")
        logger.info(f"  ROI: {roi:.2f}%")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Sortino Ratio: {sortino:.2f}")
        logger.info(f"  Max Drawdown: {drawdown_metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Stability Score: {stability:.2f}")

        return report

    def _generate_strategy_ranking(self, report: Dict) -> None:
        """Genera un archivo CSV con ranking de estrategia."""
        ranking_data = {
            "Metric": [
                "ROI %",
                "Profit Factor",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Win Rate %",
                "Stability Score",
                "Max Drawdown %",
            ],
            "Value": [
                report["summary"]["roi_pct"],
                report["summary"]["profit_factor"],
                report["summary"]["sharpe_ratio"],
                report["summary"]["sortino_ratio"],
                report["trade_stats"]["win_rate_pct"],
                report["summary"]["stability_score"],
                report["drawdown"]["max_drawdown_pct"],
            ],
        }

        df = pd.DataFrame(ranking_data)
        
        # Usar el mismo directorio que el archivo de reportes
        ranking_dir = Path(self.trades_path).parent
        ranking_path = ranking_dir / "strategy_ranking.csv"
        df.to_csv(ranking_path, index=False)

        logger.info(f"✅ Ranking guardado en {ranking_path}")

    def _empty_report(self) -> Dict:
        """Retorna un reporte vacío."""
        return {
            "summary": {
                "roi_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "stability_score": 0.0,
            },
            "drawdown": {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "recovery_time": 0,
            },
            "trade_stats": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            },
            "capital": {
                "initial": self.initial_capital,
                "final": self.initial_capital,
                "peak": self.initial_capital,
            },
        }


if __name__ == "__main__":
    analyzer = ProfitabilityAnalyzer()
    report = analyzer.generate_profitability_report()

    print("\n" + "=" * 60)
    print("📊 REPORTE DE RENTABILIDAD")
    print("=" * 60)
    print(json.dumps(report, indent=2))
