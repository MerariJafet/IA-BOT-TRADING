"""
Dashboard interactivo de terminal para visualizar progreso de tests,
rentabilidad, métricas en tiempo real y tiempo estimado.

Usa la librería 'rich' para crear una interfaz interactiva en terminal.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

LIVE_TRADES_PATH = DATA_DIR / "live_trades.parquet"
PORTFOLIO_STATE_PATH = DATA_DIR / "portfolio_state.json"
PROFITABILITY_REPORT_PATH = REPORTS_DIR / "profitability_report.json"
MONITORING_LOG_PATH = DATA_DIR / "monitoring_log.json"
ALERTS_PATH = DATA_DIR / "alerts.json"
RETRAIN_HISTORY_PATH = DATA_DIR / "retrain_history.json"


class TerminalDashboard:
    """
    Dashboard interactivo de terminal que muestra:
    - Progreso de tests/backtest
    - Rentabilidad actual (ROI, Sharpe, PnL)
    - Métricas de riesgo (VaR, Beta, Drawdown)
    - Alertas activas
    - Historial de reentrenamientos
    - Tiempo estimado hasta completar evaluación
    """

    def __init__(
        self,
        refresh_interval: float = 2.0,
        test_total_steps: int = 100,
        backtest_days: int = 30,
    ):
        """
        Args:
            refresh_interval: Intervalo de actualización en segundos
            test_total_steps: Total de pasos de prueba (para barra de progreso)
            backtest_days: Días de backtesting (para calcular tiempo restante)
        """
        self.console = Console()
        self.refresh_interval = refresh_interval
        self.test_total_steps = test_total_steps
        self.backtest_days = backtest_days

        self.start_time = datetime.now()
        self.current_step = 0

    def _load_portfolio_state(self) -> Optional[Dict]:
        """Carga el estado del portfolio."""
        if not PORTFOLIO_STATE_PATH.exists():
            return None

        try:
            with open(PORTFOLIO_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_profitability_report(self) -> Optional[Dict]:
        """Carga el reporte de rentabilidad."""
        if not PROFITABILITY_REPORT_PATH.exists():
            return None

        try:
            with open(PROFITABILITY_REPORT_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_monitoring_log(self) -> Optional[Dict]:
        """Carga el log de monitoreo."""
        if not MONITORING_LOG_PATH.exists():
            return None

        try:
            with open(MONITORING_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_alerts(self) -> List[Dict]:
        """Carga las alertas activas."""
        if not ALERTS_PATH.exists():
            return []

        try:
            with open(ALERTS_PATH, "r") as f:
                data = json.load(f)
                return data.get("alerts", [])
        except Exception:
            return []

    def _load_retrain_history(self) -> List[Dict]:
        """Carga el historial de reentrenamientos."""
        if not RETRAIN_HISTORY_PATH.exists():
            return []

        try:
            with open(RETRAIN_HISTORY_PATH, "r") as f:
                data = json.load(f)
                return data.get("retrains", [])
        except Exception:
            return []

    def _create_header(self) -> Panel:
        """Crea el header del dashboard."""
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split(".")[0]

        # Calcular tiempo restante estimado
        if self.current_step > 0:
            time_per_step = elapsed.total_seconds() / self.current_step
            remaining_steps = max(0, self.test_total_steps - self.current_step)
            remaining_time = timedelta(seconds=time_per_step * remaining_steps)
            remaining_str = str(remaining_time).split(".")[0]
        else:
            remaining_str = "Calculando..."

        header_text = Text()
        header_text.append("🤖 IA BOT TRADING - DASHBOARD INTERACTIVO\n", style="bold cyan")
        header_text.append(f"Tiempo transcurrido: {elapsed_str} | ", style="green")
        header_text.append(f"Tiempo restante estimado: {remaining_str}\n", style="yellow")
        header_text.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")

        return Panel(header_text, style="bold blue", border_style="blue")

    def _create_progress_panel(self) -> Panel:
        """Crea el panel de progreso de tests/backtest."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        # Task principal: progreso de backtest
        progress.add_task(
            f"[cyan]Backtest ({self.backtest_days} días)",
            total=self.test_total_steps,
            completed=self.current_step,
        )

        return Panel(progress, title="📊 Progreso de Evaluación", border_style="green")

    def _create_profitability_panel(self) -> Panel:
        """Crea el panel de rentabilidad."""
        report = self._load_profitability_report()

        table = Table(title="💰 Rentabilidad Actual", show_header=True, header_style="bold magenta")
        table.add_column("Métrica", style="cyan", width=25)
        table.add_column("Valor", justify="right", style="green")

        if report and "summary" in report:
            summary = report["summary"]
            roi = summary.get("roi_pct", 0.0)
            sharpe = summary.get("sharpe_ratio", 0.0)
            profit_factor = summary.get("profit_factor", 0.0)
            win_rate = summary.get("win_rate", 0.0)
            max_dd = summary.get("max_drawdown_pct", 0.0)

            # Colorear según performance
            roi_color = "green" if roi > 0 else "red"
            sharpe_color = "green" if sharpe > 0.5 else "yellow" if sharpe > 0 else "red"

            table.add_row("ROI (%)", f"[{roi_color}]{roi:.2f}%[/{roi_color}]")
            table.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]")
            table.add_row("Profit Factor", f"{profit_factor:.2f}")
            table.add_row("Win Rate (%)", f"{win_rate:.1f}%")
            table.add_row("Max Drawdown (%)", f"[red]{max_dd:.2f}%[/red]")
        else:
            table.add_row("Estado", "[yellow]Sin datos disponibles[/yellow]")

        return Panel(table, border_style="magenta")

    def _create_portfolio_panel(self) -> Panel:
        """Crea el panel de estado del portfolio."""
        state = self._load_portfolio_state()

        table = Table(title="📈 Estado del Portfolio", show_header=True, header_style="bold blue")
        table.add_column("Métrica", style="cyan", width=25)
        table.add_column("Valor", justify="right", style="yellow")

        if state:
            total_value = state.get("total_value", 0.0)
            total_pnl = state.get("total_pnl", 0.0)
            num_positions = len(state.get("positions", []))
            num_strategies = len(state.get("strategies", []))

            pnl_color = "green" if total_pnl > 0 else "red"

            table.add_row("Valor Total", f"${total_value:,.2f}")
            table.add_row("PnL Total", f"[{pnl_color}]${total_pnl:,.2f}[/{pnl_color}]")
            table.add_row("Posiciones Abiertas", f"{num_positions}")
            table.add_row("Estrategias Activas", f"{num_strategies}")

            # VaR y Beta si existen
            if "risk_metrics" in state:
                var_95 = state["risk_metrics"].get("var_95", 0.0)
                beta = state["risk_metrics"].get("beta", 0.0)
                table.add_row("VaR (95%)", f"[red]${var_95:,.2f}[/red]")
                table.add_row("Beta", f"{beta:.2f}")
        else:
            table.add_row("Estado", "[yellow]Sin datos disponibles[/yellow]")

        return Panel(table, border_style="blue")

    def _create_alerts_panel(self) -> Panel:
        """Crea el panel de alertas activas."""
        alerts = self._load_alerts()

        table = Table(title="🚨 Alertas Activas", show_header=True, header_style="bold red")
        table.add_column("Tipo", style="cyan", width=20)
        table.add_column("Mensaje", style="white", width=40)
        table.add_column("Severidad", justify="center", width=10)

        if alerts:
            # Mostrar solo las últimas 5 alertas
            for alert in alerts[-5:]:
                severity = alert.get("severity", "INFO")
                severity_color = {
                    "INFO": "blue",
                    "WARNING": "yellow",
                    "CRITICAL": "red",
                }.get(severity, "white")

                table.add_row(
                    alert.get("type", "UNKNOWN"),
                    alert.get("message", "")[:40],
                    f"[{severity_color}]{severity}[/{severity_color}]",
                )
        else:
            table.add_row("Sin alertas", "Sistema operando normalmente", "[green]OK[/green]")

        return Panel(table, border_style="red")

    def _create_retrain_panel(self) -> Panel:
        """Crea el panel de historial de reentrenamientos."""
        retrains = self._load_retrain_history()

        table = Table(
            title="🔄 Historial de Reentrenamientos",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("Fecha", style="cyan", width=20)
        table.add_column("Razón", style="white", width=30)
        table.add_column("Estado", justify="center", width=10)

        if retrains:
            # Mostrar solo los últimos 3 reentrenamientos
            for retrain in retrains[-3:]:
                timestamp = retrain.get("timestamp", "")
                reasons = retrain.get("reasons", [])
                success = retrain.get("success", False)

                status_color = "green" if success else "red"
                status_text = "✅ OK" if success else "❌ FAIL"

                table.add_row(
                    timestamp[:19],
                    ", ".join(reasons)[:30],
                    f"[{status_color}]{status_text}[/{status_color}]",
                )
        else:
            table.add_row("Sin reentrenamientos", "Sistema estable", "[green]OK[/green]")

        return Panel(table, border_style="yellow")

    def _create_layout(self) -> Layout:
        """Crea el layout del dashboard."""
        layout = Layout()

        # Estructura principal
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="progress", size=5),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        # Dividir main en dos columnas
        layout["main"].split_row(
            Layout(name="left_column"),
            Layout(name="right_column"),
        )

        # Dividir columnas en secciones
        layout["left_column"].split_column(
            Layout(name="profitability"),
            Layout(name="portfolio"),
        )

        layout["right_column"].split_column(
            Layout(name="alerts"),
            Layout(name="retrain"),
        )

        # Footer
        footer_text = Text()
        footer_text.append("Presiona ", style="dim")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" para salir | Actualización automática cada ", style="dim")
        footer_text.append(f"{self.refresh_interval}s", style="bold yellow")

        layout["footer"].update(Panel(footer_text, style="dim", border_style="dim"))

        return layout

    def _update_layout(self, layout: Layout):
        """Actualiza el contenido del layout."""
        layout["header"].update(self._create_header())
        layout["progress"].update(self._create_progress_panel())
        layout["profitability"].update(self._create_profitability_panel())
        layout["portfolio"].update(self._create_portfolio_panel())
        layout["alerts"].update(self._create_alerts_panel())
        layout["retrain"].update(self._create_retrain_panel())

    def run(self, duration_seconds: Optional[int] = None):
        """
        Ejecuta el dashboard interactivo.

        Args:
            duration_seconds: Duración máxima en segundos (None = infinito)
        """
        layout = self._create_layout()

        try:
            with Live(layout, console=self.console, refresh_per_second=1) as live:
                start = time.time()

                while True:
                    self._update_layout(layout)

                    # Incrementar progreso (simulado)
                    self.current_step = min(
                        self.test_total_steps,
                        int((time.time() - start) / self.refresh_interval * 5),
                    )

                    live.update(layout)
                    time.sleep(self.refresh_interval)

                    # Salir si se alcanza la duración
                    if duration_seconds and (time.time() - start) >= duration_seconds:
                        break

                    # Salir si llegamos al 100%
                    if self.current_step >= self.test_total_steps:
                        break

        except KeyboardInterrupt:
            self.console.print("\n[bold red]Dashboard detenido por el usuario.[/bold red]")


def main():
    """Función principal para ejecutar el dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard interactivo de terminal")
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Intervalo de actualización en segundos (default: 2.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Total de pasos de prueba (default: 100)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Días de backtesting (default: 30)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duración máxima en segundos (default: infinito)",
    )

    args = parser.parse_args()

    dashboard = TerminalDashboard(
        refresh_interval=args.refresh,
        test_total_steps=args.steps,
        backtest_days=args.days,
    )

    dashboard.run(duration_seconds=args.duration)


if __name__ == "__main__":
    main()
