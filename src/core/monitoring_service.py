"""
Monitoring Service - Monitoreo continuo de performance y riesgo del portafolio.

Este módulo observa métricas clave en tiempo real, detecta drift de performance,
y genera alertas cuando se requiere intervención o reentrenamiento.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class MonitoringService:
    """Servicio de monitoreo continuo de portafolio."""

    LIVE_TRADES_PATH = "data/live_trades.parquet"
    PORTFOLIO_STATE_PATH = "reports/portfolio_state.json"
    PERFORMANCE_REPORT_PATH = "reports/performance_report.json"
    MONITORING_LOG_PATH = "data/monitoring_log.json"
    ALERTS_PATH = "data/alerts.json"

    def __init__(
        self,
        sharpe_drift_threshold: float = 0.2,
        roi_threshold: float = 0.0,
        sharpe_threshold: float = 0.3,
        var_limit: float = -10000.0,
    ):
        """
        Inicializa el servicio de monitoreo.

        Args:
            sharpe_drift_threshold: % máximo de caída en Sharpe antes de alertar
            roi_threshold: ROI mínimo aceptable
            sharpe_threshold: Sharpe ratio mínimo aceptable
            var_limit: VaR máximo aceptable (valor negativo)
        """
        self.sharpe_drift_threshold = sharpe_drift_threshold
        self.roi_threshold = roi_threshold
        self.sharpe_threshold = sharpe_threshold
        self.var_limit = var_limit

        # Estado histórico
        self.historical_metrics: List[Dict] = []
        self.baseline_sharpe: Optional[float] = None

        logger.info("📡 MonitoringService inicializado")
        logger.info(f"  Sharpe drift threshold: {sharpe_drift_threshold:.1%}")
        logger.info(f"  ROI threshold: {roi_threshold:.1%}")
        logger.info(f"  Sharpe threshold: {sharpe_threshold}")

    def load_live_trades(self) -> pd.DataFrame:
        """
        Carga trades en vivo desde archivo.

        Returns:
            DataFrame con trades
        """
        trades_path = Path(self.LIVE_TRADES_PATH)

        if not trades_path.exists() or trades_path.stat().st_size == 0:
            logger.warning("⚠️ No hay trades para monitorear")
            return pd.DataFrame()

        return pd.read_parquet(trades_path)

    def load_portfolio_state(self) -> Optional[Dict]:
        """
        Carga estado actual del portafolio.

        Returns:
            Diccionario con estado o None
        """
        state_path = Path(self.PORTFOLIO_STATE_PATH)

        if not state_path.exists():
            logger.warning("⚠️ No hay estado de portafolio disponible")
            return None

        with open(state_path, "r") as f:
            return json.load(f)

    def calculate_current_metrics(
        self, trades_df: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """
        Calcula métricas actuales desde trades.

        Args:
            trades_df: DataFrame con trades

        Returns:
            Diccionario con métricas o None
        """
        if len(trades_df) == 0:
            return None

        # Filtrar solo trades completados
        filled_trades = trades_df[trades_df["status"] == "FILLED"]

        if len(filled_trades) == 0:
            return None

        # Calcular métricas básicas
        total_pnl = filled_trades["pnl"].sum() if "pnl" in filled_trades.columns else 0.0
        avg_pnl = filled_trades["pnl"].mean() if "pnl" in filled_trades.columns else 0.0

        # ROI (asumiendo capital inicial de 10000)
        initial_capital = 10000.0
        roi = (total_pnl / initial_capital) * 100

        # Win rate
        winning_trades = (filled_trades["pnl"] > 0).sum() if "pnl" in filled_trades.columns else 0
        total_trades = len(filled_trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

        # Sharpe ratio
        if "pnl" in filled_trades.columns and len(filled_trades) > 1:
            returns = filled_trades["pnl"] / initial_capital
            sharpe = self._calculate_sharpe(returns)
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = filled_trades[filled_trades["pnl"] > 0]["pnl"].sum() if "pnl" in filled_trades.columns else 0.0
        gross_loss = abs(filled_trades[filled_trades["pnl"] < 0]["pnl"].sum()) if "pnl" in filled_trades.columns else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_pnl": float(total_pnl),
            "avg_pnl": float(avg_pnl),
            "roi_pct": float(roi),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
            "profit_factor": float(profit_factor),
            "total_trades": int(total_trades),
        }

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sharpe ratio.

        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anualizada

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        periods_per_year = 252
        risk_free_period = risk_free_rate / periods_per_year

        excess_returns = returns - risk_free_period
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()

        if std_excess == 0 or np.isnan(std_excess):
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
        return float(sharpe)

    def detect_drift(self, current_metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Detecta drift en performance.

        Args:
            current_metrics: Métricas actuales

        Returns:
            Tupla (drift_detected, reasons)
        """
        drift_detected = False
        reasons = []

        current_sharpe = current_metrics.get("sharpe_ratio", 0.0)

        # Establecer baseline si es la primera vez
        if self.baseline_sharpe is None and current_sharpe != 0.0:
            self.baseline_sharpe = current_sharpe
            logger.info(f"📊 Baseline Sharpe establecido: {self.baseline_sharpe:.2f}")

        # Detectar caída en Sharpe
        if self.baseline_sharpe is not None and self.baseline_sharpe > 0:
            sharpe_drop = (self.baseline_sharpe - current_sharpe) / self.baseline_sharpe

            if sharpe_drop > self.sharpe_drift_threshold:
                drift_detected = True
                reasons.append(
                    f"Sharpe cayó {sharpe_drop:.1%} desde baseline ({self.baseline_sharpe:.2f} → {current_sharpe:.2f})"
                )

        # ROI bajo
        if current_metrics.get("roi_pct", 0.0) < self.roi_threshold:
            drift_detected = True
            reasons.append(f"ROI bajo: {current_metrics['roi_pct']:.2f}%")

        # Sharpe absoluto bajo
        if current_sharpe < self.sharpe_threshold:
            drift_detected = True
            reasons.append(f"Sharpe bajo: {current_sharpe:.2f} < {self.sharpe_threshold}")

        # Profit factor bajo
        if current_metrics.get("profit_factor", 0.0) < 1.0:
            drift_detected = True
            reasons.append(f"Profit factor < 1.0: {current_metrics['profit_factor']:.2f}")

        return drift_detected, reasons

    def generate_alert(self, alert_type: str, message: str, severity: str = "WARNING") -> None:
        """
        Genera una alerta.

        Args:
            alert_type: Tipo de alerta
            message: Mensaje de la alerta
            severity: Severidad (INFO, WARNING, CRITICAL)
        """
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "severity": severity,
            "message": message,
        }

        # Cargar alertas existentes
        alerts_path = Path(self.ALERTS_PATH)
        alerts_path.parent.mkdir(parents=True, exist_ok=True)

        if alerts_path.exists():
            with open(alerts_path, "r") as f:
                alerts_data = json.load(f)
        else:
            alerts_data = {"alerts": []}

        # Agregar nueva alerta
        alerts_data["alerts"].append(alert)

        # Mantener solo últimas 100 alertas
        alerts_data["alerts"] = alerts_data["alerts"][-100:]

        # Guardar
        with open(alerts_path, "w") as f:
            json.dump(alerts_data, f, indent=2)

        # Log
        if severity == "CRITICAL":
            logger.error(f"🚨 ALERTA CRÍTICA: {message}")
        elif severity == "WARNING":
            logger.warning(f"⚠️ ALERTA: {message}")
        else:
            logger.info(f"ℹ️ ALERTA: {message}")

    def log_monitoring_event(self, metrics: Dict, drift_detected: bool, reasons: List[str]) -> None:
        """
        Registra evento de monitoreo.

        Args:
            metrics: Métricas actuales
            drift_detected: Si se detectó drift
            reasons: Razones del drift
        """
        log_path = Path(self.MONITORING_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Cargar log existente
        if log_path.exists():
            with open(log_path, "r") as f:
                log_data = json.load(f)
        else:
            log_data = {"events": []}

        # Agregar evento
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "drift_detected": drift_detected,
            "drift_reasons": reasons,
        }

        log_data["events"].append(event)

        # Mantener solo últimos 1000 eventos
        log_data["events"] = log_data["events"][-1000:]

        # Guardar
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"📝 Evento de monitoreo registrado")

    def update_performance_report(self, metrics: Dict) -> None:
        """
        Actualiza reporte de performance diario.

        Args:
            metrics: Métricas actuales
        """
        report_path = Path(self.PERFORMANCE_REPORT_PATH)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Cargar reporte existente
        if report_path.exists():
            with open(report_path, "r") as f:
                report = json.load(f)
        else:
            report = {
                "daily_snapshots": [],
                "summary": {},
            }

        # Agregar snapshot diario
        snapshot = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }

        report["daily_snapshots"].append(snapshot)

        # Mantener solo últimos 30 días
        report["daily_snapshots"] = report["daily_snapshots"][-30:]

        # Actualizar resumen
        if len(report["daily_snapshots"]) > 0:
            recent_metrics = [s["metrics"] for s in report["daily_snapshots"]]

            report["summary"] = {
                "last_update": datetime.utcnow().isoformat(),
                "days_tracked": len(report["daily_snapshots"]),
                "avg_roi": float(np.mean([m["roi_pct"] for m in recent_metrics])),
                "avg_sharpe": float(np.mean([m["sharpe_ratio"] for m in recent_metrics])),
                "avg_win_rate": float(np.mean([m["win_rate"] for m in recent_metrics])),
                "total_trades": sum([m["total_trades"] for m in recent_metrics]),
            }

        # Guardar
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Performance report actualizado: {report_path}")

    def run_monitoring_cycle(self) -> Dict:
        """
        Ejecuta un ciclo completo de monitoreo.

        Returns:
            Diccionario con resultado del monitoreo
        """
        logger.info("🔄 Iniciando ciclo de monitoreo...")

        # Cargar trades
        trades = self.load_live_trades()

        if len(trades) == 0:
            msg = "No hay trades para monitorear"
            logger.warning(f"⚠️ {msg}")
            return {"status": "NO_DATA", "message": msg}

        # Calcular métricas
        metrics = self.calculate_current_metrics(trades)

        if metrics is None:
            msg = "No se pudieron calcular métricas"
            logger.warning(f"⚠️ {msg}")
            return {"status": "ERROR", "message": msg}

        logger.info("📊 Métricas actuales:")
        logger.info(f"  ROI: {metrics['roi_pct']:.2f}%")
        logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")

        # Detectar drift
        drift_detected, reasons = self.detect_drift(metrics)

        if drift_detected:
            logger.warning("⚠️ DRIFT DETECTADO:")
            for reason in reasons:
                logger.warning(f"  - {reason}")

            # Generar alertas
            self.generate_alert(
                "PERFORMANCE_DRIFT",
                f"Drift detectado: {'; '.join(reasons)}",
                severity="WARNING" if len(reasons) < 3 else "CRITICAL",
            )

        # Registrar evento
        self.log_monitoring_event(metrics, drift_detected, reasons)

        # Actualizar reporte de performance
        self.update_performance_report(metrics)

        result = {
            "status": "SUCCESS",
            "drift_detected": drift_detected,
            "drift_reasons": reasons,
            "metrics": metrics,
            "requires_retrain": drift_detected and len(reasons) >= 2,
        }

        logger.info(f"✅ Ciclo de monitoreo completado")

        return result

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """
        Obtiene alertas recientes.

        Args:
            limit: Número máximo de alertas

        Returns:
            Lista de alertas
        """
        alerts_path = Path(self.ALERTS_PATH)

        if not alerts_path.exists():
            return []

        with open(alerts_path, "r") as f:
            alerts_data = json.load(f)

        return alerts_data["alerts"][-limit:]

    def get_performance_summary(self) -> Dict:
        """
        Obtiene resumen de performance.

        Returns:
            Diccionario con resumen
        """
        report_path = Path(self.PERFORMANCE_REPORT_PATH)

        if not report_path.exists():
            return {}

        with open(report_path, "r") as f:
            report = json.load(f)

        return report.get("summary", {})


def run_monitoring_session() -> None:
    """Ejecuta sesión de monitoreo."""
    print("=" * 60)
    print("📡 MONITORING SERVICE - CONTINUOUS PERFORMANCE TRACKING")
    print("=" * 60)

    # Crear servicio de monitoreo
    monitor = MonitoringService(
        sharpe_drift_threshold=0.2,
        roi_threshold=0.0,
        sharpe_threshold=0.3,
    )

    # Ejecutar ciclo
    result = monitor.run_monitoring_cycle()

    print("\n" + "=" * 60)
    print("📊 RESULTADO DEL MONITOREO")
    print("=" * 60)
    print(f"Status: {result.get('status')}")

    if result.get("drift_detected"):
        print(f"\n⚠️ DRIFT DETECTADO:")
        for reason in result.get("drift_reasons", []):
            print(f"  - {reason}")

        print(f"\n🔄 Reentrenamiento requerido: {result.get('requires_retrain')}")

    if "metrics" in result:
        print(f"\n📈 Métricas:")
        metrics = result["metrics"]
        print(f"  ROI: {metrics['roi_pct']:.2f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

    # Mostrar alertas recientes
    recent_alerts = monitor.get_recent_alerts(limit=5)
    if recent_alerts:
        print(f"\n🚨 Últimas {len(recent_alerts)} alertas:")
        for alert in recent_alerts:
            print(f"  [{alert['severity']}] {alert['message']}")

    print("\n✅ Monitoreo completado. Revisar data/monitoring_log.json")


if __name__ == "__main__":
    run_monitoring_session()
