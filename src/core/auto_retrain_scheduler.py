"""
Auto-Retrain Scheduler - Programador de reentrenamiento automático.

Este módulo monitorea condiciones de performance y dispara reentrenamiento
automático cuando se detectan degradaciones sostenidas.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.live_retrainer import LiveRetrainer
from src.core.logger import get_logger
from src.core.monitoring_service import MonitoringService

logger = get_logger(__name__)


class AutoRetrainScheduler:
    """Programador de reentrenamiento automático."""

    SCHEDULER_STATE_PATH = "data/scheduler_state.json"
    RETRAIN_HISTORY_PATH = "data/retrain_history.json"

    def __init__(
        self,
        consecutive_sessions_threshold: int = 3,
        pnl_threshold: float = 0.0,
        sharpe_threshold: float = 0.3,
        enable_auto_retrain: bool = True,
    ):
        """
        Inicializa el scheduler.

        Args:
            consecutive_sessions_threshold: Sesiones consecutivas malas antes de reentrenar
            pnl_threshold: PnL mínimo aceptable
            sharpe_threshold: Sharpe mínimo aceptable
            enable_auto_retrain: Si el reentrenamiento automático está habilitado
        """
        self.consecutive_sessions_threshold = consecutive_sessions_threshold
        self.pnl_threshold = pnl_threshold
        self.sharpe_threshold = sharpe_threshold
        self.enable_auto_retrain = enable_auto_retrain

        # Servicios
        self.monitoring_service = MonitoringService()
        self.retrainer = LiveRetrainer(min_trades_threshold=5)

        # Estado
        self.bad_sessions_count = 0
        self.last_retrain_timestamp: Optional[str] = None

        # Cargar estado
        self._load_state()

        logger.info("⏰ AutoRetrainScheduler inicializado")
        logger.info(
            f"  Threshold de sesiones consecutivas: {consecutive_sessions_threshold}"
        )
        logger.info(f"  Auto-retrain habilitado: {enable_auto_retrain}")

    def _load_state(self) -> None:
        """Carga estado del scheduler."""
        state_path = Path(self.SCHEDULER_STATE_PATH)

        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)

            self.bad_sessions_count = state.get("bad_sessions_count", 0)
            self.last_retrain_timestamp = state.get("last_retrain_timestamp")

            logger.info(f"📂 Estado del scheduler cargado")
            logger.info(f"  Sesiones malas consecutivas: {self.bad_sessions_count}")

    def _save_state(self) -> None:
        """Guarda estado del scheduler."""
        state_path = Path(self.SCHEDULER_STATE_PATH)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "bad_sessions_count": self.bad_sessions_count,
            "last_retrain_timestamp": self.last_retrain_timestamp,
            "last_update": datetime.utcnow().isoformat(),
        }

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("💾 Estado del scheduler guardado")

    def check_retrain_conditions(
        self, monitoring_result: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Verifica si se deben cumplir condiciones para reentrenar.

        Args:
            monitoring_result: Resultado del monitoreo

        Returns:
            Tupla (should_retrain, reasons)
        """
        should_retrain = False
        reasons = []

        # Verificar si el monitoreo detectó drift
        if monitoring_result.get("drift_detected"):
            metrics = monitoring_result.get("metrics", {})

            # PnL bajo
            if metrics.get("roi_pct", 0.0) < self.pnl_threshold:
                self.bad_sessions_count += 1
                reasons.append(f"PnL bajo: {metrics['roi_pct']:.2f}%")

            # Sharpe bajo
            if metrics.get("sharpe_ratio", 0.0) < self.sharpe_threshold:
                if not reasons:  # Solo incrementar si no se incrementó por PnL
                    self.bad_sessions_count += 1
                reasons.append(
                    f"Sharpe bajo: {metrics['sharpe_ratio']:.2f} < {self.sharpe_threshold}"
                )

            # Profit factor bajo
            if metrics.get("profit_factor", 0.0) < 1.0:
                reasons.append(f"Profit factor < 1.0: {metrics['profit_factor']:.2f}")

        else:
            # Sesión buena, resetear contador
            if self.bad_sessions_count > 0:
                logger.info(
                    f"✅ Sesión buena detectada, reseteando contador (era {self.bad_sessions_count})"
                )
            self.bad_sessions_count = 0

        # Verificar threshold de sesiones consecutivas
        if self.bad_sessions_count >= self.consecutive_sessions_threshold:
            should_retrain = True
            reasons.append(
                f"Sesiones malas consecutivas: {self.bad_sessions_count}/{self.consecutive_sessions_threshold}"
            )

        return should_retrain, reasons

    def execute_retrain(self, reasons: List[str]) -> Tuple[bool, str]:
        """
        Ejecuta reentrenamiento.

        Args:
            reasons: Razones del reentrenamiento

        Returns:
            Tupla (success, message)
        """
        if not self.enable_auto_retrain:
            msg = "Auto-retrain deshabilitado"
            logger.warning(f"⚠️ {msg}")
            return False, msg

        logger.info("🔄 Ejecutando reentrenamiento automático...")
        logger.info(f"  Razones: {'; '.join(reasons)}")

        # Ejecutar reentrenamiento
        success, message = self.retrainer.retrain()

        if success:
            # Actualizar timestamp
            self.last_retrain_timestamp = datetime.utcnow().isoformat()

            # Resetear contador de sesiones malas
            self.bad_sessions_count = 0

            # Registrar en historial
            self._log_retrain_event(reasons, success, message)

            logger.info("✅ Reentrenamiento automático completado")

        else:
            logger.error(f"❌ Reentrenamiento falló: {message}")

        return success, message

    def _log_retrain_event(self, reasons: List[str], success: bool, message: str) -> None:
        """
        Registra evento de reentrenamiento.

        Args:
            reasons: Razones del reentrenamiento
            success: Si fue exitoso
            message: Mensaje del resultado
        """
        history_path = Path(self.RETRAIN_HISTORY_PATH)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Cargar historial
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {"retrains": []}

        # Agregar evento
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "reasons": reasons,
            "success": success,
            "message": message,
            "weights_after": self.retrainer.get_current_weights(),
        }

        history["retrains"].append(event)

        # Mantener solo últimos 50 eventos
        history["retrains"] = history["retrains"][-50:]

        # Guardar
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"📝 Evento de reentrenamiento registrado")

    def run_scheduler_cycle(self) -> Dict:
        """
        Ejecuta ciclo completo del scheduler.

        Returns:
            Diccionario con resultado
        """
        logger.info("⏰ Iniciando ciclo del scheduler...")

        # Ejecutar monitoreo
        monitoring_result = self.monitoring_service.run_monitoring_cycle()

        if monitoring_result.get("status") != "SUCCESS":
            msg = f"Monitoreo falló: {monitoring_result.get('message')}"
            logger.warning(f"⚠️ {msg}")
            return {"status": "MONITORING_FAILED", "message": msg}

        # Verificar condiciones de reentrenamiento
        should_retrain, reasons = self.check_retrain_conditions(monitoring_result)

        result = {
            "status": "SUCCESS",
            "monitoring_result": monitoring_result,
            "should_retrain": should_retrain,
            "retrain_reasons": reasons,
            "bad_sessions_count": self.bad_sessions_count,
        }

        if should_retrain:
            logger.warning("⚠️ CONDICIONES DE REENTRENAMIENTO CUMPLIDAS")
            for reason in reasons:
                logger.warning(f"  - {reason}")

            # Ejecutar reentrenamiento
            retrain_success, retrain_message = self.execute_retrain(reasons)

            result["retrain_executed"] = True
            result["retrain_success"] = retrain_success
            result["retrain_message"] = retrain_message

        else:
            logger.info(
                f"✅ No se requiere reentrenamiento (sesiones malas: {self.bad_sessions_count}/{self.consecutive_sessions_threshold})"
            )
            result["retrain_executed"] = False

        # Guardar estado
        self._save_state()

        logger.info("✅ Ciclo del scheduler completado")

        return result

    def get_retrain_history(self, limit: int = 10) -> List[Dict]:
        """
        Obtiene historial de reentrenamientos.

        Args:
            limit: Número máximo de eventos

        Returns:
            Lista de eventos
        """
        history_path = Path(self.RETRAIN_HISTORY_PATH)

        if not history_path.exists():
            return []

        with open(history_path, "r") as f:
            history = json.load(f)

        return history["retrains"][-limit:]

    def force_retrain(self, reason: str = "Manual trigger") -> Tuple[bool, str]:
        """
        Fuerza un reentrenamiento manual.

        Args:
            reason: Razón del reentrenamiento

        Returns:
            Tupla (success, message)
        """
        logger.info(f"🔧 Reentrenamiento manual forzado: {reason}")

        return self.execute_retrain([reason])


def run_scheduler_session() -> None:
    """Ejecuta sesión del scheduler."""
    print("=" * 60)
    print("⏰ AUTO-RETRAIN SCHEDULER - ADAPTIVE RETRAINING")
    print("=" * 60)

    # Crear scheduler
    scheduler = AutoRetrainScheduler(
        consecutive_sessions_threshold=3,
        pnl_threshold=0.0,
        sharpe_threshold=0.3,
        enable_auto_retrain=True,
    )

    # Ejecutar ciclo
    result = scheduler.run_scheduler_cycle()

    print("\n" + "=" * 60)
    print("📊 RESULTADO DEL SCHEDULER")
    print("=" * 60)
    print(f"Status: {result.get('status')}")
    print(f"Sesiones malas consecutivas: {result.get('bad_sessions_count')}")

    if result.get("should_retrain"):
        print(f"\n⚠️ REENTRENAMIENTO REQUERIDO:")
        for reason in result.get("retrain_reasons", []):
            print(f"  - {reason}")

        if result.get("retrain_executed"):
            print(f"\n✅ Reentrenamiento ejecutado: {result.get('retrain_success')}")
            print(f"   Mensaje: {result.get('retrain_message')}")

    # Mostrar historial reciente
    history = scheduler.get_retrain_history(limit=5)
    if history:
        print(f"\n📜 Últimos {len(history)} reentrenamientos:")
        for event in history:
            status = "✅" if event["success"] else "❌"
            print(f"  {status} {event['timestamp']}: {event['message']}")

    print("\n✅ Scheduler completado. Revisar data/scheduler_state.json")


if __name__ == "__main__":
    run_scheduler_session()
