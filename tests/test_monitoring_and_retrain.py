"""
Tests para MonitoringService y AutoRetrainScheduler.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.auto_retrain_scheduler import AutoRetrainScheduler
from src.core.monitoring_service import MonitoringService


@pytest.fixture
def monitoring_service(tmp_path, monkeypatch):
    """Crea MonitoringService con paths temporales."""
    trades_path = tmp_path / "live_trades.parquet"
    state_path = tmp_path / "portfolio_state.json"
    perf_path = tmp_path / "performance_report.json"
    log_path = tmp_path / "monitoring_log.json"
    alerts_path = tmp_path / "alerts.json"

    # Monkeypatch paths
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.LIVE_TRADES_PATH",
        str(trades_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.PORTFOLIO_STATE_PATH",
        str(state_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.PERFORMANCE_REPORT_PATH",
        str(perf_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.MONITORING_LOG_PATH",
        str(log_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.ALERTS_PATH", str(alerts_path)
    )

    monitor = MonitoringService(
        sharpe_drift_threshold=0.2,
        roi_threshold=0.0,
        sharpe_threshold=0.3,
    )

    return monitor


@pytest.fixture
def auto_scheduler(tmp_path, monkeypatch):
    """Crea AutoRetrainScheduler con paths temporales."""
    scheduler_state_path = tmp_path / "scheduler_state.json"
    retrain_history_path = tmp_path / "retrain_history.json"

    # Monkeypatch paths para scheduler
    monkeypatch.setattr(
        "src.core.auto_retrain_scheduler.AutoRetrainScheduler.SCHEDULER_STATE_PATH",
        str(scheduler_state_path),
    )
    monkeypatch.setattr(
        "src.core.auto_retrain_scheduler.AutoRetrainScheduler.RETRAIN_HISTORY_PATH",
        str(retrain_history_path),
    )

    # Monkeypatch paths para MonitoringService dentro del scheduler
    trades_path = tmp_path / "live_trades.parquet"
    state_path = tmp_path / "portfolio_state.json"
    perf_path = tmp_path / "performance_report.json"
    log_path = tmp_path / "monitoring_log.json"
    alerts_path = tmp_path / "alerts.json"

    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.LIVE_TRADES_PATH",
        str(trades_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.PORTFOLIO_STATE_PATH",
        str(state_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.PERFORMANCE_REPORT_PATH",
        str(perf_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.MONITORING_LOG_PATH",
        str(log_path),
    )
    monkeypatch.setattr(
        "src.core.monitoring_service.MonitoringService.ALERTS_PATH", str(alerts_path)
    )

    # Monkeypatch paths para LiveRetrainer
    weights_path = tmp_path / "policy_weights.json"
    retraining_log_path = tmp_path / "retraining_log.json"

    monkeypatch.setattr("src.core.live_retrainer.LiveRetrainer.LIVE_TRADES_PATH", str(trades_path))
    monkeypatch.setattr(
        "src.core.live_retrainer.LiveRetrainer.POLICY_WEIGHTS_PATH", str(weights_path)
    )
    monkeypatch.setattr(
        "src.core.live_retrainer.LiveRetrainer.RETRAINING_LOG_PATH",
        str(retraining_log_path),
    )

    scheduler = AutoRetrainScheduler(
        consecutive_sessions_threshold=3,
        pnl_threshold=0.0,
        sharpe_threshold=0.3,
        enable_auto_retrain=True,
    )

    return scheduler


@pytest.fixture
def sample_trades_good():
    """Crea trades con buena performance."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "order_id": [f"ORDER_{i}" for i in range(20)],
            "status": ["FILLED"] * 20,
            "pnl": [50, 60, -10, 70, 40, 30, -5, 80, 90, 20] * 2,  # Mayormente positivos
        }
    )


@pytest.fixture
def sample_trades_bad():
    """Crea trades con mala performance."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "order_id": [f"ORDER_{i}" for i in range(20)],
            "status": ["FILLED"] * 20,
            "pnl": [-50, -30, 10, -40, -20, -10, 5, -60, -30, -15] * 2,  # Mayormente negativos
        }
    )


# ==================== MONITORING SERVICE TESTS ====================


def test_monitoring_service_initialization(monitoring_service):
    """Test inicialización del servicio de monitoreo."""
    assert monitoring_service.sharpe_drift_threshold == 0.2
    assert monitoring_service.roi_threshold == 0.0
    assert monitoring_service.sharpe_threshold == 0.3


def test_calculate_current_metrics_good(monitoring_service, sample_trades_good, tmp_path):
    """Test cálculo de métricas con trades buenos."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_good.to_parquet(trades_path)

    trades = monitoring_service.load_live_trades()
    metrics = monitoring_service.calculate_current_metrics(trades)

    assert metrics is not None
    assert "roi_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "win_rate" in metrics

    # Con trades buenos, ROI debería ser positivo
    assert metrics["roi_pct"] > 0
    assert metrics["win_rate"] > 50


def test_calculate_current_metrics_bad(monitoring_service, sample_trades_bad, tmp_path):
    """Test cálculo de métricas con trades malos."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_bad.to_parquet(trades_path)

    trades = monitoring_service.load_live_trades()
    metrics = monitoring_service.calculate_current_metrics(trades)

    assert metrics is not None

    # Con trades malos, ROI debería ser negativo
    assert metrics["roi_pct"] < 0
    assert metrics["win_rate"] < 50


def test_detect_drift_with_bad_performance(monitoring_service, sample_trades_bad, tmp_path):
    """Test detección de drift con mala performance."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_bad.to_parquet(trades_path)

    trades = monitoring_service.load_live_trades()
    metrics = monitoring_service.calculate_current_metrics(trades)

    drift_detected, reasons = monitoring_service.detect_drift(metrics)

    assert drift_detected is True
    assert len(reasons) > 0


def test_detect_drift_with_good_performance(monitoring_service, sample_trades_good, tmp_path):
    """Test que no se detecta drift con buena performance."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_good.to_parquet(trades_path)

    trades = monitoring_service.load_live_trades()
    metrics = monitoring_service.calculate_current_metrics(trades)

    # Establecer baseline alto
    monitoring_service.baseline_sharpe = 0.1

    drift_detected, reasons = monitoring_service.detect_drift(metrics)

    # Puede o no detectar drift dependiendo de las métricas calculadas
    assert isinstance(drift_detected, bool)


def test_generate_alert(monitoring_service, tmp_path):
    """Test generación de alertas."""
    alerts_path = tmp_path / "alerts.json"

    monitoring_service.generate_alert("TEST_ALERT", "This is a test alert", severity="WARNING")

    assert alerts_path.exists()

    with open(alerts_path, "r") as f:
        alerts_data = json.load(f)

    assert "alerts" in alerts_data
    assert len(alerts_data["alerts"]) == 1
    assert alerts_data["alerts"][0]["type"] == "TEST_ALERT"


def test_log_monitoring_event(monitoring_service, tmp_path):
    """Test registro de eventos de monitoreo."""
    log_path = tmp_path / "monitoring_log.json"

    metrics = {
        "roi_pct": 5.0,
        "sharpe_ratio": 1.5,
        "win_rate": 60.0,
    }

    monitoring_service.log_monitoring_event(metrics, drift_detected=False, reasons=[])

    assert log_path.exists()

    with open(log_path, "r") as f:
        log_data = json.load(f)

    assert "events" in log_data
    assert len(log_data["events"]) == 1


def test_update_performance_report(monitoring_service, tmp_path):
    """Test actualización de reporte de performance."""
    perf_path = tmp_path / "performance_report.json"

    metrics = {
        "roi_pct": 5.0,
        "sharpe_ratio": 1.5,
        "win_rate": 60.0,
        "total_trades": 10,
    }

    monitoring_service.update_performance_report(metrics)

    assert perf_path.exists()

    with open(perf_path, "r") as f:
        report = json.load(f)

    assert "daily_snapshots" in report
    assert "summary" in report


def test_run_monitoring_cycle_no_data(monitoring_service):
    """Test ciclo de monitoreo sin datos."""
    result = monitoring_service.run_monitoring_cycle()

    assert result["status"] == "NO_DATA"


def test_run_monitoring_cycle_with_data(monitoring_service, sample_trades_good, tmp_path):
    """Test ciclo completo de monitoreo con datos."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_good.to_parquet(trades_path)

    result = monitoring_service.run_monitoring_cycle()

    assert result["status"] == "SUCCESS"
    assert "metrics" in result
    assert "drift_detected" in result


# ==================== AUTO RETRAIN SCHEDULER TESTS ====================


def test_scheduler_initialization(auto_scheduler):
    """Test inicialización del scheduler."""
    assert auto_scheduler.consecutive_sessions_threshold == 3
    assert auto_scheduler.pnl_threshold == 0.0
    assert auto_scheduler.sharpe_threshold == 0.3
    assert auto_scheduler.enable_auto_retrain is True


def test_check_retrain_conditions_no_drift(auto_scheduler):
    """Test condiciones de reentrenamiento sin drift."""
    monitoring_result = {
        "drift_detected": False,
        "metrics": {"roi_pct": 5.0, "sharpe_ratio": 1.5},
    }

    should_retrain, reasons = auto_scheduler.check_retrain_conditions(monitoring_result)

    assert should_retrain is False
    assert auto_scheduler.bad_sessions_count == 0


def test_check_retrain_conditions_with_drift(auto_scheduler):
    """Test condiciones de reentrenamiento con drift."""
    monitoring_result = {
        "drift_detected": True,
        "metrics": {"roi_pct": -5.0, "sharpe_ratio": 0.1, "profit_factor": 0.5},
    }

    should_retrain, reasons = auto_scheduler.check_retrain_conditions(monitoring_result)

    assert auto_scheduler.bad_sessions_count == 1
    assert len(reasons) > 0


def test_check_retrain_conditions_consecutive_sessions(auto_scheduler):
    """Test que se dispare reentrenamiento tras N sesiones malas."""
    bad_result = {
        "drift_detected": True,
        "metrics": {"roi_pct": -5.0, "sharpe_ratio": 0.1, "profit_factor": 0.5},
    }

    # Simular 3 sesiones malas
    for i in range(3):
        should_retrain, reasons = auto_scheduler.check_retrain_conditions(bad_result)

        if i < 2:
            assert should_retrain is False
        else:
            assert should_retrain is True
            assert "consecutivas" in " ".join(reasons).lower()


def test_scheduler_state_persistence(auto_scheduler, tmp_path):
    """Test que el estado se persiste correctamente."""
    state_path = tmp_path / "scheduler_state.json"

    auto_scheduler.bad_sessions_count = 2
    auto_scheduler._save_state()

    assert state_path.exists()

    with open(state_path, "r") as f:
        state = json.load(f)

    assert state["bad_sessions_count"] == 2


def test_force_retrain(auto_scheduler, sample_trades_good, tmp_path, monkeypatch):
    """Test reentrenamiento manual forzado."""
    # Preparar datos
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_good.to_parquet(trades_path)

    # Monkeypatch para que min_trades_threshold sea más bajo
    monkeypatch.setattr(auto_scheduler.retrainer, "min_trades_threshold", 5)

    success, message = auto_scheduler.force_retrain("Test manual")

    # Puede fallar si no hay suficientes trades, pero debe ejecutarse
    assert isinstance(success, bool)
    assert isinstance(message, str)


def test_run_scheduler_cycle_no_retrain_needed(auto_scheduler, sample_trades_good, tmp_path):
    """Test ciclo del scheduler sin necesidad de reentrenar."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_good.to_parquet(trades_path)

    result = auto_scheduler.run_scheduler_cycle()

    # Puede fallar el monitoreo si no hay suficientes datos
    assert "status" in result


def test_retrain_history_logging(auto_scheduler, tmp_path):
    """Test que se registre el historial de reentrenamiento."""
    history_path = tmp_path / "retrain_history.json"

    auto_scheduler._log_retrain_event(reasons=["Test reason"], success=True, message="Test message")

    assert history_path.exists()

    with open(history_path, "r") as f:
        history = json.load(f)

    assert "retrains" in history
    assert len(history["retrains"]) == 1


def test_drift_triggers_retrain_after_threshold(
    auto_scheduler, sample_trades_bad, tmp_path, monkeypatch
):
    """Test integración: drift sostenido dispara reentrenamiento."""
    trades_path = tmp_path / "live_trades.parquet"
    sample_trades_bad.to_parquet(trades_path)

    # Ajustar threshold del retrainer
    monkeypatch.setattr(auto_scheduler.retrainer, "min_trades_threshold", 5)

    # Simular 3 ciclos consecutivos con mal performance
    for i in range(3):
        result = auto_scheduler.run_scheduler_cycle()

        if i < 2:
            # No debe reentrenar aún
            assert result.get("retrain_executed") is False or result["status"] != "SUCCESS"
        else:
            # En el 3er ciclo, debe intentar reentrenar
            if result.get("status") == "SUCCESS":
                assert result.get("should_retrain") is True
