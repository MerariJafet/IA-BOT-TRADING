from datetime import datetime, timedelta

from src.core.watchdog import Watchdog


def test_watchdog_is_stale(tmp_path):
    heartbeat = tmp_path / "heartbeat.txt"
    log_file = tmp_path / "watchdog.log"
    watchdog = Watchdog(heartbeat_file=heartbeat, log_file=log_file, stale_after=timedelta(seconds=2))

    watchdog.record_heartbeat()
    assert watchdog.is_stale() is False

    old_timestamp = (datetime.utcnow() - timedelta(seconds=5)).isoformat()
    heartbeat.write_text(old_timestamp, encoding="utf-8")

    assert watchdog.is_stale() is True
