from datetime import datetime

import schedule

from src.core.scheduler import SchedulerService


def test_scheduler_runs_all_jobs(tmp_path):
    heartbeat_file = tmp_path / "heartbeat.txt"
    counters = {"heartbeat": 0, "kill": 0, "watchdog": 0, "report": 0}

    def heartbeat():
        counters["heartbeat"] += 1
        heartbeat_file.write_text("beat", encoding="utf-8")

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
        scheduler_instance=schedule.Scheduler(),
    )
    scheduler.start()

    for job in scheduler.jobs().values():
        job.next_run = datetime.now()

    scheduler.run_pending()

    assert heartbeat_file.exists()
    assert counters == {"heartbeat": 1, "kill": 1, "watchdog": 1, "report": 1}


def test_scheduler_trigger_specific_job():
    counters = {"kill": 0}

    def kill_switch_job():
        counters["kill"] += 1
        return False

    scheduler = SchedulerService(
        heartbeat_fn=lambda: None,
        kill_switch_fn=kill_switch_job,
        watchdog_check_fn=lambda: False,
        daily_report_fn=lambda: None,
        scheduler_instance=schedule.Scheduler(),
    )
    scheduler.start()

    scheduler.trigger("kill_switch")

    assert counters["kill"] == 1
