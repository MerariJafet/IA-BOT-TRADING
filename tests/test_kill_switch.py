import pytest

import src.core.kill_switch as kill_switch


@pytest.fixture(autouse=True)
def reset_flag(tmp_path, monkeypatch):
    flag_path = tmp_path / "kill_switch.flag"
    monkeypatch.setattr(kill_switch, "KILL_SWITCH_FLAG", flag_path)
    return flag_path


def test_kill_switch_not_triggered(reset_flag):
    state = {"pnl_acumulado": -0.01, "drawdown_relativo": -0.02}
    assert kill_switch.check_kill_conditions(state, pnl=0.0, errors_last_minute=0) is False


def test_kill_switch_triggers_on_threshold(reset_flag):
    state = {"pnl_acumulado": -0.08, "drawdown_relativo": -0.1}
    assert kill_switch.check_kill_conditions(state, pnl=-10.0, errors_last_minute=0) is True


def test_kill_switch_triggers_on_errors(reset_flag):
    state = {"pnl_acumulado": 0.0, "drawdown_relativo": -0.01}
    assert kill_switch.check_kill_conditions(state, pnl=0.0, errors_last_minute=3) is True


def test_kill_switch_triggers_on_flag(reset_flag):
    reset_flag.touch()
    assert kill_switch.check_kill_conditions({}, pnl=0.0, errors_last_minute=0) is True
