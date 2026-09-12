"""Runtime deadlines use controlled clocks and budgets, not deployment values."""

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_terminal.execution import manager as module
from open_terminal.execution.policy import ExecutionPolicy, Limits


@pytest.fixture
def running(tmp_path, monkeypatch):
    unit = Limits(100, 4096, 4)
    policy = ExecutionPolicy(
        tmp_path,
        unit,
        unit,
        unit * 2,
        unit,
        2,
        1,
        max_runtime=2,
        terminate_grace=1,
        cleanup_timeout=0.1,
    )
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "time", lambda: 1000 + clock.now - 100)
    monkeypatch.setattr(module.Launch, "_ensure_reaper", lambda self: None)
    tree = Mock()
    tree.is_empty.return_value = False
    tree.task_usage.return_value = {"memory_events": {"oom_kill": 0}}
    tree.kill.side_effect = lambda path: setattr(tree.is_empty, "return_value", True)
    manager = module.Manager(policy, tree)
    launch = module.Launch(
        manager, ("owner", "task"), tmp_path, ["/bin/true"], 10001, 10001, "/", {}
    )
    manager._tasks["task"] = launch
    manager._owners["owner"] = {"task"}
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    os.write(launch._status_write, b"READY 100000000000 1000000000000\n")
    launch.started(process)
    yield launch, tree, clock, process
    launch.abort()


def test_deadline_terminates_then_kills_after_grace(running):
    launch, tree, clock, _process = running
    assert launch.snapshot()["deadline"] == 1002
    clock.now = 101.9
    assert not launch._advance()
    tree.terminate.assert_not_called()
    clock.now = 102
    assert not launch._advance()
    tree.terminate.assert_called_once_with(launch.path)
    assert launch.snapshot()["state"] == "stopping"
    assert launch.end_reason == "timed_out"
    assert launch.manager._tasks
    clock.now = 102.9
    assert not launch._advance()
    tree.kill.assert_not_called()
    clock.now = 103
    assert launch._advance()
    tree.kill.assert_called_once_with(launch.path)
    assert launch.snapshot()["state"] == "finished"
    assert launch.finished_at == 1003
    assert not launch.manager._tasks
    launch.stop(reason="cancelled")
    assert launch.end_reason == "timed_out"
    tree.remove_task.assert_called_once()


def test_clean_exit_during_grace_releases_early(running):
    launch, tree, clock, process = running
    clock.now = 102
    launch._advance()
    tree.is_empty.return_value = True
    process.poll.return_value = 0
    clock.now = 102.1
    assert launch._advance()
    assert launch.end_reason == "timed_out"
    process.kill.assert_not_called()


@pytest.mark.parametrize("oom,reason", [(0, "completed"), (1, "oom")])
def test_natural_exit_and_oom_are_not_timeouts(running, oom, reason):
    launch, tree, _clock, process = running
    tree.task_usage.return_value = {"memory_events": {"oom_kill": oom}}
    process.poll.return_value = -9 if oom else 0
    assert launch._advance()
    assert launch.end_reason == reason
    tree.terminate.assert_not_called()


def test_force_stop_preserves_cancellation_reason(running):
    launch, tree, _clock, _process = running
    launch.stop(force=True)
    assert launch.end_reason == "cancelled"
    assert launch.snapshot()["state"] == "finished"
    tree.terminate.assert_not_called()


def test_wall_clock_changes_do_not_extend_runtime(running, monkeypatch):
    launch, _tree, clock, _process = running
    monkeypatch.setattr(module.time, "time", lambda: 1)
    clock.now = 102
    launch._advance()
    assert launch.end_reason == "timed_out"
    assert launch.deadline == 1002


def test_delayed_handshake_uses_launcher_start_time(running, tmp_path):
    previous, _tree, clock, _ = running
    previous.abort()
    clock.now = 110
    launch = module.Launch(
        previous.manager, ("other", "other"), tmp_path, ["true"], 10001, 10001, "/", {}
    )
    process = Mock()
    process.poll.return_value = None
    os.write(launch._status_write, b"READY 100000000000 1000000000000\n")
    launch.started(process)
    try:
        assert launch._deadline_monotonic == 102
        launch._advance()
        assert launch.end_reason == "timed_out"
    finally:
        launch.abort()


@pytest.mark.parametrize("field", ["max_runtime", "terminate_grace"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_deadline_configuration(running, field, value):
    launch, *_ = running
    with pytest.raises(ValueError):
        replace(launch.manager.policy, **{field: value})


def test_real_process_ignoring_term_is_killed_at_deadline(tmp_path):
    import signal
    import subprocess
    import sys
    import time

    unit = Limits(100, 4096, 4)
    policy = ExecutionPolicy(
        tmp_path,
        unit,
        unit,
        unit * 2,
        unit,
        2,
        1,
        max_runtime=0.15,
        terminate_grace=0.1,
        cleanup_timeout=1,
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); time.sleep(30)',
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    tree = Mock()
    tree.task_usage.return_value = {"memory_events": {"oom_kill": 0}}
    tree.is_empty.side_effect = lambda _: process.poll() is not None
    tree.terminate.side_effect = lambda _: os.killpg(process.pid, signal.SIGTERM)

    def kill(_):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    tree.kill.side_effect = kill
    manager = module.Manager(policy, tree)
    launch = module.Launch(
        manager, ("owner", "task"), tmp_path, ["true"], 10001, 10001, "/", {}
    )
    manager._tasks["task"] = launch
    manager._owners["owner"] = {"task"}
    try:
        assert process.stdout.readline() == b"ready\n"
        os.write(
            launch._status_write,
            f"READY {time.monotonic_ns()} {time.time_ns()}\n".encode(),
        )
        launch.started(process)
        assert process.wait(timeout=3) == -signal.SIGKILL
        end = time.monotonic() + 2
        while launch.snapshot()["state"] != "finished" and time.monotonic() < end:
            time.sleep(0.01)
        assert launch.snapshot()["state"] == "finished"
        assert launch.end_reason == "timed_out"
        assert not manager._tasks
        tree.terminate.assert_called_once()
    finally:
        process.kill()
        process.wait()
        process.stdout.close()
        launch.abort()


def test_timeout_keeps_reservation_when_cgroup_cannot_be_emptied(running):
    launch, tree, clock, _process = running
    clock.now = 102
    launch._advance()
    clock.now = 103
    tree.kill.side_effect = None

    def still_populated(_):
        clock.now += 0.05
        return False

    tree.is_empty.side_effect = still_populated
    with pytest.raises(RuntimeError, match="reservation retained"):
        launch._advance()
    assert launch.end_reason == "timed_out"
    assert launch.snapshot()["state"] == "stopping"
    assert launch.manager._tasks
    tree.remove_task.assert_not_called()
    tree.is_empty.side_effect = None
    tree.is_empty.return_value = True
    assert launch._advance()
    assert not launch.manager._tasks
    tree.remove_task.assert_called_once()


def test_output_cleanup_does_not_shorten_descendant_grace(running, monkeypatch):
    launch, tree, clock, process = running
    clock.now = 102
    launch._advance()
    process.poll.return_value = 0
    killed_at = []

    def kill(_):
        killed_at.append(clock.now)
        tree.is_empty.return_value = True

    tree.kill.side_effect = kill
    monkeypatch.setattr(
        module.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    launch.stop(reason="completed")
    assert killed_at[0] >= 103
    assert launch.end_reason == "timed_out"
